#pragma once

#include <algorithm>
#include <cassert>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include "cache_index.h"
#include "nivo/status.h"

namespace nivo::detail {
// A selection owns only stable identities, never entries, key views or pins.
// Each candidate is returned at most once, even if the caller rejects it.
// Reaching the end is final: pressure must not restart an exhausted selection.
class VictimSelection final {
 public:
  struct Candidate {
    EntryId id;
    std::uint64_t order;
  };

  explicit VictimSelection(std::vector<Candidate> candidates)
      : candidates_(std::move(candidates)) {}
  VictimSelection(VictimSelection&&) noexcept = default;
  VictimSelection& operator=(VictimSelection&&) noexcept = default;
  VictimSelection(const VictimSelection&) = delete;
  VictimSelection& operator=(const VictimSelection&) = delete;

  [[nodiscard]] std::optional<EntryId> next() noexcept {
    if (position_ >= candidates_.size()) return std::nullopt;
    return candidates_[position_++].id;
  }

 private:
  std::vector<Candidate> candidates_;
  std::size_t position_ = 0;
};

// Compile-time implementations share intrusive storage, but not a runtime
// algorithm branch. Only live indexed entries are linked. Links are repaired
// synchronously during relocation; external selections never retain them.
template <bool RefreshOnAccess>
class OrderedEvictionPolicy final {
 public:
  void record_access(Entry& entry) noexcept {
    if constexpr (RefreshOnAccess) stamp(entry);
  }
  void insert(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(entry.version != invalid_version);
    stamp(entry);
    entry.policy_previous = nullptr;
    entry.policy_next = head_;
    if (head_) head_->policy_previous = &entry;
    head_ = &entry;
    ++size_;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(size_ != 0 && (entry.policy_previous || head_ == &entry));
    unlink(entry);
  }
  // Unlinks a chain of entries detached from the index, linked through
  // hash_next, under one lock acquisition.
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) unlink(*chain);
  }
  // Compaction preserves logical recency and does not count as an access.
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(old.version == replacement.version);
    assert(size_ != 0 && (old.policy_previous || head_ == &old));
    replacement.access_sequence.store(old.access_sequence.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
    replacement.policy_previous = old.policy_previous;
    replacement.policy_next = old.policy_next;
    if (old.policy_previous) old.policy_previous->policy_next = &replacement;
    else head_ = &replacement;
    if (old.policy_next) old.policy_next->policy_previous = &replacement;
    old.policy_previous = old.policy_next = nullptr;
  }
  // Exclusive pool maintenance freezes membership and accesses while taking
  // this snapshot. LRU is exact at that boundary, not sampled or buffered.
  // FIFO orders publication only. Both preserve the existing ordering rules.
  [[nodiscard]] VictimSelection select_victims() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<VictimSelection::Candidate> candidates;
    candidates.reserve(size_);
    for (Entry* entry = head_; entry; entry = entry->policy_next)
      candidates.push_back({CacheIndex::identity(*entry),
                            entry->access_sequence.load(std::memory_order_relaxed)});
    std::sort(candidates.begin(), candidates.end(),
        [](const VictimSelection::Candidate& left, const VictimSelection::Candidate& right) {
          return left.order < right.order;
        });
    return VictimSelection(std::move(candidates));
  }

 private:
  void stamp(Entry& entry) noexcept {
    const std::uint64_t next = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::uint64_t current = entry.access_sequence.load(std::memory_order_relaxed);
    // A delayed reader must not overwrite a more recent reader's timestamp.
    while (current < next && !entry.access_sequence.compare_exchange_weak(
        current, next, std::memory_order_relaxed)) {}
  }
  void unlink(Entry& entry) noexcept {
    if (entry.policy_previous) entry.policy_previous->policy_next = entry.policy_next;
    else head_ = entry.policy_next;
    if (entry.policy_next) entry.policy_next->policy_previous = entry.policy_previous;
    entry.policy_previous = entry.policy_next = nullptr;
    --size_;
  }
  std::atomic<std::uint64_t> sequence_{0};
  mutable std::mutex mutex_;
  Entry* head_ = nullptr;
  std::size_t size_ = 0;
};

using LruEvictionPolicy = OrderedEvictionPolicy<true>;
using FifoEvictionPolicy = OrderedEvictionPolicy<false>;

// Per-pool runtime selection with inline, allocation-free access dispatch.
// Adding an algorithm changes this adapter/configuration, not CachePool's
// ownership, expiry, compaction or quota logic. There are no user callbacks.
//
// Lock order: maintenance -> index stripe -> policy. Selection takes only the
// policy lock and releases it before Core resolves candidates in the index.
// insert/remove/relocate run at the corresponding index commit boundary and
// do not allocate. Only select_victims may allocate (on the pressure path).
class EvictionPolicyEngine final {
 public:
  [[nodiscard]] static Status validate(EvictionPolicy policy) noexcept {
    switch (policy) {
      case EvictionPolicy::lru:
      case EvictionPolicy::fifo: return Status::ok;
    }
    return Status::invalid_argument;
  }

  explicit EvictionPolicyEngine(EvictionPolicy policy) {
    switch (policy) {
      case EvictionPolicy::lru: break;
      case EvictionPolicy::fifo: policy_.emplace<FifoEvictionPolicy>(); break;
      default: throw std::invalid_argument("Invalid eviction policy");
    }
  }

  void record_access(Entry& entry) noexcept {
    // A reader may already have pinned an entry when another thread removes
    // it. Only its lifetime-bound atomic metadata may be touched here, never
    // queue links or membership. A late access cannot resurrect an old version.
    std::visit([&entry]<typename Policy>(Policy& policy) noexcept -> void {
      policy.record_access(entry);
    }, policy_);
  }

  void insert(Entry& entry) noexcept {
    std::visit([&entry]<typename Policy>(Policy& policy) noexcept -> void {
      policy.insert(entry);
    }, policy_);
  }
  void remove(Entry& entry) noexcept {
    std::visit([&entry]<typename Policy>(Policy& policy) noexcept -> void {
      policy.remove(entry);
    }, policy_);
  }
  void remove_chain(Entry* chain) noexcept {
    std::visit([chain]<typename Policy>(Policy& policy) noexcept -> void {
      policy.remove_chain(chain);
    }, policy_);
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::visit([&old, &replacement]<typename Policy>(Policy& policy) noexcept -> void {
      policy.relocate(old, replacement);
    }, policy_);
  }
  [[nodiscard]] VictimSelection select_victims() const {
    return std::visit([]<typename Policy>(const Policy& policy) -> VictimSelection {
      return policy.select_victims();
    }, policy_);
  }

 private:
  std::variant<LruEvictionPolicy, FifoEvictionPolicy> policy_;
};
}  // namespace nivo::detail
