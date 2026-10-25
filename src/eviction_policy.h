#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "cache_entry.h"

namespace nivo::detail {
// Reads record recency without locking or editing queue links. Maintenance
// materializes exact LRU order only under pressure; FIFO ignores read accesses.
// This deliberately trades O(n log n) victim ordering on the slow path for an
// allocation-free, queue-lock-free hit path, without approximating LRU.
class EvictionQueue final {
 public:
  explicit EvictionQueue(EvictionPolicy policy) : policy_(policy) {}

  void record_access(Entry& entry) noexcept {
    if (policy_ == EvictionPolicy::lru) stamp(entry);
  }
  void insert(Entry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    stamp(entry);
    entry.policy_previous = nullptr;
    entry.policy_next = head_;
    if (head_) head_->policy_previous = &entry;
    head_ = &entry;
    ++size_;
  }
  void remove(Entry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    unlink(entry);
  }
  // Unlinks a chain of entries detached from the index, linked through
  // hash_next, under one lock acquisition.
  void remove_chain(Entry* chain) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) unlink(*chain);
  }
  // Compaction preserves logical recency and does not count as an access.
  void relocate(Entry& old, Entry& replacement) {
    std::lock_guard<std::mutex> lock(mutex_);
    replacement.access_sequence.store(old.access_sequence.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
    replacement.policy_previous = old.policy_previous;
    replacement.policy_next = old.policy_next;
    if (old.policy_previous) old.policy_previous->policy_next = &replacement;
    else head_ = &replacement;
    if (old.policy_next) old.policy_next->policy_previous = &replacement;
    old.policy_previous = old.policy_next = nullptr;
  }
  // Raw snapshots require the caller's exclusive pool maintenance guard. It
  // prevents publication, removal, relocation and new read accesses until done.
  std::vector<Entry*> entries(bool oldest_first = false) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Entry*> result;
    result.reserve(size_);
    for (Entry* entry = head_; entry; entry = entry->policy_next) result.push_back(entry);
    if (oldest_first) {
      std::sort(result.begin(), result.end(), [](const Entry* left, const Entry* right) {
        return left->access_sequence.load(std::memory_order_relaxed) <
               right->access_sequence.load(std::memory_order_relaxed);
      });
    }
    return result;
  }
  void clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    while (head_) unlink(*head_);
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
  const EvictionPolicy policy_;
  std::atomic<std::uint64_t> sequence_{0};
  mutable std::mutex mutex_;
  Entry* head_ = nullptr;
  std::size_t size_ = 0;
};
}  // namespace nivo::detail
