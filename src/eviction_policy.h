#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

#include "cache_index.h"
#include "nivo/status.h"

namespace nivo::detail {
// Fields reserved in every Entry for whichever policy tracks it.
inline constexpr std::size_t policy_entry_bytes =
    sizeof(Entry::policy_previous) + sizeof(Entry::policy_next) + sizeof(Entry::policy_state);

// Policy contract shared by every algorithm below:
//  - record_access runs for find() hits under the shared pool guard, possibly
//    on a version another thread has already removed (a reader may still pin
//    it). It must never relink such an entry or resurrect it.
//  - publish/remove/remove_chain/relocate run at the matching index commit
//    under the stripe lock and never allocate, except that publish may grow a
//    side table; it then returns false without changing anything.
//  - Selection (begin_selection/next_victim/end_selection) runs only under the
//    exclusive pool guard. Nothing else can publish, access or select then, so
//    a policy walks its own structure and only has to repair its cursor when
//    Core removes a returned candidate. Each step is O(1) amortized; no policy
//    copies or sorts its members.

// Doubly linked list over policy_previous/policy_next. The head is the most
// recently placed entry; the tail is considered for eviction first.
class PolicyList final {
 public:
  bool empty() const noexcept { return head_ == nullptr; }
  Entry* head() const noexcept { return head_; }
  Entry* tail() const noexcept { return tail_; }
  void push_head(Entry& entry) noexcept {
    entry.policy_previous = nullptr;
    entry.policy_next = head_;
    if (head_) head_->policy_previous = &entry;
    else tail_ = &entry;
    head_ = &entry;
  }
  void unlink(Entry& entry) noexcept {
    if (entry.policy_previous) entry.policy_previous->policy_next = entry.policy_next;
    else head_ = entry.policy_next;
    if (entry.policy_next) entry.policy_next->policy_previous = entry.policy_previous;
    else tail_ = entry.policy_previous;
    entry.policy_previous = entry.policy_next = nullptr;
  }
  // The replacement takes the exact position of old.
  void substitute(Entry& old, Entry& replacement) noexcept {
    replacement.policy_previous = old.policy_previous;
    replacement.policy_next = old.policy_next;
    if (old.policy_previous) old.policy_previous->policy_next = &replacement;
    else head_ = &replacement;
    if (old.policy_next) old.policy_next->policy_previous = &replacement;
    else tail_ = &replacement;
    old.policy_previous = old.policy_next = nullptr;
  }
  // Appends other on the tail side, leaving it empty.
  void splice_tail(PolicyList& other) noexcept {
    if (other.empty()) return;
    if (empty()) {
      head_ = other.head_;
    } else {
      tail_->policy_next = other.head_;
      other.head_->policy_previous = tail_;
    }
    tail_ = other.tail_;
    other.head_ = other.tail_ = nullptr;
  }

 private:
  Entry* head_ = nullptr;
  Entry* tail_ = nullptr;
};

// LRU (refresh on access) and FIFO (publication order) share one exact list.
// LRU moves a found entry to the head under the policy lock: O(1), no search,
// and exact at every selection. policy_state is 1 while the entry is linked,
// which is how a late access to a removed version is ignored.
template <bool RefreshOnAccess>
class OrderedEvictionPolicy final {
 public:
  void record_access(Entry& entry) noexcept {
    if constexpr (RefreshOnAccess) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (entry.policy_state.load(std::memory_order_relaxed) != linked || list_.head() == &entry) return;
      list_.unlink(entry);
      list_.push_head(entry);
    }
  }
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) detach(*replaced);
    else ++size_;
    entry.policy_state.store(linked, std::memory_order_relaxed);
    list_.push_head(entry);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry);
    --size_;
  }
  // Unlinks entries detached from the index, chained through hash_next.
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) {
      detach(*chain);
      --size_;
    }
  }
  // Compaction preserves the position and does not count as an access.
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(old.policy_state.load(std::memory_order_relaxed) == linked);
    if (cursor_ == &old) cursor_ = &replacement;
    replacement.policy_state.store(linked, std::memory_order_relaxed);
    old.policy_state.store(0, std::memory_order_relaxed);
    list_.substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cursor_ = list_.tail();
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry* entry = cursor_;
    if (!entry) return std::nullopt;
    cursor_ = entry->policy_previous;
    return CacheIndex::identity(*entry);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cursor_ = nullptr;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ * policy_entry_bytes;
  }

 private:
  static constexpr std::uint64_t linked = 1;
  void detach(Entry& entry) noexcept {
    assert(entry.policy_state.load(std::memory_order_relaxed) == linked && size_ != 0);
    if (cursor_ == &entry) cursor_ = entry.policy_previous;
    entry.policy_state.store(0, std::memory_order_relaxed);
    list_.unlink(entry);
  }
  mutable std::mutex mutex_;
  PolicyList list_;
  Entry* cursor_ = nullptr;
  std::size_t size_ = 0;
};

using LruEvictionPolicy = OrderedEvictionPolicy<true>;
using FifoEvictionPolicy = OrderedEvictionPolicy<false>;

// Shared intrusive segment accounting. Callers serialize operations. Selection
// cursors advance before returning a victim and survive deletion/relocation.
class PolicySegments {
 public:
  struct Snapshot {
    std::array<std::size_t, 3> entries;
    std::array<std::size_t, 3> bytes;
  };
  Snapshot snapshot() const noexcept { return {counts_, bytes_}; }
 protected:
  static std::size_t share(std::size_t bytes, std::size_t percent) noexcept {
    return bytes / 100 * percent + bytes % 100 * percent / 100;
  }
  static std::size_t segment(const Entry& entry) noexcept {
    return static_cast<std::size_t>(entry.policy_state.load(std::memory_order_relaxed) & 3) - 1;
  }
  std::size_t total_bytes() const noexcept { return bytes_[0] + bytes_[1] + bytes_[2]; }
  std::size_t count() const noexcept { return counts_[0] + counts_[1] + counts_[2]; }
  void attach(Entry& entry, std::size_t queue) noexcept {
    entry.policy_state.store(queue + 1, std::memory_order_relaxed);
    lists_[queue].push_head(entry);
    bytes_[queue] += entry.charge;
    ++counts_[queue];
  }
  void detach(Entry& entry) noexcept {
    const std::size_t queue = segment(entry);
    assert(queue < 3 && counts_[queue]);
    if (cursors_[queue] == &entry) cursors_[queue] = entry.policy_previous;
    lists_[queue].unlink(entry);
    bytes_[queue] -= entry.charge;
    --counts_[queue];
    entry.policy_state.store(0, std::memory_order_relaxed);
  }
  void move(Entry& entry, std::size_t queue) noexcept { detach(entry); attach(entry, queue); }
  void substitute(Entry& old, Entry& replacement) noexcept {
    const std::size_t queue = segment(old);
    replacement.policy_state.store(old.policy_state.load(std::memory_order_relaxed), std::memory_order_relaxed);
    old.policy_state.store(0, std::memory_order_relaxed);
    if (cursors_[queue] == &old) cursors_[queue] = &replacement;
    lists_[queue].substitute(old, replacement);
    bytes_[queue] = bytes_[queue] - old.charge + replacement.charge;
  }
  void start_selection() noexcept {
    selecting_ = true;
    for (std::size_t queue = 0; queue < 3; ++queue) cursors_[queue] = lists_[queue].tail();
  }
  std::optional<EntryId> take(std::size_t queue) noexcept {
    Entry* entry = cursors_[queue];
    if (!entry) return std::nullopt;
    cursors_[queue] = entry->policy_previous;
    return CacheIndex::identity(*entry);
  }
  void stop_selection() noexcept { selecting_ = false; cursors_.fill(nullptr); }
  std::array<PolicyList, 3> lists_;
  std::array<std::size_t, 3> counts_{};
  std::array<std::size_t, 3> bytes_{};
  std::array<Entry*, 3> cursors_{};
  bool selecting_ = false;
};

// Exact SLRU: probation (0) and protected (1). A probation hit promotes;
// excess protected bytes demote its LRU entry to probation's MRU position.
class SlruEvictionPolicy final : private PolicySegments {
 public:
  explicit SlruEvictionPolicy(const SlruConfig& config, std::size_t capacity) noexcept
      : capacity_(capacity), protected_percent_(config.protected_percent) {}
  void record_access(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entry.policy_state.load(std::memory_order_relaxed)) return;
    move(entry, 1);
    balance();
  }
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) substitute(*replaced, entry);
    else attach(entry, 0);
    balance();
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry);
    if (!selecting_) balance();
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) detach(*chain);
    if (!selecting_) balance();
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    start_selection();
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return take(cursors_[0] ? 0 : 1);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_selection();
    balance();
  }
  void set_capacity(std::size_t capacity) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    balance();
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return count() * policy_entry_bytes;
  }
  Snapshot snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return PolicySegments::snapshot();
  }
 private:
  void balance() noexcept {
    const std::size_t limit = share(capacity_, protected_percent_);
    while (bytes_[1] > limit) move(*lists_[1].tail(), 0);
  }
  mutable std::mutex mutex_;
  std::size_t capacity_;
  const std::size_t protected_percent_;
};

// Fixed storage FIFO of full-width key hashes with intrusive bucket chains.
// Nodes are recycled without allocations, tombstones, or unbounded history.
class TwoQGhost final {
 public:
  explicit TwoQGhost(std::size_t capacity)
      : nodes_(capacity), buckets_(capacity ? std::bit_ceil(capacity) : 0, none) {
    for (std::size_t i = 0; i < capacity; ++i) nodes_[i].next = i + 1 < capacity ? i + 1 : none;
    free_ = capacity ? 0 : none;
  }
  bool contains(std::size_t hash) const noexcept {
    if (buckets_.empty()) return false;
    for (std::size_t i = buckets_[bucket(hash)]; i != none; i = nodes_[i].hash_next)
      if (nodes_[i].hash == hash) return true;
    return false;
  }
  void clear() noexcept {
    while (tail_ != none) erase(tail_);
  }
  bool take(std::size_t hash) noexcept {
    if (buckets_.empty()) return false;
    for (std::size_t i = buckets_[bucket(hash)]; i != none; i = nodes_[i].hash_next) {
      if (nodes_[i].hash == hash) { erase(i); return true; }
    }
    return false;
  }
  void insert(std::size_t hash) noexcept {
    if (nodes_.empty()) return;
    take(hash);
    if (free_ == none) erase(tail_);
    const std::size_t i = free_;
    Node& node = nodes_[i];
    free_ = node.next;
    node.hash = hash;
    node.previous = none;
    node.next = head_;
    if (head_ != none) nodes_[head_].previous = i;
    else tail_ = i;
    head_ = i;
    const std::size_t b = bucket(hash);
    node.hash_previous = none;
    node.hash_next = buckets_[b];
    if (node.hash_next != none) nodes_[node.hash_next].hash_previous = i;
    buckets_[b] = i;
    ++size_;
  }
  std::size_t size() const noexcept { return size_; }
  std::size_t bytes() const noexcept { return nodes_.capacity() * sizeof(Node) + buckets_.capacity() * sizeof(std::size_t); }
 private:
  static constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
  struct Node {
    std::size_t hash = 0;
    std::size_t previous = none;
    std::size_t next = none;
    std::size_t hash_previous = none;
    std::size_t hash_next = none;
  };
  std::size_t bucket(std::size_t hash) const noexcept { return hash & (buckets_.size() - 1); }
  void erase(std::size_t i) noexcept {
    Node& node = nodes_[i];
    if (node.previous != none) nodes_[node.previous].next = node.next;
    else head_ = node.next;
    if (node.next != none) nodes_[node.next].previous = node.previous;
    else tail_ = node.previous;
    if (node.hash_previous != none) nodes_[node.hash_previous].hash_next = node.hash_next;
    else buckets_[bucket(node.hash)] = node.hash_next;
    if (node.hash_next != none) nodes_[node.hash_next].hash_previous = node.hash_previous;
    node.next = free_;
    free_ = i;
    --size_;
  }
  std::vector<Node> nodes_;
  std::vector<std::size_t> buckets_;
  std::size_t free_ = none;
  std::size_t head_ = none;
  std::size_t tail_ = none;
  std::size_t size_ = 0;
};

class TwoQEvictionPolicy final : private PolicySegments {
 public:
  explicit TwoQEvictionPolicy(const TwoQConfig& config)
      : ghost_(config.ghost_capacity), a1in_percent_(config.a1in_percent) {}
  static constexpr std::uint64_t history_hint = std::uint64_t{1} << 63;
  std::uint64_t prepare_insert(std::string_view key) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return ghost_.contains(CacheIndex::hash(key)) ? history_hint : 0;
  }
  void reset_history() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ghost_.clear();
  }
  void record_access(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entry.policy_state.load(std::memory_order_relaxed) == 2) move(entry, 1);
  }
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) substitute(*replaced, entry);
    else {
      const bool remembered = ghost_.take(CacheIndex::hash(entry.key));
      const bool hint = (entry.policy_state.load(std::memory_order_relaxed) & history_hint) != 0;
      attach(entry, remembered || hint ? 1 : 0);
    }
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (&entry == pending_ && segment(entry) == 0) ghost_.insert(CacheIndex::hash(entry.key));
    pending_ = nullptr;
    detach(entry);
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) {
      if (pending_ == chain) pending_ = nullptr;
      detach(*chain);
    }
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_ == &old) pending_ = &replacement;
    substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    start_selection();
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool from_in = cursors_[0] && (!cursors_[1] || bytes_[0] > share(total_bytes(), a1in_percent_));
    const std::size_t queue = from_in ? 0 : 1;
    pending_ = cursors_[queue];
    return take(queue);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = nullptr;
    stop_selection();
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return count() * policy_entry_bytes + ghost_.bytes();
  }
  Snapshot snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return PolicySegments::snapshot();
  }
  std::size_t ghost_size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return ghost_.size();
  }
 private:
  mutable std::mutex mutex_;
  TwoQGhost ghost_;
  const std::size_t a1in_percent_;
  Entry* pending_ = nullptr;
};

// Four independently mixed rows of saturating 8-bit counters. The remaining
// 56 bits hold an epoch. Aging is lazy, including on estimates, so no request
// clears the whole table. A CAS merges concurrent updates and aging. Samples
// are batched through 64 cache-line-separated counters; the shared clock is
// written at most once per 64 requests. There is no estimator mutex.
class TinyLfuSketch final {
 public:
  explicit TinyLfuSketch(const WTinyLfuConfig& config)
      : width_(config.sketch_width), sample_size_(config.sample_size),
        cells_(new std::atomic<std::uint64_t>[4 * width_]{}), clock_(new ClockState) {}
  void record(std::size_t hash) noexcept {
    const std::uint64_t mixed = mix(hash);
    const std::uint64_t local = clock_->samples[mixed & 63].value.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((local & 63) == 0) clock_->requests.fetch_add(64, std::memory_order_relaxed);
    const std::uint64_t epoch = clock_->requests.load(std::memory_order_relaxed) / sample_size_;
    for (std::size_t row = 0; row < 4; ++row) {
      std::atomic<std::uint64_t>& cell = cells_[offset(mixed, row)];
      std::uint64_t old = cell.load(std::memory_order_relaxed);
      for (;;) {
        // Another reader may have advanced the cell while this one paused.
        const std::uint64_t current_epoch = std::max(epoch, old >> 8);
        const std::uint64_t frequency = aged(old, current_epoch);
        const std::uint64_t updated = current_epoch << 8 | std::min<std::uint64_t>(255, frequency + 1);
        if (updated == old || cell.compare_exchange_weak(old, updated, std::memory_order_relaxed)) break;
      }
    }
  }
  // Only under the exclusive pool guard, with all readers stopped.
  void clear() noexcept {
    for (std::size_t i = 0; i < 4 * width_; ++i) cells_[i].store(0, std::memory_order_relaxed);
    for (Samples& sample : clock_->samples) sample.value.store(0, std::memory_order_relaxed);
    clock_->requests.store(0, std::memory_order_relaxed);
  }
  std::uint32_t estimate(std::size_t hash) const noexcept {
    const std::uint64_t mixed = mix(hash);
    const std::uint64_t epoch = clock_->requests.load(std::memory_order_relaxed) / sample_size_;
    std::uint64_t result = 255;
    for (std::size_t row = 0; row < 4; ++row) {
      const std::uint64_t cell = cells_[offset(mixed, row)].load(std::memory_order_relaxed);
      result = std::min(result, aged(cell, std::max(epoch, cell >> 8)));
    }
    return static_cast<std::uint32_t>(result);
  }
  std::size_t bytes() const noexcept { return 4 * width_ * sizeof(std::atomic<std::uint64_t>) + sizeof(ClockState); }
 private:
  static std::uint64_t mix(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }
  std::size_t offset(std::uint64_t hash, std::size_t row) const noexcept {
    return row * width_ + (mix(hash + (row + 1) * 0x9e3779b97f4a7c15ULL) & (width_ - 1));
  }
  static std::uint64_t aged(std::uint64_t cell, std::uint64_t epoch) noexcept {
    const std::uint64_t elapsed = epoch - (cell >> 8);
    return elapsed >= 8 ? 0 : (cell & 255) >> elapsed;
  }
  struct alignas(64) Samples { std::atomic<std::uint64_t> value{0}; };
  const std::size_t width_;
  const std::size_t sample_size_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> cells_;
  struct ClockState {
    std::array<Samples, 64> samples;
    alignas(64) std::atomic<std::uint64_t> requests{0};
  };
  std::unique_ptr<ClockState> clock_;
};

// Window (0), main probation (1), main protected (2). The core allocates before
// publication, so admission happens on the existing window LRU when pressure
// arrives. A winning candidate moves only after Core actually removes its
// victim; rejected/pinned victims cannot accidentally admit it. Large candidates
// may need several wins before fitting in main. Queue shares are soft targets:
// window borrows free main bytes and oversized entries can exceed its target,
// but total resident bytes remain bounded by the core's quota at every step.
class WTinyLfuEvictionPolicy final : private PolicySegments {
 public:
  explicit WTinyLfuEvictionPolicy(const WTinyLfuConfig& config, std::size_t capacity)
      : sketch_(config), capacity_(capacity), window_percent_(config.window_percent),
        protected_percent_(config.protected_percent) {}
  void reset_history() noexcept { sketch_.clear(); }
  void record_request(std::string_view key) noexcept { sketch_.record(CacheIndex::hash(key)); }
  void record_access(Entry& entry) noexcept {
    record_request(entry.key);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entry.policy_state.load(std::memory_order_relaxed)) return;
    const std::size_t queue = segment(entry);
    move(entry, queue == 0 ? 0 : 2);
    balance_protected();
  }
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) substitute(*replaced, entry);
    else attach(entry, 0);
    balance_protected();
    fill_main();
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry* candidate = &entry == pending_victim_ ? pending_candidate_ : nullptr;
    if (&entry == pending_candidate_) pending_candidate_ = nullptr;
    if (&entry == pending_victim_) pending_victim_ = nullptr;
    detach(entry);
    if (candidate && candidate != &entry && main_bytes() <= main_limit() &&
        candidate->charge <= main_limit() - main_bytes()) {
      move(*candidate, 1);
      // Appending at the head is reachable from a nonempty cursor. When the
      // previous victim was the last probation entry, start at the new one.
      if (!cursors_[1]) cursors_[1] = candidate;
      ++admissions_;
    }
    if (!selecting_) balance_protected();
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) {
      if (pending_candidate_ == chain) pending_candidate_ = nullptr;
      if (pending_victim_ == chain) pending_victim_ = nullptr;
      detach(*chain);
    }
    if (!selecting_) balance_protected();
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_candidate_ == &old) pending_candidate_ = &replacement;
    if (pending_victim_ == &old) pending_victim_ = &replacement;
    substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    fill_main();
    start_selection();
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_candidate_ = pending_victim_ = nullptr;
    Entry* candidate = cursors_[0];
    if (candidate && (bytes_[0] >= window_limit() || (!cursors_[1] && !cursors_[2]))) {
      Entry* victim = cursors_[1];
      if (candidate->charge <= main_limit() && victim &&
          sketch_.estimate(CacheIndex::hash(candidate->key)) > sketch_.estimate(CacheIndex::hash(victim->key))) {
        pending_candidate_ = candidate;
        pending_victim_ = victim;
        return take(1);
      }
      ++rejections_;
      return take(0);
    }
    if (cursors_[1]) return take(1);
    if (cursors_[2]) return take(2);
    return take(0);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_candidate_ = pending_victim_ = nullptr;
    stop_selection();
    balance_protected();
    fill_main();
  }
  void set_capacity(std::size_t capacity) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    balance_protected();
    fill_main();
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return count() * policy_entry_bytes + sketch_.bytes();
  }
  Snapshot snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return PolicySegments::snapshot();
  }
  std::uint32_t frequency(std::string_view key) const noexcept { return sketch_.estimate(CacheIndex::hash(key)); }
  std::uint64_t admissions() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return admissions_; }
 private:
  std::size_t window_limit() const noexcept { return std::min(capacity_, std::max<std::size_t>(1, share(capacity_, window_percent_))); }
  std::size_t main_limit() const noexcept { return capacity_ - window_limit(); }
  std::size_t main_bytes() const noexcept { return bytes_[1] + bytes_[2]; }
  void balance_protected() noexcept {
    const std::size_t limit = share(main_limit(), protected_percent_);
    while (bytes_[2] > limit) move(*lists_[2].tail(), 1);
  }
  void fill_main() noexcept {
    while (bytes_[0] > window_limit() && main_bytes() < main_limit()) {
      Entry* candidate = lists_[0].tail();
      if (candidate->charge > main_limit() - main_bytes()) break;
      move(*candidate, 1);
    }
  }
  TinyLfuSketch sketch_;
  mutable std::mutex mutex_;
  std::size_t capacity_;
  const std::size_t window_percent_;
  const std::size_t protected_percent_;
  Entry* pending_candidate_ = nullptr;
  Entry* pending_victim_ = nullptr;
  std::uint64_t admissions_ = 0;
  std::uint64_t rejections_ = 0;
};

// O(1) LFU over one list per frequency level and a bitmask of nonempty levels.
// Insertions start at 1 and finds increment up to max_frequency, so counters
// stay bounded. After aging_window(size) finds, every count is halved: lists
// 2k and 2k+1 merge into list k, O(levels) splices regardless of pool size.
// Entries store (epoch, frequency) and derive the current level lazily as
// frequency >> (epoch_ - epoch), which always names the list holding them.
// Ties within a level go to the least recently placed entry, and after a
// merge the formerly lower half is evicted first, so order is deterministic.
class LfuEvictionPolicy final {
 public:
  static constexpr std::uint32_t levels = 16;
  static constexpr std::uint32_t initial_frequency = 1;
  static constexpr std::uint32_t max_frequency = levels - 1;
  static constexpr std::size_t min_aging_window = 64;
  static constexpr std::size_t aging_window_per_entry = 8;
  static constexpr std::size_t aging_window(std::size_t size) noexcept {
    return std::max(min_aging_window, size * aging_window_per_entry);
  }

  void record_access(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t state = entry.policy_state.load(std::memory_order_relaxed);
    if (!(state & linked)) return;
    const std::uint32_t level = level_of(state);
    detach(entry, level);
    attach(entry, std::min(level + 1, max_frequency));
    if (++accesses_ >= aging_window(size_)) age();
  }
  // A replacement keeps the key's current frequency, so rewriting a hot key
  // does not make it the next victim. It counts as placement, not an access.
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint32_t level = initial_frequency;
    if (replaced) {
      const std::uint64_t state = replaced->policy_state.load(std::memory_order_relaxed);
      level = level_of(state);
      detach(*replaced, level);
    } else {
      ++size_;
    }
    attach(entry, level);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry, level_of(entry.policy_state.load(std::memory_order_relaxed)));
    --size_;
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) {
      detach(*chain, level_of(chain->policy_state.load(std::memory_order_relaxed)));
      --size_;
    }
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t state = old.policy_state.load(std::memory_order_relaxed);
    assert(state & linked);
    if (cursor_ == &old) cursor_ = &replacement;
    replacement.policy_state.store(state, std::memory_order_relaxed);
    old.policy_state.store(0, std::memory_order_relaxed);
    lists_[level_of(state)].substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cursor_level_ = nonempty_ ? static_cast<std::uint32_t>(std::countr_zero(nonempty_)) : levels;
    cursor_ = cursor_level_ < levels ? lists_[cursor_level_].tail() : nullptr;
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!cursor_) {
      if (cursor_level_ + 1 >= levels) return std::nullopt;
      const std::uint32_t higher = nonempty_ >> (cursor_level_ + 1);
      if (!higher) {
        cursor_level_ = levels;
        return std::nullopt;
      }
      cursor_level_ += 1 + static_cast<std::uint32_t>(std::countr_zero(higher));
      cursor_ = lists_[cursor_level_].tail();
    }
    Entry* entry = cursor_;
    cursor_ = entry->policy_previous;
    return CacheIndex::identity(*entry);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    cursor_ = nullptr;
    cursor_level_ = levels;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ * policy_entry_bytes;
  }
  // Current frequency of a linked entry; for tests and diagnostics.
  std::uint32_t frequency(const Entry& entry) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return level_of(entry.policy_state.load(std::memory_order_relaxed));
  }
  std::uint64_t aging_epochs() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return epoch_;
  }

 private:
  // policy_state: epoch << 8 | linked | frequency (low four bits).
  static constexpr std::uint64_t linked = 0x80;
  static constexpr std::uint64_t frequency_mask = 0x0f;
  static constexpr std::uint32_t epoch_shift = 8;
  std::uint32_t level_of(std::uint64_t state) const noexcept {
    const std::uint64_t elapsed = epoch_ - (state >> epoch_shift);
    const std::uint32_t frequency = static_cast<std::uint32_t>(state & frequency_mask);
    return elapsed >= 4 ? 0 : frequency >> elapsed;
  }
  void attach(Entry& entry, std::uint32_t level) noexcept {
    entry.policy_state.store(epoch_ << epoch_shift | linked | level, std::memory_order_relaxed);
    lists_[level].push_head(entry);
    nonempty_ |= 1u << level;
  }
  void detach(Entry& entry, std::uint32_t level) noexcept {
    assert((entry.policy_state.load(std::memory_order_relaxed) & linked) && size_ != 0);
    if (cursor_ == &entry) cursor_ = entry.policy_previous;
    entry.policy_state.store(0, std::memory_order_relaxed);
    lists_[level].unlink(entry);
    if (lists_[level].empty()) nonempty_ &= ~(1u << level);
  }
  void age() noexcept {
    std::array<PolicyList, levels> aged{};
    std::uint32_t nonempty = 0;
    for (std::uint32_t level = 0; level < levels / 2; ++level) {
      aged[level].splice_tail(lists_[2 * level + 1]);
      aged[level].splice_tail(lists_[2 * level]);
      if (!aged[level].empty()) nonempty |= 1u << level;
    }
    lists_ = aged;
    nonempty_ = nonempty;
    ++epoch_;
    accesses_ = 0;
  }
  mutable std::mutex mutex_;
  std::array<PolicyList, levels> lists_{};
  std::uint32_t nonempty_ = 0;
  std::uint64_t epoch_ = 0;
  std::size_t accesses_ = 0;
  std::size_t size_ = 0;
  Entry* cursor_ = nullptr;
  std::uint32_t cursor_level_ = levels;
};

// Second-chance CLOCK over a circular list. find() sets the reference bit with
// a relaxed atomic RMW and no lock, skipped when already set, so hot entries
// are written at most once per sweep. The hand clears bits under the policy
// lock while the exclusive pool guard excludes readers, so a set never races a
// clear in practice; if it did, the only effect would be one more chance.
// Removed versions may still be marked by late readers; they are off the ring
// and their bits are never read. New versions start unreferenced just behind
// the hand, which makes them the last entries the current sweep examines.
// A selection stamps each returned entry with its epoch, so nothing is
// returned twice, and it ends after at most two sweeps: the first clears every
// bit it passes, the second returns whatever is left.
class ClockEvictionPolicy final {
 public:
  void record_access(Entry& entry) noexcept {
    if (!(entry.policy_state.load(std::memory_order_relaxed) & referenced))
      entry.policy_state.fetch_or(referenced, std::memory_order_relaxed);
  }
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) detach(*replaced);
    else ++size_;
    entry.policy_state.store(0, std::memory_order_relaxed);
    if (selecting_) ++unreturned_;
    link_behind_hand(entry);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry);
    --size_;
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) {
      detach(*chain);
      --size_;
    }
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    replacement.policy_state.store(old.policy_state.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
    if (old.policy_next == &old) {
      replacement.policy_previous = replacement.policy_next = &replacement;
    } else {
      replacement.policy_previous = old.policy_previous;
      replacement.policy_next = old.policy_next;
      old.policy_previous->policy_next = &replacement;
      old.policy_next->policy_previous = &replacement;
    }
    if (hand_ == &old) hand_ = &replacement;
    old.policy_previous = old.policy_next = nullptr;
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++epoch_;
    selecting_ = true;
    unreturned_ = size_;
    steps_left_ = 2 * size_;
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    while (unreturned_ != 0 && steps_left_ != 0) {
      --steps_left_;
      Entry* entry = hand_;
      hand_ = entry->policy_next;
      const std::uint64_t state = entry->policy_state.load(std::memory_order_relaxed);
      if (state >> 1 == epoch_) continue;  // already returned by this selection
      if (state & referenced) {
        entry->policy_state.store(state & ~referenced, std::memory_order_relaxed);
        continue;
      }
      entry->policy_state.store(epoch_ << 1, std::memory_order_relaxed);
      --unreturned_;
      return CacheIndex::identity(*entry);
    }
    return std::nullopt;
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    selecting_ = false;
    unreturned_ = steps_left_ = 0;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ * policy_entry_bytes;
  }

 private:
  // policy_state: selection epoch << 1 | reference bit.
  static constexpr std::uint64_t referenced = 1;
  void link_behind_hand(Entry& entry) noexcept {
    if (!hand_) {
      entry.policy_previous = entry.policy_next = &entry;
      hand_ = &entry;
      return;
    }
    Entry* last = hand_->policy_previous;
    entry.policy_previous = last;
    entry.policy_next = hand_;
    last->policy_next = &entry;
    hand_->policy_previous = &entry;
  }
  void detach(Entry& entry) noexcept {
    assert(entry.policy_next && size_ != 0);
    if (selecting_ && entry.policy_state.load(std::memory_order_relaxed) >> 1 != epoch_) --unreturned_;
    if (entry.policy_next == &entry) {
      hand_ = nullptr;
    } else {
      entry.policy_previous->policy_next = entry.policy_next;
      entry.policy_next->policy_previous = entry.policy_previous;
      if (hand_ == &entry) hand_ = entry.policy_next;
    }
    entry.policy_previous = entry.policy_next = nullptr;
  }
  mutable std::mutex mutex_;
  Entry* hand_ = nullptr;
  std::size_t size_ = 0;
  std::uint64_t epoch_ = 0;
  bool selecting_ = false;
  std::size_t unreturned_ = 0;
  std::size_t steps_left_ = 0;
};

// xoshiro256** seeded through splitmix64 (Blackman and Vigna). It is small,
// fast and gives the same sequence on every platform for a given seed, which
// std::uniform_int_distribution does not guarantee.
class RandomSource final {
 public:
  explicit RandomSource(std::uint64_t seed) noexcept {
    for (std::uint64_t& word : state_) {
      seed += 0x9e3779b97f4a7c15ull;
      std::uint64_t mixed = seed;
      mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
      mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
      word = mixed ^ (mixed >> 31);
    }
  }
  std::uint64_t next() noexcept {
    const std::uint64_t result = std::rotl(state_[1] * 5, 7) * 9;
    const std::uint64_t shifted = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= shifted;
    state_[3] = std::rotl(state_[3], 45);
    return result;
  }
  // Unbiased value in [0, bound) by rejection; bound must be positive.
  std::uint64_t below(std::uint64_t bound) noexcept {
    const std::uint64_t threshold = (0 - bound) % bound;
    for (;;) {
      const std::uint64_t value = next();
      if (value >= threshold) return value % bound;
    }
  }

 private:
  std::array<std::uint64_t, 4> state_{};
};

// Uniform random eviction over a dense member array; policy_state holds the
// entry's slot. Insert and remove are O(1) (swap with the last slot). A
// selection is a lazy Fisher-Yates shuffle: slots before selected_ were
// returned, and each next_victim draws uniformly from the rest. Removal keeps
// that prefix intact, so no candidate repeats. Reads never touch the policy.
class RandomEvictionPolicy final {
 public:
  explicit RandomEvictionPolicy(std::uint64_t seed) noexcept : random_(seed) {}
  void record_access(Entry&) noexcept {}
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) {
      place(entry, slot_of(*replaced));
      return true;
    }
    if (members_.size() == members_.capacity()) {
      try {
        members_.reserve(std::max<std::size_t>(min_capacity, members_.capacity() * 2));
      } catch (const std::exception&) {
        return false;
      }
    }
    members_.push_back(&entry);
    entry.policy_state.store(members_.size() - 1, std::memory_order_relaxed);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry);
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) detach(*chain);
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    place(replacement, slot_of(old));
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_ = 0;
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (selected_ >= members_.size()) return std::nullopt;
    const std::size_t pick = selected_ + random_.below(members_.size() - selected_);
    Entry* chosen = members_[pick];
    place(*members_[selected_], pick);
    place(*chosen, selected_);
    ++selected_;
    return CacheIndex::identity(*chosen);
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    selected_ = 0;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return members_.size() * policy_entry_bytes + members_.capacity() * sizeof(Entry*);
  }

 private:
  static constexpr std::size_t min_capacity = 16;
  static std::size_t slot_of(const Entry& entry) noexcept {
    return static_cast<std::size_t>(entry.policy_state.load(std::memory_order_relaxed));
  }
  void place(Entry& entry, std::size_t slot) noexcept {
    members_[slot] = &entry;
    entry.policy_state.store(slot, std::memory_order_relaxed);
  }
  void detach(Entry& entry) noexcept {
    const std::size_t slot = slot_of(entry);
    assert(slot < members_.size() && members_[slot] == &entry);
    const std::size_t last = members_.size() - 1;
    if (slot < selected_) {
      // Keep the returned prefix contiguous: its last member fills the hole,
      // then the array's last member fills the prefix's old end.
      --selected_;
      if (selected_ != slot) place(*members_[selected_], slot);
      if (last != selected_) place(*members_[last], selected_);
    } else if (slot != last) {
      place(*members_[last], slot);
    }
    members_.pop_back();
    if (members_.empty()) {
      std::vector<Entry*>().swap(members_);
    } else if (members_.capacity() > 1024 && members_.size() * 4 < members_.capacity()) {
      try {
        members_.shrink_to_fit();
      } catch (const std::exception&) {
        // Keeping the larger buffer is harmless.
      }
    }
  }
  mutable std::mutex mutex_;
  std::vector<Entry*> members_;
  std::size_t selected_ = 0;
  RandomSource random_;
};

// SIEVE (Zhang et al., "SIEVE is Simpler than LRU", NSDI 2024), following the
// paper's pseudocode and libCacheSim's Sieve.c. One FIFO list: new entries go
// to the head. A find sets the visited bit only if it is clear (a relaxed
// load, then one relaxed fetch_or), takes no lock and never moves the entry,
// so a hot entry costs no shared write after its first hit. The hand starts
// at the tail and walks towards the head (policy_previous): a visited entry
// has its bit cleared and stays in place; the first unvisited entry is the
// victim, and the hand moves on to its head-side neighbour. Past the head it
// wraps to the tail. The hand persists across evictions; that persistence and
// the head insertion distinguish SIEVE from CLOCK, whose hand reinserts
// retained entries and places new ones just behind itself.
//
// Framework adaptation: Core may reject a candidate (pinned or the key being
// written); the hand has already moved past it, as if it had been retained.
// A selection stamps returned entries with its epoch so none repeats, and
// ends after two passes: the first clears every bit it meets, the second
// returns what remains. Clearing happens under the exclusive pool guard, so
// it never races a find. A late find on a removed version only sets a bit
// that is never read again.
class SieveEvictionPolicy final {
 public:
  void record_access(Entry& entry) noexcept {
    if (!(entry.policy_state.load(std::memory_order_relaxed) & visited))
      entry.policy_state.fetch_or(visited, std::memory_order_relaxed);
  }
  // A replacement takes the replaced version's position and visited bit.
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) {
      entry.policy_state.store(replaced->policy_state.load(std::memory_order_relaxed) & visited,
                               std::memory_order_relaxed);
      if (hand_ == replaced) hand_ = &entry;
      list_.substitute(*replaced, entry);
      return true;
    }
    ++size_;
    entry.policy_state.store(0, std::memory_order_relaxed);
    if (selecting_) ++unreturned_;
    list_.push_head(entry);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    detach(entry);
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) detach(*chain);
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    replacement.policy_state.store(old.policy_state.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
    if (hand_ == &old) hand_ = &replacement;
    list_.substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++epoch_;
    selecting_ = true;
    unreturned_ = size_;
    steps_left_ = 2 * size_;
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    while (unreturned_ != 0 && steps_left_ != 0) {
      --steps_left_;
      Entry* entry = hand_ ? hand_ : list_.tail();
      hand_ = entry->policy_previous;  // null: wrap to the tail next time
      const std::uint64_t state = entry->policy_state.load(std::memory_order_relaxed);
      if (state >> 1 == epoch_) continue;  // already returned by this selection
      if (state & visited) {
        entry->policy_state.store(state & ~visited, std::memory_order_relaxed);
        continue;
      }
      entry->policy_state.store(epoch_ << 1, std::memory_order_relaxed);
      --unreturned_;
      return CacheIndex::identity(*entry);
    }
    return std::nullopt;
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    selecting_ = false;
    unreturned_ = steps_left_ = 0;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ * policy_entry_bytes;
  }
  // For tests and diagnostics.
  bool is_visited(const Entry& entry) const noexcept {
    return entry.policy_state.load(std::memory_order_relaxed) & visited;
  }
  const Entry* hand() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return hand_;
  }

 private:
  // policy_state: selection epoch << 1 | visited bit.
  static constexpr std::uint64_t visited = 1;
  void detach(Entry& entry) noexcept {
    assert(size_ != 0);
    if (selecting_ && entry.policy_state.load(std::memory_order_relaxed) >> 1 != epoch_) --unreturned_;
    // Like an eviction, removing the hand's entry moves the hand head-wards.
    if (hand_ == &entry) hand_ = entry.policy_previous;
    list_.unlink(entry);
    --size_;
  }
  // Written by publishers; find() never touches this object.
  alignas(64) mutable std::mutex mutex_;
  PolicyList list_;
  Entry* hand_ = nullptr;
  std::size_t size_ = 0;
  std::uint64_t epoch_ = 0;
  bool selecting_ = false;
  std::size_t unreturned_ = 0;
  std::size_t steps_left_ = 0;
};

// S3-FIFO's ghost queue, implemented as the paper's section 4.2 suggests: a
// bucket-based hash table of key fingerprints stamped with a virtual
// insertion time (the number of ghost insertions so far). With limit L and
// time N, entries stamped at or before N - L have left the queue, so FIFO
// order costs no list. An entry is also dropped when it is found (a ghost
// hit) or when its slot is needed by a collision, as in the paper. No value,
// key or pointer is retained.
//
// Each slot is 8 bytes: a 32-bit stamp and a 32-bit fingerprint (the paper's
// 4-byte fingerprint; bucket bits come from a different part of the hash, and
// bit 0 is forced to 1 so a zero slot means empty). Buckets hold 8 slots, one
// cache line. The table holds at most 0.75 entries per slot, and growth only
// happens on publish (the one place that may allocate; failure just keeps the
// smaller table). Growing discards the old fingerprints, because the bucket
// bits of their hashes were not stored; this happens only while the entry
// count passes its previous peak. The table is freed once the policy tracks
// nothing, since the limit is then zero. Memory is therefore bounded by about
// 10.7 to 21.3 bytes per ghost entry at the peak limit, and the limit itself
// is capped at max_limit.
class GhostTable final {
 public:
  static constexpr std::size_t max_limit = std::size_t{1} << 30;

  void set_limit(std::size_t limit) noexcept { limit_ = std::min(limit, max_limit); }
  std::size_t limit() const noexcept { return limit_; }
  // Called where allocation is allowed. Returns false only if growth failed.
  bool reserve() noexcept {
    const std::size_t wanted = buckets_for(limit_);
    if (wanted <= bucket_count_) return true;
    std::unique_ptr<Bucket[]> grown(new (std::nothrow) Bucket[wanted]);
    if (!grown) return false;
    buckets_ = std::move(grown);
    bucket_count_ = wanted;
    return true;
  }
  void release() noexcept {
    buckets_.reset();
    bucket_count_ = 0;
  }
  void insert(std::size_t hash) noexcept {
    if (!bucket_count_ || !limit_) return;
    const std::uint32_t stamp = static_cast<std::uint32_t>(++clock_);
    const std::uint64_t value = std::uint64_t{stamp} << 32 | fingerprint(hash);
    std::array<std::uint64_t, slots_per_bucket>& slots = bucket(hash);
    std::uint64_t* target = &slots[0];
    std::uint32_t oldest_age = 0;
    bool refresh = false;
    for (std::uint64_t& slot : slots) {
      // A key evicted again refreshes its stamp; otherwise reuse an empty or
      // expired slot, or else displace the bucket's oldest entry.
      if (slot && static_cast<std::uint32_t>(slot) == fingerprint(hash)) {
        target = &slot;
        refresh = true;
        break;
      }
      const std::uint32_t age = slot ? age_of(slot) : std::numeric_limits<std::uint32_t>::max();
      if (age > oldest_age) {
        oldest_age = age;
        target = &slot;
      }
    }
    if (!refresh && *target && age_of(*target) < limit_) ++collisions_;
    *target = value;
    // Clear stale stamps before the 32-bit clock could alias them.
    if ((clock_ & (sweep_interval - 1)) == 0) sweep();
  }
  // True for a live entry, which the hit consumes.
  bool take(std::size_t hash) noexcept {
    if (!bucket_count_) return false;
    for (std::uint64_t& slot : bucket(hash)) {
      if (!slot || static_cast<std::uint32_t>(slot) != fingerprint(hash)) continue;
      const bool live = age_of(slot) < limit_;
      slot = 0;
      return live;
    }
    return false;
  }
  // Live entries; O(table), for tests and diagnostics.
  std::size_t size() const noexcept {
    std::size_t live = 0;
    for (std::size_t index = 0; index < bucket_count_; ++index)
      for (const std::uint64_t slot : buckets_[index].slots)
        if (slot && age_of(slot) < limit_) ++live;
    return live;
  }
  std::size_t bytes() const noexcept { return bucket_count_ * sizeof(Bucket); }
  std::uint64_t collisions() const noexcept { return collisions_; }

 private:
  static constexpr std::size_t slots_per_bucket = 8;
  static constexpr std::uint64_t sweep_interval = std::uint64_t{1} << 30;
  static_assert(max_limit <= sweep_interval);
  struct alignas(64) Bucket {
    std::array<std::uint64_t, slots_per_bucket> slots{};
  };
  static std::size_t buckets_for(std::size_t limit) noexcept {
    if (!limit) return 0;
    const std::size_t per_bucket = slots_per_bucket * 3 / 4;
    return std::bit_ceil((limit + per_bucket - 1) / per_bucket);
  }
  static std::uint32_t fingerprint(std::size_t hash) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(hash) >> 32) | 1u;
  }
  std::array<std::uint64_t, slots_per_bucket>& bucket(std::size_t hash) noexcept {
    return buckets_[static_cast<std::uint64_t>(hash) & (bucket_count_ - 1)].slots;
  }
  std::uint32_t age_of(std::uint64_t slot) const noexcept {
    return static_cast<std::uint32_t>(clock_) - static_cast<std::uint32_t>(slot >> 32);
  }
  void sweep() noexcept {
    for (std::size_t index = 0; index < bucket_count_; ++index)
      for (std::uint64_t& slot : buckets_[index].slots)
        if (slot && age_of(slot) >= limit_) slot = 0;
  }
  std::unique_ptr<Bucket[]> buckets_;
  std::size_t bucket_count_ = 0;
  std::size_t limit_ = 0;
  std::uint64_t clock_ = 0;
  std::uint64_t collisions_ = 0;
};

// S3-FIFO (Yang et al., "FIFO queues are all you need for cache eviction",
// SOSP 2023), following Algorithm 1 and section 4:
//  - read: a hit sets freq = min(freq + 1, max_frequency) with a lock-free
//    CAS that is skipped once the counter is saturated, so popular entries
//    cost no shared write after their first max_frequency hits.
//  - insert: a key found in the ghost queue enters the head of main, any
//    other key the head of small, with freq 0.
//  - evict: from small while it holds at least small_queue_percent of the
//    tracked bytes, otherwise from main.
//  - evictS: a small tail with freq >= promotion_frequency moves to the head
//    of main; otherwise it is evicted and its fingerprint enters the ghost.
//  - evictM: a main tail with freq > 0 is reinserted at the head with
//    freq - 1; otherwise it is evicted.
// Variants, each chosen to follow the paper where sources differ:
//  - Promotion clears the counter, per the paper's text ("its access bits are
//    cleared during the move"); libCacheSim keeps it.
//  - promotion_frequency 2 is the pseudocode's "freq > 1"; the text's "accessed
//    more than once" agrees. Some write-ups use one access instead.
//  - The ghost holds ghost_queue_percent of the tracked entry count, by
//    default 90%, the main queue's share, as the paper sizes it in entries;
//    libCacheSim sizes it in bytes.
//  - "Cache size" in the 10% rule is the bytes the policy tracks (indexed
//    entries), so drafts and pinned retired versions do not skew the split.
//  - Algorithm 1's evictS may evict from main and then still evict from small
//    in the same call. Here each candidate re-applies the 10% rule, which is
//    equivalent (moving a promoted entry is what fills main), and pressure
//    stops as soon as the request fits rather than overshooting.
// Framework adaptation: a candidate that Core rejects (pinned or the key being
// written) stays in place and is not recorded in the ghost. Only a candidate
// Core actually removes during selection counts as evicted; erase, expiry and
// replacement never enter the ghost. Returned entries carry the selection's
// epoch, so they are skipped (rotated to their queue's head) if reached again.
// Steps per selection are bounded by about (max_frequency + 2) * entries.
class S3FifoEvictionPolicy final {
 public:
  explicit S3FifoEvictionPolicy(const S3FifoConfig& config) noexcept
      : max_frequency_(config.max_frequency),
        promotion_frequency_(config.promotion_frequency),
        small_percent_(config.small_queue_percent),
        ghost_percent_(config.ghost_queue_percent) {}

  void record_access(Entry& entry) noexcept {
    std::uint64_t state = entry.policy_state.load(std::memory_order_relaxed);
    while ((state & linked) && (state & frequency_mask) < max_frequency_) {
      if (entry.policy_state.compare_exchange_weak(state, state + 1, std::memory_order_relaxed,
                                                   std::memory_order_relaxed)) return;
    }
  }
  // A replacement keeps the replaced version's queue, position and counter.
  bool publish(Entry& entry, Entry* replaced) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (replaced) {
      const std::uint64_t state = replaced->policy_state.load(std::memory_order_relaxed);
      entry.policy_state.store(state & (frequency_mask | in_main | linked), std::memory_order_relaxed);
      replaced->policy_state.store(0, std::memory_order_relaxed);
      std::size_t& bytes = (state & in_main) ? main_bytes_ : small_bytes_;
      bytes = bytes - replaced->charge + entry.charge;
      if (pending_ == replaced) pending_ = &entry;
      ((state & in_main) ? main_ : small_).substitute(*replaced, entry);
      return true;
    }
    ++count_;
    update_ghost_limit();
    ghost_.reserve();
    const bool remembered = ghost_.take(hash_of(entry));
    entry.policy_state.store(linked | (remembered ? in_main : 0), std::memory_order_relaxed);
    if (remembered) {
      main_.push_head(entry);
      main_bytes_ += entry.charge;
      ++main_count_;
      ++ghost_hits_;
    } else {
      small_.push_head(entry);
      small_bytes_ += entry.charge;
    }
    if (selecting_) ++(remembered ? unreturned_main_ : unreturned_small_);
    return true;
  }
  void remove(Entry& entry) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    // Only the candidate Core just accepted is an eviction.
    if (selecting_ && &entry == pending_ && pending_small_) ghost_.insert(hash_of(entry));
    detach(entry);
  }
  void remove_chain(Entry* chain) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (; chain; chain = chain->hash_next) detach(*chain);
  }
  void relocate(Entry& old, Entry& replacement) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t state = old.policy_state.load(std::memory_order_relaxed);
    replacement.policy_state.store(state, std::memory_order_relaxed);
    old.policy_state.store(0, std::memory_order_relaxed);
    if (pending_ == &old) pending_ = &replacement;
    ((state & in_main) ? main_ : small_).substitute(old, replacement);
  }
  void begin_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++epoch_;
    selecting_ = true;
    unreturned_small_ = small_count();
    unreturned_main_ = count_ - unreturned_small_;
    steps_left_ = (max_frequency_ + 3) * count_ + 2;
  }
  std::optional<EntryId> next_victim() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = nullptr;
    while ((unreturned_small_ || unreturned_main_) && steps_left_) {
      --steps_left_;
      const bool small_due = small_bytes_ * 100 >= (small_bytes_ + main_bytes_) * small_percent_;
      const bool from_small = unreturned_small_ && (small_due || !unreturned_main_);
      PolicyList& queue = from_small ? small_ : main_;
      Entry* entry = queue.tail();
      const std::uint64_t state = entry->policy_state.load(std::memory_order_relaxed);
      if (state >> epoch_shift == epoch_) {  // returned earlier and rejected
        queue.unlink(*entry);
        queue.push_head(*entry);
        continue;
      }
      const std::uint64_t frequency = state & frequency_mask;
      if (from_small && frequency >= promotion_frequency_) {
        small_.unlink(*entry);
        small_bytes_ -= entry->charge;
        main_.push_head(*entry);
        main_bytes_ += entry->charge;
        ++main_count_;
        entry->policy_state.store(linked | in_main, std::memory_order_relaxed);
        --unreturned_small_;
        ++unreturned_main_;
        ++promotions_;
        continue;
      }
      if (!from_small && frequency > 0) {
        main_.unlink(*entry);
        main_.push_head(*entry);
        entry->policy_state.store(state - 1, std::memory_order_relaxed);
        continue;
      }
      entry->policy_state.store(epoch_ << epoch_shift | (state & ~epoch_bits), std::memory_order_relaxed);
      --(from_small ? unreturned_small_ : unreturned_main_);
      pending_ = entry;
      pending_small_ = from_small;
      return CacheIndex::identity(*entry);
    }
    return std::nullopt;
  }
  void end_selection() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    selecting_ = false;
    pending_ = nullptr;
    unreturned_small_ = unreturned_main_ = steps_left_ = 0;
  }
  std::size_t metadata_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_ * policy_entry_bytes + ghost_.bytes();
  }

  // For tests and diagnostics.
  struct Snapshot {
    std::size_t small_entries = 0;
    std::size_t main_entries = 0;
    std::size_t small_bytes = 0;
    std::size_t main_bytes = 0;
    std::size_t ghost_entries = 0;
    std::size_t ghost_limit = 0;
    std::size_t ghost_bytes = 0;
    std::uint64_t ghost_hits = 0;
    std::uint64_t promotions = 0;
  };
  Snapshot snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return {small_count(), count_ - small_count(), small_bytes_, main_bytes_, ghost_.size(),
            ghost_.limit(), ghost_.bytes(), ghost_hits_, promotions_};
  }
  bool in_main_queue(const Entry& entry) const noexcept {
    return entry.policy_state.load(std::memory_order_relaxed) & in_main;
  }
  std::uint32_t frequency(const Entry& entry) const noexcept {
    return static_cast<std::uint32_t>(entry.policy_state.load(std::memory_order_relaxed) & frequency_mask);
  }

 private:
  // policy_state: selection epoch << 5 | linked | in_main | frequency (3 bits).
  static constexpr std::uint64_t frequency_mask = 0x7;
  static constexpr std::uint64_t in_main = 0x8;
  static constexpr std::uint64_t linked = 0x10;
  static constexpr std::uint32_t epoch_shift = 5;
  static constexpr std::uint64_t epoch_bits = ~std::uint64_t{0} << epoch_shift;

  static std::size_t hash_of(const Entry& entry) noexcept {
    // Mixed so the bucket bits and the fingerprint bits are independent.
    std::uint64_t value = static_cast<std::uint64_t>(CacheIndex::hash(entry.key));
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return static_cast<std::size_t>(value ^ (value >> 31));
  }
  std::size_t small_count() const noexcept { return count_ - main_count_; }
  void update_ghost_limit() noexcept {
    ghost_.set_limit(count_ / 100 * ghost_percent_ + count_ % 100 * ghost_percent_ / 100);
  }
  void detach(Entry& entry) noexcept {
    const std::uint64_t state = entry.policy_state.load(std::memory_order_relaxed);
    assert((state & linked) && count_ != 0);
    const bool main = state & in_main;
    if (selecting_ && state >> epoch_shift != epoch_) --(main ? unreturned_main_ : unreturned_small_);
    if (pending_ == &entry) pending_ = nullptr;
    entry.policy_state.store(0, std::memory_order_relaxed);
    (main ? main_ : small_).unlink(entry);
    (main ? main_bytes_ : small_bytes_) -= entry.charge;
    if (main) --main_count_;
    if (--count_ == 0) ghost_.release();
    update_ghost_limit();
  }

  // Read by find(); never written after construction.
  const std::uint64_t max_frequency_;
  const std::uint64_t promotion_frequency_;
  const std::size_t small_percent_;
  const std::size_t ghost_percent_;
  // Written by publishers and selection.
  alignas(64) mutable std::mutex mutex_;
  PolicyList small_;
  PolicyList main_;
  GhostTable ghost_;
  std::size_t count_ = 0;
  std::size_t main_count_ = 0;
  std::size_t small_bytes_ = 0;
  std::size_t main_bytes_ = 0;
  Entry* pending_ = nullptr;
  bool pending_small_ = false;
  std::uint64_t epoch_ = 0;
  bool selecting_ = false;
  std::size_t unreturned_small_ = 0;
  std::size_t unreturned_main_ = 0;
  std::size_t steps_left_ = 0;
  std::uint64_t ghost_hits_ = 0;
  std::uint64_t promotions_ = 0;
};

// Tracks nothing and never offers a victim, so pressure fails instead.
class NoEvictionPolicy final {
 public:
  void record_access(Entry&) noexcept {}
  bool publish(Entry&, Entry*) noexcept { return true; }
  void remove(Entry&) noexcept {}
  void remove_chain(Entry*) noexcept {}
  void relocate(Entry&, Entry&) noexcept {}
  void begin_selection() noexcept {}
  std::optional<EntryId> next_victim() noexcept { return std::nullopt; }
  void end_selection() noexcept {}
  std::size_t metadata_bytes() const noexcept { return 0; }
};

class EvictionPolicyEngine;

// One victim search. It must live entirely inside the exclusive pool guard.
// Candidates are stable identities that Core resolves again under the stripe
// lock, where it may still reject them. Each candidate is returned at most
// once, and reaching the end is final: pressure never restarts an exhausted
// search, so a pool full of pinned entries fails instead of spinning.
class VictimSelection final {
 public:
  explicit VictimSelection(EvictionPolicyEngine& engine) noexcept : engine_(engine) {}
  VictimSelection(const VictimSelection&) = delete;
  VictimSelection& operator=(const VictimSelection&) = delete;
  ~VictimSelection();
  [[nodiscard]] std::optional<EntryId> next() noexcept;

 private:
  EvictionPolicyEngine& engine_;
  bool exhausted_ = false;
};

// Per-pool runtime selection with inline, allocation-free access dispatch.
// Adding an algorithm changes this adapter and the configuration, not
// CachePool's ownership, expiry, compaction or quota logic. No user callbacks.
//
// Lock order: maintenance -> index stripe -> policy. Selection takes only the
// policy lock and releases it before Core resolves a candidate in the index.
class alignas(64) EvictionPolicyEngine final {
  // Declared before the constructor, which takes their addresses.
  using AccessHook = void (*)(void*, Entry&) noexcept;
  template <typename Policy>
  static void access_hook(void* policy, Entry& entry) noexcept {
    static_cast<Policy*>(policy)->record_access(entry);
  }
 public:
  [[nodiscard]] static Status validate(EvictionPolicy policy, const S3FifoConfig& s3_fifo = {},
                                       const SlruConfig& slru = {}, const TwoQConfig& two_q = {},
                                       const WTinyLfuConfig& w_tinylfu = {}) noexcept {
    switch (policy) {
      case EvictionPolicy::w_tinylfu:
        return w_tinylfu.window_percent >= 1 && w_tinylfu.window_percent <= 99 &&
               w_tinylfu.protected_percent >= 1 && w_tinylfu.protected_percent <= 99 &&
               w_tinylfu.sketch_width >= 64 && w_tinylfu.sketch_width <= (1U << 20) &&
               std::has_single_bit(w_tinylfu.sketch_width) && w_tinylfu.sample_size >= 8192
                   ? Status::ok : Status::invalid_argument;
      case EvictionPolicy::two_q:
        return two_q.a1in_percent >= 1 && two_q.a1in_percent <= 99 && two_q.ghost_capacity <= (1U << 20)
                   ? Status::ok : Status::invalid_argument;
      case EvictionPolicy::slru:
        return slru.protected_percent >= 1 && slru.protected_percent <= 99
                   ? Status::ok : Status::invalid_argument;
      case EvictionPolicy::lru:
      case EvictionPolicy::fifo:
      case EvictionPolicy::lfu:
      case EvictionPolicy::clock:
      case EvictionPolicy::random:
      case EvictionPolicy::no_eviction:
      case EvictionPolicy::sieve: return Status::ok;
      case EvictionPolicy::s3_fifo: {
        const bool valid = s3_fifo.small_queue_percent >= 1 && s3_fifo.small_queue_percent <= 99 &&
                           s3_fifo.ghost_queue_percent <= 400 && s3_fifo.max_frequency >= 1 &&
                           s3_fifo.max_frequency <= 7 && s3_fifo.promotion_frequency >= 1 &&
                           s3_fifo.promotion_frequency <= s3_fifo.max_frequency;
        return valid ? Status::ok : Status::invalid_argument;
      }
    }
    return Status::invalid_argument;
  }

  // A zero seed draws a nondeterministic one; only the random policy uses it.
  explicit EvictionPolicyEngine(EvictionPolicy policy, std::uint64_t seed = 0,
                                const S3FifoConfig& s3_fifo = {}, const SlruConfig& slru = {},
                                const TwoQConfig& two_q = {},
                                const WTinyLfuConfig& w_tinylfu = {}, std::size_t capacity = 1024) {
    if (validate(policy, s3_fifo, slru, two_q, w_tinylfu) != Status::ok)
      throw std::invalid_argument("Invalid eviction policy");
    switch (policy) {
      case EvictionPolicy::w_tinylfu: policy_.emplace<WTinyLfuEvictionPolicy>(w_tinylfu, capacity); break;
      case EvictionPolicy::two_q: policy_.emplace<TwoQEvictionPolicy>(two_q); break;
      case EvictionPolicy::slru: policy_.emplace<SlruEvictionPolicy>(slru, capacity); break;
      case EvictionPolicy::lru: break;
      case EvictionPolicy::fifo: policy_.emplace<FifoEvictionPolicy>(); break;
      case EvictionPolicy::lfu: policy_.emplace<LfuEvictionPolicy>(); break;
      case EvictionPolicy::clock: policy_.emplace<ClockEvictionPolicy>(); break;
      case EvictionPolicy::random:
        policy_.emplace<RandomEvictionPolicy>(seed != 0 ? seed : nondeterministic_seed());
        break;
      case EvictionPolicy::no_eviction: policy_.emplace<NoEvictionPolicy>(); break;
      case EvictionPolicy::sieve: policy_.emplace<SieveEvictionPolicy>(); break;
      case EvictionPolicy::s3_fifo: policy_.emplace<S3FifoEvictionPolicy>(s3_fifo); break;
    }
    std::visit([this]<typename Policy>(Policy& selected) noexcept -> void {
      access_ = &access_hook<Policy>;
      access_target_ = &selected;
    }, policy_);
  }
  EvictionPolicyEngine(const EvictionPolicyEngine&) = delete;
  EvictionPolicyEngine& operator=(const EvictionPolicyEngine&) = delete;

  // Runs on every find() hit. It reads only this line, which nothing writes
  // after construction, and then the policy's own access path. Dispatching
  // through std::visit would read the variant's index, which shares cache
  // lines with state that publishers write.
  void record_access(Entry& entry) noexcept { access_(access_target_, entry); }
  // Preserve a 2Q ghost hit across pre-publication reclamation, which can
  // otherwise overwrite a small ghost queue before this draft is published.
  std::uint64_t prepare_insert(std::string_view key) const noexcept {
    const TwoQEvictionPolicy* selected = std::get_if<TwoQEvictionPolicy>(&policy_);
    return selected ? selected->prepare_insert(key) : 0;
  }
  void reset_history() noexcept {
    if (TwoQEvictionPolicy* selected = std::get_if<TwoQEvictionPolicy>(&policy_)) selected->reset_history();
    if (WTinyLfuEvictionPolicy* selected = std::get_if<WTinyLfuEvictionPolicy>(&policy_)) selected->reset_history();
  }
  // Only W-TinyLFU observes absent requests. Hits are recorded by its access hook.
  void record_miss(std::string_view key) noexcept {
    if (WTinyLfuEvictionPolicy* selected = std::get_if<WTinyLfuEvictionPolicy>(&policy_)) selected->record_request(key);
  }
  void set_capacity(std::size_t capacity) noexcept {
    if (SlruEvictionPolicy* selected = std::get_if<SlruEvictionPolicy>(&policy_)) selected->set_capacity(capacity);
    if (WTinyLfuEvictionPolicy* selected = std::get_if<WTinyLfuEvictionPolicy>(&policy_)) selected->set_capacity(capacity);
  }
  // Links a newly published version, unlinking the version it replaces.
  // Call before the index commit; false means nothing changed.
  [[nodiscard]] bool publish(Entry& entry, Entry* replaced) noexcept {
    return std::visit([&entry, replaced]<typename Policy>(Policy& policy) noexcept -> bool {
      return policy.publish(entry, replaced);
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
  // Requires the exclusive pool guard for the selection's whole lifetime.
  [[nodiscard]] VictimSelection select_victims() noexcept {
    std::visit([]<typename Policy>(Policy& policy) noexcept -> void {
      policy.begin_selection();
    }, policy_);
    return VictimSelection(*this);
  }
  [[nodiscard]] std::size_t metadata_bytes() const noexcept {
    return std::visit([]<typename Policy>(const Policy& policy) noexcept -> std::size_t {
      return policy.metadata_bytes();
    }, policy_);
  }
  // Direct access for white-box tests of a specific algorithm.
  template <typename Policy>
  Policy* get_if() noexcept { return std::get_if<Policy>(&policy_); }

 private:
  friend class VictimSelection;
  std::optional<EntryId> next_victim() noexcept {
    return std::visit([]<typename Policy>(Policy& policy) noexcept -> std::optional<EntryId> {
      return policy.next_victim();
    }, policy_);
  }
  void end_selection() noexcept {
    std::visit([]<typename Policy>(Policy& policy) noexcept -> void {
      policy.end_selection();
    }, policy_);
  }
  static std::uint64_t nondeterministic_seed() noexcept {
    try {
      std::random_device device;
      const std::uint64_t seed = static_cast<std::uint64_t>(device()) << 32 | device();
      if (seed != 0) return seed;
    } catch (const std::exception&) {
      // Fall back to the clock below.
    }
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1;
  }

  AccessHook access_ = nullptr;
  void* access_target_ = nullptr;
  alignas(64) std::variant<LruEvictionPolicy, FifoEvictionPolicy, LfuEvictionPolicy, ClockEvictionPolicy,
                           RandomEvictionPolicy, NoEvictionPolicy, SieveEvictionPolicy,
                           S3FifoEvictionPolicy, SlruEvictionPolicy, TwoQEvictionPolicy, WTinyLfuEvictionPolicy> policy_;
};

inline VictimSelection::~VictimSelection() { engine_.end_selection(); }
inline std::optional<EntryId> VictimSelection::next() noexcept {
  if (exhausted_) return std::nullopt;
  std::optional<EntryId> candidate = engine_.next_victim();
  exhausted_ = !candidate;
  return candidate;
}
}  // namespace nivo::detail
