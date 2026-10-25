#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <shared_mutex>

#include "cache_index.h"
#include "eviction_policy.h"
#include "nivo/status.h"

namespace nivo::detail {
struct AtomicCounters {
  std::atomic<std::uint64_t> hits{0}, misses{0}, allocations{0}, allocation_failures{0};
  std::atomic<std::uint64_t> inserts{0}, replacements{0}, erases{0}, evictions{0}, expirations{0};
  Counters snapshot() const noexcept {
    return {hits.load(std::memory_order_relaxed), misses.load(std::memory_order_relaxed),
            allocations.load(std::memory_order_relaxed), allocation_failures.load(std::memory_order_relaxed),
            inserts.load(std::memory_order_relaxed), replacements.load(std::memory_order_relaxed),
            erases.load(std::memory_order_relaxed), evictions.load(std::memory_order_relaxed),
            expirations.load(std::memory_order_relaxed)};
  }
};

// Lock order: pool maintenance guard -> index stripe -> eviction queue.
// The allocator owns its storage lock. Destruction of detached entries occurs
// outside index/policy locks. Exclusive maintenance prevents new references and
// allocations during compaction, quota pressure, clear, resize and snapshots.
// Ordinary operations share the guard and synchronize only within each module.
// Expiry reaping is an ordinary operation: it holds the shared guard and one
// index stripe at a time, for a bounded batch, and never scans the whole pool.
class CachePool final {
 public:
  CachePool(PoolId id, PoolConfig config, std::shared_ptr<const Owner> owner,
            std::shared_ptr<const Clock> clock)
      : id_(id), config_(std::move(config)), owner_(std::move(owner)), clock_(std::move(clock)),
        budget_(std::make_shared<MemoryAllocator>(config_.capacity_bytes)),
        index_(config_.capacity_bytes), eviction_(config_.eviction) {
    budget_->reserve_initial(config_.initial_reserve_bytes, config_.initial_reserve_timeout);
  }
  ~CachePool() { close(); }
  const std::string& name() const noexcept { return config_.name; }
  std::size_t capacity() const {
    std::shared_lock<SharedMutex> lock(maintenance_);
    return config_.capacity_bytes;
  }
  std::optional<MemoryAllocator::TimePoint> maintain_reserve() {
    return budget_->maintain_initial_reserve();
  }
  void maintain_storage() noexcept { budget_->maintain(); }

  Result<EntryPtr> allocate(std::string_view key, std::size_t value_size) {
    std::shared_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    if (value_size > std::numeric_limits<std::size_t>::max() - key.size() ||
        key.size() + value_size > std::numeric_limits<std::size_t>::max() - sizeof(Entry))
      return Status::invalid_argument;
    const std::size_t charge = key.size() + value_size;
    if (reserve(charge)) return allocate_reserved(key, value_size, charge);
    lock.unlock();
    // Upgrade by releasing and reacquiring; capacity/closed must be rechecked.
    std::unique_lock<SharedMutex> maintenance(maintenance_);
    if (closed_) return Status::closed;
    if (charge <= config_.capacity_bytes) {
      if (reserve(charge)) return allocate_reserved(key, value_size, charge);
      prune_locked(clock_->now());
      if (make_room(config_.capacity_bytes - charge, key) && reserve(charge))
        return allocate_reserved(key, value_size, charge);
    }
    counters_.allocation_failures.fetch_add(1, std::memory_order_relaxed);
    return Status::resource_exhausted;
  }

  Result<EntryPtr> publish(EntryPtr& draft, WriteOptions options) {
    std::shared_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    if (!draft || draft->owner != owner_) return Status::invalid_argument;
    const Duration now = clock_->now();
    if (options.ttl.count() > std::numeric_limits<Duration::rep>::max() - now.count())
      return Status::invalid_argument;
    EntryPtr retired;
    EntryPtr expired_entry;
    const Status status = index_.write(draft->key, [&](CacheIndex::Slot& slot) -> Status {
      if (options.ttl.count() != 0 && !slot.prepare_expiry(now)) return Status::resource_exhausted;
      if (slot.get() && expired(*slot.get(), now)) {
        eviction_.remove(*slot.get());
        expired_entry = slot.remove();
        counters_.expirations.fetch_add(1, std::memory_order_relaxed);
      }
      Entry* old = slot.get();
      if (options.mode == WriteMode::insert_only && old) return Status::already_exists;
      if ((options.mode == WriteMode::replace_only || options.mode == WriteMode::compare_exchange) && !old)
        return Status::not_found;
      if (options.mode == WriteMode::compare_exchange && old->version != options.expected_version)
        return Status::version_mismatch;
      const Version version = owner_->next_version();
      if (version == invalid_version) return Status::resource_exhausted;
      draft->version = version;
      draft->expires_at = options.ttl.count() == 0 ? never_expires : now + options.ttl;
      if (old) eviction_.remove(*old);
      eviction_.insert(*draft);
      retired = slot.replace(*draft);
      if (old) counters_.replacements.fetch_add(1, std::memory_order_relaxed);
      else counters_.inserts.fetch_add(1, std::memory_order_relaxed);
      return Status::ok;
    });
    if (status != Status::ok) return status;
    return std::move(draft);
  }

  Result<EntryPtr> lookup(std::string_view key, bool touch) {
    std::shared_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    const Duration now = clock_->now();
    EntryPtr entry = index_.find(key);
    if (entry && expired(*entry, now)) {
      EntryPtr retired;
      index_.write(key, [&](CacheIndex::Slot& slot) {
        if (slot.get() == &*entry) {
          eviction_.remove(*slot.get());
          retired = slot.remove();
          counters_.expirations.fetch_add(1, std::memory_order_relaxed);
        }
      });
      entry.reset();
    }
    if (!entry) {
      if (touch) counters_.misses.fetch_add(1, std::memory_order_relaxed);
      return Status::not_found;
    }
    if (touch) {
      counters_.hits.fetch_add(1, std::memory_order_relaxed);
      eviction_.record_access(*entry);
    }
    return entry;
  }

  Status erase(std::string_view key, Version expected_version) {
    std::shared_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    const Duration now = clock_->now();
    EntryPtr retired;
    return index_.write(key, [&](CacheIndex::Slot& slot) -> Status {
      Entry* entry = slot.get();
      if (!entry) return Status::not_found;
      const bool stale = expired(*entry, now);
      if (!stale && expected_version && entry->version != expected_version) return Status::version_mismatch;
      eviction_.remove(*entry);
      retired = slot.remove();
      if (stale) {
        counters_.expirations.fetch_add(1, std::memory_order_relaxed);
        return Status::not_found;
      }
      counters_.erases.fetch_add(1, std::memory_order_relaxed);
      return Status::ok;
    });
  }

  Status resize(std::size_t capacity) {
    std::unique_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    prune_locked(clock_->now());
    if (!make_room(capacity, {})) return Status::resource_exhausted;
    config_.capacity_bytes = capacity;
    budget_->resize(capacity);
    return Status::ok;
  }
  Status clear() {
    std::unique_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    counters_.erases.fetch_add(index_.stats().items, std::memory_order_relaxed);
    clear_locked();
    budget_->trim();
    return Status::ok;
  }
  void close() {
    std::unique_lock<SharedMutex> lock(maintenance_);
    if (closed_) return;
    closed_ = true;
    budget_->trim(true);
    clear_locked();
  }
  PoolId id() const noexcept { return id_; }
  Duration now() const noexcept { return clock_->now(); }
  struct StorageUse {
    std::size_t reserved = 0;
    std::size_t unused = 0;  // as MemoryStats::unused_bytes
  };
  // Takes only the allocator lock, never the maintenance guard.
  StorageUse storage_use() const {
    const AllocatorStats storage = budget_->inspect();
    return {storage.reserved_bytes,
            storage.reserved_bytes - storage.metadata_bytes - storage.active_bytes};
  }
  // Background compaction bookkeeping, used only by the cache worker thread.
  struct AutoCompaction {
    bool pending = false;  // entries were reaped since the last check
    bool ran = false;
    std::chrono::steady_clock::time_point last{};
  };
  AutoCompaction& auto_compaction() noexcept { return auto_compaction_; }
  // Reclaims up to limit entries expired at now, examining at most batch due
  // entries per stripe lock hold. Each batch takes the shared guard only, so
  // reads and writes continue; resume() runs between batches with no locks
  // held and may stop early by returning false. A rotating start stripe keeps
  // bounded calls fair. Cost follows the number of due entries, not pool size.
  template <typename Resume>
  Result<std::size_t> reap(Duration now, std::size_t limit, std::size_t batch, Resume&& resume) {
    std::size_t count = 0;
    const std::size_t start = reap_cursor_.load(std::memory_order_relaxed);
    for (std::size_t offset = 0; offset < CacheIndex::stripe_count; ++offset) {
      const std::size_t stripe = (start + offset) % CacheIndex::stripe_count;
      for (;;) {
        if (count >= limit) {
          reap_cursor_.store(stripe, std::memory_order_relaxed);
          return count;
        }
        Reaped step;
        {
          std::shared_lock<SharedMutex> lock(maintenance_);
          if (closed_) return Status::closed;
          step = reap_stripe(stripe, now, std::min(batch, limit - count));
        }
        count += step.count;
        if (step.examined && !resume()) {
          reap_cursor_.store(step.more ? stripe : (stripe + 1) % CacheIndex::stripe_count,
                             std::memory_order_relaxed);
          return count;
        }
        if (!step.more) break;
      }
    }
    return count;
  }

  PoolStats snapshot() const {
    std::unique_lock<SharedMutex> lock(maintenance_);
    const CacheIndex::Stats index = index_.stats();
    return {id_, config_.name, config_.capacity_bytes, budget_->used.load(std::memory_order_relaxed),
            index.bytes, index.items, index.expiring, counters_.snapshot(), budget_->stats()};
  }
  MemoryStats memory_snapshot() const {
    std::unique_lock<SharedMutex> lock(maintenance_);
    // Remote handle releases may continue, so storage gauges must come from a
    // single allocator snapshot. Indexed storage cannot disappear in this scope.
    const AllocatorStats storage = budget_->inspect();
    MemoryStats result;
    result.reserved_bytes = storage.reserved_bytes;
    result.entry_metadata_bytes = storage.live_allocations * sizeof(Entry);
    result.kv_bytes = storage.requested_bytes - result.entry_metadata_bytes;
    result.indexed_kv_bytes = index_.stats().bytes;
    result.retained_kv_bytes = result.kv_bytes - result.indexed_kv_bytes;
    result.allocator_metadata_bytes = storage.metadata_bytes;
    result.internal_fragmentation_bytes = storage.active_bytes - storage.requested_bytes;
    result.unused_bytes = storage.reserved_bytes - storage.metadata_bytes - storage.active_bytes;
    result.reclaimable_bytes = storage.reclaimable_bytes;
    return result;
  }

  Result<CompactionStats> compact(Duration now) {
    std::unique_lock<SharedMutex> lock(maintenance_);
    if (closed_) return Status::closed;
    CompactionStats result;
    result.pools = 1;
    result.reserved_bytes_before = budget_->stats().reserved_bytes;
    try {
      result.expired_items = prune_locked(now);
      std::vector<Entry*> entries = eviction_.entries();
      std::vector<MemoryAllocator::Allocation> allocations;
      std::vector<Entry*> candidates;
      allocations.reserve(entries.size());
      candidates.reserve(entries.size());
      for (Entry* entry : entries) {
        if (entry->references.load(std::memory_order_acquire) == 1 && entry->allocation.slab) {
          candidates.push_back(entry);
          allocations.push_back(entry->allocation);
        }
      }
      {
        MemoryAllocator::Compaction compaction(*budget_, allocations);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
          if (!compaction.should_move(allocations[i])) continue;
          Entry& source = *candidates[i];
          const MemoryAllocator::Allocation allocation = compaction.allocate(allocations[i]);
          EntryPtr destination(new (allocation.data) Entry{});
          destination->owner = source.owner;
          destination->budget = source.budget;
          destination->allocation = allocation;
          destination->pool = source.pool;
          destination->version = source.version;
          destination->expires_at = source.expires_at;
          std::byte* value = entry_value_data(*destination);
          char* key = reinterpret_cast<char*>(value + source.value.size());
          std::memcpy(key, source.key.data(), source.key.size());
          if (!source.value.empty()) std::memcpy(value, source.value.data(), source.value.size());
          destination->key = {key, source.key.size()};
          destination->value = {value, source.value.size()};
          destination->charge = source.charge;
          EntryPtr retired;
          index_.write(source.key, [&](CacheIndex::Slot& slot) {
            eviction_.relocate(source, *destination);
            retired = slot.replace(*destination);
            source.charge = 0;  // Transfer the existing reservation to the copy.
          });
          ++result.moved_items;
        }
      }
      result.skipped_items = index_.stats().items - result.moved_items;
      result.reserved_bytes_after = budget_->stats().reserved_bytes;
      return result;
    } catch (const std::bad_alloc&) {
      budget_->trim();
      return Status::resource_exhausted;
    }
  }

 private:
  bool reserve(std::size_t charge) noexcept {
    if (charge > config_.capacity_bytes) return false;
    std::size_t used = budget_->used.load(std::memory_order_relaxed);
    while (used <= config_.capacity_bytes - charge) {
      if (budget_->used.compare_exchange_weak(used, used + charge, std::memory_order_relaxed)) return true;
    }
    return false;
  }
  Result<EntryPtr> allocate_reserved(std::string_view key, std::size_t value_size, std::size_t charge) {
    MemoryAllocator::Allocation allocation;
    try {
      allocation = budget_->allocate(sizeof(Entry) + charge);
    } catch (const std::bad_alloc&) {
      budget_->used.fetch_sub(charge, std::memory_order_relaxed);
      counters_.allocation_failures.fetch_add(1, std::memory_order_relaxed);
      return Status::resource_exhausted;
    }
    EntryPtr entry(new (allocation.data) Entry{});
    entry->owner = owner_;
    entry->budget = budget_;
    entry->allocation = allocation;
    entry->pool = id_;
    entry->charge = charge;
    std::byte* value = entry_value_data(*entry);
    char* key_data = reinterpret_cast<char*>(value + value_size);
    std::memcpy(key_data, key.data(), key.size());
    entry->key = {key_data, key.size()};
    entry->value = {value, value_size};
    std::fill(entry->value.begin(), entry->value.end(), std::byte{0});
    counters_.allocations.fetch_add(1, std::memory_order_relaxed);
    return entry;
  }
  void unlink(Entry& entry) {
    EntryPtr retired;
    index_.write(entry.key, [&](CacheIndex::Slot& slot) {
      eviction_.remove(entry);
      retired = slot.remove();
    });
  }
  struct Reaped {
    std::size_t count = 0;
    bool examined = false;
    bool more = false;
  };
  // Requires the maintenance guard in either mode. The policy queue is locked
  // once per batch, after the stripe lock is released; the detached entries
  // are unreachable through the index, so nothing else touches their links.
  Reaped reap_stripe(std::size_t stripe, Duration now, std::size_t limit) {
    CacheIndex::Expired expired = index_.expire(stripe, now, limit);
    if (expired.chain) {
      eviction_.remove_chain(expired.chain);
      counters_.expirations.fetch_add(expired.count, std::memory_order_relaxed);
      while (expired.chain) {
        Entry* next = expired.chain->hash_next;
        expired.chain->hash_next = nullptr;
        EntryPtr released(expired.chain);
        expired.chain = next;
      }
    }
    return {expired.count, expired.examined != 0, expired.more};
  }
  std::size_t prune_locked(Duration now) {
    std::size_t count = 0;
    for (std::size_t stripe = 0; stripe < CacheIndex::stripe_count; ++stripe) {
      for (;;) {
        const Reaped step = reap_stripe(stripe, now, locked_reap_batch);
        count += step.count;
        if (!step.more) break;
      }
    }
    return count;
  }
  bool make_room(std::size_t target, std::string_view protected_key) {
    if (budget_->used.load(std::memory_order_relaxed) <= target) return true;
    for (Entry* entry : eviction_.entries(true)) {
      if (budget_->used.load(std::memory_order_relaxed) <= target) break;
      if (entry->key != protected_key && entry->references.load(std::memory_order_acquire) == 1) {
        unlink(*entry);
        counters_.evictions.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return budget_->used.load(std::memory_order_relaxed) <= target;
  }
  void clear_locked() noexcept {
    eviction_.clear();
    index_.clear();
  }

  // Exclusive callers already block all traffic, so larger batches are fine.
  static constexpr std::size_t locked_reap_batch = 1024;

  const PoolId id_;
  PoolConfig config_;
  const std::shared_ptr<const Owner> owner_;
  const std::shared_ptr<const Clock> clock_;
  std::shared_ptr<MemoryAllocator> budget_;
  mutable SharedMutex maintenance_;
  bool closed_ = false;
  CacheIndex index_;
  EvictionQueue eviction_;
  AtomicCounters counters_;
  std::atomic<std::size_t> reap_cursor_{0};
  AutoCompaction auto_compaction_;
};
}  // namespace nivo::detail
