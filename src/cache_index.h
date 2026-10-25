#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <vector>

#include "cache_entry.h"
#include "expiry_wheel.h"
#include "shared_mutex.h"

namespace nivo::detail {
// Fixed buckets avoid rehash synchronization. Entries own their keys and links;
// the table owns one reference per indexed entry, with no per-key allocation.
// Each stripe also tracks its indexed TTL entries in an expiry wheel under the
// same lock, so every index change keeps the wheel exact for free.
class CacheIndex final {
 public:
  static constexpr std::size_t stripe_count = 64;
  explicit CacheIndex(std::size_t capacity)
      : buckets_(std::clamp(std::bit_ceil(std::min(capacity / 128, std::size_t{1} << 20)),
                            stripe_count, std::size_t{1} << 20), nullptr) {}
  CacheIndex(const CacheIndex&) = delete;
  CacheIndex& operator=(const CacheIndex&) = delete;
  ~CacheIndex() { clear(); }

  static std::size_t hash(std::string_view key) noexcept {
    return std::hash<std::string_view>{}(key);
  }
  std::size_t bucket_for(std::string_view key) const noexcept {
    return hash(key) & (buckets_.size() - 1);
  }
  std::size_t bucket_count() const noexcept { return buckets_.size(); }

  EntryPtr find(std::string_view key) const {
    const std::size_t bucket = bucket_for(key);
    std::shared_lock<SharedMutex> lock(stripes_[bucket % stripe_count].mutex);
    Entry* entry = buckets_[bucket];
    while (entry && entry->key != key) entry = entry->hash_next;
    return pin_entry(entry);
  }

 private:
  struct alignas(64) Stripe {
    mutable SharedMutex mutex;
    std::size_t items = 0;
    std::size_t bytes = 0;
    // Allocated by the first TTL publication, so pools without TTLs pay nothing.
    std::unique_ptr<ExpiryWheel> wheel;
    // Mirrors wheel->size() so reapers can skip idle stripes without locking.
    std::atomic<std::size_t> expiring{0};
  };

 public:
  // A slot is valid only for the duration of write(). Returned references from
  // replace/remove adopt the old index reference and can be released afterward.
  class Slot final {
   public:
    Entry* get() const noexcept { return entry_; }
    EntryPtr remove() noexcept {
      Entry* old = entry_;
      if (old) {
        forget(*old);
        *link_ = old->hash_next;
        old->hash_next = nullptr;
        entry_ = nullptr;
        --stripe_.items;
        stripe_.bytes -= old->charge;
      }
      return EntryPtr(old);
    }
    // A deadline requires a successful prepare_expiry() on this slot first.
    EntryPtr replace(Entry& entry) noexcept {
      Entry* old = entry_;
      if (old) forget(*old);
      track(entry);
      entry.references.fetch_add(1, std::memory_order_relaxed);
      entry.hash_next = old ? old->hash_next : *link_;
      *link_ = &entry;
      entry_ = &entry;
      stripe_.bytes += entry.charge;
      if (old) {
        old->hash_next = nullptr;
        stripe_.bytes -= old->charge;
      } else {
        ++stripe_.items;
      }
      return EntryPtr(old);
    }
    // Ensures the stripe can track a deadline. Fails only if allocation fails.
    bool prepare_expiry(Duration now) noexcept {
      if (!stripe_.wheel) stripe_.wheel.reset(new (std::nothrow) ExpiryWheel(now));
      return stripe_.wheel != nullptr;
    }
   private:
    friend class CacheIndex;
    void track(Entry& entry) noexcept {
      if (entry.expires_at == never_expires) return;
      assert(stripe_.wheel);
      stripe_.wheel->insert(entry);
      stripe_.expiring.store(stripe_.wheel->size(), std::memory_order_relaxed);
    }
    void forget(Entry& entry) noexcept {
      if (entry.expiry_slot == expiry_unlinked) return;
      stripe_.wheel->remove(entry);
      stripe_.expiring.store(stripe_.wheel->size(), std::memory_order_relaxed);
    }
    Slot(Entry** link, Stripe& stripe) : link_(link), entry_(*link), stripe_(stripe) {}
    Entry** link_;
    Entry* entry_;
    Stripe& stripe_;
  };

  template <typename Function>
  std::invoke_result_t<Function, Slot&> write(std::string_view key, Function&& function) {
    const std::size_t bucket = bucket_for(key);
    Stripe& stripe = stripes_[bucket % stripe_count];
    std::unique_lock<SharedMutex> lock(stripe.mutex);
    Entry** link = &buckets_[bucket];
    while (*link && (*link)->key != key) link = &(*link)->hash_next;
    Slot slot(link, stripe);
    return function(slot);
  }

  struct Stats { std::size_t items = 0; std::size_t bytes = 0; std::size_t expiring = 0; };
  Stats stats() const {
    Stats result;
    for (const Stripe& stripe : stripes_) {
      std::shared_lock<SharedMutex> lock(stripe.mutex);
      result.items += stripe.items;
      result.bytes += stripe.bytes;
      if (stripe.wheel) result.expiring += stripe.wheel->size();
    }
    return result;
  }

  struct Expired {
    Entry* chain = nullptr;  // linked through hash_next; owns the index references
    std::size_t count = 0;
    std::size_t examined = 0;
    bool more = false;       // due entries remain beyond the examined limit
  };
  // Examines at most limit due entries of one stripe under its lock, never
  // entries whose deadline is still in the future. Expired entries leave the
  // index and are returned so the caller can unlink them from the policy and
  // release them outside this lock; the rest cascade to finer wheel levels.
  Expired expire(std::size_t stripe_id, Duration now, std::size_t limit) {
    Expired result;
    Stripe& stripe = stripes_[stripe_id];
    if (stripe.expiring.load(std::memory_order_relaxed) == 0) return result;
    std::unique_lock<SharedMutex> lock(stripe.mutex);
    ExpiryWheel* wheel = stripe.wheel.get();
    if (!wheel) return result;
    // Another reaper may have advanced further with a later clock reading.
    now = std::max(now, wheel->current());
    wheel->advance(now);
    for (; result.examined < limit; ++result.examined) {
      Entry* entry = wheel->pop_pending();
      if (!entry) break;
      if (!expired(*entry, now)) {
        wheel->insert(*entry);
        continue;
      }
      Entry** link = &buckets_[bucket_for(entry->key)];
      while (*link != entry) link = &(*link)->hash_next;
      *link = entry->hash_next;
      --stripe.items;
      stripe.bytes -= entry->charge;
      entry->hash_next = result.chain;
      result.chain = entry;
      ++result.count;
    }
    result.more = wheel->has_pending();
    stripe.expiring.store(wheel->size(), std::memory_order_relaxed);
    return result;
  }

  // Pool maintenance prevents concurrent operations while clearing. Separate
  // stripes still protect each chain; destruction happens outside their locks.
  void clear() noexcept {
    for (std::size_t stripe_id = 0; stripe_id < stripe_count; ++stripe_id) {
      Entry* detached = nullptr;
      {
        Stripe& stripe = stripes_[stripe_id];
        std::unique_lock<SharedMutex> lock(stripe.mutex);
        for (std::size_t bucket = stripe_id; bucket < buckets_.size(); bucket += stripe_count) {
          Entry* entry = std::exchange(buckets_[bucket], nullptr);
          while (entry) {
            Entry* next = entry->hash_next;
            entry->hash_next = detached;
            entry->expiry_previous = entry->expiry_next = nullptr;
            entry->expiry_slot = expiry_unlinked;
            detached = entry;
            entry = next;
          }
        }
        stripe.items = 0;
        stripe.bytes = 0;
        if (stripe.wheel) stripe.wheel->reset();
        stripe.expiring.store(0, std::memory_order_relaxed);
      }
      while (detached) {
        Entry* next = detached->hash_next;
        detached->hash_next = nullptr;
        EntryPtr released(detached);
        detached = next;
      }
    }
  }

 private:
  std::vector<Entry*> buckets_;
  std::array<Stripe, stripe_count> stripes_;
};
}  // namespace nivo::detail
