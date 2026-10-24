#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "cache_entry.h"
#include "shared_mutex.h"

namespace nivo::detail {
// Fixed buckets avoid rehash synchronization. Entries own their keys and links;
// the table owns one reference per indexed entry, with no per-key allocation.
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
        *link_ = old->hash_next;
        old->hash_next = nullptr;
        entry_ = nullptr;
        --stripe_.items;
        stripe_.bytes -= old->charge;
      }
      return EntryPtr(old);
    }
    EntryPtr replace(Entry& entry) noexcept {
      Entry* old = entry_;
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
   private:
    friend class CacheIndex;
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

  struct Stats { std::size_t items = 0; std::size_t bytes = 0; };
  Stats stats() const {
    Stats result;
    for (const Stripe& stripe : stripes_) {
      std::shared_lock<SharedMutex> lock(stripe.mutex);
      result.items += stripe.items;
      result.bytes += stripe.bytes;
    }
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
            detached = entry;
            entry = next;
          }
        }
        stripe.items = 0;
        stripe.bytes = 0;
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
