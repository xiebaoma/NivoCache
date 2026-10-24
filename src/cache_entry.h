#pragma once

#include <atomic>
#include <limits>
#include <optional>

#include "memory_allocator.h"
#include "nivo/handle.h"

namespace nivo::detail {
struct Owner {
  // Zero is reserved for invalid versions. Never wrap and reuse a version.
  Version next_version() const noexcept {
    Version current = version.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<Version>::max()) {
      if (version.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
        return current + 1;
    }
    return invalid_version;
  }
  mutable std::atomic<Version> version{0};
};

struct Entry {
  std::atomic<std::size_t> references{1};
  std::shared_ptr<const Owner> owner;
  std::shared_ptr<MemoryAllocator> budget;
  MemoryAllocator::Allocation allocation;
  std::string_view key;
  std::span<std::byte> value;
  PoolId pool = invalid_pool;
  Version version = invalid_version;
  std::size_t charge = 0;
  std::optional<Duration> expires_at;
  // The index and eviction policy own separate intrusive links.
  Entry* hash_next = nullptr;
  Entry* policy_previous = nullptr;
  Entry* policy_next = nullptr;
  std::atomic<std::uint64_t> access_sequence{0};
};

// Call only with an existing owner or an index lock protecting the entry.
inline EntryPtr pin_entry(Entry* entry) noexcept {
  if (entry) entry->references.fetch_add(1, std::memory_order_relaxed);
  return EntryPtr(entry);
}
inline bool expired(const Entry& entry, Duration now) noexcept {
  return entry.expires_at && now >= *entry.expires_at;
}
}  // namespace nivo::detail
