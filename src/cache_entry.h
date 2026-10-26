#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "memory_allocator.h"
#include "nivo/handle.h"

namespace nivo::detail {
// A maximal deadline keeps the read-path expiry test to a single comparison.
inline constexpr Duration never_expires = Duration::max();
inline constexpr std::uint16_t expiry_unlinked = std::numeric_limits<std::uint16_t>::max();

// Scoped to the originating cache/index. Versions are never reused and survive
// compaction; the hash only routes an identity lookup to the existing bucket.
// No borrowed key or address is retained, including after removal/replacement.
struct EntryId {
  Version version = invalid_version;
  std::size_t key_hash = 0;
  bool operator==(const EntryId&) const noexcept = default;
};

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

// Entry storage is followed by the value bytes and then the key bytes. Aligning
// Entry to max_align_t keeps the value start suitably aligned for any
// fundamental type, which typed zero-copy reads rely on.
struct alignas(std::max_align_t) Entry {
  std::atomic<std::size_t> references{1};
  std::shared_ptr<const Owner> owner;
  std::shared_ptr<MemoryAllocator> budget;
  MemoryAllocator::Allocation allocation;
  std::string_view key;
  std::span<std::byte> value;
  PoolId pool = invalid_pool;
  std::uint16_t expiry_slot = expiry_unlinked;  // fills padding before version
  Version version = invalid_version;
  std::size_t charge = 0;
  Duration expires_at = never_expires;
  // The index, eviction policy and expiry wheel own separate intrusive links.
  Entry* hash_next = nullptr;
  Entry* policy_previous = nullptr;
  Entry* policy_next = nullptr;
  Entry* expiry_previous = nullptr;
  Entry* expiry_next = nullptr;
  // Interpreted only by the pool's eviction policy (see eviction_policy.h).
  // An unpublished 2Q draft may carry a history hint in the high bit.
  std::atomic<std::uint64_t> policy_state{0};
};

static_assert(sizeof(Entry) % alignof(std::max_align_t) == 0);

inline std::byte* entry_value_data(Entry& entry) noexcept {
  return reinterpret_cast<std::byte*>(&entry + 1);
}

// Call only with an existing owner or an index lock protecting the entry.
inline EntryPtr pin_entry(Entry* entry) noexcept {
  if (entry) entry->references.fetch_add(1, std::memory_order_relaxed);
  return EntryPtr(entry);
}
inline bool expired(const Entry& entry, Duration now) noexcept {
  return now >= entry.expires_at;
}
}  // namespace nivo::detail
