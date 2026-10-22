#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "nivo/config.h"

namespace nivo {
struct Counters {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t allocations = 0;
  std::uint64_t allocation_failures = 0;
  std::uint64_t inserts = 0;
  std::uint64_t replacements = 0;
  std::uint64_t erases = 0;
  std::uint64_t evictions = 0;
  std::uint64_t expirations = 0;
};
struct PoolStats {
  PoolId id = invalid_pool;
  std::string name;
  std::size_t capacity_bytes = 0;
  std::size_t allocated_bytes = 0;  // drafts + indexed entries + detached pinned entries
  std::size_t indexed_bytes = 0;
  std::size_t indexed_items = 0;  // may include expired entries awaiting discovery
  Counters counters;
};
struct CacheStats {
  std::size_t capacity_bytes = 0;
  std::size_t assigned_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t indexed_bytes = 0;
  std::size_t indexed_items = 0;
  std::size_t pools = 0;
  bool closed = false;
  Counters counters;
};
}  // namespace nivo
