#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nivo/config.h"

namespace nivo {
// Item storage mappings only: excludes indexes, pool objects and OS page tables.
// These figures describe allocator-owned bytes, not process RSS.
struct AllocatorStats {
  std::size_t requested_bytes = 0;  // live entry metadata, keys and values
  std::size_t active_bytes = 0;     // live size-class slots (including rounding)
  std::size_t reserved_bytes = 0;   // mapped regions and large blocks, including page rounding
  std::size_t live_allocations = 0;
  std::size_t slabs = 0;
  std::size_t large_allocations = 0;
  std::uint64_t system_allocations = 0;
  std::uint64_t reused_allocations = 0;
  std::size_t regions = 0;         // independently releasable slab batches
  std::size_t empty_slabs = 0;     // includes empty slabs pinned by a live region
  std::size_t reserve_limit_bytes = 0;  // adaptive reserve budget; excludes startup reserve, not an RSS cap
  std::size_t refill_batch_size = 1;    // target; cache aggregate reports the maximum
  std::uint64_t batch_refills = 0;
  std::uint64_t batch_fallbacks = 0;     // failed batches retried with one slab
  std::size_t metadata_bytes = 0;       // region and slab headers
  std::size_t reclaimable_bytes = 0;    // wholly idle regions, including headers
};

// Current item-storage gauges, in bytes. reserved_bytes is the outstanding
// OS mapping size including page rounding, NOT process RSS. It excludes indexes,
// queue nodes, pool/cache objects, temporary compaction plans and OS page tables.
// Mapped pages need not be resident. These are not estimates of total RSS.
struct MemoryStats {
  std::size_t reserved_bytes = 0;
  // Every allocated key+value: indexed entries, drafts and detached snapshots.
  // An entry is counted once regardless of the number of handles holding it.
  std::size_t kv_bytes = 0;
  // Indexed data includes expired entries until discovered or explicitly pruned.
  std::size_t indexed_kv_bytes = 0;
  std::size_t retained_kv_bytes = 0;  // kv_bytes - indexed_kv_bytes: drafts/old snapshots
  std::size_t entry_metadata_bytes = 0;
  std::size_t allocator_metadata_bytes = 0;
  std::size_t internal_fragmentation_bytes = 0;  // live slot rounding only
  // Unoccupied slab payload and mapping padding, including unusable tails.
  // Not all unused bytes are reusable or releasable while a region is live.
  std::size_t unused_bytes = 0;
  // Entirely empty regions that trim/compact can release, including headers.
  // This overlaps allocator_metadata_bytes and unused_bytes; do not add it.
  std::size_t reclaimable_bytes = 0;
  // reserved = kv + entry_metadata + allocator_metadata + internal_fragmentation + unused
};
struct PoolMemoryStats {
  PoolId id = invalid_pool;
  std::string name;
  MemoryStats memory;
};
struct CacheMemoryStats {
  MemoryStats total;  // sum of the pool samples below
  std::vector<PoolMemoryStats> pools;
  // Current resident memory of the ENTIRE hosting process, including other
  // caches/libraries and application data. Never sum this across cache instances.
  // Sampled separately from pool storage. Unavailable is nullopt, not zero.
  std::optional<std::size_t> process_rss_bytes;
};
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
struct CompactionStats {
  std::size_t pools = 0;
  std::size_t moved_items = 0;
  // Surviving indexed entries not moved: pinned, in a region containing pins
  // or drafts, directly allocated, or already packed without a reclaim gain.
  std::size_t skipped_items = 0;
  std::size_t expired_items = 0;
  std::size_t reserved_bytes_before = 0;
  std::size_t reserved_bytes_after = 0;
};
struct PoolStats {
  PoolId id = invalid_pool;
  std::string name;
  std::size_t capacity_bytes = 0;
  std::size_t allocated_bytes = 0;  // drafts + indexed entries + detached pinned entries
  std::size_t indexed_bytes = 0;
  std::size_t indexed_items = 0;  // may include expired entries awaiting discovery
  std::size_t ttl_items = 0;      // indexed entries with a TTL, expired or not
  Counters counters;
  AllocatorStats allocator;
};
// Background reaper activity. Counters.expirations counts every expiry,
// including lazy discovery and prune_expired().
struct ReaperStats {
  std::uint64_t passes = 0;      // completed or interrupted background passes
  std::uint64_t reclaimed = 0;   // expired entries removed by the background reaper
  std::uint64_t throttles = 0;   // duty-cycle sleeps taken during passes
  std::uint64_t compactions = 0; // background compactions (ReaperConfig::compact_unused_percent)
  Duration last_pass{0};         // wall time of the latest pass, including sleeps
  Duration max_pass{0};
};
struct CacheStats {
  std::size_t capacity_bytes = 0;
  std::size_t assigned_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t indexed_bytes = 0;
  std::size_t indexed_items = 0;
  std::size_t ttl_items = 0;
  std::size_t pools = 0;
  bool closed = false;
  Counters counters;
  AllocatorStats allocator;
  ReaperStats reaper;
};
}  // namespace nivo
