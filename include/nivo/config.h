#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nivo {
using PoolId = std::uint32_t;
using Version = std::uint64_t;
inline constexpr PoolId invalid_pool = 0;
inline constexpr Version invalid_version = 0;
using Duration = std::chrono::milliseconds;

// Implementations must be thread safe, nondecreasing, nonnegative, and must not
// call back into Cache. Millisecond ticks are local to this cache's clock domain.
class Clock {
 public:
  virtual ~Clock() = default;
  [[nodiscard]] virtual Duration now() const noexcept = 0;
};

// Capacity-driven eviction, selected per pool and fixed for its lifetime.
// Only find() hits change recency; W-TinyLFU also counts find() misses for
// admission. peek(), failed writes and compaction never
// change policy state. Each successful publication is a new version: the
// replaced version leaves the policy at once, even while handles still pin it.
// Pinned entries, unpublished drafts and the key being written are never
// chosen, so a request fails with resource_exhausted when only they remain.
//  lru:    evicts the entry least recently found or published. Exact order;
//          a find moves the entry under the pool's policy lock in O(1).
//  fifo:   evicts in publication order; finds never change it. Replacing a
//          key publishes it again, so it becomes the newest entry.
//  lfu:    evicts the lowest access count, least recently placed among equal
//          counts. Counts start at 1, saturate at 15 and are all halved after
//          every 8 finds per entry (at least 64), so old popularity decays.
//          A replacement keeps the key's current count.
//  clock:  second-chance approximation of LRU. A find sets a reference bit
//          without locking; the hand clears set bits and offers clear ones,
//          sweeping at most twice per eviction. New versions start unreferenced
//          just behind the hand.
//  random: uniform among indexed entries; reproducible with eviction_seed.
//  no_eviction: never evicts. Allocations that do not fit fail with
//          resource_exhausted and shrinking below the used bytes fails.
//          Expired entries are still reclaimed under pressure.
//  sieve:  SIEVE (Zhang et al., NSDI 2024). New keys enter the head of one
//          FIFO queue and a find only sets a visited bit, without locking or
//          moving the entry. A persistent hand walks from the tail towards
//          the head, clearing visited bits and evicting the first unvisited
//          entry; the hand stays where it stopped and wraps to the tail.
//  s3_fifo: S3-FIFO (Yang et al., SOSP 2023): small, main and ghost FIFO
//          queues tuned by S3FifoConfig. A find atomically increments a
//          capped counter without locking. Queue shares are measured in key
//          plus value bytes; the ghost queue keeps only key fingerprints.
//  slru: two LRU segments, with promotion on a probation hit.
//  two_q: A1in FIFO, bounded A1out history, and Am LRU.
//  w_tinylfu: Window LRU plus main SLRU with frequency-based admission.
// For sieve, s3_fifo, slru, two_q and w_tinylfu a replacement keeps the key's
// queue position and its visited bit or counter; it is a write, not an access.
enum class EvictionPolicy {
  lru, fifo, lfu, clock, random, no_eviction, sieve, s3_fifo, slru, two_q, w_tinylfu
};

// Parameters for EvictionPolicy::s3_fifo, ignored by other policies. The
// defaults follow the S3-FIFO paper's Algorithm 1.
struct S3FifoConfig {
  // The small queue's share of the bytes the policy tracks, 1 to 99.
  std::uint32_t small_queue_percent = 10;
  // Ghost capacity in entries, as a percentage of the tracked entry count, so
  // 90 matches the main queue's share. 0 disables the ghost queue; at most 400.
  std::uint32_t ghost_queue_percent = 90;
  // Access counter cap, 1 to 7. The paper uses two bits (3).
  std::uint32_t max_frequency = 3;
  // A small-queue tail with at least this count moves to the main queue
  // (the paper's "freq > 1"), 1 to max_frequency.
  std::uint32_t promotion_frequency = 2;
};
// Segment targets use a share of pool capacity in key plus value bytes,
// excluding drafts and retired versions pinned by handles. Replacements
// preserve segment and order unless a larger value forces demotion.
struct SlruConfig {
  std::uint32_t protected_percent = 80;  // 1 to 99; remainder is probation
};
// Full 2Q: A1in hits remain FIFO; only an A1out history hit enters Am.
// A1in's target is a share of indexed bytes (it may borrow unused Am space).
struct TwoQConfig {
  std::uint32_t a1in_percent = 25;  // 1 to 99
  // Fixed maximum of 64-bit historical key hashes, no Value or Entry pointers.
  // Hash collisions can cause false history hits. 0 disables history.
  std::size_t ghost_capacity = 4096;  // at most 1,048,576
};
// Window TinyLFU with a main SLRU and a four-row Count-Min Sketch. All find()
// requests, including misses, train admission; peek and writes do not. Ties
// retain the resident victim. This variant uses no Doorkeeper. Window/main
// byte shares are soft targets: unused space can be borrowed, and an indivisible
// large entry can exceed the window target. Admission is deferred until allocation
// pressure; a candidate too large for main remains in window until evicted.
struct WTinyLfuConfig {
  std::uint32_t window_percent = 1;  // 1 to 99 of pool capacity in bytes
  std::uint32_t protected_percent = 80;  // 1 to 99 of main capacity
  // Power of two, 64 to 1,048,576. Each of four rows has this many 8-byte
  // atomic epoch/counter cells; storage is fixed at pool creation.
  std::size_t sketch_width = 16384;
  // Halve frequencies after this many find requests (at least 8192).
  // Batched accounting delays an epoch by at most 63 * 64 requests.
  std::size_t sample_size = 65536;
};
enum class StorageMode { memory, hybrid };  // hybrid is explicitly unsupported in v0.1

// Background reclamation of expired entries. Expired entries are never
// returned either way; reaping only frees their memory without waiting for a
// later access. Each pass costs O(expired entries): TTL deadlines are kept in
// per-stripe timing wheels, so entries that are not due are never visited.
struct ReaperConfig {
  // Delay from the end of one pass to the start of the next, on the steady
  // clock. Zero disables the background worker; expired entries are then
  // reclaimed on access, under allocation pressure, or by prune_expired().
  Duration interval{1000};
  // Due entries examined per index-stripe lock hold, which bounds how long
  // any single read or write can wait for the reaper. Must be positive.
  std::size_t batch_size = 64;
  // Duty cycle: after work_slice of continuous reaping, sleep for sleep_slice
  // before continuing, so large expiry bursts drain smoothly over time.
  // A zero sleep_slice disables throttling.
  Duration work_slice{5};
  Duration sleep_slice{10};
  // Optional background compaction, off by default. Reaping frees slots, but
  // slabs that still hold live entries stay mapped. After the reaper reclaims
  // entries from a pool, it compacts that pool when the unused space inside
  // its mapped storage (MemoryStats::unused_bytes) is at least
  // compact_unused_percent of its reserved bytes and at least
  // compact_min_unused_bytes. Compaction takes the exclusive pool guard, so it
  // briefly blocks that pool's reads and writes; other pools are unaffected.
  // Each pool is compacted at most once per compact_cooldown. Needs a nonzero
  // interval, since the background worker performs it.
  std::uint32_t compact_unused_percent = 0;  // zero disables; at most 100
  std::size_t compact_min_unused_bytes = 8 * 1024 * 1024;
  Duration compact_cooldown{60000};
};

struct CacheConfig {
  std::size_t capacity_bytes = 64 * 1024 * 1024;
  std::size_t max_key_bytes = 1024;
  std::size_t max_value_bytes = 16 * 1024 * 1024;
  StorageMode storage = StorageMode::memory;
  std::shared_ptr<const Clock> clock;  // null selects steady_clock; also drives TTL reaping
  ReaperConfig reaper;
};

struct PoolConfig {
  std::string name;
  std::size_t capacity_bytes = 0;
  EvictionPolicy eviction = EvictionPolicy::lru;
  // Zero disables startup reservation. Otherwise add_pool() synchronously
  // reserves and write-touches slab mappings, rounded up to whole slabs plus
  // region headers and OS page rounding. Must not exceed capacity_bytes. Direct large allocations
  // bypass this reserve. This is resident-page preparation, not memory locking.
  std::size_t initial_reserve_bytes = 0;
  // Positive elapsed-time retention from completion of preparation, independent
  // of CacheConfig::clock. Unused regions revert to normal retention at expiry,
  // even without requests. Trim/clear, resize, compact and close may release
  // them earlier. Startup storage is never replenished automatically.
  Duration initial_reserve_timeout{30000};
  // Seeds EvictionPolicy::random; zero draws a nondeterministic seed.
  std::uint64_t eviction_seed = 0;
  S3FifoConfig s3_fifo;
  SlruConfig slru;
  TwoQConfig two_q;
  WTinyLfuConfig w_tinylfu;
};

enum class WriteMode { upsert, insert_only, replace_only, compare_exchange };
struct WriteOptions {
  WriteMode mode = WriteMode::upsert;
  Version expected_version = invalid_version;  // nonzero only for compare_exchange
  Duration ttl{0};  // zero = no expiration; positive = starts at publication
};

struct Capabilities {
  bool memory = true;
  bool hybrid = false;
  bool persistence = false;
  bool async = false;
  bool chained_items = false;
  bool object_cache = false;
};
}  // namespace nivo
