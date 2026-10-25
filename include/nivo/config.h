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

// Selected per pool at runtime; the choice is fixed for the pool's lifetime.
// LRU refreshes on every find hit; FIFO orders successful publications only.
enum class EvictionPolicy { lru, fifo };
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
