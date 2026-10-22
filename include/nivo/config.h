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

enum class EvictionPolicy { lru, fifo };
enum class StorageMode { memory, hybrid };  // hybrid is explicitly unsupported in v0.1

struct CacheConfig {
  std::size_t capacity_bytes = 64 * 1024 * 1024;
  std::size_t max_key_bytes = 1024;
  std::size_t max_value_bytes = 16 * 1024 * 1024;
  StorageMode storage = StorageMode::memory;
  std::shared_ptr<const Clock> clock;  // null selects steady_clock
};

struct PoolConfig {
  std::string name;
  std::size_t capacity_bytes = 0;
  EvictionPolicy eviction = EvictionPolicy::lru;
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
