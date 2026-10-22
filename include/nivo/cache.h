#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "nivo/config.h"
#include "nivo/handle.h"
#include "nivo/stats.h"
#include "nivo/status.h"

namespace nivo {
// Thread-safe synchronous facade. A key is scoped to a pool. Single-key
// operations are linearizable; put() is allocate()+publish(), not a transaction.
// No ABI stability is promised before 1.0. See docs/api-semantics.md.
class Cache final {
 public:
  [[nodiscard]] static Result<std::unique_ptr<Cache>> create(CacheConfig config = {});
  ~Cache();
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;
  Cache(Cache&&) = delete;
  Cache& operator=(Cache&&) = delete;

  [[nodiscard]] static constexpr Capabilities capabilities() noexcept { return {}; }
  [[nodiscard]] Result<PoolId> add_pool(PoolConfig config);
  // Shrinking may evict. On failure the old capacity remains; evictions remain.
  [[nodiscard]] Status resize_pool(PoolId pool, std::size_t capacity_bytes);
  [[nodiscard]] Result<std::vector<PoolStats>> pools() const;
  [[nodiscard]] Result<PoolStats> pool_stats(PoolId pool) const;
  [[nodiscard]] CacheStats stats() const;

  // Zero-initialized raw bytes; charges key.size()+value_size. May evict other
  // keys, but never this key (so an update can fail without losing its old value).
  [[nodiscard]] Result<WriteHandle> allocate(PoolId pool, std::string_view key,
                                            std::size_t value_size);
  // Consumes draft ONLY on success. Failure keeps it writable and retryable.
  [[nodiscard]] Result<ReadHandle> publish(WriteHandle&& draft, WriteOptions options = {});
  [[nodiscard]] Result<ReadHandle> put(PoolId pool, std::string_view key,
                                      std::span<const std::byte> value,
                                      WriteOptions options = {});
  [[nodiscard]] Result<ReadHandle> find(PoolId pool, std::string_view key);
  // No hit/miss counters or recency update; still discovers/removes expired data.
  [[nodiscard]] Result<ReadHandle> peek(PoolId pool, std::string_view key);
  // expected_version==0 is unconditional; otherwise erase only that version.
  [[nodiscard]] Status erase(PoolId pool, std::string_view key,
                              Version expected_version = invalid_version);
  [[nodiscard]] Status clear(PoolId pool);
  // Explicit expiry maintenance; no background threads in the reference engine.
  [[nodiscard]] Result<std::size_t> prune_expired(PoolId pool);
  // Idempotent; unlinks everything, rejects future data/control operations.
  // stats(), pool_stats(), pools(), and existing handles remain usable.
  void close();

 private:
  struct Impl;
  explicit Cache(std::unique_ptr<Impl> impl);
  [[nodiscard]] Result<ReadHandle> lookup(PoolId pool, std::string_view key, bool touch);
  std::unique_ptr<Impl> impl_;
};
}  // namespace nivo
