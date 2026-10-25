#include "nivo/cache.h"
#include "cache_pool.h"
#include "process_memory.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <utility>

namespace nivo {
namespace detail {
EntryPtr::EntryPtr(const EntryPtr& other) noexcept : entry_(other.entry_) {
  if (entry_) entry_->references.fetch_add(1, std::memory_order_relaxed);
}
EntryPtr& EntryPtr::operator=(const EntryPtr& other) noexcept {
  EntryPtr copy(other);
  std::swap(entry_, copy.entry_);
  return *this;
}
EntryPtr& EntryPtr::operator=(EntryPtr&& other) noexcept {
  EntryPtr moved(std::move(other));
  std::swap(entry_, moved.entry_);
  return *this;
}
EntryPtr::~EntryPtr() { reset(); }
std::size_t EntryPtr::use_count() const noexcept {
  return entry_ ? entry_->references.load(std::memory_order_acquire) : 0;
}
void EntryPtr::reset() noexcept {
  Entry* entry = std::exchange(entry_, nullptr);
  if (!entry || entry->references.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  // Keep storage alive through destruction of the entry's own allocator owner.
  std::shared_ptr<MemoryAllocator> allocator = std::move(entry->budget);
  const MemoryAllocator::Allocation allocation = entry->allocation;
  const std::size_t charge = entry->charge;
  entry->~Entry();
  allocator->deallocate(allocation);
  allocator->used.fetch_sub(charge, std::memory_order_relaxed);
}
}  // namespace detail

std::span<const std::byte> ReadHandle::bytes() const noexcept {
  return entry_ ? std::span<const std::byte>(entry_->value) : std::span<const std::byte>{};
}
std::string_view ReadHandle::key() const noexcept { return entry_ ? entry_->key : std::string_view{}; }
PoolId ReadHandle::pool() const noexcept { return entry_ ? entry_->pool : invalid_pool; }
Version ReadHandle::version() const noexcept { return entry_ ? entry_->version : invalid_version; }
std::size_t ReadHandle::charge() const noexcept { return entry_ ? entry_->charge : 0; }
std::span<std::byte> WriteHandle::bytes() noexcept {
  return entry_ ? std::span<std::byte>(entry_->value) : std::span<std::byte>{};
}
std::string_view WriteHandle::key() const noexcept { return entry_ ? entry_->key : std::string_view{}; }
PoolId WriteHandle::pool() const noexcept { return entry_ ? entry_->pool : invalid_pool; }
std::size_t WriteHandle::charge() const noexcept { return entry_ ? entry_->charge : 0; }

namespace {
class SteadyClock final : public Clock {
 public:
  Duration now() const noexcept override {
    return std::chrono::duration_cast<Duration>(std::chrono::steady_clock::now().time_since_epoch());
  }
};
Status validate(const WriteOptions& options) {
  if (options.ttl.count() < 0) return Status::invalid_argument;
  switch (options.mode) {
    case WriteMode::upsert:
    case WriteMode::insert_only:
    case WriteMode::replace_only:
      return options.expected_version == 0 ? Status::ok : Status::invalid_argument;
    case WriteMode::compare_exchange:
      return options.expected_version != 0 ? Status::ok : Status::invalid_argument;
  }
  return Status::invalid_argument;
}
void accumulate(Counters& a, const Counters& b) {
  a.hits += b.hits; a.misses += b.misses;
  a.allocations += b.allocations; a.allocation_failures += b.allocation_failures;
  a.inserts += b.inserts; a.replacements += b.replacements;
  a.erases += b.erases; a.evictions += b.evictions; a.expirations += b.expirations;
}
void accumulate(AllocatorStats& a, const AllocatorStats& b) {
  a.requested_bytes += b.requested_bytes;
  a.active_bytes += b.active_bytes;
  a.reserved_bytes += b.reserved_bytes;
  a.live_allocations += b.live_allocations;
  a.slabs += b.slabs;
  a.large_allocations += b.large_allocations;
  a.system_allocations += b.system_allocations;
  a.reused_allocations += b.reused_allocations;
  a.regions += b.regions;
  a.empty_slabs += b.empty_slabs;
  a.reserve_limit_bytes += b.reserve_limit_bytes;
  a.refill_batch_size = std::max(a.refill_batch_size, b.refill_batch_size);
  a.batch_refills += b.batch_refills;
  a.batch_fallbacks += b.batch_fallbacks;
  a.metadata_bytes += b.metadata_bytes;
  a.reclaimable_bytes += b.reclaimable_bytes;
}
void accumulate(MemoryStats& a, const MemoryStats& b) {
  a.reserved_bytes += b.reserved_bytes;
  a.kv_bytes += b.kv_bytes;
  a.indexed_kv_bytes += b.indexed_kv_bytes;
  a.retained_kv_bytes += b.retained_kv_bytes;
  a.entry_metadata_bytes += b.entry_metadata_bytes;
  a.allocator_metadata_bytes += b.allocator_metadata_bytes;
  a.internal_fragmentation_bytes += b.internal_fragmentation_bytes;
  a.unused_bytes += b.unused_bytes;
  a.reclaimable_bytes += b.reclaimable_bytes;
}
}  // namespace

struct Cache::Impl {
  using PoolMap = std::map<PoolId, std::unique_ptr<detail::CachePool>>;
  using SteadyTime = std::chrono::steady_clock::time_point;
  explicit Impl(CacheConfig config_) : config(std::move(config_)) {
    if (!config.clock) config.clock = std::make_shared<SteadyClock>();
  }
  ~Impl() {
    closed.store(true, std::memory_order_release);
    wake_worker();
    if (worker.joinable()) worker.join();
  }
  CacheConfig config;
  std::shared_ptr<const detail::Owner> owner = std::make_shared<detail::Owner>();
  // Registry locks are held only while accessing pool metadata. Pool addresses
  // remain stable until destruction. Slow control work never holds this lock.
  mutable detail::SharedMutex registry_mutex;
  std::mutex control_mutex;
  PoolMap pools;
  std::size_t assigned = 0;
  PoolId last_pool = 0;
  std::atomic<bool> closed{false};
  // One background thread serves startup reserves and TTL reaping. It never
  // holds worker_mutex while working, so wakeups and close() do not wait.
  std::mutex worker_mutex;
  std::condition_variable worker_wakeup;
  bool worker_signaled = false;
  std::thread worker;
  struct {
    std::atomic<std::uint64_t> passes{0}, reclaimed{0}, throttles{0}, compactions{0};
    std::atomic<Duration::rep> last_pass{0}, max_pass{0};
  } reaper_stats;

  bool reaping() const noexcept { return config.reaper.interval.count() > 0; }
  void wake_worker() {
    std::lock_guard<std::mutex> lock(worker_mutex);
    worker_signaled = true;
    worker_wakeup.notify_all();
  }
  // Visits pools in id order without holding the registry lock during work.
  template <typename Function>
  void for_each_pool(Function&& function) {
    PoolId previous = invalid_pool;
    for (;;) {
      detail::CachePool* pool;
      {
        std::shared_lock<detail::SharedMutex> registry_lock(registry_mutex);
        PoolMap::const_iterator entry = pools.upper_bound(previous);
        if (entry == pools.end()) return;
        previous = entry->first;
        pool = entry->second.get();
      }
      if (!function(*pool)) return;
    }
  }
  void run_worker() {
    SteadyTime next_reap = std::chrono::steady_clock::now() + config.reaper.interval;
    std::unique_lock<std::mutex> lock(worker_mutex);
    while (!closed.load(std::memory_order_acquire)) {
      worker_signaled = false;
      lock.unlock();
      std::optional<SteadyTime> next;
      for_each_pool([&](detail::CachePool& pool) {
        const std::optional<detail::MemoryAllocator::TimePoint> deadline = pool.maintain_reserve();
        if (deadline && (!next || *deadline < *next)) next = deadline;
        return true;
      });
      if (reaping()) {
        if (std::chrono::steady_clock::now() >= next_reap) {
          reap_pass();
          next_reap = std::chrono::steady_clock::now() + config.reaper.interval;
        }
        if (!next || next_reap < *next) next = next_reap;
      }
      lock.lock();
      if (next) worker_wakeup.wait_until(lock, *next, [this] { return worker_ready(); });
      else worker_wakeup.wait(lock, [this] { return worker_ready(); });
    }
  }
  bool worker_ready() const noexcept {
    return worker_signaled || closed.load(std::memory_order_acquire);
  }
  // Runs between reaping batches with no cache locks held. Enforces the
  // work/sleep duty cycle and stops the pass on close().
  class Throttle final {
   public:
    explicit Throttle(Impl& impl) : impl_(impl), slice_(std::chrono::steady_clock::now()) {}
    bool stopped() const noexcept { return stopped_ || impl_.closed.load(std::memory_order_acquire); }
    bool operator()() {
      if (stopped()) return false;
      const ReaperConfig& reaper = impl_.config.reaper;
      if (reaper.sleep_slice.count() == 0) return true;
      if (std::chrono::steady_clock::now() - slice_ < reaper.work_slice) return true;
      impl_.reaper_stats.throttles.fetch_add(1, std::memory_order_relaxed);
      {
        std::unique_lock<std::mutex> lock(impl_.worker_mutex);
        stopped_ = impl_.worker_wakeup.wait_for(lock, reaper.sleep_slice, [this] {
          return impl_.closed.load(std::memory_order_acquire);
        });
      }
      slice_ = std::chrono::steady_clock::now();
      return !stopped_;
    }
   private:
    Impl& impl_;
    SteadyTime slice_;
    bool stopped_ = false;
  };
  // Checks fragmentation only after reaping freed slots in this pool. A
  // pending check survives the cooldown, so a later pass still performs it.
  void maybe_compact(detail::CachePool& pool) {
    const ReaperConfig& reaper = config.reaper;
    detail::CachePool::AutoCompaction& state = pool.auto_compaction();
    if (reaper.compact_unused_percent == 0 || !state.pending) return;
    const SteadyTime now = std::chrono::steady_clock::now();
    if (state.ran && now - state.last < reaper.compact_cooldown) return;
    state.pending = false;
    const detail::CachePool::StorageUse use = pool.storage_use();
    if (use.unused < reaper.compact_min_unused_bytes ||
        use.unused / reaper.compact_unused_percent < use.reserved / 100) return;
    state.ran = true;
    state.last = now;
    if (pool.compact(pool.now())) reaper_stats.compactions.fetch_add(1, std::memory_order_relaxed);
  }
  // One throttled sweep of every pool. Each pool uses a fresh clock reading.
  void reap_pass() {
    const SteadyTime begin = std::chrono::steady_clock::now();
    Throttle throttle(*this);
    for_each_pool([&](detail::CachePool& pool) {
      Result<std::size_t> reaped = pool.reap(pool.now(), std::numeric_limits<std::size_t>::max(),
                                             config.reaper.batch_size, throttle);
      if (reaped) {
        reaper_stats.reclaimed.fetch_add(reaped.value(), std::memory_order_relaxed);
        if (reaped.value() != 0) pool.auto_compaction().pending = true;
      }
      if (throttle.stopped()) return false;
      maybe_compact(pool);
      // Lets spare slabs left by reaping decay back to the OS while idle.
      pool.maintain_storage();
      return !throttle.stopped();
    });
    const Duration elapsed =
        std::chrono::duration_cast<Duration>(std::chrono::steady_clock::now() - begin);
    reaper_stats.passes.fetch_add(1, std::memory_order_relaxed);
    reaper_stats.last_pass.store(elapsed.count(), std::memory_order_relaxed);
    if (elapsed.count() > reaper_stats.max_pass.load(std::memory_order_relaxed))
      reaper_stats.max_pass.store(elapsed.count(), std::memory_order_relaxed);
  }
  bool valid_key(std::string_view key) const {
    return !key.empty() && key.size() <= config.max_key_bytes;
  }
  Result<detail::CachePool*> get_pool(PoolId pool, bool allow_closed = false) const {
    std::shared_lock<detail::SharedMutex> lock(registry_mutex);
    if (!allow_closed && closed.load(std::memory_order_acquire)) return Status::closed;
    PoolMap::const_iterator it = pools.find(pool);
    if (it == pools.end()) return Status::pool_not_found;
    return it->second.get();
  }
};

Cache::Cache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Cache::~Cache() { close(); }
Result<std::unique_ptr<Cache>> Cache::create(CacheConfig config) {
  if (config.capacity_bytes == 0 || config.max_key_bytes == 0 || config.max_value_bytes == 0)
    return Status::invalid_argument;
  if (config.storage == StorageMode::hybrid) return Status::unsupported;
  if (config.storage != StorageMode::memory) return Status::invalid_argument;
  const ReaperConfig& reaper = config.reaper;
  if (reaper.interval.count() < 0 || reaper.batch_size == 0 || reaper.work_slice.count() < 0 ||
      reaper.sleep_slice.count() < 0 || reaper.compact_unused_percent > 100 ||
      reaper.compact_cooldown.count() < 0) return Status::invalid_argument;
  return std::unique_ptr<Cache>(new Cache(std::make_unique<Impl>(std::move(config))));
}
Result<PoolId> Cache::add_pool(PoolConfig config) {
  std::lock_guard<std::mutex> control(impl_->control_mutex);
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (config.name.empty() || config.capacity_bytes == 0) return Status::invalid_argument;
  if (config.initial_reserve_bytes > config.capacity_bytes || config.initial_reserve_timeout.count() <= 0)
    return Status::invalid_argument;
  if (config.eviction != EvictionPolicy::lru && config.eviction != EvictionPolicy::fifo)
    return Status::invalid_argument;
  {
    std::shared_lock<detail::SharedMutex> lock(impl_->registry_mutex);
    for (const Impl::PoolMap::value_type& entry : impl_->pools)
      if (entry.second->name() == config.name) return Status::already_exists;
    if (config.capacity_bytes > impl_->config.capacity_bytes - impl_->assigned ||
        impl_->last_pool == std::numeric_limits<PoolId>::max()) return Status::resource_exhausted;
  }
  const PoolId id = impl_->last_pool + 1;
  const std::size_t capacity = config.capacity_bytes;
  const bool reserve = config.initial_reserve_bytes != 0;
  const bool needs_worker = reserve || impl_->reaping();
  try {
    std::unique_ptr<detail::CachePool> pool = std::make_unique<detail::CachePool>(
        id, std::move(config), impl_->owner, impl_->config.clock);
    {
      std::unique_lock<detail::SharedMutex> lock(impl_->registry_mutex);
      impl_->pools.emplace(id, std::move(pool));
      if (needs_worker && !impl_->worker.joinable()) {
        try {
          impl_->worker = std::thread([state = impl_.get()] { state->run_worker(); });
        } catch (...) {
          impl_->pools.erase(id);
          throw;
        }
      }
      impl_->last_pool = id;
      impl_->assigned += capacity;
    }
  } catch (const std::invalid_argument&) {
    return Status::invalid_argument;
  } catch (const std::bad_alloc&) {
    return Status::resource_exhausted;
  } catch (const std::system_error&) {
    return Status::resource_exhausted;
  }
  if (reserve) impl_->wake_worker();
  return id;
}
Status Cache::resize_pool(PoolId pool, std::size_t capacity_bytes) {
  std::lock_guard<std::mutex> control(impl_->control_mutex);
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (capacity_bytes == 0) return Status::invalid_argument;
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  const std::size_t other = impl_->assigned - found.value()->capacity();
  if (capacity_bytes > impl_->config.capacity_bytes - other) return Status::resource_exhausted;
  const Status status = found.value()->resize(capacity_bytes);
  if (status == Status::ok) {
    std::unique_lock<detail::SharedMutex> lock(impl_->registry_mutex);
    impl_->assigned = other + capacity_bytes;
  }
  return status;
}
Result<std::vector<PoolStats>> Cache::pools() const {
  std::shared_lock<detail::SharedMutex> lock(impl_->registry_mutex);
  std::vector<PoolStats> result;
  result.reserve(impl_->pools.size());
  for (const Impl::PoolMap::value_type& entry : impl_->pools) result.push_back(entry.second->snapshot());
  return result;
}
Result<PoolStats> Cache::pool_stats(PoolId pool) const {
  Result<detail::CachePool*> found = impl_->get_pool(pool, true);
  if (!found) return found.status();
  return found.value()->snapshot();
}
CacheStats Cache::stats() const {
  std::shared_lock<detail::SharedMutex> lock(impl_->registry_mutex);
  CacheStats result;
  result.capacity_bytes = impl_->config.capacity_bytes;
  result.assigned_bytes = impl_->assigned;
  result.pools = impl_->pools.size();
  result.closed = impl_->closed.load(std::memory_order_acquire);
  result.reaper.passes = impl_->reaper_stats.passes.load(std::memory_order_relaxed);
  result.reaper.reclaimed = impl_->reaper_stats.reclaimed.load(std::memory_order_relaxed);
  result.reaper.throttles = impl_->reaper_stats.throttles.load(std::memory_order_relaxed);
  result.reaper.compactions = impl_->reaper_stats.compactions.load(std::memory_order_relaxed);
  result.reaper.last_pass = Duration(impl_->reaper_stats.last_pass.load(std::memory_order_relaxed));
  result.reaper.max_pass = Duration(impl_->reaper_stats.max_pass.load(std::memory_order_relaxed));
  for (const Impl::PoolMap::value_type& entry : impl_->pools) {
    const PoolStats pool = entry.second->snapshot();
    result.allocated_bytes += pool.allocated_bytes;
    result.indexed_bytes += pool.indexed_bytes;
    result.indexed_items += pool.indexed_items;
    result.ttl_items += pool.ttl_items;
    accumulate(result.counters, pool.counters);
    accumulate(result.allocator, pool.allocator);
  }
  return result;
}
CacheMemoryStats Cache::memory_stats() const {
  CacheMemoryStats result;
  {
    std::shared_lock<detail::SharedMutex> lock(impl_->registry_mutex);
    result.pools.reserve(impl_->pools.size());
    for (const Impl::PoolMap::value_type& entry : impl_->pools) {
      const MemoryStats memory = entry.second->memory_snapshot();
      result.pools.push_back({entry.first, entry.second->name(), memory});
      accumulate(result.total, memory);
    }
  }
  result.process_rss_bytes = detail::process_rss_bytes();
  return result;
}
Result<MemoryStats> Cache::memory_stats(PoolId pool) const {
  Result<detail::CachePool*> found = impl_->get_pool(pool, true);
  if (!found) return found.status();
  return found.value()->memory_snapshot();
}
Result<WriteHandle> Cache::allocate(PoolId pool, std::string_view key, std::size_t value_size) {
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (!impl_->valid_key(key) || value_size > impl_->config.max_value_bytes ||
      value_size > std::numeric_limits<std::size_t>::max() - key.size()) return Status::invalid_argument;
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  Result<detail::EntryPtr> entry = found.value()->allocate(key, value_size);
  if (!entry) return entry.status();
  return WriteHandle(std::move(entry).value());
}
Result<ReadHandle> Cache::publish(WriteHandle&& draft, WriteOptions options) {
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (!draft || draft.entry_->owner != impl_->owner) return Status::invalid_argument;
  if (Status status = validate(options); status != Status::ok) return status;
  Result<detail::CachePool*> found = impl_->get_pool(draft.pool());
  if (!found) return found.status();
  Result<detail::EntryPtr> entry = found.value()->publish(draft.entry_, options);
  if (!entry) return entry.status();
  return ReadHandle(std::move(entry).value());
}
Result<ReadHandle> Cache::put(PoolId pool, std::string_view key,
                            std::span<const std::byte> value, WriteOptions options) {
  if (Status status = validate(options); status != Status::ok) return status;
  Result<WriteHandle> draft = allocate(pool, key, value.size());
  if (!draft) return draft.status();
  std::copy(value.begin(), value.end(), draft.value().bytes().begin());
  return publish(std::move(draft).value(), options);
}
Result<ReadHandle> Cache::lookup(PoolId pool, std::string_view key, bool touch) {
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (!impl_->valid_key(key)) return Status::invalid_argument;
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  Result<detail::EntryPtr> entry = found.value()->lookup(key, touch);
  if (!entry) return entry.status();
  return ReadHandle(std::move(entry).value());
}
Result<ReadHandle> Cache::find(PoolId pool, std::string_view key) { return lookup(pool, key, true); }
Result<ReadHandle> Cache::peek(PoolId pool, std::string_view key) { return lookup(pool, key, false); }
Status Cache::erase(PoolId pool, std::string_view key, Version expected_version) {
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  if (!impl_->valid_key(key)) return Status::invalid_argument;
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  return found.value()->erase(key, expected_version);
}
Status Cache::clear(PoolId pool) {
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  return found.value()->clear();
}
Result<std::size_t> Cache::prune_expired(PoolId pool, std::size_t max_items) {
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  detail::CachePool& target = *found.value();
  return target.reap(target.now(), max_items, impl_->config.reaper.batch_size, [] { return true; });
}
Result<std::size_t> Cache::prune_expired() {
  if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
  const Duration now = impl_->config.clock->now();
  std::size_t total = 0;
  Status status = Status::ok;
  impl_->for_each_pool([&](detail::CachePool& pool) {
    Result<std::size_t> reaped =
        pool.reap(now, std::numeric_limits<std::size_t>::max(), impl_->config.reaper.batch_size,
                  [] { return true; });
    if (!reaped) {
      status = reaped.status();
      return false;
    }
    total += reaped.value();
    return true;
  });
  if (status != Status::ok) return status;
  return total;
}
Result<CompactionStats> Cache::compact(PoolId pool) {
  Result<detail::CachePool*> found = impl_->get_pool(pool);
  if (!found) return found.status();
  return found.value()->compact(impl_->config.clock->now());
}
Result<CompactionStats> Cache::compact() {
  try {
    std::vector<detail::CachePool*> pools;
    {
      std::shared_lock<detail::SharedMutex> lock(impl_->registry_mutex);
      if (impl_->closed.load(std::memory_order_acquire)) return Status::closed;
      for (const Impl::PoolMap::value_type& entry : impl_->pools) pools.push_back(entry.second.get());
    }
    const Duration now = impl_->config.clock->now();
    CompactionStats result;
    for (detail::CachePool* pool : pools) {
      Result<CompactionStats> compacted = pool->compact(now);
      if (!compacted) return compacted.status();
      const CompactionStats& stats = compacted.value();
      result.pools += stats.pools;
      result.moved_items += stats.moved_items;
      result.skipped_items += stats.skipped_items;
      result.expired_items += stats.expired_items;
      result.reserved_bytes_before += stats.reserved_bytes_before;
      result.reserved_bytes_after += stats.reserved_bytes_after;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Status::resource_exhausted;
  }
}
void Cache::close() {
  std::lock_guard<std::mutex> control(impl_->control_mutex);
  impl_->closed.store(true, std::memory_order_release);
  // Control serialization prevents adding/removing pools. No registry lock is
  // needed while waiting for in-flight operations in each pool to finish.
  for (const Impl::PoolMap::value_type& entry : impl_->pools) entry.second->close();
  impl_->wake_worker();
}
}  // namespace nivo
