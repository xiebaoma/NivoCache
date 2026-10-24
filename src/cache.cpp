#include "nivo/cache.h"
#include "memory_allocator.h"
#include "process_memory.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>

namespace nivo {
namespace detail {
struct Owner {};
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
};
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
  using Queue = std::list<detail::EntryPtr>;
  using Index = std::map<std::string, Queue::iterator, std::less<>>;
  struct Pool {
    PoolId id;
    PoolConfig config;
    std::shared_ptr<detail::MemoryAllocator> budget;
    Queue queue;  // front = most recent
    Index index;
    std::size_t indexed_bytes = 0;
    Counters counters;
    Pool(PoolId id_, PoolConfig config_)
        : id(id_), config(std::move(config_)),
          budget(std::make_shared<detail::MemoryAllocator>(config.capacity_bytes)) {
      budget->reserve_initial(config.initial_reserve_bytes, config.initial_reserve_timeout);
    }
  };
  using PoolMap = std::map<PoolId, Pool>;
  explicit Impl(CacheConfig config_) : config(std::move(config_)) {
    if (!config.clock) config.clock = std::make_shared<SteadyClock>();
  }
  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closed = true;
    }
    reserve_wakeup.notify_all();
    if (reserve_worker.joinable()) reserve_worker.join();
  }
  CacheConfig config;
  std::shared_ptr<const detail::Owner> owner = std::make_shared<detail::Owner>();
  mutable std::mutex mutex;
  PoolMap pools;
  std::size_t assigned = 0;
  PoolId last_pool = 0;
  Version last_version = 0;
  bool closed = false;
  std::condition_variable reserve_wakeup;
  std::thread reserve_worker;

  void maintain_reserves() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!closed) {
      std::optional<detail::MemoryAllocator::TimePoint> next;
      for (PoolMap::value_type& pool_entry : pools) {
        const std::optional<detail::MemoryAllocator::TimePoint> deadline =
            pool_entry.second.budget->maintain_initial_reserve();
        if (deadline && (!next || *deadline < *next)) next = deadline;
      }
      if (next) reserve_wakeup.wait_until(lock, *next);
      else reserve_wakeup.wait(lock);
    }
  }

  bool valid_key(std::string_view key) const {
    return !key.empty() && key.size() <= config.max_key_bytes;
  }
  static bool expired(const detail::Entry& e, Duration now) {
    return e.expires_at && now >= *e.expires_at;
  }
  static void unlink(Pool& p, Queue::iterator it) {
    p.indexed_bytes -= (*it)->charge;
    p.index.erase(p.index.find((*it)->key));
    p.queue.erase(it);
  }
  static std::size_t prune(Pool& p, Duration now) {
    std::size_t count = 0;
    for (Queue::iterator it = p.queue.begin(); it != p.queue.end();) {
      Queue::iterator candidate = it++;
      if (expired(**candidate, now)) { unlink(p, candidate); ++count; }
    }
    p.counters.expirations += count;
    return count;
  }
  static CompactionStats compact_pool(Pool& p, Duration now) {
    CompactionStats result;
    result.pools = 1;
    result.reserved_bytes_before = p.budget->stats().reserved_bytes;
    result.expired_items = prune(p, now);
    try {
      std::vector<Queue::iterator> entries;
      std::vector<detail::MemoryAllocator::Allocation> allocations;
      entries.reserve(p.index.size());
      allocations.reserve(p.index.size());
      for (Queue::iterator it = p.queue.begin(); it != p.queue.end(); ++it) {
        if (it->use_count() == 1 && (*it)->allocation.slab) {
          entries.push_back(it);
          allocations.push_back((*it)->allocation);
        }
      }
      detail::MemoryAllocator::Compaction compaction(*p.budget, allocations);
      for (std::size_t i = 0; i < entries.size(); ++i) {
        if (!compaction.should_move(allocations[i])) continue;
        detail::Entry& source = **entries[i];
        const detail::MemoryAllocator::Allocation allocation = compaction.allocate(allocations[i]);
        detail::Entry* destination = new (allocation.data) detail::Entry{};
        destination->owner = source.owner;
        destination->budget = source.budget;
        destination->allocation = allocation;
        destination->pool = source.pool;
        destination->version = source.version;
        destination->expires_at = source.expires_at;
        char* key = reinterpret_cast<char*>(destination + 1);
        std::byte* value = reinterpret_cast<std::byte*>(key + source.key.size());
        std::memcpy(key, source.key.data(), source.key.size());
        if (!source.value.empty()) std::memcpy(value, source.value.data(), source.value.size());
        destination->key = {key, source.key.size()};
        destination->value = {value, source.value.size()};
        // Everything after storage allocation is non-throwing. Transfer the
        // existing charge and queue position without a second logical charge.
        // With the cache lock held, a sole queue reference cannot gain a pin.
        destination->charge = std::exchange(source.charge, 0);
        *entries[i] = detail::EntryPtr(destination);
        ++result.moved_items;
      }
    } catch (const std::bad_alloc&) {
      p.budget->trim();
      throw;
    }
    result.skipped_items = p.index.size() - result.moved_items;
    result.reserved_bytes_after = p.budget->stats().reserved_bytes;
    return result;
  }
  static void expire_key(Pool& p, std::string_view key, Duration now) {
    Index::iterator it = p.index.find(key);
    if (it != p.index.end() && expired(**it->second, now)) {
      unlink(p, it->second);
      ++p.counters.expirations;
    }
  }
  // target is the maximum total charge allowed before the new reservation.
  static bool make_room(Pool& p, std::size_t target, std::string_view protected_key) {
    Queue::iterator it = p.queue.end();
    while (p.budget->used.load(std::memory_order_relaxed) > target && it != p.queue.begin()) {
      --it;
      if ((*it)->key != protected_key && it->use_count() == 1) {
        Queue::iterator victim = it++;
        unlink(p, victim);
        ++p.counters.evictions;
      }
    }
    return p.budget->used.load(std::memory_order_relaxed) <= target;
  }
  static PoolStats snapshot(const Pool& p) {
    return {p.id, p.config.name, p.config.capacity_bytes,
            p.budget->used.load(std::memory_order_relaxed), p.indexed_bytes,
            p.index.size(), p.counters, p.budget->stats()};
  }
  static MemoryStats memory_snapshot(const Pool& p) {
    // requested_bytes and live_allocations come from one locked allocator
    // snapshot. Reading the independently updated quota would race with the
    // gap between storage release and quota release on a handle's last reset.
    const AllocatorStats storage = p.budget->inspect();
    MemoryStats result;
    result.reserved_bytes = storage.reserved_bytes;
    result.entry_metadata_bytes = storage.live_allocations * sizeof(detail::Entry);
    result.kv_bytes = storage.requested_bytes - result.entry_metadata_bytes;
    result.indexed_kv_bytes = p.indexed_bytes;
    result.retained_kv_bytes = result.kv_bytes - result.indexed_kv_bytes;
    result.allocator_metadata_bytes = storage.metadata_bytes;
    result.internal_fragmentation_bytes = storage.active_bytes - storage.requested_bytes;
    result.unused_bytes = storage.reserved_bytes - storage.metadata_bytes - storage.active_bytes;
    result.reclaimable_bytes = storage.reclaimable_bytes;
    return result;
  }
};

Cache::Cache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Cache::~Cache() { close(); }
Result<std::unique_ptr<Cache>> Cache::create(CacheConfig config) {
  if (config.capacity_bytes == 0 || config.max_key_bytes == 0 || config.max_value_bytes == 0)
    return Status::invalid_argument;
  if (config.storage == StorageMode::hybrid) return Status::unsupported;
  if (config.storage != StorageMode::memory) return Status::invalid_argument;
  return std::unique_ptr<Cache>(new Cache(std::make_unique<Impl>(std::move(config))));
}
Result<PoolId> Cache::add_pool(PoolConfig config) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (config.name.empty() || config.capacity_bytes == 0) return Status::invalid_argument;
  if (config.initial_reserve_bytes > config.capacity_bytes || config.initial_reserve_timeout.count() <= 0)
    return Status::invalid_argument;
  if (config.eviction != EvictionPolicy::lru && config.eviction != EvictionPolicy::fifo)
    return Status::invalid_argument;
  for (const Impl::PoolMap::value_type& pool_entry : impl_->pools) {
    const Impl::Pool& p = pool_entry.second;
    if (p.config.name == config.name) return Status::already_exists;
  }
  if (config.capacity_bytes > impl_->config.capacity_bytes - impl_->assigned ||
      impl_->last_pool == std::numeric_limits<PoolId>::max()) return Status::resource_exhausted;
  const PoolId id = impl_->last_pool + 1;
  const std::size_t capacity = config.capacity_bytes;
  const bool reserve = config.initial_reserve_bytes != 0;
  try {
    impl_->pools.try_emplace(id, id, std::move(config));
    if (reserve && !impl_->reserve_worker.joinable()) {
      try {
        impl_->reserve_worker = std::thread([state = impl_.get()] { state->maintain_reserves(); });
      } catch (...) {
        impl_->pools.erase(id);
        throw;
      }
    }
  } catch (const std::invalid_argument&) {
    return Status::invalid_argument;
  } catch (const std::bad_alloc&) {
    return Status::resource_exhausted;
  } catch (const std::system_error&) {
    return Status::resource_exhausted;
  }
  impl_->last_pool = id;
  impl_->assigned += capacity;
  if (reserve) impl_->reserve_wakeup.notify_one();
  return id;
}
Status Cache::resize_pool(PoolId pool, std::size_t capacity_bytes) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (capacity_bytes == 0) return Status::invalid_argument;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  Impl::Pool& p = it->second;
  const std::size_t other = impl_->assigned - p.config.capacity_bytes;
  if (capacity_bytes > impl_->config.capacity_bytes - other) return Status::resource_exhausted;
  Impl::prune(p, impl_->config.clock->now());
  if (!Impl::make_room(p, capacity_bytes, {})) return Status::resource_exhausted;
  p.config.capacity_bytes = capacity_bytes;
  p.budget->resize(capacity_bytes);
  impl_->assigned = other + capacity_bytes;
  return Status::ok;
}
Result<std::vector<PoolStats>> Cache::pools() const {
  std::lock_guard lock(impl_->mutex);
  std::vector<PoolStats> result;
  result.reserve(impl_->pools.size());
  for (const Impl::PoolMap::value_type& pool_entry : impl_->pools) {
    result.push_back(Impl::snapshot(pool_entry.second));
  }
  return result;
}
Result<PoolStats> Cache::pool_stats(PoolId pool) const {
  std::lock_guard lock(impl_->mutex);
  Impl::PoolMap::const_iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  return Impl::snapshot(it->second);
}
CacheStats Cache::stats() const {
  std::lock_guard lock(impl_->mutex);
  CacheStats result;
  result.capacity_bytes = impl_->config.capacity_bytes;
  result.assigned_bytes = impl_->assigned;
  result.pools = impl_->pools.size();
  result.closed = impl_->closed;
  for (const Impl::PoolMap::value_type& pool_entry : impl_->pools) {
    const Impl::Pool& p = pool_entry.second;
    result.allocated_bytes += p.budget->used.load(std::memory_order_relaxed);
    result.indexed_bytes += p.indexed_bytes;
    result.indexed_items += p.index.size();
    accumulate(result.counters, p.counters);
    accumulate(result.allocator, p.budget->stats());
  }
  return result;
}
CacheMemoryStats Cache::memory_stats() const {
  CacheMemoryStats result;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    result.pools.reserve(impl_->pools.size());
    for (const Impl::PoolMap::value_type& pool_entry : impl_->pools) {
      const Impl::Pool& pool = pool_entry.second;
      const MemoryStats memory = Impl::memory_snapshot(pool);
      result.pools.push_back({pool.id, pool.config.name, memory});
      accumulate(result.total, memory);
    }
  }
  result.process_rss_bytes = detail::process_rss_bytes();
  return result;
}
Result<MemoryStats> Cache::memory_stats(PoolId pool) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Impl::PoolMap::const_iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  return Impl::memory_snapshot(it->second);
}
Result<WriteHandle> Cache::allocate(PoolId pool, std::string_view key, std::size_t value_size) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (!impl_->valid_key(key) || value_size > impl_->config.max_value_bytes ||
      value_size > std::numeric_limits<std::size_t>::max() - key.size()) return Status::invalid_argument;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  Impl::Pool& p = it->second;
  const std::size_t charge = key.size() + value_size;
  if (charge > std::numeric_limits<std::size_t>::max() - sizeof(detail::Entry))
    return Status::invalid_argument;
  if (charge > p.config.capacity_bytes) {
    ++p.counters.allocation_failures;
    return Status::resource_exhausted;
  }
  const std::size_t target = p.config.capacity_bytes - charge;
  if (p.budget->used.load(std::memory_order_relaxed) > target) {
    Impl::prune(p, impl_->config.clock->now());
    if (!Impl::make_room(p, target, key)) {
      ++p.counters.allocation_failures;
      return Status::resource_exhausted;
    }
  }
  detail::MemoryAllocator::Allocation allocation;
  try {
    allocation = p.budget->allocate(sizeof(detail::Entry) + charge);
  } catch (const std::bad_alloc&) {
    ++p.counters.allocation_failures;
    return Status::resource_exhausted;
  }
  detail::EntryPtr entry(new (allocation.data) detail::Entry{});
  entry->owner = impl_->owner;
  entry->budget = p.budget;
  entry->allocation = allocation;
  char* key_data = reinterpret_cast<char*>(allocation.data) + sizeof(detail::Entry);
  std::memcpy(key_data, key.data(), key.size());
  entry->key = std::string_view(key_data, key.size());
  std::byte* value_data = reinterpret_cast<std::byte*>(key_data + key.size());
  entry->value = std::span<std::byte>(value_data, value_size);
  std::fill(entry->value.begin(), entry->value.end(), std::byte{0});
  entry->pool = pool;
  entry->charge = charge;
  p.budget->used.fetch_add(charge, std::memory_order_relaxed);
  ++p.counters.allocations;
  return WriteHandle(std::move(entry));
}
Result<ReadHandle> Cache::publish(WriteHandle&& draft, WriteOptions options) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (!draft || draft.entry_->owner != impl_->owner) return Status::invalid_argument;
  if (Status status = validate(options); status != Status::ok) return status;
  const Duration now = impl_->config.clock->now();
  if (options.ttl.count() > std::numeric_limits<Duration::rep>::max() - now.count())
    return Status::invalid_argument;
  Impl::Pool& p = impl_->pools.at(draft.pool());
  Impl::expire_key(p, draft.key(), now);
  Impl::Index::iterator it = p.index.find(draft.key());
  const bool exists = it != p.index.end();
  if (options.mode == WriteMode::insert_only && exists) return Status::already_exists;
  if ((options.mode == WriteMode::replace_only || options.mode == WriteMode::compare_exchange) && !exists)
    return Status::not_found;
  if (options.mode == WriteMode::compare_exchange && (*it->second)->version != options.expected_version)
    return Status::version_mismatch;
  if (impl_->last_version == std::numeric_limits<Version>::max()) return Status::resource_exhausted;
  // Complete potentially throwing work before changing the indexed value.
  if (!exists) {
    p.queue.push_front(draft.entry_);
    try { p.index.emplace(std::string(draft.key()), p.queue.begin()); }
    catch (...) { p.queue.pop_front(); throw; }
  } else {
    p.indexed_bytes -= (*it->second)->charge;
    *it->second = draft.entry_;
    p.queue.splice(p.queue.begin(), p.queue, it->second);
  }
  draft.entry_->version = ++impl_->last_version;
  draft.entry_->expires_at = options.ttl.count() == 0 ? std::nullopt : std::optional<Duration>(now + options.ttl);
  p.indexed_bytes += draft.charge();
  if (exists) ++p.counters.replacements; else ++p.counters.inserts;
  ReadHandle result(draft.entry_);
  draft.reset();
  return result;
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
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (!impl_->valid_key(key)) return Status::invalid_argument;
  Impl::PoolMap::iterator pool_it = impl_->pools.find(pool);
  if (pool_it == impl_->pools.end()) return Status::pool_not_found;
  Impl::Pool& p = pool_it->second;
  Impl::expire_key(p, key, impl_->config.clock->now());
  Impl::Index::iterator it = p.index.find(key);
  if (it == p.index.end()) {
    if (touch) ++p.counters.misses;
    return Status::not_found;
  }
  if (touch) {
    ++p.counters.hits;
    if (p.config.eviction == EvictionPolicy::lru) p.queue.splice(p.queue.begin(), p.queue, it->second);
  }
  return ReadHandle(*it->second);
}

Result<ReadHandle> Cache::find(PoolId pool, std::string_view key) {
  return lookup(pool, key, true);
}

Result<ReadHandle> Cache::peek(PoolId pool, std::string_view key) { return lookup(pool, key, false); }
Status Cache::erase(PoolId pool, std::string_view key, Version expected_version) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (!impl_->valid_key(key)) return Status::invalid_argument;
  Impl::PoolMap::iterator pool_it = impl_->pools.find(pool);
  if (pool_it == impl_->pools.end()) return Status::pool_not_found;
  Impl::Pool& p = pool_it->second;
  Impl::expire_key(p, key, impl_->config.clock->now());
  Impl::Index::iterator it = p.index.find(key);
  if (it == p.index.end()) return Status::not_found;
  if (expected_version && (*it->second)->version != expected_version) return Status::version_mismatch;
  Impl::unlink(p, it->second);
  ++p.counters.erases;
  return Status::ok;
}
Status Cache::clear(PoolId pool) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  Impl::Pool& p = it->second;
  p.counters.erases += p.index.size();
  p.index.clear(); p.queue.clear(); p.indexed_bytes = 0;
  p.budget->trim();
  return Status::ok;
}
Result<std::size_t> Cache::prune_expired(PoolId pool) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  return Impl::prune(it->second, impl_->config.clock->now());
}
Result<CompactionStats> Cache::compact(PoolId pool) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  try {
    return Impl::compact_pool(it->second, impl_->config.clock->now());
  } catch (const std::bad_alloc&) {
    return Status::resource_exhausted;
  }
}
Result<CompactionStats> Cache::compact() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  const Duration now = impl_->config.clock->now();
  CompactionStats result;
  try {
    for (Impl::PoolMap::value_type& pool_entry : impl_->pools) {
      const CompactionStats pool = Impl::compact_pool(pool_entry.second, now);
      result.pools += pool.pools;
      result.moved_items += pool.moved_items;
      result.skipped_items += pool.skipped_items;
      result.expired_items += pool.expired_items;
      result.reserved_bytes_before += pool.reserved_bytes_before;
      result.reserved_bytes_after += pool.reserved_bytes_after;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Status::resource_exhausted;
  }
}
void Cache::close() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->closed = true;
  impl_->reserve_wakeup.notify_all();
  for (Impl::PoolMap::value_type& pool_entry : impl_->pools) {
    Impl::Pool& p = pool_entry.second;
    p.budget->trim(true);
    p.index.clear(); p.queue.clear(); p.indexed_bytes = 0;
  }
}
}  // namespace nivo
