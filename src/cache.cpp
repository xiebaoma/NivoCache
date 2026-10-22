#include "nivo/cache.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <utility>

namespace nivo {
namespace detail {
struct Owner {};
struct Budget { std::atomic<std::size_t> used{0}; };
struct Entry {
  std::shared_ptr<const Owner> owner;
  std::shared_ptr<Budget> budget;
  std::string key;
  std::vector<std::byte> value;
  PoolId pool = invalid_pool;
  Version version = invalid_version;
  std::size_t charge = 0;
  std::optional<Duration> expires_at;
  ~Entry() { if (budget) budget->used.fetch_sub(charge, std::memory_order_relaxed); }
};
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
}  // namespace

struct Cache::Impl {
  using Queue = std::list<std::shared_ptr<detail::Entry>>;
  using Index = std::map<std::string, Queue::iterator, std::less<>>;
  struct Pool {
    PoolId id;
    PoolConfig config;
    std::shared_ptr<detail::Budget> budget = std::make_shared<detail::Budget>();
    Queue queue;  // front = most recent
    Index index;
    std::size_t indexed_bytes = 0;
    Counters counters;
    Pool(PoolId id_, PoolConfig config_) : id(id_), config(std::move(config_)) {}
  };
  using PoolMap = std::map<PoolId, Pool>;
  explicit Impl(CacheConfig config_) : config(std::move(config_)) {
    if (!config.clock) config.clock = std::make_shared<SteadyClock>();
  }
  CacheConfig config;
  std::shared_ptr<const detail::Owner> owner = std::make_shared<detail::Owner>();
  mutable std::mutex mutex;
  PoolMap pools;
  std::size_t assigned = 0;
  PoolId last_pool = 0;
  Version last_version = 0;
  bool closed = false;

  bool valid_key(std::string_view key) const {
    return !key.empty() && key.size() <= config.max_key_bytes;
  }
  static bool expired(const detail::Entry& e, Duration now) {
    return e.expires_at && now >= *e.expires_at;
  }
  static void unlink(Pool& p, Queue::iterator it) {
    p.indexed_bytes -= (*it)->charge;
    p.index.erase((*it)->key);
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
            p.index.size(), p.counters};
  }
};

Cache::Cache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Cache::~Cache() = default;
Result<std::unique_ptr<Cache>> Cache::create(CacheConfig config) {
  if (config.capacity_bytes == 0 || config.max_key_bytes == 0 || config.max_value_bytes == 0)
    return Status::invalid_argument;
  if (config.storage == StorageMode::hybrid) return Status::unsupported;
  if (config.storage != StorageMode::memory) return Status::invalid_argument;
  return std::unique_ptr<Cache>(new Cache(std::make_unique<Impl>(std::move(config))));
}
Result<PoolId> Cache::add_pool(PoolConfig config) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  if (config.name.empty() || config.capacity_bytes == 0) return Status::invalid_argument;
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
  impl_->pools.try_emplace(id, id, std::move(config));
  impl_->last_pool = id;
  impl_->assigned += capacity;
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
  }
  return result;
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
  if (charge > p.config.capacity_bytes) {
    ++p.counters.allocation_failures;
    return Status::resource_exhausted;
  }
  Impl::prune(p, impl_->config.clock->now());
  if (!Impl::make_room(p, p.config.capacity_bytes - charge, key)) {
    ++p.counters.allocation_failures;
    return Status::resource_exhausted;
  }
  std::shared_ptr<detail::Entry> entry = std::make_shared<detail::Entry>();
  entry->owner = impl_->owner;
  entry->budget = p.budget;
  entry->key = key;
  entry->value.resize(value_size);
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
  return Status::ok;
}
Result<std::size_t> Cache::prune_expired(PoolId pool) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return Status::closed;
  Impl::PoolMap::iterator it = impl_->pools.find(pool);
  if (it == impl_->pools.end()) return Status::pool_not_found;
  return Impl::prune(it->second, impl_->config.clock->now());
}
void Cache::close() {
  std::lock_guard lock(impl_->mutex);
  impl_->closed = true;
  for (Impl::PoolMap::value_type& pool_entry : impl_->pools) {
    Impl::Pool& p = pool_entry.second;
    p.index.clear(); p.queue.clear(); p.indexed_bytes = 0;
  }
}
}  // namespace nivo
