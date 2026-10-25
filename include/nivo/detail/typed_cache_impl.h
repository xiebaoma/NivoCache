// Member definitions for TypedReadHandle and TypedCache. Included by
// nivo/typed_cache.h; do not include directly.
#pragma once

#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace nivo {

// ---------------------------------------------------------------------------
// TypedReadHandle
// ---------------------------------------------------------------------------

template <class V>
const V* TypedReadHandle<V>::get() const noexcept {
  if constexpr (is_zero_copy_value_v<V>) {
    return state_.value;
  } else {
    return state_.value.get();
  }
}

template <class V>
Version TypedReadHandle<V>::version() const noexcept {
  if constexpr (is_zero_copy_value_v<V>) {
    return state_.handle.version();
  } else {
    return state_.version;
  }
}

// ---------------------------------------------------------------------------
// TypedCache: construction
// ---------------------------------------------------------------------------

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedCache<K, V, KeyCodecT, ValueCodecT>::TypedCache(TypedCacheConfig config) {
  const std::size_t capacity = config.cache.capacity_bytes;
  Result<std::unique_ptr<Cache>> created = Cache::create(std::move(config.cache));
  if (!created) throw CacheError(created.status());
  owned_ = std::move(created).value();
  Result<PoolId> added = owned_->add_pool(
      {.name = std::move(config.pool_name), .capacity_bytes = capacity, .eviction = config.eviction});
  if (!added) throw CacheError(added.status());
  cache_ = owned_.get();
  pool_ = added.value();
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedCache<K, V, KeyCodecT, ValueCodecT>::TypedCache(Cache& cache, PoolId pool)
    : cache_(&cache), pool_(pool) {
  Result<PoolStats> stats = cache.pool_stats(pool);
  if (!stats) throw CacheError(stats.status());
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedCache<K, V, KeyCodecT, ValueCodecT>::TypedCache(TypedCache&& other) noexcept
    : owned_(std::move(other.owned_)),
      cache_(std::exchange(other.cache_, nullptr)),
      pool_(std::exchange(other.pool_, invalid_pool)) {}

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedCache<K, V, KeyCodecT, ValueCodecT>& TypedCache<K, V, KeyCodecT, ValueCodecT>::operator=(
    TypedCache&& other) noexcept {
  if (this != &other) {
    owned_ = std::move(other.owned_);
    cache_ = std::exchange(other.cache_, nullptr);
    pool_ = std::exchange(other.pool_, invalid_pool);
  }
  return *this;
}

// ---------------------------------------------------------------------------
// TypedCache: reads
// ---------------------------------------------------------------------------

template <class K, class V, class KeyCodecT, class ValueCodecT>
std::optional<V> TypedCache<K, V, KeyCodecT, ValueCodecT>::get_value(const K& key)
  requires std::is_copy_constructible_v<V>
{
  TypedReadHandle<V> handle = get(key);
  if (!handle) return std::nullopt;
  return std::optional<V>(*handle);
}

// ---------------------------------------------------------------------------
// TypedCache: writes
// ---------------------------------------------------------------------------

template <class K, class V, class KeyCodecT, class ValueCodecT>
bool TypedCache<K, V, KeyCodecT, ValueCodecT>::put(const K& key, const V& value, Duration ttl) {
  Result<Version> written = write(key, value, {.mode = WriteMode::upsert, .ttl = ttl});
  if (written) return true;
  if (written.status() == Status::resource_exhausted) return false;
  throw CacheError(written.status());
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
bool TypedCache<K, V, KeyCodecT, ValueCodecT>::insert(const K& key, const V& value, Duration ttl) {
  Result<Version> written = write(key, value, {.mode = WriteMode::insert_only, .ttl = ttl});
  if (written) return true;
  if (written.status() == Status::already_exists ||
      written.status() == Status::resource_exhausted)
    return false;
  throw CacheError(written.status());
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
bool TypedCache<K, V, KeyCodecT, ValueCodecT>::erase(const K& key) {
  const Status status = erase(key, invalid_version);
  if (status == Status::ok) return true;
  if (status == Status::not_found) return false;
  throw CacheError(status);
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
Result<Version> TypedCache<K, V, KeyCodecT, ValueCodecT>::write(const K& key, const V& value,
                                                                WriteOptions options) {
  try {
    const typename KeyCodecT::Encoded encoded = KeyCodecT::encode(key);
    const std::string_view key_bytes = KeyCodecT::view(encoded);
    std::size_t size = sizeof(V);
    if constexpr (!zero_copy) size = static_cast<std::size_t>(ValueCodecT::encoded_size(value));
    Result<WriteHandle> draft = cache_->allocate(pool_, key_bytes, size);
    if (!draft) return draft.status();
    std::span<std::byte> out = draft.value().bytes();
    if constexpr (zero_copy) {
      std::memcpy(out.data(), std::addressof(value), sizeof(V));
    } else {
      ValueCodecT::encode(value, out);
    }
    Result<ReadHandle> published = cache_->publish(std::move(draft).value(), options);
    if (!published) return published.status();
    return published.value().version();
  } catch (const CacheError& error) {
    return error.status();
  }
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
Status TypedCache<K, V, KeyCodecT, ValueCodecT>::erase(const K& key, Version expected_version) {
  try {
    const typename KeyCodecT::Encoded encoded = KeyCodecT::encode(key);
    return cache_->erase(pool_, KeyCodecT::view(encoded), expected_version);
  } catch (const CacheError& error) {
    return error.status();
  }
}

// ---------------------------------------------------------------------------
// TypedCache: private helpers
// ---------------------------------------------------------------------------

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedCacheConfig TypedCache<K, V, KeyCodecT, ValueCodecT>::make_config(std::size_t capacity_bytes) {
  TypedCacheConfig config;
  config.cache.capacity_bytes = capacity_bytes;
  return config;
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
TypedReadHandle<V> TypedCache<K, V, KeyCodecT, ValueCodecT>::lookup_or_throw(const K& key,
                                                                            bool touch) {
  Result<TypedReadHandle<V>> found = lookup(key, touch);
  if (found) return std::move(found).value();
  if (found.status() == Status::not_found) return TypedReadHandle<V>{};
  throw CacheError(found.status());
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
Result<TypedReadHandle<V>> TypedCache<K, V, KeyCodecT, ValueCodecT>::lookup(const K& key,
                                                                           bool touch) {
  try {
    const typename KeyCodecT::Encoded encoded = KeyCodecT::encode(key);
    const std::string_view key_bytes = KeyCodecT::view(encoded);
    Result<ReadHandle> found =
        touch ? cache_->find(pool_, key_bytes) : cache_->peek(pool_, key_bytes);
    if (!found) return found.status();
    return wrap(std::move(found).value());
  } catch (const CacheError& error) {
    return error.status();
  }
}

template <class K, class V, class KeyCodecT, class ValueCodecT>
Result<TypedReadHandle<V>> TypedCache<K, V, KeyCodecT, ValueCodecT>::wrap(ReadHandle handle) {
  using State = typename TypedReadHandle<V>::State;
  const std::span<const std::byte> bytes = handle.bytes();
  if constexpr (zero_copy) {
    // A size mismatch means another value type was stored in this pool.
    if (bytes.size() != sizeof(V)) return Status::invalid_argument;
    const V* value = std::launder(reinterpret_cast<const V*>(bytes.data()));
    return TypedReadHandle<V>(State{std::move(handle), value});
  } else {
    std::shared_ptr<const V> value = std::make_shared<V>(ValueCodecT::decode(bytes));
    const Version version = handle.version();
    handle.reset();  // decoded copy no longer needs the cache entry
    return TypedReadHandle<V>(State{std::move(value), version});
  }
}

}  // namespace nivo
