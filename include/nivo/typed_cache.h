// Typed key/value cache:
//
//   nivo::NivoCache<int, Student> cache(64 * 1024 * 1024);
//   cache.put(10001, Student{20, "Alice", 95.5});
//   nivo::TypedReadHandle<Student> student = cache.get(10001);
//   if (student) std::cout << student->name;
//
// Supported key types (a key is encoded to bytes; equal bytes = same key):
//   - integral and enum types
//   - float and double (-0.0 equals 0.0; NaN is rejected)
//   - std::string, std::string_view (must not be empty)
//   - trivially copyable types without padding bytes, e.g. std::array<char, N>
//   - anything else: specialize nivo::KeyCodec<K> (see KeyCodecFor)
// Pointers and long double are deliberately unsupported.
//
// Supported value types:
//   - trivially copyable types (int, double, char[N], POD structs): stored as
//     raw bytes and read in place without copying (zero copy)
//   - std::string and std::vector<T> of a trivially copyable T
//   - types that list their fields with a serialize() member:
//       struct Course {
//         std::string title;
//         std::vector<int> student_ids;
//
//         template <class Archive>
//         void serialize(Archive& archive) { archive(title, student_ids); }
//       };
//     Fields may be trivially copyable types, std::string, std::vector of any
//     supported field type, or nested types with serialize(). serialize() must
//     only pass fields to the archive, and the type must be
//     default-constructible.
//   - anything else: specialize nivo::ValueCodec<V> (see ValueCodecFor)
// Non-zero-copy values are encoded on write and decoded into a copy on read.
#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "nivo/cache.h"

namespace nivo {

// Thrown for failures that indicate misuse or an unusable cache
// (invalid_argument, pool_not_found, closed, ...). Misses and capacity
// pressure are reported through return values instead.
class CacheError : public std::runtime_error {
 public:
  explicit CacheError(Status status)
      : std::runtime_error(std::string("nivo: ") + to_string(status)), status_(status) {}
  [[nodiscard]] Status status() const noexcept { return status_; }

 private:
  Status status_;
};

// True for values stored as raw bytes and read in place. Cache value storage
// is aligned to alignof(std::max_align_t). Pointer members inside such a value
// are stored as plain addresses; the cache does not own or follow them.
template <class V>
inline constexpr bool is_zero_copy_value_v =
    std::is_trivially_copyable_v<V> && !std::is_pointer_v<V> &&
    !std::is_member_pointer_v<V> && alignof(V) <= alignof(std::max_align_t);

// ---------------------------------------------------------------------------
// Customization points
// ---------------------------------------------------------------------------

template <class K, class Enable = void>
struct KeyCodec;
template <class V, class Enable = void>
struct ValueCodec;

// A KeyCodec maps a key to a canonical byte string:
//   using Encoded = ...;                                  // owns or views bytes
//   static Encoded encode(const K& key);                  // may throw CacheError
//   static std::string_view view(const Encoded& encoded) noexcept;
template <class C, class K>
concept KeyCodecFor = requires(const K& key, const typename C::Encoded& encoded) {
  { C::encode(key) } -> std::same_as<typename C::Encoded>;
  { C::view(encoded) } -> std::convertible_to<std::string_view>;
};

// A ValueCodec writes a value into a buffer of exactly encoded_size() bytes:
//   static std::size_t encoded_size(const V& value);
//   static void encode(const V& value, std::span<std::byte> out);
//   static V decode(std::span<const std::byte> in);   // throw CacheError on malformed input
template <class C, class V>
concept ValueCodecFor = requires(const V& value, std::span<std::byte> out,
                                 std::span<const std::byte> in) {
  { C::encoded_size(value) } -> std::convertible_to<std::size_t>;
  C::encode(value, out);
  { C::decode(in) } -> std::same_as<V>;
};

namespace detail {
class SizeArchive;
class WriteArchive;
class ReadArchive;
}  // namespace detail

// True for types with a serialize(Archive&) member (see the top of this file).
template <class V>
concept Serializable = requires(V& value, detail::SizeArchive& size_archive,
                                detail::WriteArchive& write_archive,
                                detail::ReadArchive& read_archive) {
  value.serialize(size_archive);
  value.serialize(write_archive);
  value.serialize(read_archive);
};

template <class K, class V, class KeyCodecT, class ValueCodecT>
class TypedCache;

// ---------------------------------------------------------------------------
// TypedReadHandle
// ---------------------------------------------------------------------------

// Copyable immutable snapshot of one value; empty on a miss. The value stays
// valid after erase, replace, expiry, close and cache destruction.
//   zero-copy V: points into cache memory and pins the entry (it counts
//                against capacity and cannot be evicted until the last copy is
//                released).
//   other V:     holds a decoded copy; cache memory is not pinned.
template <class V>
class TypedReadHandle {
 public:
  TypedReadHandle() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return get() != nullptr; }
  [[nodiscard]] const V* get() const noexcept;
  [[nodiscard]] const V& operator*() const noexcept { return *get(); }
  [[nodiscard]] const V* operator->() const noexcept { return get(); }
  [[nodiscard]] Version version() const noexcept;
  void reset() noexcept { state_ = State{}; }

 private:
  template <class K, class T, class KC, class VC>
  friend class TypedCache;

  struct ZeroCopyState {
    ReadHandle handle;
    const V* value = nullptr;
  };
  struct DecodedState {
    std::shared_ptr<const V> value;
    Version version = invalid_version;
  };
  using State = std::conditional_t<is_zero_copy_value_v<V>, ZeroCopyState, DecodedState>;

  explicit TypedReadHandle(State state) : state_(std::move(state)) {}
  State state_;
};

// ---------------------------------------------------------------------------
// TypedCache
// ---------------------------------------------------------------------------

struct TypedCacheConfig {
  CacheConfig cache;  // cache.capacity_bytes is also the pool capacity
  std::string pool_name = "default";
  EvictionPolicy eviction = EvictionPolicy::lru;
};

// Type-safe facade over one pool. Use a pool with exactly one TypedCache
// instantiation: values of a different type with the same encoded size are not
// detected. Capacity is charged as encoded key bytes + stored value bytes.
// Thread safety and linearizability follow nivo::Cache.
//
// Error model:
//   - Constructors throw CacheError on failure.
//   - get/peek/get_value/contains: a missing or expired key yields an empty
//     result; other failures throw CacheError.
//   - put/insert/erase(key): expected outcomes are reported as bool (see each
//     method); other failures throw CacheError.
//   - find/write/erase(key, version): report every Status as a value instead
//     of throwing CacheError.
//   - std::bad_alloc and exceptions from user codecs may propagate.
// A moved-from TypedCache may only be destroyed or assigned to.
template <class K, class V, class KeyCodecT = KeyCodec<K>, class ValueCodecT = ValueCodec<V>>
class TypedCache {
  static_assert(KeyCodecFor<KeyCodecT, K>,
                "nivo::TypedCache: no usable KeyCodec for this key type. Use an integral, "
                "enum, float/double, std::string or padding-free trivially copyable key, "
                "or specialize nivo::KeyCodec<K>.");
  static_assert(is_zero_copy_value_v<V> || ValueCodecFor<ValueCodecT, V>,
                "nivo::TypedCache: value type is not trivially copyable. Add a "
                "serialize(Archive&) member that lists its fields, or specialize "
                "nivo::ValueCodec<V>.");

 public:
  using key_type = K;
  using value_type = V;
  using handle_type = TypedReadHandle<V>;
  static constexpr bool zero_copy = is_zero_copy_value_v<V>;
  static constexpr std::size_t default_capacity_bytes = 64 * 1024 * 1024;

  // Owning mode: creates a private Cache with one pool spanning its capacity.
  explicit TypedCache(std::size_t capacity_bytes = default_capacity_bytes)
      : TypedCache(make_config(capacity_bytes)) {}
  explicit TypedCache(TypedCacheConfig config);
  // Borrowing mode: uses an existing pool. `cache` must outlive this object.
  TypedCache(Cache& cache, PoolId pool);

  TypedCache(const TypedCache&) = delete;
  TypedCache& operator=(const TypedCache&) = delete;
  TypedCache(TypedCache&& other) noexcept;
  TypedCache& operator=(TypedCache&& other) noexcept;
  ~TypedCache() = default;

  // --- Reads ---------------------------------------------------------------

  // Updates hit/miss counters and LRU order.
  [[nodiscard]] handle_type get(const K& key) { return lookup_or_throw(key, true); }
  // No counters or recency update; still discovers expiry.
  [[nodiscard]] handle_type peek(const K& key) { return lookup_or_throw(key, false); }
  // Copy of the value; releases any pin before returning.
  [[nodiscard]] std::optional<V> get_value(const K& key)
    requires std::is_copy_constructible_v<V>;
  [[nodiscard]] bool contains(const K& key) { return static_cast<bool>(peek(key)); }

  // --- Writes --------------------------------------------------------------

  // Upsert. Returns false only on resource_exhausted (the value cannot be
  // cached right now). Oversized keys/values throw invalid_argument.
  bool put(const K& key, const V& value, Duration ttl = Duration{0});
  // Insert-only. Returns false on already_exists or resource_exhausted.
  bool insert(const K& key, const V& value, Duration ttl = Duration{0});
  // Returns true if an entry was removed, false if the key was absent.
  bool erase(const K& key);

  // --- Status-returning variants --------------------------------------------

  [[nodiscard]] Result<handle_type> find(const K& key) { return lookup(key, true); }
  // Full WriteOptions (insert_only, replace_only, compare_exchange, ttl).
  // Returns the new version on success.
  [[nodiscard]] Result<Version> write(const K& key, const V& value, WriteOptions options = {});
  [[nodiscard]] Status erase(const K& key, Version expected_version);

  // --- Introspection -------------------------------------------------------

  [[nodiscard]] Cache& cache() noexcept { return *cache_; }
  [[nodiscard]] const Cache& cache() const noexcept { return *cache_; }
  [[nodiscard]] PoolId pool() const noexcept { return pool_; }
  [[nodiscard]] Result<PoolStats> stats() const { return cache_->pool_stats(pool_); }

 private:
  static TypedCacheConfig make_config(std::size_t capacity_bytes);
  handle_type lookup_or_throw(const K& key, bool touch);
  Result<handle_type> lookup(const K& key, bool touch);
  static Result<handle_type> wrap(ReadHandle handle);

  std::unique_ptr<Cache> owned_;  // null in borrowing mode
  Cache* cache_ = nullptr;
  PoolId pool_ = invalid_pool;
};

// Name used in user-facing examples.
template <class K, class V>
using NivoCache = TypedCache<K, V>;

}  // namespace nivo

#include "nivo/detail/typed_codecs.h"
#include "nivo/detail/typed_cache_impl.h"
