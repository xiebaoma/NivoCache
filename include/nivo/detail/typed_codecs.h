// Built-in KeyCodec and ValueCodec specializations and the archives behind
// serialize(). Included by nivo/typed_cache.h; do not include directly.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nivo {

namespace detail {
template <std::size_t Size>
struct unsigned_of_size;
template <> struct unsigned_of_size<1> { using type = std::uint8_t; };
template <> struct unsigned_of_size<2> { using type = std::uint16_t; };
template <> struct unsigned_of_size<4> { using type = std::uint32_t; };
template <> struct unsigned_of_size<8> { using type = std::uint64_t; };

// Fixed-width little-endian encoding, independent of host byte order.
template <class T>
std::array<char, sizeof(T)> little_endian_bytes(const T& value) noexcept {
  using Unsigned = typename unsigned_of_size<sizeof(T)>::type;
  const Unsigned bits = std::bit_cast<Unsigned>(value);
  std::array<char, sizeof(T)> out{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    out[i] = static_cast<char>(static_cast<unsigned char>((bits >> (8 * i)) & 0xFFu));
  }
  return out;
}

template <class T>
inline constexpr bool is_integer_key_v =
    (std::is_integral_v<T> || std::is_enum_v<T>) &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);

template <class T>
// Only float and double: long double differs across platforms (8 bytes on
// Apple arm64 and MSVC, 16 bytes with padding on x86-64 Linux), so accepting it
// would make the set of valid key types platform dependent.
inline constexpr bool is_float_key_v =
    (std::is_same_v<T, float> || std::is_same_v<T, double>) &&
    std::numeric_limits<T>::is_iec559;

template <class T>
inline constexpr bool is_raw_key_v =
    !std::is_arithmetic_v<T> && !std::is_enum_v<T> && !std::is_pointer_v<T> &&
    !std::is_member_pointer_v<T> && std::is_trivially_copyable_v<T> &&
    std::has_unique_object_representations_v<T>;
}  // namespace detail

template <class K>
struct KeyCodec<K, std::enable_if_t<detail::is_integer_key_v<K>>> {
  using Encoded = std::array<char, sizeof(K)>;
  static Encoded encode(const K& key) noexcept { return detail::little_endian_bytes(key); }
  static std::string_view view(const Encoded& encoded) noexcept {
    return {encoded.data(), encoded.size()};
  }
};

template <class K>
struct KeyCodec<K, std::enable_if_t<detail::is_float_key_v<K>>> {
  using Encoded = std::array<char, sizeof(K)>;
  static Encoded encode(const K& key) {
    if (std::isnan(key)) throw CacheError(Status::invalid_argument);
    const K normalized = key == K(0) ? K(0) : key;
    return detail::little_endian_bytes(normalized);
  }
  static std::string_view view(const Encoded& encoded) noexcept {
    return {encoded.data(), encoded.size()};
  }
};

template <>
struct KeyCodec<std::string> {
  using Encoded = std::string_view;
  static Encoded encode(const std::string& key) noexcept { return key; }
  static std::string_view view(const Encoded& encoded) noexcept { return encoded; }
};

template <>
struct KeyCodec<std::string_view> {
  using Encoded = std::string_view;
  static Encoded encode(const std::string_view& key) noexcept { return key; }
  static std::string_view view(const Encoded& encoded) noexcept { return encoded; }
};

template <class K>
struct KeyCodec<K, std::enable_if_t<detail::is_raw_key_v<K>>> {
  using Encoded = std::array<char, sizeof(K)>;
  static Encoded encode(const K& key) noexcept {
    Encoded out{};
    std::memcpy(out.data(), std::addressof(key), sizeof(K));
    return out;
  }
  static std::string_view view(const Encoded& encoded) noexcept {
    return {encoded.data(), encoded.size()};
  }
};

template <>
struct ValueCodec<std::string> {
  static std::size_t encoded_size(const std::string& value) noexcept { return value.size(); }
  static void encode(const std::string& value, std::span<std::byte> out) noexcept {
    if (!value.empty()) std::memcpy(out.data(), value.data(), value.size());
  }
  static std::string decode(std::span<const std::byte> in) {
    return std::string(reinterpret_cast<const char*>(in.data()), in.size());
  }
};

template <class T>
struct ValueCodec<std::vector<T>,
                  std::enable_if_t<is_zero_copy_value_v<T> && !std::is_same_v<T, bool> &&
                                   std::is_default_constructible_v<T>>> {
  static std::size_t encoded_size(const std::vector<T>& value) noexcept {
    return value.size() * sizeof(T);
  }
  static void encode(const std::vector<T>& value, std::span<std::byte> out) noexcept {
    if (!value.empty()) std::memcpy(out.data(), value.data(), value.size() * sizeof(T));
  }
  static std::vector<T> decode(std::span<const std::byte> in) {
    if (in.size() % sizeof(T) != 0) throw CacheError(Status::invalid_argument);
    std::vector<T> out(in.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), in.data(), in.size());
    return out;
  }
};

namespace detail {
template <class T>
inline constexpr bool is_vector_v = false;
template <class T>
inline constexpr bool is_vector_v<std::vector<T>> = !std::is_same_v<T, bool>;

template <class T>
inline constexpr bool dependent_false_v = false;

using LengthPrefix = std::uint32_t;

class SizeArchive {
 public:
  template <class... Fields>
  void operator()(const Fields&... fields) { (add(fields), ...); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  template <class T>
  void add(const T& field) {
    if constexpr (is_zero_copy_value_v<T>) {
      size_ += sizeof(T);
    } else if constexpr (std::is_same_v<T, std::string>) {
      size_ += sizeof(LengthPrefix) + field.size();
    } else if constexpr (is_vector_v<T>) {
      using Element = typename T::value_type;
      size_ += sizeof(LengthPrefix);
      if constexpr (is_zero_copy_value_v<Element>) {
        size_ += field.size() * sizeof(Element);
      } else {
        for (const Element& element : field) add(element);
      }
    } else if constexpr (Serializable<T>) {
      const_cast<T&>(field).serialize(*this);  // only reads the fields
    } else {
      static_assert(dependent_false_v<T>,
                    "nivo::Serializable: unsupported field type. Use a zero-copy type, "
                    "std::string, std::vector or a type with serialize().");
    }
  }

  std::size_t size_ = 0;
};

class WriteArchive {
 public:
  explicit WriteArchive(std::span<std::byte> out) noexcept : out_(out) {}
  template <class... Fields>
  void operator()(const Fields&... fields) { (put(fields), ...); }
  [[nodiscard]] bool done() const noexcept { return offset_ == out_.size(); }

 private:
  void put_bytes(const void* data, std::size_t size) {
    if (size > out_.size() - offset_) throw CacheError(Status::invalid_argument);
    if (size != 0) std::memcpy(out_.data() + offset_, data, size);
    offset_ += size;
  }

  void put_length(std::size_t length) {
    if (length > std::numeric_limits<LengthPrefix>::max()) throw CacheError(Status::invalid_argument);
    const LengthPrefix prefix = static_cast<LengthPrefix>(length);
    put_bytes(&prefix, sizeof(prefix));
  }

  template <class T>
  void put(const T& field) {
    if constexpr (is_zero_copy_value_v<T>) {
      put_bytes(std::addressof(field), sizeof(T));
    } else if constexpr (std::is_same_v<T, std::string>) {
      put_length(field.size());
      put_bytes(field.data(), field.size());
    } else if constexpr (is_vector_v<T>) {
      using Element = typename T::value_type;
      put_length(field.size());
      if constexpr (is_zero_copy_value_v<Element>) {
        put_bytes(field.data(), field.size() * sizeof(Element));
      } else {
        for (const Element& element : field) put(element);
      }
    } else if constexpr (Serializable<T>) {
      const_cast<T&>(field).serialize(*this);  // only reads the fields
    } else {
      static_assert(dependent_false_v<T>, "nivo::Serializable: unsupported field type.");
    }
  }

  std::span<std::byte> out_;
  std::size_t offset_ = 0;
};

// Every read is bounds-checked; malformed input throws invalid_argument.
class ReadArchive {
 public:
  explicit ReadArchive(std::span<const std::byte> in) noexcept : in_(in) {}
  template <class... Fields>
  void operator()(Fields&... fields) { (take(fields), ...); }
  [[nodiscard]] bool done() const noexcept { return offset_ == in_.size(); }

 private:
  [[nodiscard]] std::size_t remaining() const noexcept { return in_.size() - offset_; }

  void take_bytes(void* data, std::size_t size) {
    if (size > remaining()) throw CacheError(Status::invalid_argument);
    if (size != 0) std::memcpy(data, in_.data() + offset_, size);
    offset_ += size;
  }

  std::size_t take_length() {
    LengthPrefix prefix = 0;
    take_bytes(&prefix, sizeof(prefix));
    return prefix;
  }

  template <class T>
  void take(T& field) {
    if constexpr (is_zero_copy_value_v<T>) {
      take_bytes(std::addressof(field), sizeof(T));
    } else if constexpr (std::is_same_v<T, std::string>) {
      const std::size_t length = take_length();
      if (length > remaining()) throw CacheError(Status::invalid_argument);
      field.assign(reinterpret_cast<const char*>(in_.data() + offset_), length);
      offset_ += length;
    } else if constexpr (is_vector_v<T>) {
      using Element = typename T::value_type;
      const std::size_t count = take_length();
      if constexpr (is_zero_copy_value_v<Element>) {
        if (count > remaining() / sizeof(Element)) throw CacheError(Status::invalid_argument);
        field.resize(count);
        take_bytes(field.data(), count * sizeof(Element));
      } else {
        // Caps allocation on corrupt input; assumes no element encodes to zero bytes.
        if (count > remaining()) throw CacheError(Status::invalid_argument);
        field.clear();
        field.resize(count);
        for (Element& element : field) take(element);
      }
    } else if constexpr (Serializable<T>) {
      field.serialize(*this);
    } else {
      static_assert(dependent_false_v<T>, "nivo::Serializable: unsupported field type.");
    }
  }

  std::span<const std::byte> in_;
  std::size_t offset_ = 0;
};
}  // namespace detail

template <class V>
struct ValueCodec<V, std::enable_if_t<Serializable<V> && std::is_default_constructible_v<V>>> {
  static std::size_t encoded_size(const V& value) {
    detail::SizeArchive archive;
    const_cast<V&>(value).serialize(archive);  // only reads the fields
    return archive.size();
  }
  static void encode(const V& value, std::span<std::byte> out) {
    detail::WriteArchive archive(out);
    const_cast<V&>(value).serialize(archive);
    if (!archive.done()) throw CacheError(Status::invalid_argument);
  }
  static V decode(std::span<const std::byte> in) {
    V value{};
    detail::ReadArchive archive(in);
    value.serialize(archive);
    if (!archive.done()) throw CacheError(Status::invalid_argument);
    return value;
  }
};

}  // namespace nivo
