#include <nivo/nivo.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
using nivo::Status;
#define CHECK(...) do { if (!(__VA_ARGS__)) throw std::runtime_error( \
  std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + #__VA_ARGS__); } while (false)

namespace {
struct ManualClock final : nivo::Clock {
  std::atomic<nivo::Duration::rep> ticks{0};
  nivo::Duration now() const noexcept override { return nivo::Duration(ticks.load()); }
  void advance(nivo::Duration delta) { ticks.fetch_add(delta.count()); }
};

template <class Function>
Status thrown_status(Function&& function) {
  try {
    function();
  } catch (const nivo::CacheError& error) {
    return error.status();
  }
  return Status::ok;
}

struct Student {
  int age;
  char name[32];
  double score;
};

// The strictest alignment the cache guarantees (8 on Apple arm64, 16 elsewhere).
struct alignas(alignof(std::max_align_t)) Aligned {
  std::uint64_t low;
  std::uint64_t high;
};

enum class Color : std::uint8_t { red = 1, green = 2 };

// Padding-free struct usable as a raw-byte key.
struct PackedKey {
  std::uint32_t tenant;
  std::uint32_t id;
};

// Struct with padding plus a user KeyCodec that encodes fields explicitly.
struct PaddedKey {
  char kind;
  std::uint64_t id;
};

// Non-trivially-copyable value with a user ValueCodec.
struct Profile {
  std::string name;
  std::vector<int> scores;
};

// Values encoded field by field through serialize(), including nesting.
struct Lesson {
  std::string topic;
  std::vector<std::string> tags;

  template <class Archive>
  void serialize(Archive& archive) { archive(topic, tags); }
};

struct Course {
  int id = 0;
  std::string title;
  std::vector<Lesson> lessons;
  std::vector<double> weights;

  template <class Archive>
  void serialize(Archive& archive) { archive(id, title, lessons, weights); }
};

// Large zero-copy value; 512 of them in a 16 MiB pool exercise region-level
// compaction like the core tests.
struct Blob {
  int id;
  char data[14996];
};

// 1 MiB zero-copy value with the strictest supported alignment; large enough
// to bypass the slab size classes.
struct alignas(alignof(std::max_align_t)) Wide {
  std::uint64_t words[131072];
};

enum class Grade : std::uint8_t { a, b, c };

// Every supported serialize() field kind in one value.
struct Record {
  Student student;
  char code[8];
  std::array<int, 3> triple;
  Grade grade;
  std::vector<std::vector<int>> matrix;
  std::vector<std::string> notes;
  std::string payload;

  template <class Archive>
  void serialize(Archive& archive) { archive(student, code, triple, grade, matrix, notes, payload); }
};

// Trivially copyable types stay zero-copy even with serialize().
struct Point {
  int x;
  int y;

  template <class Archive>
  void serialize(Archive& archive) { archive(x, y); }
};

// An explicit ValueCodec takes precedence over serialize().
struct Tagged {
  std::string text;

  template <class Archive>
  void serialize(Archive& archive) { archive(text); }
};
}  // namespace

template <>
struct nivo::KeyCodec<PaddedKey> {
  using Encoded = std::array<char, 9>;
  static Encoded encode(const PaddedKey& key) noexcept {
    Encoded out{};
    out[0] = key.kind;
    const std::array<char, 8> id = nivo::KeyCodec<std::uint64_t>::encode(key.id);
    std::memcpy(out.data() + 1, id.data(), id.size());
    return out;
  }
  static std::string_view view(const Encoded& encoded) noexcept {
    return {encoded.data(), encoded.size()};
  }
};

template <>
struct nivo::ValueCodec<Profile> {
  // Layout: u32 name length, name bytes, int scores.
  static std::size_t encoded_size(const Profile& value) noexcept {
    return sizeof(std::uint32_t) + value.name.size() + value.scores.size() * sizeof(int);
  }
  static void encode(const Profile& value, std::span<std::byte> out) {
    const std::uint32_t length = static_cast<std::uint32_t>(value.name.size());
    std::memcpy(out.data(), &length, sizeof(length));
    std::byte* cursor = out.data() + sizeof(length);
    if (!value.name.empty()) std::memcpy(cursor, value.name.data(), value.name.size());
    cursor += value.name.size();
    if (!value.scores.empty()) std::memcpy(cursor, value.scores.data(), value.scores.size() * sizeof(int));
  }
  static Profile decode(std::span<const std::byte> in) {
    std::uint32_t length = 0;
    if (in.size() < sizeof(length)) throw nivo::CacheError(Status::invalid_argument);
    std::memcpy(&length, in.data(), sizeof(length));
    const std::size_t rest = in.size() - sizeof(length);
    if (length > rest || (rest - length) % sizeof(int) != 0)
      throw nivo::CacheError(Status::invalid_argument);
    Profile out;
    out.name.assign(reinterpret_cast<const char*>(in.data() + sizeof(length)), length);
    out.scores.resize((rest - length) / sizeof(int));
    if (!out.scores.empty())
      std::memcpy(out.scores.data(), in.data() + sizeof(length) + length, rest - length);
    return out;
  }
};

template <>
struct nivo::ValueCodec<Tagged> {
  // Raw text without a length prefix.
  static std::size_t encoded_size(const Tagged& value) noexcept { return value.text.size(); }
  static void encode(const Tagged& value, std::span<std::byte> out) noexcept {
    if (!value.text.empty()) std::memcpy(out.data(), value.text.data(), value.text.size());
  }
  static Tagged decode(std::span<const std::byte> in) {
    return Tagged{std::string(reinterpret_cast<const char*>(in.data()), in.size())};
  }
};

namespace {
// Compile-time classification.
static_assert(nivo::is_zero_copy_value_v<int>);
static_assert(nivo::is_zero_copy_value_v<Student>);
static_assert(nivo::is_zero_copy_value_v<Aligned>);
static_assert(nivo::is_zero_copy_value_v<char[16]>);
static_assert(!nivo::is_zero_copy_value_v<std::string>);
static_assert(!nivo::is_zero_copy_value_v<std::vector<int>>);
static_assert(!nivo::is_zero_copy_value_v<const char*>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<int>, int>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<double>, double>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<Color>, Color>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<std::string>, std::string>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<PackedKey>, PackedKey>);
static_assert(nivo::KeyCodecFor<nivo::KeyCodec<std::array<char, 8>>, std::array<char, 8>>);
static_assert(!nivo::KeyCodecFor<nivo::KeyCodec<Student>, Student>);  // has padding
static_assert(!nivo::KeyCodecFor<nivo::KeyCodec<const char*>, const char*>);
static_assert(!nivo::KeyCodecFor<nivo::KeyCodec<long double>, long double>);
static_assert(nivo::ValueCodecFor<nivo::ValueCodec<std::vector<double>>, std::vector<double>>);
static_assert(!nivo::ValueCodecFor<nivo::ValueCodec<std::vector<std::string>>, std::vector<std::string>>);
static_assert(nivo::Serializable<Course> && !nivo::Serializable<Profile>);
static_assert(nivo::ValueCodecFor<nivo::ValueCodec<Course>, Course>);
static_assert(nivo::TypedCache<int, Student>::zero_copy);
static_assert(!nivo::TypedCache<int, std::string>::zero_copy);
static_assert(std::is_same_v<nivo::NivoCache<int, Student>, nivo::TypedCache<int, Student>>);
static_assert(std::is_copy_constructible_v<nivo::TypedReadHandle<Student>>);
static_assert(!std::is_copy_constructible_v<nivo::TypedCache<int, int>>);
static_assert(std::is_nothrow_move_constructible_v<nivo::TypedCache<int, int>>);

std::shared_ptr<ManualClock> make_clock() { return std::make_shared<ManualClock>(); }

nivo::TypedCacheConfig small_config(std::size_t capacity, std::shared_ptr<ManualClock> clock = nullptr) {
  nivo::TypedCacheConfig config;
  config.cache.capacity_bytes = capacity;
  config.cache.clock = std::move(clock);
  return config;
}

void student_round_trip() {
  nivo::NivoCache<int, Student> cache(64 * 1024 * 1024);
  const Student student{20, "Alice", 95.5};
  CHECK(cache.put(10001, student));
  nivo::TypedReadHandle<Student> result = cache.get(10001);
  CHECK(result);
  CHECK(result->age == 20 && std::string_view(result->name) == "Alice" && result->score == 95.5);
  CHECK((*result).age == 20 && result.get() != nullptr);
  CHECK(result.version() != nivo::invalid_version);
  CHECK(reinterpret_cast<std::uintptr_t>(result.get()) % alignof(Student) == 0);
  CHECK(cache.contains(10001) && !cache.contains(10002));
  CHECK(!cache.get(10002) && cache.get(10002).get() == nullptr);
  CHECK(cache.get(10002).version() == nivo::invalid_version);
  const std::optional<Student> copy = cache.get_value(10001);
  CHECK(copy && copy->score == 95.5);
  CHECK(!cache.get_value(10002));
  // Overwrite produces a new version; the old handle keeps the old snapshot.
  CHECK(cache.put(10001, Student{21, "Alice", 97.0}));
  CHECK(result->age == 20 && cache.get(10001)->age == 21);
  CHECK(cache.get(10001).version() > result.version());
  CHECK(cache.erase(10001) && !cache.erase(10001));
  CHECK(!cache.get(10001) && result->age == 20);
  result.reset();
  CHECK(!result && result.get() == nullptr);
  // Charge is encoded key bytes plus sizeof(value).
  CHECK(cache.put(7, student));
  CHECK(cache.stats().value().indexed_bytes == sizeof(int) + sizeof(Student));
}

void default_and_scalar_types() {
  nivo::NivoCache<int, int> ints;
  CHECK(ints.put(1, 100) && ints.put(-1, -100));
  CHECK(*ints.get(1) == 100 && *ints.get(-1) == -100);
  CHECK(ints.stats().value().capacity_bytes == nivo::TypedCache<int, int>::default_capacity_bytes);

  nivo::NivoCache<std::uint64_t, double> doubles(1024 * 1024);
  CHECK(doubles.put(std::numeric_limits<std::uint64_t>::max(), 3.25));
  CHECK(*doubles.get(std::numeric_limits<std::uint64_t>::max()) == 3.25);

  nivo::NivoCache<Color, float> colors(1024 * 1024);
  CHECK(colors.put(Color::red, 1.5f) && colors.put(Color::green, 2.5f));
  CHECK(*colors.get(Color::red) == 1.5f && *colors.get(Color::green) == 2.5f);

  nivo::NivoCache<int, char[16]> chars(1024 * 1024);
  const char label[16] = "hello";
  CHECK(chars.put(1, label));
  CHECK(std::string_view(*chars.get(1)) == "hello");

  nivo::NivoCache<int, Aligned> aligned(1024 * 1024);
  for (int i = 0; i < 64; ++i) CHECK(aligned.put(i, Aligned{static_cast<std::uint64_t>(i), 7}));
  for (int i = 0; i < 64; ++i) {
    nivo::TypedReadHandle<Aligned> found = aligned.get(i);
    CHECK(found && found->low == static_cast<std::uint64_t>(i) && found->high == 7);
    CHECK(reinterpret_cast<std::uintptr_t>(found.get()) % alignof(Aligned) == 0);
  }
}

void key_encodings() {
  const std::array<char, 4> encoded = nivo::KeyCodec<std::uint32_t>::encode(0x01020304u);
  CHECK(encoded[0] == 4 && encoded[1] == 3 && encoded[2] == 2 && encoded[3] == 1);
  const std::array<char, 8> negative = nivo::KeyCodec<std::int64_t>::encode(-1);
  for (char byte : negative) CHECK(static_cast<unsigned char>(byte) == 0xFF);

  nivo::NivoCache<double, int> by_double(1024 * 1024);
  CHECK(by_double.put(-0.0, 5));
  CHECK(by_double.get(0.0) && *by_double.get(0.0) == 5);
  CHECK(by_double.put(1.5, 6) && *by_double.get(1.5) == 6);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(thrown_status([&] { (void)by_double.get(nan); }) == Status::invalid_argument);
  CHECK(thrown_status([&] { by_double.put(nan, 1); }) == Status::invalid_argument);
  CHECK(by_double.find(nan).status() == Status::invalid_argument);
  CHECK(by_double.write(nan, 1).status() == Status::invalid_argument);
  CHECK(by_double.erase(nan, nivo::invalid_version) == Status::invalid_argument);

  nivo::NivoCache<std::string, Student> by_name(1024 * 1024);
  CHECK(by_name.put("alice", Student{20, "Alice", 90.0}));
  CHECK(by_name.put(std::string("a\0b", 3), Student{30, "Binary", 80.0}));
  CHECK(by_name.get("alice")->age == 20);
  CHECK(by_name.get(std::string("a\0b", 3))->age == 30 && !by_name.get("a"));
  CHECK(thrown_status([&] { by_name.put("", Student{}); }) == Status::invalid_argument);
  CHECK(thrown_status([&] { (void)by_name.get(std::string(2048, 'k')); }) == Status::invalid_argument);

  nivo::NivoCache<std::string_view, int> by_view(1024 * 1024);
  const std::string owned = "view-key";
  CHECK(by_view.put(owned, 1) && *by_view.get("view-key") == 1);

  nivo::NivoCache<PackedKey, int> packed(1024 * 1024);
  CHECK(packed.put({1, 2}, 12) && packed.put({2, 1}, 21));
  CHECK(*packed.get({1, 2}) == 12 && *packed.get({2, 1}) == 21);

  nivo::NivoCache<std::array<char, 4>, int> arrays(1024 * 1024);
  CHECK(arrays.put({'a', 'b', 'c', 'd'}, 4) && *arrays.get({'a', 'b', 'c', 'd'}) == 4);

  // The user codec ignores padding bytes, so keys built differently still match.
  nivo::NivoCache<PaddedKey, int> padded(1024 * 1024);
  PaddedKey first;
  std::memset(&first, 0xAB, sizeof(first));
  first.kind = 'x';
  first.id = 99;
  PaddedKey second;
  std::memset(&second, 0x00, sizeof(second));
  second.kind = 'x';
  second.id = 99;
  CHECK(padded.put(first, 1) && padded.get(second) && *padded.get(second) == 1);
}

void codec_values() {
  nivo::NivoCache<std::uint64_t, std::vector<int>> vectors(1024 * 1024);
  const std::vector<int> numbers{1, 2, 3, 4, 5};
  CHECK(vectors.put(1, numbers));
  CHECK(vectors.put(2, std::vector<int>{}));
  CHECK(*vectors.get(1) == numbers && vectors.get(2) && vectors.get(2)->empty());
  CHECK(vectors.stats().value().indexed_bytes == 2 * sizeof(std::uint64_t) + numbers.size() * sizeof(int));

  nivo::NivoCache<int, std::string> strings(1024 * 1024);
  CHECK(strings.put(1, "hello") && strings.put(2, ""));
  CHECK(*strings.get(1) == "hello" && strings.get(2) && strings.get(2)->empty());
  CHECK(strings.get_value(1) == std::optional<std::string>("hello"));

  nivo::NivoCache<std::string, Profile> profiles(1024 * 1024);
  CHECK(profiles.put("bob", Profile{"Bob", {90, 85, 77}}));
  nivo::TypedReadHandle<Profile> bob = profiles.get("bob");
  CHECK(bob && bob->name == "Bob" && bob->scores == std::vector<int>({90, 85, 77}));
  CHECK(profiles.put("empty", Profile{}) && profiles.get("empty")->name.empty());

  // Decoded handles do not pin cache memory; zero-copy handles do.
  nivo::Result<nivo::PoolStats> before = strings.stats();
  nivo::TypedReadHandle<std::string> decoded = strings.get(1);
  CHECK(strings.erase(1));
  CHECK(strings.stats().value().allocated_bytes == before.value().allocated_bytes - sizeof(int) - 5);
  CHECK(*decoded == "hello");

  nivo::NivoCache<int, Student> students(1024 * 1024);
  CHECK(students.put(1, Student{1, "pinned", 1.0}));
  const std::size_t pinned_bytes = students.stats().value().allocated_bytes;
  nivo::TypedReadHandle<Student> pinned = students.get(1);
  nivo::TypedReadHandle<Student> pinned_copy = pinned;
  CHECK(students.erase(1));
  CHECK(students.stats().value().allocated_bytes == pinned_bytes);
  pinned.reset();
  CHECK(students.stats().value().allocated_bytes == pinned_bytes);
  CHECK(std::string_view(pinned_copy->name) == "pinned");
  pinned_copy.reset();
  CHECK(students.stats().value().allocated_bytes == 0);
}

void serializable_values() {
  using Codec = nivo::ValueCodec<Course>;
  const Course course{7, "Systems", {{"memory", {"slab", "mmap"}}, {"io", {}}}, {0.5, 0.25}};
  // id + title + lesson count + lessons + weight count + weights
  const std::size_t expected_size = sizeof(int) + (4 + 7) + 4 +
      ((4 + 6) + 4 + (4 + 4) + (4 + 4)) + ((4 + 2) + 4) + 4 + 2 * sizeof(double);
  CHECK(Codec::encoded_size(course) == expected_size);

  nivo::NivoCache<std::string, Course> courses(1024 * 1024);
  CHECK(courses.put("cs101", course));
  CHECK(courses.put("empty", Course{}));
  nivo::TypedReadHandle<Course> found = courses.get("cs101");
  CHECK(found && found->id == 7 && found->title == "Systems");
  CHECK(found->lessons.size() == 2 && found->lessons[0].topic == "memory");
  CHECK(found->lessons[0].tags == std::vector<std::string>({"slab", "mmap"}));
  CHECK(found->lessons[1].topic == "io" && found->lessons[1].tags.empty());
  CHECK(found->weights == std::vector<double>({0.5, 0.25}));
  nivo::TypedReadHandle<Course> empty = courses.get("empty");
  CHECK(empty && empty->id == 0 && empty->title.empty() && empty->lessons.empty());
  CHECK(courses.stats().value().indexed_bytes ==
        5 + expected_size + 5 + Codec::encoded_size(Course{}));

  // Truncated, oversized and trailing input is rejected.
  std::vector<std::byte> bytes(expected_size);
  Codec::encode(course, bytes);
  CHECK(Codec::decode(bytes).title == "Systems");
  CHECK(thrown_status([&] { (void)Codec::decode(std::span<const std::byte>(bytes).first(bytes.size() - 1)); }) ==
        Status::invalid_argument);
  std::vector<std::byte> trailing = bytes;
  trailing.push_back(std::byte{0});
  CHECK(thrown_status([&] { (void)Codec::decode(trailing); }) == Status::invalid_argument);
  std::vector<std::byte> huge_count = bytes;
  std::memset(huge_count.data() + sizeof(int) + 4 + 7, 0xFF, 4);  // lesson count
  CHECK(thrown_status([&] { (void)Codec::decode(huge_count); }) == Status::invalid_argument);
}

void write_modes_and_versions() {
  nivo::NivoCache<int, int> cache(1024);
  CHECK(cache.insert(1, 10) && !cache.insert(1, 11) && *cache.get(1) == 10);
  CHECK(cache.write(2, 20, {.mode = nivo::WriteMode::replace_only}).status() == Status::not_found);
  const nivo::Version first = cache.write(2, 20).value();
  CHECK(cache.write(2, 21, {.mode = nivo::WriteMode::compare_exchange, .expected_version = first + 100})
            .status() == Status::version_mismatch);
  const nivo::Result<nivo::Version> second =
      cache.write(2, 22, {.mode = nivo::WriteMode::compare_exchange, .expected_version = first});
  CHECK(second && second.value() > first && *cache.get(2) == 22);
  CHECK(cache.get(2).version() == second.value());
  CHECK(cache.write(2, 23, {.mode = nivo::WriteMode::upsert, .expected_version = 5}).status() ==
        Status::invalid_argument);
  CHECK(cache.erase(2, first) == Status::version_mismatch);
  CHECK(cache.erase(2, second.value()) == Status::ok);
  CHECK(cache.erase(2, nivo::invalid_version) == Status::not_found);
  CHECK(cache.find(2).status() == Status::not_found);
  CHECK(cache.find(1) && *cache.find(1).value() == 10);
  CHECK(thrown_status([&] { cache.put(3, 1, nivo::Duration(-1)); }) == Status::invalid_argument);

  // A value that cannot fit is a dropped write, not an error.
  nivo::NivoCache<int, std::vector<int>> small(1024);
  CHECK(!small.put(1, std::vector<int>(1024, 7)));
  CHECK(!small.insert(1, std::vector<int>(1024, 7)));
  CHECK(small.write(1, std::vector<int>(1024, 7)).status() == Status::resource_exhausted);
  CHECK(small.put(1, std::vector<int>(16, 7)) && small.get(1)->size() == 16);
}

void ttl_and_eviction() {
  std::shared_ptr<ManualClock> clock = make_clock();
  nivo::NivoCache<int, Student> cache(small_config(1024 * 1024, clock));
  CHECK(cache.put(1, Student{1, "short", 1.0}, 10ms));
  CHECK(cache.put(2, Student{2, "forever", 2.0}));
  clock->advance(9ms);
  CHECK(cache.get(1) && cache.peek(1));
  clock->advance(1ms);
  CHECK(!cache.get(1) && !cache.contains(1) && cache.get(2));

  // LRU eviction under capacity pressure, pool capacity from TypedCacheConfig.
  const std::size_t per_item = sizeof(int) + sizeof(Student);
  nivo::NivoCache<int, Student> lru(small_config(per_item * 3));
  CHECK(lru.put(1, Student{}) && lru.put(2, Student{}) && lru.put(3, Student{}));
  CHECK(lru.get(1));
  CHECK(lru.put(4, Student{}));
  CHECK(lru.contains(1) && !lru.contains(2) && lru.contains(3) && lru.contains(4));
  // peek does not refresh recency.
  CHECK(lru.peek(3));
  CHECK(lru.put(5, Student{}));
  CHECK(!lru.contains(3));
}

void construction_and_borrowing() {
  CHECK(thrown_status([] { nivo::NivoCache<int, int> zero(0); }) == Status::invalid_argument);
  nivo::TypedCacheConfig hybrid = small_config(1024);
  hybrid.cache.storage = nivo::StorageMode::hybrid;
  CHECK(thrown_status([&] { nivo::NivoCache<int, int> unsupported(hybrid); }) == Status::unsupported);
  nivo::TypedCacheConfig named = small_config(1024);
  named.pool_name = "";
  CHECK(thrown_status([&] { nivo::NivoCache<int, int> unnamed(named); }) == Status::invalid_argument);

  std::unique_ptr<nivo::Cache> shared = nivo::Cache::create(nivo::CacheConfig{.capacity_bytes = 1024 * 1024, .clock = nullptr}).value();
  const nivo::PoolId students_pool =
      shared->add_pool({.name = "students", .capacity_bytes = 512 * 1024}).value();
  const nivo::PoolId names_pool =
      shared->add_pool({.name = "names", .capacity_bytes = 512 * 1024}).value();
  nivo::TypedCache<int, Student> students(*shared, students_pool);
  nivo::TypedCache<int, std::string> names(*shared, names_pool);
  CHECK(students.pool() == students_pool && &students.cache() == shared.get());
  CHECK(students.put(1, Student{20, "Alice", 1.0}) && names.put(1, "Alice"));
  CHECK(students.get(1)->age == 20 && *names.get(1) == "Alice");
  CHECK(shared->stats().indexed_items == 2);
  CHECK(thrown_status([&] { nivo::TypedCache<int, int> missing(*shared, 999); }) ==
        Status::pool_not_found);

  // Closing the borrowed cache surfaces through every API style.
  nivo::TypedReadHandle<Student> survivor = students.get(1);
  shared->close();
  CHECK(thrown_status([&] { (void)students.get(1); }) == Status::closed);
  CHECK(thrown_status([&] { students.put(2, Student{}); }) == Status::closed);
  CHECK(thrown_status([&] { students.erase(1); }) == Status::closed);
  CHECK(students.find(1).status() == Status::closed);
  CHECK(students.write(2, Student{}).status() == Status::closed);
  shared.reset();
  CHECK(survivor->age == 20);

  // Owned cache: handles outlive the cache object.
  nivo::TypedReadHandle<Student> zero_copy;
  nivo::TypedReadHandle<std::vector<int>> decoded;
  {
    nivo::NivoCache<int, Student> owner(1024 * 1024);
    nivo::NivoCache<int, std::vector<int>> vectors(1024 * 1024);
    CHECK(owner.put(1, Student{42, "Owner", 1.0}) && vectors.put(1, std::vector<int>{1, 2}));
    zero_copy = owner.get(1);
    decoded = vectors.get(1);
  }
  CHECK(zero_copy->age == 42 && (*decoded)[1] == 2);

  // Move construction and assignment transfer the cache.
  nivo::NivoCache<int, int> source(1024);
  CHECK(source.put(1, 1));
  nivo::NivoCache<int, int> moved(std::move(source));
  CHECK(*moved.get(1) == 1);
  nivo::NivoCache<int, int> target(1024);
  target = std::move(moved);
  CHECK(*target.get(1) == 1);
}

void type_mismatch_is_detected() {
  std::unique_ptr<nivo::Cache> shared = nivo::Cache::create(nivo::CacheConfig{.capacity_bytes = 1024 * 1024, .clock = nullptr}).value();
  const nivo::PoolId pool = shared->add_pool({.name = "mixed", .capacity_bytes = 1024 * 1024}).value();
  nivo::TypedCache<int, std::uint64_t> wide(*shared, pool);
  nivo::TypedCache<int, std::uint32_t> narrow(*shared, pool);
  nivo::TypedCache<int, std::string> text(*shared, pool);
  nivo::TypedCache<int, std::vector<int>> vectors(*shared, pool);
  nivo::TypedCache<int, Profile> profiles(*shared, pool);
  CHECK(wide.put(1, 5));
  CHECK(thrown_status([&] { (void)narrow.get(1); }) == Status::invalid_argument);
  CHECK(narrow.find(1).status() == Status::invalid_argument);
  CHECK(text.put(2, "abc"));
  CHECK(thrown_status([&] { (void)vectors.get(2); }) == Status::invalid_argument);
  CHECK(thrown_status([&] { (void)profiles.get(2); }) == Status::invalid_argument);
  CHECK(vectors.find(2).status() == Status::invalid_argument);
}

void concurrent_readers_and_writers() {
  nivo::NivoCache<int, Student> cache(4 * 1024 * 1024);
  constexpr int keys = 64;
  constexpr int rounds = 2000;
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int writer = 0; writer < 2; ++writer) {
    threads.emplace_back([&cache, writer] {
      for (int i = 0; i < rounds; ++i) {
        const int key = (i * 7 + writer) % keys;
        Student student{key, "", static_cast<double>(i)};
        std::snprintf(student.name, sizeof(student.name), "student-%d", key);
        cache.put(key, student);
      }
    });
  }
  for (int reader = 0; reader < 4; ++reader) {
    threads.emplace_back([&cache, &failed, reader] {
      for (int i = 0; i < rounds; ++i) {
        const int key = (i + reader) % keys;
        nivo::TypedReadHandle<Student> found = cache.get(key);
        if (!found) continue;
        const std::string expected = "student-" + std::to_string(key);
        if (found->age != key || expected != found->name) failed = true;
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed);
}

// ---------------------------------------------------------------------------
// Interaction with the memory subsystem
// ---------------------------------------------------------------------------

constexpr int sparse_items = 512;
constexpr int sparse_stride = 32;

bool max_aligned(const void* address) {
  return reinterpret_cast<std::uintptr_t>(address) % alignof(std::max_align_t) == 0;
}

void fill_blob(Blob& blob, int id, int salt = 0) {
  blob.id = id;
  std::memset(blob.data, (id + salt) % 251, sizeof(blob.data));
}

// True if the blob belongs to `id` and its payload is one consistent write.
bool blob_consistent(const Blob& blob, int id) {
  if (blob.id != id) return false;
  for (char byte : blob.data)
    if (byte != blob.data[0]) return false;
  return true;
}

bool blob_matches(const Blob& blob, int id, int salt = 0) {
  return blob_consistent(blob, id) &&
         static_cast<unsigned char>(blob.data[0]) == static_cast<unsigned char>((id + salt) % 251);
}

// Fills the pool, then erases all but every 32nd key to leave sparse regions.
void populate_sparse_blobs(nivo::NivoCache<int, Blob>& blobs) {
  std::unique_ptr<Blob> blob = std::make_unique<Blob>();
  for (int i = 0; i < sparse_items; ++i) {
    fill_blob(*blob, i);
    CHECK(blobs.put(i, *blob));
  }
  for (int i = 0; i < sparse_items; ++i)
    if (i % sparse_stride != 0) CHECK(blobs.erase(i));
}

std::string course_title(int id) { return std::string(15000, static_cast<char>('a' + id % 26)); }

void zero_copy_values_survive_compaction() {
  std::shared_ptr<ManualClock> clock = make_clock();
  nivo::NivoCache<int, Blob> blobs(small_config(16 * 1024 * 1024, clock));
  populate_sparse_blobs(blobs);
  std::unique_ptr<Blob> expiring = std::make_unique<Blob>();
  fill_blob(*expiring, sparse_items);
  CHECK(blobs.put(sparse_items, *expiring, 50ms));

  std::vector<nivo::Version> versions;
  for (int i = 0; i < sparse_items; i += sparse_stride) versions.push_back(blobs.peek(i).version());
  const nivo::PoolStats before = blobs.stats().value();

  const nivo::CompactionStats compacted = blobs.cache().compact(blobs.pool()).value();
  CHECK(compacted.moved_items > 0);
  CHECK(compacted.reserved_bytes_after < compacted.reserved_bytes_before);

  // Contents, alignment, versions, accounting and TTL are all preserved.
  for (int i = 0; i < sparse_items; i += sparse_stride) {
    nivo::TypedReadHandle<Blob> found = blobs.peek(i);
    CHECK(found && blob_matches(*found, i) && max_aligned(found.get()));
    CHECK(found.version() == versions[static_cast<std::size_t>(i / sparse_stride)]);
  }
  for (int i = 1; i < sparse_items; i += sparse_stride) CHECK(!blobs.contains(i));
  const nivo::PoolStats after = blobs.stats().value();
  CHECK(after.indexed_items == before.indexed_items && after.indexed_bytes == before.indexed_bytes);
  CHECK(after.allocated_bytes == before.allocated_bytes);
  clock->advance(49ms);
  CHECK(blobs.get(sparse_items) && blob_matches(*blobs.get(sparse_items), sparse_items));
  clock->advance(1ms);
  CHECK(!blobs.get(sparse_items));

  // The cache keeps working normally after compaction.
  std::unique_ptr<Blob> fresh = std::make_unique<Blob>();
  for (int i = 1; i < sparse_items; i += sparse_stride) {
    fill_blob(*fresh, i, 7);
    CHECK(blobs.put(i, *fresh));
  }
  for (int i = 1; i < sparse_items; i += sparse_stride) CHECK(blob_matches(*blobs.get(i), i, 7));
}

void pinned_handles_during_compaction() {
  nivo::NivoCache<int, Blob> blobs(16 * 1024 * 1024);
  populate_sparse_blobs(blobs);
  nivo::TypedReadHandle<Blob> pinned = blobs.get(0);
  const Blob* address = pinned.get();
  const std::size_t entry_bytes = sizeof(int) + sizeof(Blob);

  // A zero-copy handle pins its entry: compaction must not move it.
  const nivo::CompactionStats first = blobs.cache().compact(blobs.pool()).value();
  CHECK(first.skipped_items >= 1);
  CHECK(pinned.get() == address && blob_matches(*pinned, 0));
  CHECK(blobs.peek(0).get() == address);

  // Erasing a pinned key unlinks it but keeps the bytes until release.
  CHECK(blobs.erase(0) && !blobs.contains(0));
  const std::size_t allocated = blobs.stats().value().allocated_bytes;
  CHECK(blobs.stats().value().indexed_bytes + entry_bytes == allocated);
  (void)blobs.cache().compact(blobs.pool()).value();
  CHECK(pinned.get() == address && blob_matches(*pinned, 0));
  pinned.reset();
  CHECK(blobs.stats().value().allocated_bytes == allocated - entry_bytes);
  for (int i = sparse_stride; i < sparse_items; i += sparse_stride)
    CHECK(blob_matches(*blobs.peek(i), i));

  // Decoded handles never pin, so their entries remain movable.
  nivo::NivoCache<int, std::string> strings(16 * 1024 * 1024);
  for (int i = 0; i < sparse_items; ++i) CHECK(strings.put(i, course_title(i)));
  for (int i = 0; i < sparse_items; ++i)
    if (i % sparse_stride != 0) CHECK(strings.erase(i));
  nivo::TypedReadHandle<std::string> decoded = strings.get(0);
  const nivo::CompactionStats moved = strings.cache().compact(strings.pool()).value();
  CHECK(moved.moved_items > 0 && moved.skipped_items < moved.moved_items);
  CHECK(*decoded == course_title(0) && *strings.get(0) == course_title(0));
}

void codec_values_survive_compaction() {
  nivo::NivoCache<int, Course> courses(16 * 1024 * 1024);
  for (int i = 0; i < sparse_items; ++i) {
    CHECK(courses.put(i, Course{i, course_title(i), {{"lesson", {std::to_string(i)}}}, {i * 0.5}}));
  }
  for (int i = 0; i < sparse_items; ++i)
    if (i % sparse_stride != 0) CHECK(courses.erase(i));
  const std::size_t indexed = courses.stats().value().indexed_bytes;
  const nivo::CompactionStats compacted = courses.cache().compact(courses.pool()).value();
  CHECK(compacted.moved_items > 0 && compacted.reserved_bytes_after < compacted.reserved_bytes_before);
  CHECK(courses.stats().value().indexed_bytes == indexed);
  for (int i = 0; i < sparse_items; i += sparse_stride) {
    nivo::TypedReadHandle<Course> found = courses.get(i);
    CHECK(found && found->id == i && found->title == course_title(i));
    CHECK(found->lessons.size() == 1 && found->lessons[0].tags[0] == std::to_string(i));
    CHECK(found->weights == std::vector<double>({i * 0.5}));
  }
}

void values_across_size_classes() {
  // Sizes around every power of two up to 4 MiB plus random sizes, so values
  // land in many slab size classes and in direct large allocations.
  std::vector<std::size_t> sizes;
  for (std::size_t size = 1; size <= 4 * 1024 * 1024; size *= 2) {
    sizes.push_back(size - 1);
    sizes.push_back(size);
    sizes.push_back(size + 1);
  }
  std::mt19937 random(42);
  std::uniform_int_distribution<std::size_t> distribution(0, 200000);
  for (int i = 0; i < 200; ++i) sizes.push_back(distribution(random));

  nivo::NivoCache<int, std::string> strings(64 * 1024 * 1024);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const int key = static_cast<int>(i);
    CHECK(strings.put(key, std::string(sizes[i], static_cast<char>('a' + i % 26))));
  }
  const nivo::PoolStats stats = strings.stats().value();
  CHECK(stats.counters.evictions == 0 && stats.indexed_items == sizes.size());
  CHECK(stats.allocator.slabs > 0 && stats.allocator.large_allocations > 0);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    nivo::TypedReadHandle<std::string> found = strings.get(static_cast<int>(i));
    CHECK(found && *found == std::string(sizes[i], static_cast<char>('a' + i % 26)));
  }
  for (std::size_t i = 0; i < sizes.size(); ++i) CHECK(strings.erase(static_cast<int>(i)));
  CHECK(strings.stats().value().allocated_bytes == 0 && strings.stats().value().indexed_items == 0);
  CHECK(strings.cache().memory_stats(strings.pool()).value().kv_bytes == 0);

  // Zero-copy values stay aligned in direct large allocations and behind
  // keys of every length.
  std::unique_ptr<Wide> wide = std::make_unique<Wide>();
  for (std::size_t i = 0; i < std::size(wide->words); ++i) wide->words[i] = i * 3;
  nivo::NivoCache<int, Wide> wides(8 * 1024 * 1024);
  CHECK(wides.put(1, *wide) && wides.put(2, *wide));
  CHECK(wides.stats().value().allocator.large_allocations >= 2);
  nivo::TypedReadHandle<Wide> found_wide = wides.get(2);
  CHECK(found_wide && max_aligned(found_wide.get()));
  CHECK(std::memcmp(found_wide->words, wide->words, sizeof(wide->words)) == 0);

  nivo::NivoCache<std::string, Aligned> by_key(1024 * 1024);
  for (std::size_t length = 1; length <= 64; ++length)
    CHECK(by_key.put(std::string(length, 'k'), Aligned{length, length * 2}));
  for (std::size_t length = 1; length <= 64; ++length) {
    nivo::TypedReadHandle<Aligned> found = by_key.get(std::string(length, 'k'));
    CHECK(found && max_aligned(found.get()) && found->low == length && found->high == length * 2);
  }
}

void growing_and_shrinking_overwrites() {
  nivo::NivoCache<int, std::string> strings(16 * 1024 * 1024);
  std::vector<std::pair<std::size_t, nivo::TypedReadHandle<std::string>>> snapshots;
  constexpr int rounds = 2000;
  for (int round = 0; round < rounds; ++round) {
    const std::size_t size = static_cast<std::size_t>(round) * 7919 % 70000;
    CHECK(strings.put(1, std::string(size, static_cast<char>('a' + round % 26))));
    const nivo::PoolStats stats = strings.stats().value();
    CHECK(stats.indexed_items == 1 && stats.indexed_bytes == sizeof(int) + size);
    if (round % 100 == 0) snapshots.emplace_back(size, strings.get(1));
  }
  CHECK(strings.stats().value().counters.replacements == rounds - 1);
  for (std::size_t i = 0; i < snapshots.size(); ++i) {
    const std::size_t size = snapshots[i].first;
    const char fill = static_cast<char>('a' + (i * 100) % 26);
    CHECK(*snapshots[i].second == std::string(size, fill));
  }
  CHECK(strings.erase(1) && strings.stats().value().allocated_bytes == 0);

  // A pinned zero-copy snapshot keeps exactly one old entry alive.
  nivo::NivoCache<int, Student> students(1024 * 1024);
  const std::size_t entry_bytes = sizeof(int) + sizeof(Student);
  CHECK(students.put(1, Student{0, "v0", 0.0}));
  nivo::TypedReadHandle<Student> first = students.get(1);
  for (int version = 1; version <= 100; ++version) CHECK(students.put(1, Student{version, "vN", 0.0}));
  CHECK(students.stats().value().allocated_bytes == 2 * entry_bytes);
  CHECK(first->age == 0 && students.get(1)->age == 100);
  first.reset();
  CHECK(students.stats().value().allocated_bytes == entry_bytes);
}

void memory_accounting_for_typed_values() {
  nivo::NivoCache<std::string, Course> courses(4 * 1024 * 1024);
  std::size_t expected = 0;
  for (int i = 0; i < 100; ++i) {
    const std::string key = "course-" + std::to_string(i);
    const Course course{i, std::string(static_cast<std::size_t>(i) * 13, 't'),
                        {{"l", std::vector<std::string>(static_cast<std::size_t>(i % 5), "tag")}},
                        std::vector<double>(static_cast<std::size_t>(i), 1.0)};
    CHECK(courses.put(key, course));
    expected += key.size() + nivo::ValueCodec<Course>::encoded_size(course);
  }
  const nivo::MemoryStats memory = courses.cache().memory_stats(courses.pool()).value();
  CHECK(courses.stats().value().indexed_bytes == expected);
  CHECK(memory.indexed_kv_bytes == expected && memory.kv_bytes == expected);
  CHECK(memory.retained_kv_bytes == 0);
  CHECK(memory.reserved_bytes == memory.kv_bytes + memory.entry_metadata_bytes +
        memory.allocator_metadata_bytes + memory.internal_fragmentation_bytes + memory.unused_bytes);

  // A decoded handle does not retain memory; erased entries are released.
  nivo::TypedReadHandle<Course> held = courses.get("course-99");
  CHECK(courses.erase("course-99"));
  CHECK(courses.cache().memory_stats(courses.pool()).value().retained_kv_bytes == 0);
  CHECK(held->id == 99);
  CHECK(courses.cache().clear(courses.pool()) == Status::ok);
  CHECK(courses.stats().value().indexed_items == 0);
  CHECK(courses.cache().memory_stats(courses.pool()).value().kv_bytes == 0);
}

void pinned_entries_and_capacity() {
  // Two zero-copy pins fill the pool: the third write is dropped, not evicting.
  const std::size_t student_bytes = sizeof(int) + sizeof(Student);
  nivo::NivoCache<int, Student> students(small_config(student_bytes * 2));
  CHECK(students.put(1, Student{1, "one", 1.0}) && students.put(2, Student{2, "two", 2.0}));
  nivo::TypedReadHandle<Student> first = students.get(1);
  nivo::TypedReadHandle<Student> second = students.get(2);
  CHECK(!students.put(3, Student{3, "three", 3.0}));
  CHECK(students.write(3, Student{}).status() == Status::resource_exhausted);
  CHECK(students.contains(1) && students.contains(2) && !students.contains(3));
  CHECK(students.stats().value().counters.evictions == 0);
  first.reset();
  second.reset();
  CHECK(students.put(3, Student{3, "three", 3.0}) && students.contains(3));
  CHECK(students.stats().value().counters.evictions == 1);

  // Decoded handles do not pin, so the same pressure evicts normally.
  const std::size_t string_bytes = sizeof(int) + 8;
  nivo::NivoCache<int, std::string> strings(small_config(string_bytes * 2));
  CHECK(strings.put(1, "value-01") && strings.put(2, "value-02"));
  nivo::TypedReadHandle<std::string> one = strings.get(1);
  nivo::TypedReadHandle<std::string> two = strings.get(2);
  CHECK(strings.put(3, "value-03") && strings.contains(3));
  CHECK(strings.stats().value().counters.evictions == 1);
  CHECK(*one == "value-01" && *two == "value-02");
}

void pool_maintenance_with_typed_values() {
  // Shrinking a pool evicts the least recently used typed values.
  const std::size_t student_bytes = sizeof(int) + sizeof(Student);
  nivo::NivoCache<int, Student> students(small_config(student_bytes * 100));
  for (int i = 0; i < 100; ++i) CHECK(students.put(i, Student{i, "s", 0.0}));
  CHECK(students.cache().resize_pool(students.pool(), student_bytes * 10) == Status::ok);
  CHECK(students.stats().value().indexed_items == 10);
  for (int i = 90; i < 100; ++i) CHECK(students.contains(i));
  CHECK(!students.contains(0) && !students.contains(89));
  CHECK(students.put(1000, Student{1000, "new", 0.0}) && !students.contains(90));

  // clear() unlinks everything; a pinned handle keeps its bytes alive.
  nivo::TypedReadHandle<Student> survivor = students.get(99);
  CHECK(students.cache().clear(students.pool()) == Status::ok);
  CHECK(!students.contains(99) && survivor->age == 99);
  CHECK(students.stats().value().allocated_bytes == student_bytes);
  survivor.reset();
  CHECK(students.stats().value().allocated_bytes == 0);

  // prune_expired() removes only expired typed entries.
  std::shared_ptr<ManualClock> clock = make_clock();
  nivo::NivoCache<int, std::vector<int>> vectors(small_config(1024 * 1024, clock));
  for (int i = 0; i < 100; ++i) {
    const nivo::Duration ttl = i % 2 == 0 ? nivo::Duration(10ms) : nivo::Duration(0);
    CHECK(vectors.put(i, std::vector<int>(static_cast<std::size_t>(i), i), ttl));
  }
  clock->advance(10ms);
  CHECK(vectors.cache().prune_expired(vectors.pool()).value() == 50);
  CHECK(vectors.stats().value().indexed_items == 50);
  for (int i = 0; i < 100; ++i) CHECK(vectors.contains(i) == (i % 2 == 1));
  CHECK(*vectors.get(99) == std::vector<int>(99, 99));
}

void serialize_field_kinds() {
  static_assert(nivo::Serializable<Record> && !nivo::TypedCache<int, Record>::zero_copy);
  static_assert(nivo::Serializable<Point> && nivo::TypedCache<int, Point>::zero_copy);
  static_assert(nivo::Serializable<Tagged> && !nivo::TypedCache<int, Tagged>::zero_copy);

  Record record{};
  record.student = Student{20, "Alice", 95.5};
  std::memcpy(record.code, "CS-101\0", 8);
  record.triple = {1, -2, 3};
  record.grade = Grade::c;
  record.matrix = {{1, 2, 3}, {}, {4}};
  record.notes = {"", std::string("a\0b", 3), std::string(1000, 'n')};
  record.payload = std::string(100000, 'p');

  nivo::NivoCache<int, Record> records(4 * 1024 * 1024);
  CHECK(records.put(1, record) && records.put(2, Record{}));
  nivo::TypedReadHandle<Record> found = records.get(1);
  CHECK(found && found->student.age == 20 && std::string_view(found->student.name) == "Alice");
  CHECK(found->student.score == 95.5 && std::memcmp(found->code, "CS-101\0", 8) == 0);
  CHECK(found->triple == record.triple && found->grade == Grade::c);
  CHECK(found->matrix == record.matrix && found->notes == record.notes);
  CHECK(found->payload == record.payload);
  nivo::TypedReadHandle<Record> empty = records.get(2);
  CHECK(empty && empty->matrix.empty() && empty->notes.empty() && empty->payload.empty());

  // Round trips are deterministic: re-encoding a decoded value is byte-identical.
  using Codec = nivo::ValueCodec<Record>;
  std::vector<std::byte> encoded(Codec::encoded_size(record));
  Codec::encode(record, encoded);
  std::vector<std::byte> reencoded(Codec::encoded_size(*found));
  Codec::encode(*found, reencoded);
  CHECK(encoded == reencoded);
  // Every strict prefix is rejected.
  for (std::size_t length = 0; length < 200; ++length) {
    CHECK(thrown_status([&] { (void)Codec::decode(std::span<const std::byte>(encoded).first(length)); }) ==
          Status::invalid_argument);
  }

  nivo::NivoCache<int, Point> points(1024);
  CHECK(points.put(1, Point{3, 4}) && points.get(1)->y == 4);
  CHECK(points.stats().value().indexed_bytes == sizeof(int) + sizeof(Point));

  nivo::NivoCache<int, Tagged> tagged(1024);
  CHECK(tagged.put(1, Tagged{"abc"}) && tagged.get(1)->text == "abc");
  CHECK(tagged.stats().value().indexed_bytes == sizeof(int) + 3);  // no length prefix
}

void key_type_coverage() {
  nivo::NivoCache<bool, int> flags(1024);
  CHECK(flags.put(true, 1) && flags.put(false, 0));
  CHECK(*flags.get(true) == 1 && *flags.get(false) == 0);

  nivo::NivoCache<std::int8_t, int> small_ints(1024);
  CHECK(small_ints.put(-1, -1) && small_ints.put(127, 127) && small_ints.put(-128, -128));
  CHECK(*small_ints.get(-1) == -1 && *small_ints.get(-128) == -128 && !small_ints.get(0));

  nivo::NivoCache<Grade, std::string> grades(1024);
  CHECK(grades.put(Grade::a, "top") && grades.put(Grade::c, "pass"));
  CHECK(*grades.get(Grade::a) == "top" && !grades.get(Grade::b));

  using Uuid = std::array<std::uint8_t, 16>;
  nivo::NivoCache<Uuid, int> uuids(1024 * 1024);
  for (int i = 0; i < 256; ++i) {
    Uuid uuid{};
    uuid[static_cast<std::size_t>(i % 16)] = static_cast<std::uint8_t>(i);
    CHECK(uuids.put(uuid, i));
  }
  Uuid probe{};
  probe[5] = 5;
  CHECK(*uuids.get(probe) == 5);

  // Many keys, including the extremes of the key type.
  nivo::NivoCache<std::uint64_t, std::uint64_t> numbers(64 * 1024 * 1024);
  constexpr std::uint64_t count = 100000;
  for (std::uint64_t i = 0; i < count; ++i) CHECK(numbers.put(i * 2654435761u, i));
  CHECK(numbers.put(std::numeric_limits<std::uint64_t>::max(), 1));
  CHECK(numbers.stats().value().indexed_items == count + 1);
  for (std::uint64_t i = 0; i < count; ++i) CHECK(*numbers.get(i * 2654435761u) == i);
  CHECK(*numbers.get(std::numeric_limits<std::uint64_t>::max()) == 1 && !numbers.get(1));
}

// Writers, readers, erasers and a compactor run together; every value a
// reader sees must be one complete write for that key.
void concurrent_zero_copy_with_compaction() {
  nivo::NivoCache<int, Blob> blobs(16 * 1024 * 1024);
  constexpr int keys = 64;
  constexpr int rounds = 1500;
  std::atomic<bool> failed{false};
  std::atomic<bool> stop{false};
  std::atomic<int> compactions{0};
  std::vector<std::thread> threads;
  for (int writer = 0; writer < 2; ++writer) {
    threads.emplace_back([&blobs, writer] {
      std::unique_ptr<Blob> blob = std::make_unique<Blob>();
      for (int i = 0; i < rounds; ++i) {
        const int key = (i * 7 + writer) % keys;
        fill_blob(*blob, key, i);
        blobs.put(key, *blob);
        if (i % 5 == 0) blobs.erase((key + 3) % keys);
      }
    });
  }
  for (int reader = 0; reader < 3; ++reader) {
    threads.emplace_back([&blobs, &failed, reader] {
      for (int i = 0; i < rounds * 2; ++i) {
        const int key = (i + reader * 11) % keys;
        nivo::TypedReadHandle<Blob> found = blobs.get(key);
        if (found && (!blob_consistent(*found, key) || !max_aligned(found.get()))) failed = true;
      }
    });
  }
  std::thread compactor([&blobs, &stop, &failed, &compactions] {
    while (!stop.load()) {
      const nivo::Result<nivo::CompactionStats> compacted = blobs.cache().compact(blobs.pool());
      if (!compacted && compacted.status() != Status::resource_exhausted) failed = true;
      compactions.fetch_add(1);
      std::this_thread::sleep_for(1ms);
    }
  });
  for (std::thread& thread : threads) thread.join();
  stop = true;
  compactor.join();
  CHECK(!failed && compactions.load() > 0);
  for (int key = 0; key < keys; ++key) {
    nivo::TypedReadHandle<Blob> found = blobs.get(key);
    CHECK(!found || blob_consistent(*found, key));
  }
}

std::string tagged_value(int key, int round) {
  const std::size_t length = 100 + static_cast<std::size_t>(round % 50) * 300;
  std::string value = std::to_string(key) + ":" + std::to_string(round) + "|";
  value.append(length, static_cast<char>('a' + key % 26));
  return value;
}

bool tagged_value_valid(const std::string& value, int key) {
  const std::string prefix = std::to_string(key) + ":";
  if (value.compare(0, prefix.size(), prefix) != 0) return false;
  const std::size_t bar = value.find('|');
  if (bar == std::string::npos) return false;
  const int round = std::stoi(value.substr(prefix.size(), bar - prefix.size()));
  return value == tagged_value(key, round);
}

void concurrent_codec_values_with_compaction() {
  nivo::NivoCache<int, std::string> strings(8 * 1024 * 1024);
  constexpr int keys = 128;
  constexpr int rounds = 2000;
  std::atomic<bool> failed{false};
  std::atomic<bool> stop{false};
  std::atomic<int> compactions{0};
  std::vector<std::thread> threads;
  for (int writer = 0; writer < 2; ++writer) {
    threads.emplace_back([&strings, writer] {
      for (int i = 0; i < rounds; ++i) {
        const int key = (i * 13 + writer) % keys;
        strings.put(key, tagged_value(key, i));
        if (i % 7 == 0) strings.erase((key + 5) % keys);
      }
    });
  }
  for (int reader = 0; reader < 3; ++reader) {
    threads.emplace_back([&strings, &failed, reader] {
      for (int i = 0; i < rounds * 2; ++i) {
        const int key = (i + reader * 17) % keys;
        nivo::TypedReadHandle<std::string> found = strings.get(key);
        if (found && !tagged_value_valid(*found, key)) failed = true;
      }
    });
  }
  std::thread compactor([&strings, &stop, &failed, &compactions] {
    while (!stop.load()) {
      const nivo::Result<nivo::CompactionStats> compacted = strings.cache().compact(strings.pool());
      if (!compacted && compacted.status() != Status::resource_exhausted) failed = true;
      compactions.fetch_add(1);
      std::this_thread::sleep_for(1ms);
    }
  });
  for (std::thread& thread : threads) thread.join();
  stop = true;
  compactor.join();
  CHECK(!failed && compactions.load() > 0);
  for (int key = 0; key < keys; ++key) {
    nivo::TypedReadHandle<std::string> found = strings.get(key);
    CHECK(!found || tagged_value_valid(*found, key));
  }
}
}  // namespace

int main() {
  const std::pair<const char*, std::function<void()>> cases[] = {
    {"student_round_trip", student_round_trip},
    {"default_and_scalar_types", default_and_scalar_types},
    {"key_encodings", key_encodings},
    {"codec_values", codec_values},
    {"serializable_values", serializable_values},
    {"write_modes_and_versions", write_modes_and_versions},
    {"ttl_and_eviction", ttl_and_eviction},
    {"construction_and_borrowing", construction_and_borrowing},
    {"type_mismatch_is_detected", type_mismatch_is_detected},
    {"concurrent_readers_and_writers", concurrent_readers_and_writers},
    {"zero_copy_values_survive_compaction", zero_copy_values_survive_compaction},
    {"pinned_handles_during_compaction", pinned_handles_during_compaction},
    {"codec_values_survive_compaction", codec_values_survive_compaction},
    {"values_across_size_classes", values_across_size_classes},
    {"growing_and_shrinking_overwrites", growing_and_shrinking_overwrites},
    {"memory_accounting_for_typed_values", memory_accounting_for_typed_values},
    {"pinned_entries_and_capacity", pinned_entries_and_capacity},
    {"pool_maintenance_with_typed_values", pool_maintenance_with_typed_values},
    {"serialize_field_kinds", serialize_field_kinds},
    {"key_type_coverage", key_type_coverage},
    {"concurrent_zero_copy_with_compaction", concurrent_zero_copy_with_compaction},
    {"concurrent_codec_values_with_compaction", concurrent_codec_values_with_compaction},
  };
  unsigned failures = 0;
  for (const std::pair<const char*, std::function<void()>>& test_case : cases) {
    const char* name = test_case.first;
    const std::function<void()>& test = test_case.second;
    try { test(); std::cout << "PASS " << name << '\n'; }
    catch (const std::exception& e) { ++failures; std::cerr << "FAIL " << name << ": " << e.what() << '\n'; }
  }
  return failures == 0 ? 0 : 1;
}
