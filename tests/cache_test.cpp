#include <nivo/nivo.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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
std::span<const std::byte> bytes(std::string_view s) {
  return std::as_bytes(std::span(s.data(), s.size()));
}
std::string text(const nivo::ReadHandle& handle) {
  std::span<const std::byte> b = handle.bytes();
  if (b.empty()) return {};
  return {reinterpret_cast<const char*>(b.data()), b.size()};
}
struct Fixture {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache;
  nivo::PoolId pool;
  explicit Fixture(std::size_t capacity = 1024) {
    nivo::Result<std::unique_ptr<nivo::Cache>> result =
        nivo::Cache::create({.capacity_bytes = capacity * 2, .clock = clock});
    CHECK(result);
    cache = std::move(result).value();
    nivo::Result<nivo::PoolId> added =
        cache->add_pool({.name = "default", .capacity_bytes = capacity});
    CHECK(added);
    pool = added.value();
  }
  void put(std::string_view key, std::string_view value, nivo::WriteOptions options = {}) {
    CHECK(cache->put(pool, key, bytes(value), options));
  }
};

void configuration() {
  CHECK(nivo::Cache::create({.capacity_bytes = 0}).status() == Status::invalid_argument);
  CHECK(nivo::Cache::create({.max_key_bytes = 0}).status() == Status::invalid_argument);
  CHECK(nivo::Cache::create({.storage = nivo::StorageMode::hybrid}).status() == Status::unsupported);
  CHECK(!nivo::Cache::capabilities().persistence);
  Fixture f(10);
  CHECK(f.cache->add_pool({"", 1}).status() == Status::invalid_argument);
  CHECK(f.cache->add_pool({"default", 1}).status() == Status::already_exists);
  CHECK(f.cache->add_pool({"large", 11}).status() == Status::resource_exhausted);
  CHECK(f.cache->add_pool({"second", 10}));
  CHECK(f.cache->pools().value().size() == 2);
  CHECK(f.cache->pool_stats(999).status() == Status::pool_not_found);
}
void unpublished_and_aborted() {
  Fixture f;
  {
    nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "key", 4);
    CHECK(draft && draft.value().charge() == 7);
    CHECK(std::all_of(draft.value().bytes().begin(), draft.value().bytes().end(),
                      [](std::byte b) { return b == std::byte{0}; }));
    CHECK(f.cache->find(f.pool, "key").status() == Status::not_found);
    CHECK(f.cache->stats().allocated_bytes == 7);
  }
  CHECK(f.cache->stats().allocated_bytes == 0);
  CHECK(f.cache->stats().indexed_items == 0);
}
void publish_and_snapshot() {
  Fixture f;
  nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "a", 3);
  std::copy(bytes("one").begin(), bytes("one").end(), draft.value().bytes().begin());
  nivo::Result<nivo::ReadHandle> first = f.cache->publish(std::move(draft).value());
  CHECK(first && !draft.value());
  CHECK(first.value().pool() == f.pool);
  CHECK(first.value().version() != 0);
  f.put("a", "two");
  nivo::Result<nivo::ReadHandle> second = f.cache->find(f.pool, "a");
  CHECK(text(first.value()) == "one");
  CHECK(text(second.value()) == "two");
  CHECK(first.value().version() != second.value().version());
  CHECK(f.cache->stats().allocated_bytes == 8);
  CHECK(f.cache->erase(f.pool, "a") == Status::ok);
  CHECK(f.cache->stats().indexed_bytes == 0);
  CHECK(text(second.value()) == "two");
  first.value().reset(); second.value().reset();
  CHECK(f.cache->stats().allocated_bytes == 0);
}
void write_conditions_and_retry() {
  Fixture f;
  nivo::Result<nivo::ReadHandle> original = f.cache->put(f.pool, "a", bytes("one"));
  const nivo::Version version = original.value().version();
  nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "a", 0);
  CHECK(f.cache->publish(std::move(draft).value(), {.mode = nivo::WriteMode::insert_only}).status()
        == Status::already_exists);
  CHECK(draft.value());
  CHECK(f.cache->publish(std::move(draft).value(),
        {.mode = nivo::WriteMode::compare_exchange, .expected_version = version + 1}).status()
        == Status::version_mismatch);
  nivo::Result<nivo::ReadHandle> replaced = f.cache->publish(std::move(draft).value(),
      {.mode = nivo::WriteMode::compare_exchange, .expected_version = version});
  CHECK(replaced && !draft.value());
  CHECK(f.cache->erase(f.pool, "a", version) == Status::version_mismatch);
  CHECK(f.cache->erase(f.pool, "a", replaced.value().version()) == Status::ok);
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = nivo::WriteMode::replace_only}).status() == Status::not_found);
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = nivo::WriteMode::compare_exchange,
        .expected_version = version}).status() == Status::not_found);
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = nivo::WriteMode::compare_exchange}).status() == Status::invalid_argument);
  CHECK(f.cache->put(f.pool, "a", {}, {.expected_version = version}).status() == Status::invalid_argument);
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = static_cast<nivo::WriteMode>(99)}).status() == Status::invalid_argument);
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = nivo::WriteMode::insert_only}));
  CHECK(f.cache->put(f.pool, "a", {}, {.mode = nivo::WriteMode::replace_only}));
}
void ttl_boundary_and_publish_time() {
  Fixture f;
  nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "a", 1);
  f.clock->advance(100ms);
  nivo::Result<nivo::ReadHandle> snapshot =
      f.cache->publish(std::move(draft).value(), {.ttl = 10ms});
  f.clock->advance(9ms);
  CHECK(f.cache->find(f.pool, "a"));
  f.clock->advance(1ms);
  CHECK(f.cache->find(f.pool, "a").status() == Status::not_found);
  CHECK(snapshot.value().bytes().size() == 1);
  CHECK(f.cache->stats().counters.expirations == 1);
  CHECK(f.cache->stats().allocated_bytes == 2);
  f.put("a", "new", {.mode = nivo::WriteMode::insert_only});
  f.clock->advance(100000ms);
  CHECK(f.cache->find(f.pool, "a"));
  CHECK(f.cache->put(f.pool, "b", {}, {.ttl = -1ms}).status() == Status::invalid_argument);
  CHECK(f.cache->put(f.pool, "b", {}, {.ttl = nivo::Duration::max()}).status() == Status::invalid_argument);
}
void expiry_maintenance() {
  Fixture f;
  f.put("a", "a", {.ttl = 1ms});
  f.put("b", "b", {.ttl = 2ms});
  f.put("c", "c");
  f.clock->advance(2ms);
  CHECK(f.cache->stats().indexed_items == 3);
  CHECK(f.cache->prune_expired(f.pool).value() == 2);
  CHECK(f.cache->stats().indexed_items == 1);
  CHECK(f.cache->prune_expired(f.pool).value() == 0);
  f.put("a", "a", {.ttl = 1ms});
  f.clock->advance(1ms);
  CHECK(f.cache->erase(f.pool, "a") == Status::not_found);
  f.put("a", "a", {.ttl = 1ms});
  f.clock->advance(1ms);
  CHECK(f.cache->peek(f.pool, "a").status() == Status::not_found);
}
void allocation_prunes_only_under_pressure() {
  Fixture roomy(64);
  roomy.put("expired", "value", {.ttl = 1ms});
  roomy.clock->advance(1ms);
  nivo::Result<nivo::WriteHandle> roomy_draft =
      roomy.cache->allocate(roomy.pool, "new", 1);
  CHECK(roomy_draft);
  CHECK(roomy.cache->pool_stats(roomy.pool).value().indexed_items == 1);
  CHECK(roomy.cache->pool_stats(roomy.pool).value().counters.expirations == 0);
  CHECK(roomy.cache->prune_expired(roomy.pool).value() == 1);

  Fixture pressured(4);
  pressured.put("a", "1", {.ttl = 1ms});
  pressured.put("b", "1");
  pressured.clock->advance(1ms);
  nivo::Result<nivo::WriteHandle> pressured_draft =
      pressured.cache->allocate(pressured.pool, "c", 1);
  CHECK(pressured_draft);
  nivo::PoolStats stats = pressured.cache->pool_stats(pressured.pool).value();
  CHECK(stats.indexed_items == 1);
  CHECK(stats.counters.expirations == 1);
  CHECK(stats.counters.evictions == 0);
  CHECK(pressured.cache->peek(pressured.pool, "b"));
}
void lru_and_peek() {
  Fixture f(6);
  f.put("a", "a"); f.put("b", "b"); f.put("c", "c");
  CHECK(f.cache->find(f.pool, "a"));
  CHECK(f.cache->peek(f.pool, "b"));
  f.put("d", "d");
  CHECK(f.cache->peek(f.pool, "b").status() == Status::not_found);
  CHECK(f.cache->peek(f.pool, "a"));
  nivo::CacheStats stats = f.cache->stats();
  CHECK(stats.counters.hits == 1 && stats.counters.misses == 0);
  CHECK(stats.counters.evictions == 1);
}
void fifo_policy() {
  Fixture f(6);
  nivo::PoolId pool = f.cache->add_pool({"fifo", 6, nivo::EvictionPolicy::fifo}).value();
  CHECK(f.cache->put(pool, "a", bytes("a")));
  CHECK(f.cache->put(pool, "b", bytes("b")));
  CHECK(f.cache->put(pool, "c", bytes("c")));
  CHECK(f.cache->find(pool, "a"));
  CHECK(f.cache->put(pool, "d", bytes("d")));
  CHECK(f.cache->peek(pool, "a").status() == Status::not_found);
}
void pins_and_capacity() {
  Fixture f(4);
  nivo::Result<nivo::ReadHandle> pinned = f.cache->put(f.pool, "a", bytes("123"));
  CHECK(pinned);
  CHECK(f.cache->allocate(f.pool, "b", 0).status() == Status::resource_exhausted);
  CHECK(f.cache->erase(f.pool, "a") == Status::ok);
  CHECK(f.cache->allocate(f.pool, "b", 0).status() == Status::resource_exhausted);
  pinned.value().reset();
  nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "b", 3);
  CHECK(draft);
  CHECK(f.cache->allocate(f.pool, "c", 0).status() == Status::resource_exhausted);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocated_bytes == 4);
  draft.value().reset();
  CHECK(f.cache->stats().allocated_bytes == 0);
}
void update_protects_old_key() {
  Fixture f(4);
  f.put("a", "123");
  CHECK(f.cache->put(f.pool, "a", {}).status() == Status::resource_exhausted);
  CHECK(text(f.cache->find(f.pool, "a").value()) == "123");
  CHECK(f.cache->stats().counters.evictions == 0);
}
void eviction_skips_pins() {
  Fixture f(6);
  nivo::Result<nivo::ReadHandle> pinned = f.cache->put(f.pool, "a", bytes("a"));
  f.put("b", "b"); f.put("c", "c");
  f.put("d", "d");
  CHECK(f.cache->peek(f.pool, "a"));
  CHECK(f.cache->peek(f.pool, "b").status() == Status::not_found);
  CHECK(f.cache->stats().allocated_bytes == 6);
}
void pool_isolation_resize() {
  Fixture f(8);
  nivo::PoolId other = f.cache->add_pool({"other", 8}).value();
  f.put("a", "one");
  CHECK(f.cache->put(other, "a", bytes("two")));
  CHECK(text(f.cache->find(f.pool, "a").value()) == "one");
  CHECK(text(f.cache->find(other, "a").value()) == "two");
  CHECK(f.cache->resize_pool(f.pool, 9) == Status::resource_exhausted);
  nivo::Result<nivo::ReadHandle> pin = f.cache->find(f.pool, "a");
  CHECK(f.cache->resize_pool(f.pool, 2) == Status::resource_exhausted);
  CHECK(f.cache->pool_stats(f.pool).value().capacity_bytes == 8);
  pin.value().reset();
  CHECK(f.cache->resize_pool(f.pool, 2) == Status::ok);
  CHECK(f.cache->peek(f.pool, "a").status() == Status::not_found);
  CHECK(f.cache->resize_pool(other, 14) == Status::ok);
  CHECK(f.cache->stats().assigned_bytes == 16);
  CHECK(f.cache->resize_pool(other, 0) == Status::invalid_argument);
}
void foreign_and_empty_handles() {
  Fixture a, b;
  nivo::Result<nivo::WriteHandle> draft = a.cache->allocate(a.pool, "k", 1);
  CHECK(b.cache->publish(std::move(draft).value()).status() == Status::invalid_argument);
  CHECK(draft.value());
  CHECK(a.cache->publish(std::move(draft).value()));
  CHECK(a.cache->publish(std::move(draft).value()).status() == Status::invalid_argument);
  nivo::ReadHandle read;
  nivo::WriteHandle write;
  CHECK(!read && read.bytes().empty() && read.version() == 0 && read.pool() == 0);
  CHECK(!write && write.bytes().empty() && write.key().empty() && write.charge() == 0);
}
void binary_keys_and_validation() {
  Fixture f;
  const std::string key("a\0b", 3), value("x\0y", 3);
  f.put(key, value);
  CHECK(text(f.cache->find(f.pool, key).value()) == value);
  CHECK(f.cache->find(f.pool, "a").status() == Status::not_found);
  f.put("empty-value", "");
  CHECK(f.cache->find(f.pool, "empty-value").value().bytes().empty());
  CHECK(f.cache->allocate(f.pool, "", 1).status() == Status::invalid_argument);
  CHECK(f.cache->find(f.pool, "").status() == Status::invalid_argument);
  CHECK(f.cache->erase(f.pool, "") == Status::invalid_argument);
  CHECK(f.cache->allocate(f.pool, std::string(1025, 'a'), 0).status() == Status::invalid_argument);
  CHECK(f.cache->allocate(f.pool, "k", std::numeric_limits<std::size_t>::max()).status() == Status::invalid_argument);
  CHECK(f.cache->allocate(999, "a", 1).status() == Status::pool_not_found);
  CHECK(f.cache->find(999, "a").status() == Status::pool_not_found);
}
void close_and_lifetime() {
  Fixture f;
  nivo::Result<nivo::ReadHandle> read = f.cache->put(f.pool, "a", bytes("alive"));
  nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, "b", 2);
  f.cache->close(); f.cache->close();
  CHECK(f.cache->stats().closed && f.cache->stats().indexed_items == 0);
  CHECK(f.cache->stats().allocated_bytes == 9);
  CHECK(f.cache->find(f.pool, "a").status() == Status::closed);
  CHECK(f.cache->peek(f.pool, "a").status() == Status::closed);
  CHECK(f.cache->allocate(f.pool, "a", 0).status() == Status::closed);
  CHECK(f.cache->publish(std::move(draft).value()).status() == Status::closed);
  CHECK(draft.value());
  CHECK(f.cache->add_pool({"closed", 1}).status() == Status::closed);
  CHECK(f.cache->clear(f.pool) == Status::closed);
  CHECK(f.cache->erase(f.pool, "a") == Status::closed);
  CHECK(f.cache->resize_pool(f.pool, 1) == Status::closed);
  CHECK(f.cache->prune_expired(f.pool).status() == Status::closed);
  CHECK(f.cache->pools() && f.cache->pool_stats(f.pool));
  f.cache.reset();
  CHECK(text(read.value()) == "alive");
  draft.value().bytes()[0] = std::byte{42};
}
void destructor_without_close() {
  nivo::ReadHandle survivor;
  nivo::WriteHandle draft;
  {
    Fixture f;
    survivor = std::move(f.cache->put(f.pool, "key", bytes("survives"))).value();
    draft = std::move(f.cache->allocate(f.pool, "draft", 1)).value();
  }
  CHECK(text(survivor) == "survives");
  draft.bytes()[0] = std::byte{1};
}
void result_contract() {
  static_assert(!std::is_copy_constructible_v<nivo::WriteHandle>);
  static_assert(std::is_copy_constructible_v<nivo::ReadHandle>);
  nivo::Result<int> error(Status::not_found);
  CHECK(!error && error.status() == Status::not_found);
  bool threw = false;
  try { (void)error.value(); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { nivo::Result<int> invalid(Status::ok); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}
void concurrent_cas() {
  Fixture f;
  nivo::Result<nivo::ReadHandle> initial = f.cache->put(f.pool, "a", bytes("zero"));
  const nivo::Version version = initial.value().version();
  std::barrier gate(8);
  std::atomic<int> winners{0}, conflicts{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] {
      gate.arrive_and_wait();
      nivo::Result<nivo::ReadHandle> result = f.cache->put(f.pool, "a", bytes(std::to_string(i)),
          {.mode = nivo::WriteMode::compare_exchange, .expected_version = version});
      if (result) ++winners;
      else if (result.status() == Status::version_mismatch) ++conflicts;
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(winners == 1 && conflicts == 7);
  CHECK(text(initial.value()) == "zero");
}
void concurrent_operations() {
  Fixture f(1024 * 1024);
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int i = 0; i < 6; ++i) {
    threads.emplace_back([&, i] {
      try {
        const std::string key = std::to_string(i);
        for (int j = 0; j < 500; ++j) {
          const std::string value = std::to_string(j);
          nivo::Result<nivo::ReadHandle> put = f.cache->put(f.pool, key, bytes(value));
          CHECK(put);
          nivo::Result<nivo::ReadHandle> found = f.cache->find(f.pool, key);
          CHECK(found && text(found.value()) == value);
          CHECK(f.cache->erase(f.pool, key, found.value().version()) == Status::ok);
          CHECK(text(put.value()) == value);
        }
      } catch (...) { failed = true; }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed);
  CHECK(f.cache->stats().indexed_items == 0 && f.cache->stats().allocated_bytes == 0);
  CHECK(f.cache->stats().counters.inserts == 3000);
}
void concurrent_close() {
  Fixture f;
  nivo::Result<nivo::ReadHandle> pinned = f.cache->put(f.pool, "a", bytes("stable"));
  std::barrier gate(4);
  std::atomic<bool> failed{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 3; ++i) {
    readers.emplace_back([&] {
      gate.arrive_and_wait();
      for (int j = 0; j < 500; ++j) {
        nivo::Result<nivo::ReadHandle> result = f.cache->find(f.pool, "a");
        if (result ? text(result.value()) != "stable" : result.status() != Status::closed) failed = true;
      }
    });
  }
  gate.arrive_and_wait();
  f.cache->close();
  for (std::thread& reader : readers) reader.join();
  CHECK(!failed && text(pinned.value()) == "stable");
}
}  // namespace

int main() {
  const std::pair<const char*, std::function<void()>> cases[] = {
    {"configuration", configuration}, {"unpublished_and_aborted", unpublished_and_aborted},
    {"publish_and_snapshot", publish_and_snapshot}, {"write_conditions_and_retry", write_conditions_and_retry},
    {"ttl_boundary_and_publish_time", ttl_boundary_and_publish_time}, {"expiry_maintenance", expiry_maintenance},
    {"allocation_prunes_only_under_pressure", allocation_prunes_only_under_pressure},
    {"lru_and_peek", lru_and_peek}, {"fifo_policy", fifo_policy},
    {"pins_and_capacity", pins_and_capacity}, {"update_protects_old_key", update_protects_old_key},
    {"eviction_skips_pins", eviction_skips_pins}, {"pool_isolation_resize", pool_isolation_resize},
    {"foreign_and_empty_handles", foreign_and_empty_handles}, {"binary_keys_and_validation", binary_keys_and_validation},
    {"close_and_lifetime", close_and_lifetime}, {"destructor_without_close", destructor_without_close},
    {"result_contract", result_contract}, {"concurrent_cas", concurrent_cas},
    {"concurrent_operations", concurrent_operations}, {"concurrent_close", concurrent_close}
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
