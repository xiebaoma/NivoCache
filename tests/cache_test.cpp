#include <nivo/nivo.h>
#include "../src/memory_allocator.h"
#include "../src/process_memory.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
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

void allocator_size_boundaries() {
  nivo::detail::MemoryAllocator allocator(1024 * 1024);
  for (std::size_t size = 1; size <= 20000; ++size) {
    const nivo::detail::MemoryAllocator::Allocation block = allocator.allocate(size);
    CHECK(reinterpret_cast<std::uintptr_t>(block.data) % alignof(std::max_align_t) == 0);
    std::byte* data = static_cast<std::byte*>(block.data);
    std::fill_n(data, size, std::byte{0x5a});
    const nivo::AllocatorStats stats = allocator.stats();
    CHECK(stats.live_allocations == 1 && stats.requested_bytes == size);
    CHECK(stats.active_bytes >= size && stats.reserved_bytes >= stats.active_bytes);
    if (size > 128) CHECK(stats.active_bytes <= size + (size + 7) / 8);
    allocator.deallocate(block);
    CHECK(allocator.stats().live_allocations == 0);
    CHECK(allocator.stats().active_bytes == 0);
  }
  allocator.trim();
  CHECK(allocator.stats().reserved_bytes == 0);
}

void allocator_reuse_and_reclassification() {
  nivo::detail::MemoryAllocator allocator(65536);
  const nivo::detail::MemoryAllocator::Allocation first = allocator.allocate(256);
  const nivo::detail::MemoryAllocator::Allocation second = allocator.allocate(256);
  allocator.deallocate(first);
  const nivo::detail::MemoryAllocator::Allocation reused = allocator.allocate(255);
  CHECK(reused.data == first.data);
  CHECK(allocator.stats().system_allocations == 1);
  allocator.deallocate(second);
  allocator.deallocate(reused);
  CHECK(allocator.stats().slabs == 1);
  const nivo::detail::MemoryAllocator::Allocation different = allocator.allocate(1024);
  CHECK(allocator.stats().system_allocations == 1);
  CHECK(allocator.stats().active_bytes == 1024);
  allocator.deallocate(different);
  CHECK(allocator.stats().reused_allocations >= 2);
  allocator.trim();
  CHECK(allocator.stats().slabs == 0);
}

void allocator_mixed_slabs() {
  nivo::detail::MemoryAllocator allocator(65536);
  std::vector<nivo::detail::MemoryAllocator::Allocation> blocks;
  std::mt19937 random(42);
  std::size_t requested = 0;
  for (std::size_t i = 0; i < 4000; ++i) {
    const std::size_t size = 1 + random() % 20000;
    blocks.push_back(allocator.allocate(size));
    requested += size;
    std::fill_n(static_cast<std::byte*>(blocks.back().data), size,
                std::byte{static_cast<unsigned char>(size % 251)});
  }
  CHECK(allocator.stats().requested_bytes == requested);
  CHECK(allocator.stats().slabs > 1);
  std::shuffle(blocks.begin(), blocks.end(), random);
  for (const nivo::detail::MemoryAllocator::Allocation& block : blocks) {
    const std::byte* data = static_cast<const std::byte*>(block.data);
    CHECK(std::all_of(data, data + block.requested, [&block](std::byte value) {
      return value == std::byte{static_cast<unsigned char>(block.requested % 251)};
    }));
    allocator.deallocate(block);
  }
  CHECK(allocator.stats().requested_bytes == 0);
  CHECK(allocator.stats().active_bytes == 0);
  CHECK(allocator.stats().large_allocations == 0);
  CHECK(allocator.stats().slabs == 1);
  CHECK(allocator.stats().reserved_bytes > 65536);
  CHECK(allocator.stats().regions == 1);
  allocator.trim();
  CHECK(allocator.stats().reserved_bytes == 0);
}

void allocator_full_slab_reuse() {
  nivo::detail::MemoryAllocator allocator(4096);
  std::vector<nivo::detail::MemoryAllocator::Allocation> blocks;
  for (std::size_t i = 0; i < 100; ++i) blocks.push_back(allocator.allocate(256));
  const std::uint64_t system_allocations = allocator.stats().system_allocations;
  // Reopen full slabs, then fill them again without allocating another slab.
  for (std::size_t i = 0; i < blocks.size(); i += 2) allocator.deallocate(blocks[i]);
  for (std::size_t i = 0; i < blocks.size(); i += 2) blocks[i] = allocator.allocate(256);
  CHECK(allocator.stats().system_allocations == system_allocations);
  for (const nivo::detail::MemoryAllocator::Allocation& block : blocks) allocator.deallocate(block);
  CHECK(allocator.stats().slabs == 1);
}

void allocator_large_and_failure() {
  nivo::detail::MemoryAllocator allocator(1);
  CHECK(allocator.stats().reserved_bytes == 0);
  const nivo::detail::MemoryAllocator::Allocation block = allocator.allocate(5 * 1024 * 1024 + 1);
  CHECK(allocator.stats().large_allocations == 1);
  CHECK(allocator.stats().reserved_bytes == block.requested);
  allocator.deallocate(block);
  CHECK(allocator.stats().reserved_bytes == 0);
  for (const std::size_t size : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
    bool threw = false;
    try { (void)allocator.allocate(size); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(threw);
  }
  CHECK(allocator.stats().live_allocations == 0);
  CHECK(allocator.stats().system_allocations == 1);
  const nivo::detail::MemoryAllocator::Allocation small = allocator.allocate(64);
  CHECK(allocator.stats().reserved_bytes > 4096);
  CHECK(allocator.stats().slabs == 1);
  allocator.trim(true);
  CHECK(allocator.stats().reserved_bytes > 4096);
  CHECK(allocator.stats().slabs == 1);
  allocator.deallocate(small);
  CHECK(allocator.stats().reserved_bytes == 0);
}

void allocator_concurrent_remote_free() {
  nivo::detail::MemoryAllocator allocator(65536);
  std::array<std::vector<nivo::detail::MemoryAllocator::Allocation>, 8> batches;
  std::array<std::mutex, 8> handoff_mutexes;
  std::barrier gate(8);
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < batches.size(); ++worker) {
    threads.emplace_back([&, worker] {
      for (std::size_t round = 0; round < 20; ++round) {
        std::vector<nivo::detail::MemoryAllocator::Allocation> local;
        for (std::size_t i = 0; i < 128; ++i)
          local.push_back(allocator.allocate(16 + (i * 37) % 2000));
        {
          std::lock_guard<std::mutex> lock(handoff_mutexes[worker]);
          batches[worker].swap(local);
        }
        gate.arrive_and_wait();
        const std::size_t peer = (worker + 1) % batches.size();
        // Explicitly synchronize batch ownership; allocator calls remain
        // concurrent and run outside these handoff locks.
        {
          std::lock_guard<std::mutex> lock(handoff_mutexes[peer]);
          batches[peer].swap(local);
        }
        for (const nivo::detail::MemoryAllocator::Allocation& block : local)
          allocator.deallocate(block);
        gate.arrive_and_wait();
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(allocator.stats().live_allocations == 0);
  CHECK(allocator.stats().slabs == 1);
}

void allocator_cache_storage_and_zeroing() {
  Fixture f(1024 * 1024);
  CHECK(f.cache->stats().allocator.reserved_bytes == 0);
  const std::string key(500, 'k');
  nivo::WriteHandle first = std::move(f.cache->allocate(f.pool, key, 257)).value();
  const std::byte* address = first.bytes().data();
  std::fill(first.bytes().begin(), first.bytes().end(), std::byte{0x7f});
  const nivo::AllocatorStats active = f.cache->pool_stats(f.pool).value().allocator;
  CHECK(active.live_allocations == 1 && active.system_allocations == 1);
  CHECK(active.requested_bytes > first.charge());
  first.reset();
  nivo::WriteHandle second = std::move(f.cache->allocate(f.pool, key, 257)).value();
  CHECK(second.bytes().data() == address);
  CHECK(second.key() == key);
  CHECK(std::all_of(second.bytes().begin(), second.bytes().end(),
                    [](std::byte value) { return value == std::byte{0}; }));
  second.reset();
  CHECK(f.cache->stats().allocator.active_bytes == 0);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocator.reserved_bytes == 0);
}

void allocator_handle_copy_move_and_pins() {
  Fixture f(65536);
  nivo::ReadHandle original = std::move(f.cache->put(f.pool, "key", bytes("snapshot"))).value();
  nivo::ReadHandle copy(original);
  nivo::ReadHandle assigned;
  assigned = original;
  nivo::ReadHandle moved(std::move(copy));
  CHECK(!copy && text(moved) == "snapshot");
  copy = std::move(assigned);
  CHECK(!assigned);
  // Exercise aliasing assignments through references, including self-move.
  nivo::ReadHandle& alias = original;
  original = alias;
  original = std::move(alias);
  CHECK(text(original) == "snapshot");
  f.put("key", "replacement");
  CHECK(f.cache->stats().allocator.live_allocations == 2);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocator.live_allocations == 1);
  original.reset(); moved.reset();
  CHECK(text(copy) == "snapshot");
  f.cache->close();
  CHECK(f.cache->stats().allocator.reserved_bytes > 0);
  std::thread remote([handle = std::move(copy)]() mutable { handle.reset(); });
  remote.join();
  CHECK(f.cache->stats().allocator.live_allocations == 0);
  CHECK(f.cache->stats().allocator.reserved_bytes == 0);
}

void allocator_resize_and_pool_stats() {
  Fixture f(65536);
  const nivo::PoolId other = f.cache->add_pool({"other", 65536}).value();
  nivo::WriteHandle first = std::move(f.cache->allocate(f.pool, "a", 128)).value();
  nivo::WriteHandle second = std::move(f.cache->allocate(other, "b", 128)).value();
  CHECK(f.cache->stats().allocator.slabs == 2);
  CHECK(f.cache->pool_stats(f.pool).value().allocator.slabs == 1);
  CHECK(f.cache->resize_pool(f.pool, 1) == Status::resource_exhausted);
  first.reset();
  CHECK(f.cache->resize_pool(f.pool, 1) == Status::ok);
  CHECK(f.cache->pool_stats(f.pool).value().allocator.reserved_bytes == 0);
  CHECK(f.cache->stats().allocator.live_allocations == 1);
  second.reset();
  f.cache->close();
  CHECK(f.cache->stats().allocator.reserved_bytes == 0);
}

void allocator_large_cache_value_and_overflow() {
  Fixture f(32 * 1024 * 1024);
  nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "large", 16 * 1024 * 1024)).value();
  CHECK(draft.bytes().front() == std::byte{0} && draft.bytes().back() == std::byte{0});
  draft.bytes().back() = std::byte{42};
  nivo::ReadHandle read = std::move(f.cache->publish(std::move(draft))).value();
  CHECK(f.cache->stats().allocator.large_allocations == 1);
  f.cache.reset();
  CHECK(read.bytes().back() == std::byte{42});
  read.reset();
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
      {.capacity_bytes = maximum, .max_value_bytes = maximum})).value();
  const nivo::PoolId pool = cache->add_pool({"overflow", maximum}).value();
  CHECK(cache->allocate(pool, "k", maximum - 1).status() == Status::invalid_argument);
  CHECK(cache->stats().allocated_bytes == 0 && cache->stats().allocator.reserved_bytes == 0);
}

void allocator_handles_outlive_cache_on_threads() {
  std::vector<nivo::ReadHandle> handles;
  std::vector<nivo::WriteHandle> drafts;
  {
    Fixture f(1024 * 1024);
    for (int i = 0; i < 300; ++i) {
      handles.push_back(std::move(f.cache->put(f.pool, std::to_string(i), bytes("alive"))).value());
      drafts.push_back(std::move(f.cache->allocate(f.pool, "draft", 100)).value());
    }
  }
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < 6; ++worker) {
    threads.emplace_back([&, worker] {
      for (std::size_t i = worker; i < handles.size(); i += 6) {
        if (text(handles[i]) != "alive") failed = true;
        handles[i].reset();
        drafts[i].bytes()[99] = std::byte{42};
        drafts[i].reset();
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed);
}

struct AllocatorProbe {
  using Allocator = nivo::detail::MemoryAllocator;
  struct SystemBlock { void* data; std::size_t bytes; };
  std::atomic<std::int64_t> milliseconds{0};
  std::vector<std::size_t> requests;
  std::vector<SystemBlock> live;
  std::size_t fail_above = std::numeric_limits<std::size_t>::max();
  bool fail_all = false;
  bool invalid_delete = false;

  static Allocator::TimePoint now(void* context) noexcept {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    return Allocator::TimePoint{} + std::chrono::milliseconds(probe.milliseconds.load());
  }
  static void* allocate(void* context, std::size_t bytes) {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    probe.requests.push_back(bytes);
    if (probe.fail_all || bytes > probe.fail_above) throw std::bad_alloc();
    void* data = ::operator new(bytes);
    try { probe.live.push_back({data, bytes}); }
    catch (...) { ::operator delete(data); throw; }
    return data;
  }
  static void deallocate(void* context, void* data) noexcept {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    std::vector<SystemBlock>::iterator it = std::find_if(probe.live.begin(), probe.live.end(),
        [data](const SystemBlock& block) { return block.data == data; });
    if (it == probe.live.end()) { probe.invalid_delete = true; return; }
    probe.live.erase(it);
    ::operator delete(data);
  }
  Allocator::Hooks hooks() noexcept { return {this, now, allocate, deallocate}; }
};

// Return allocations even when an assertion throws, so failures retain their
// diagnostic instead of tripping the allocator's outstanding-allocation check.
struct AllocatorBlocks {
  using Allocation = nivo::detail::MemoryAllocator::Allocation;
  nivo::detail::MemoryAllocator& allocator;
  std::vector<Allocation> blocks;
  explicit AllocatorBlocks(nivo::detail::MemoryAllocator& owner) : allocator(owner) {}
  ~AllocatorBlocks() { clear(); }
  void add(std::size_t size = 15000) {
    const Allocation block = allocator.allocate(size);
    try { blocks.push_back(block); }
    catch (...) { allocator.deallocate(block); throw; }
    std::fill_n(static_cast<std::byte*>(block.data), size, std::byte{0x5a});
  }
  void release(std::size_t index) noexcept {
    if (blocks[index].data) {
      allocator.deallocate(blocks[index]);
      blocks[index] = {};
    }
  }
  void clear() noexcept {
    for (std::size_t i = 0; i < blocks.size(); ++i) release(i);
    blocks.clear();
  }
};

void adaptive_refill_growth() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  bool saw_two = false, saw_four = false, saw_eight = false;
  for (std::size_t i = 0; i < 1000; ++i) {
    blocks.add();
    const nivo::AllocatorStats stats = allocator.stats();
    saw_two = saw_two || stats.refill_batch_size == 2;
    saw_four = saw_four || stats.refill_batch_size == 4;
    saw_eight = saw_eight || stats.refill_batch_size == 8;
    CHECK(stats.refill_batch_size <= 8);
    CHECK(stats.empty_slabs * 65536 <= stats.reserve_limit_bytes);
  }
  const nivo::AllocatorStats stats = allocator.stats();
  CHECK(saw_two && saw_four && saw_eight);
  CHECK(stats.batch_refills > 0 && stats.system_allocations < stats.slabs / 2);
  CHECK(stats.regions == probe.requests.size());
  CHECK(stats.system_allocations == probe.requests.size());
  const std::size_t header_bytes = probe.requests.front() - 65536;
  CHECK(probe.requests.back() == header_bytes + 8 * 65536);
  blocks.clear();
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
  CHECK(allocator.stats().reserved_bytes == 0);
}

void adaptive_refill_slow_and_recycled_traffic() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 80; ++i) {
    probe.milliseconds.fetch_add(1000);
    blocks.add();
  }
  CHECK(allocator.stats().refill_batch_size == 1);
  CHECK(allocator.stats().batch_refills == 0);
  CHECK(allocator.stats().system_allocations == allocator.stats().slabs);
  blocks.clear();
  const std::uint64_t calls = allocator.stats().system_allocations;
  for (std::size_t i = 0; i < 1000; ++i) {
    blocks.add(128 + (i % 4) * 128);
    blocks.clear();
  }
  CHECK(allocator.stats().system_allocations == calls);
  CHECK(allocator.stats().refill_batch_size == 1);
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
}

void adaptive_refill_cooling() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 1000; ++i) blocks.add();
  blocks.clear();
  CHECK(allocator.stats().refill_batch_size == 8);
  CHECK(allocator.stats().empty_slabs > 1 && allocator.stats().empty_slabs <= 8);
  probe.milliseconds.fetch_add(499);
  CHECK(allocator.stats().refill_batch_size == 8);
  probe.milliseconds.fetch_add(1);
  CHECK(allocator.stats().refill_batch_size == 4);
  CHECK(allocator.stats().empty_slabs <= 4);
  probe.milliseconds.fetch_add(500);
  CHECK(allocator.stats().refill_batch_size == 2);
  probe.milliseconds.fetch_add(500);
  CHECK(allocator.stats().refill_batch_size == 1);
  CHECK(allocator.stats().empty_slabs <= 1);
  const std::uint64_t batches = allocator.stats().batch_refills;
  for (std::size_t i = 0; i < 16; ++i) {
    probe.milliseconds.fetch_add(1000);
    blocks.add();
  }
  CHECK(allocator.stats().batch_refills == batches);
  CHECK(probe.requests.back() == probe.requests.front());
}

void adaptive_refill_pinned_region_and_reclassification() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  // Four 15,360-byte slots per slab: the ninth item starts the first batch.
  for (std::size_t i = 0; i < 9; ++i) blocks.add();
  const nivo::AllocatorStats before = allocator.stats();
  CHECK(before.slabs == 4 && before.regions == 3 && before.empty_slabs == 1);
  blocks.add(256);
  CHECK(allocator.stats().system_allocations == before.system_allocations);
  CHECK(reinterpret_cast<std::uintptr_t>(blocks.blocks[9].slab) -
        reinterpret_cast<std::uintptr_t>(blocks.blocks[8].slab) == 65536);
  for (std::size_t i = 0; i < 8; ++i) blocks.release(i);
  allocator.trim(true);
  CHECK(allocator.stats().regions == 1 && allocator.stats().slabs == 2);
  blocks.release(9);
  CHECK(allocator.stats().regions == 1 && allocator.stats().empty_slabs == 1);
  CHECK(static_cast<std::byte*>(blocks.blocks[8].data)[14999] == std::byte{0x5a});
  blocks.release(8);
  CHECK(allocator.stats().reserved_bytes == 0);
  CHECK(probe.live.empty() && !probe.invalid_delete);
}

void adaptive_refill_reserve_budget() {
  AllocatorProbe probe;
  constexpr std::size_t capacity = 4 * 1024 * 1024;
  nivo::detail::MemoryAllocator allocator(capacity, probe.hooks());
  AllocatorBlocks blocks(allocator);
  CHECK(allocator.stats().reserve_limit_bytes == capacity / 16);
  bool crossed_capacity = false;
  for (std::size_t i = 0; i < 400; ++i) {
    const nivo::AllocatorStats before = allocator.stats();
    blocks.add();
    const nivo::AllocatorStats after = allocator.stats();
    CHECK(after.empty_slabs * 65536 <= after.reserve_limit_bytes);
    if (after.system_allocations != before.system_allocations && after.reserved_bytes > capacity) {
      CHECK(after.slabs == before.slabs + 1);
      CHECK(after.empty_slabs == 0);
      crossed_capacity = true;
    }
  }
  CHECK(crossed_capacity && allocator.stats().batch_refills > 0);
  const std::size_t header_bytes = probe.requests.front() - 65536;
  for (const std::size_t request : probe.requests) CHECK(request <= header_bytes + 5 * 65536);
  blocks.clear();
  CHECK(allocator.stats().empty_slabs * 65536 <= allocator.stats().reserve_limit_bytes);
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
  nivo::detail::MemoryAllocator huge(std::numeric_limits<std::size_t>::max());
  CHECK(huge.stats().reserve_limit_bytes == 8 * 65536);
}

void adaptive_refill_resize() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 100; ++i) blocks.add();
  allocator.resize(65536);
  CHECK(allocator.stats().reserve_limit_bytes == 65536);
  CHECK(allocator.stats().refill_batch_size == 1);
  const std::uint64_t batches = allocator.stats().batch_refills;
  for (std::size_t i = 0; i < 100; ++i) blocks.add();
  CHECK(allocator.stats().batch_refills == batches);
  CHECK(probe.requests.back() == probe.requests.front());
  blocks.clear();
  CHECK(allocator.stats().empty_slabs <= 1);
  allocator.trim();
  allocator.resize(64 * 1024 * 1024);
  CHECK(allocator.stats().reserve_limit_bytes == 8 * 65536);
  for (std::size_t i = 0; i < 100; ++i) blocks.add();
  CHECK(allocator.stats().batch_refills > batches);
}

void adaptive_refill_failure_fallback() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 8; ++i) blocks.add();
  probe.fail_above = probe.requests.front();
  blocks.add();
  CHECK(probe.requests.size() == 4);
  CHECK(probe.requests[2] > probe.fail_above && probe.requests[3] == probe.fail_above);
  CHECK(allocator.stats().batch_fallbacks == 1);
  CHECK(allocator.stats().refill_batch_size == 1);
  CHECK(allocator.stats().slabs == 3 && allocator.stats().live_allocations == 9);
  for (std::size_t i = 0; i < 3; ++i) blocks.add();
  const nivo::AllocatorStats before = allocator.stats();
  probe.fail_all = true;
  bool threw = false;
  try { blocks.add(); } catch (const std::bad_alloc&) { threw = true; }
  CHECK(threw);
  CHECK(allocator.stats().reserved_bytes == before.reserved_bytes);
  CHECK(allocator.stats().live_allocations == before.live_allocations);
  probe.fail_all = false;
  blocks.add();
  blocks.clear();
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
}

void adaptive_refill_batch_and_fallback_failure() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 8; ++i) blocks.add();
  const nivo::AllocatorStats before = allocator.stats();
  probe.fail_all = true;
  bool threw = false;
  try { blocks.add(); } catch (const std::bad_alloc&) { threw = true; }
  CHECK(threw && probe.requests.size() == 4);
  CHECK(allocator.stats().batch_fallbacks == 1);
  CHECK(allocator.stats().reserved_bytes == before.reserved_bytes);
  CHECK(allocator.stats().system_allocations == before.system_allocations);
  CHECK(probe.live.size() == 2);
  probe.fail_all = false;
  blocks.add();
  CHECK(static_cast<std::byte*>(blocks.blocks.front().data)[14999] == std::byte{0x5a});
}

void adaptive_refill_large_bypass() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 50; ++i) blocks.add(128 * 1024);
  CHECK(allocator.stats().large_allocations == 50);
  CHECK(allocator.stats().regions == 0 && allocator.stats().slabs == 0);
  CHECK(allocator.stats().refill_batch_size == 1 && allocator.stats().batch_refills == 0);
  blocks.clear();
  CHECK(allocator.stats().reserved_bytes == 0 && probe.live.empty());
}

void adaptive_refill_concurrent_remote_free() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  using Batch = std::vector<nivo::detail::MemoryAllocator::Allocation>;
  std::array<Batch, 8> batches;
  std::array<std::mutex, 8> handoff_mutexes;
  std::barrier gate(8);
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < batches.size(); ++worker) {
    threads.emplace_back([&, worker] {
      for (std::size_t round = 0; round < 5; ++round) {
        Batch local;
        for (std::size_t i = 0; i < 64; ++i) {
          local.push_back(allocator.allocate(15000));
          std::fill_n(static_cast<std::byte*>(local.back().data), 15000, std::byte{0x5a});
        }
        {
          std::lock_guard<std::mutex> lock(handoff_mutexes[worker]);
          batches[worker].swap(local);
        }
        gate.arrive_and_wait();
        const std::size_t peer = (worker + 1) % batches.size();
        {
          std::lock_guard<std::mutex> lock(handoff_mutexes[peer]);
          batches[peer].swap(local);
        }
        for (const nivo::detail::MemoryAllocator::Allocation& block : local) {
          if (static_cast<std::byte*>(block.data)[14999] != std::byte{0x5a}) failed = true;
          allocator.deallocate(block);
        }
        gate.arrive_and_wait();
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed && !probe.invalid_delete);
  CHECK(allocator.stats().batch_refills > 0);
  CHECK(allocator.stats().live_allocations == 0);
  allocator.trim();
  CHECK(allocator.stats().reserved_bytes == 0 && probe.live.empty());
}

void adaptive_cache_pool_limits_and_lifetime() {
  Fixture f(8 * 1024 * 1024);
  const nivo::PoolId other = f.cache->add_pool({"other", 8 * 1024 * 1024}).value();
  std::vector<nivo::ReadHandle> reads;
  std::vector<nivo::WriteHandle> drafts;
  const std::string value(15000, 'v');
  for (std::size_t i = 0; i < 128; ++i) {
    reads.push_back(std::move(f.cache->put(f.pool, std::to_string(i), bytes(value))).value());
    drafts.push_back(std::move(f.cache->allocate(other, "draft", 15000)).value());
  }
  CHECK(f.cache->stats().allocator.live_allocations == 256);
  CHECK(f.cache->resize_pool(f.pool, 1024 * 1024) == Status::resource_exhausted);
  CHECK(f.cache->pool_stats(f.pool).value().allocator.reserve_limit_bytes == 8 * 65536);
  CHECK(f.cache->resize_pool(f.pool, 4 * 1024 * 1024) == Status::ok);
  const nivo::AllocatorStats first_stats = f.cache->pool_stats(f.pool).value().allocator;
  const nivo::AllocatorStats second_stats = f.cache->pool_stats(other).value().allocator;
  const nivo::AllocatorStats total = f.cache->stats().allocator;
  CHECK(first_stats.reserve_limit_bytes == 4 * 65536);
  CHECK(second_stats.reserve_limit_bytes == 8 * 65536);
  CHECK(total.reserve_limit_bytes == 12 * 65536);
  CHECK(total.regions == first_stats.regions + second_stats.regions);
  CHECK(total.refill_batch_size == std::max(first_stats.refill_batch_size, second_stats.refill_batch_size));
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocator.live_allocations == 256);
  f.cache->close();
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      for (std::size_t i = worker; i < reads.size(); i += 4) {
        if (text(reads[i]) != value) failed = true;
        reads[i].reset();
        drafts[i].bytes().back() = std::byte{42};
        drafts[i].reset();
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed);
  CHECK(f.cache->stats().allocator.live_allocations == 0);
  CHECK(f.cache->stats().allocator.reserved_bytes == 0);
  CHECK(f.cache->stats().allocator.regions == 0);
}

std::string compact_key(std::size_t i) { return std::to_string(100000 + i); }
std::string compact_value(std::size_t i) { return std::string(15000, static_cast<char>(i % 251)); }
void populate_sparse(nivo::Cache& cache, nivo::PoolId pool, nivo::WriteOptions options = {}) {
  for (std::size_t i = 0; i < 512; ++i)
    CHECK(cache.put(pool, compact_key(i), bytes(compact_value(i)), options));
  for (std::size_t i = 0; i < 512; ++i)
    if (i % 32 != 0) CHECK(cache.erase(pool, compact_key(i)) == Status::ok);
}
void verify_sparse(nivo::Cache& cache, nivo::PoolId pool) {
  for (std::size_t i = 0; i < 512; i += 32) {
    const nivo::Result<nivo::ReadHandle> found = cache.peek(pool, compact_key(i));
    CHECK(found && found.value().key() == compact_key(i));
    CHECK(found.value().pool() == pool && found.value().charge() == 15006);
    CHECK(text(found.value()) == compact_value(i));
  }
}
void compaction_reclaims_sparse_regions() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  const nivo::PoolStats before = f.cache->pool_stats(f.pool).value();
  const nivo::Result<nivo::CompactionStats> compacted = f.cache->compact(f.pool);
  CHECK(compacted);
  const nivo::CompactionStats result = compacted.value();
  const nivo::PoolStats after = f.cache->pool_stats(f.pool).value();
  CHECK(result.pools == 1 && result.moved_items == 16 && result.skipped_items == 0);
  CHECK(result.expired_items == 0 && result.reserved_bytes_before >= result.reserved_bytes_after);
  CHECK(result.reserved_bytes_after == after.allocator.reserved_bytes);
  CHECK(after.allocator.reserved_bytes < before.allocator.reserved_bytes / 2);
  CHECK(after.allocator.slabs == 4 && after.allocator.regions == 4);
  CHECK(after.allocator.live_allocations == before.allocator.live_allocations);
  CHECK(after.allocator.requested_bytes == before.allocator.requested_bytes);
  CHECK(after.allocator.active_bytes == before.allocator.active_bytes);
  CHECK(after.allocator.empty_slabs == 0 && after.allocator.refill_batch_size == 1);
  CHECK(after.allocated_bytes == before.allocated_bytes && after.indexed_bytes == before.indexed_bytes);
  CHECK(after.counters.allocations == before.counters.allocations);
  CHECK(after.counters.inserts == before.counters.inserts && after.counters.replacements == before.counters.replacements);
  CHECK(after.counters.evictions == before.counters.evictions && after.counters.erases == before.counters.erases);
  CHECK(after.counters.hits == before.counters.hits && after.counters.misses == before.counters.misses);
  verify_sparse(*f.cache, f.pool);
  const nivo::CompactionStats repeated = f.cache->compact(f.pool).value();
  CHECK(repeated.moved_items == 0 && repeated.skipped_items == 16);
  CHECK(repeated.reserved_bytes_before == repeated.reserved_bytes_after);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocated_bytes == 0 && f.cache->stats().allocator.reserved_bytes == 0);
}

void compaction_at_full_logical_capacity() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  const std::size_t used = f.cache->stats().allocated_bytes;
  CHECK(f.cache->resize_pool(f.pool, used) == Status::ok);
  CHECK(f.cache->compact(f.pool).value().moved_items == 16);
  const nivo::PoolStats stats = f.cache->pool_stats(f.pool).value();
  CHECK(stats.allocated_bytes == used && stats.capacity_bytes == used && stats.counters.evictions == 0);
  verify_sparse(*f.cache, f.pool);
}

void compaction_small_pool_and_idle_storage() {
  Fixture f(4096);
  for (std::size_t i = 0; i < 6; ++i) f.put(compact_key(i), std::string(600, static_cast<char>('a' + i)));
  for (std::size_t i = 1; i < 5; ++i) CHECK(f.cache->erase(f.pool, compact_key(i)) == Status::ok);
  const nivo::CompactionStats compacted = f.cache->compact(f.pool).value();
  CHECK(compacted.moved_items == 2 && compacted.reserved_bytes_after < compacted.reserved_bytes_before);
  CHECK(f.cache->stats().allocator.slabs == 1);
  CHECK(text(f.cache->peek(f.pool, compact_key(0)).value()) == std::string(600, 'a'));
  CHECK(text(f.cache->peek(f.pool, compact_key(5)).value()) == std::string(600, 'f'));
  CHECK(f.cache->erase(f.pool, compact_key(0)) == Status::ok);
  CHECK(f.cache->erase(f.pool, compact_key(5)) == Status::ok);
  CHECK(f.cache->stats().allocator.reserved_bytes > 0);
  const nivo::CompactionStats empty = f.cache->compact(f.pool).value();
  CHECK(empty.moved_items == 0 && empty.skipped_items == 0 && empty.reserved_bytes_after == 0);
}

void compaction_all_pools_share_expiry_cutoff() {
  struct CountingClock final : nivo::Clock {
    mutable std::atomic<nivo::Duration::rep> ticks{0};
    std::atomic<nivo::Duration::rep> step{0};
    mutable std::atomic<std::size_t> calls{0};
    nivo::Duration now() const noexcept override {
      ++calls;
      return nivo::Duration(ticks.fetch_add(step.load()));
    }
  };
  std::shared_ptr<CountingClock> clock = std::make_shared<CountingClock>();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create({.clock = clock})).value();
  const nivo::PoolId first = cache->add_pool({"first", 4096}).value();
  const nivo::PoolId second = cache->add_pool({"second", 4096}).value();
  CHECK(cache->put(first, "key", bytes("value"), {.ttl = 10ms}));
  CHECK(cache->put(second, "key", bytes("value"), {.ttl = 10ms}));
  clock->ticks = 5;
  clock->step = 10;
  const std::size_t calls = clock->calls.load();
  CHECK(cache->compact().value().expired_items == 0);
  CHECK(clock->calls == calls + 1 && cache->stats().indexed_items == 2);
  const nivo::CompactionStats expired = cache->compact().value();
  CHECK(expired.expired_items == 2 && expired.reserved_bytes_after == 0);
}

void compaction_preserves_versions_and_ttl() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool, {.ttl = 10s});
  f.put(compact_key(0), compact_value(0), {.ttl = 1s});
  const nivo::Version version = f.cache->peek(f.pool, compact_key(32)).value().version();
  f.clock->advance(5s);
  const nivo::CompactionStats result = f.cache->compact(f.pool).value();
  CHECK(result.expired_items == 1 && result.moved_items == 15);
  CHECK(f.cache->peek(f.pool, compact_key(0)).status() == Status::not_found);
  CHECK(f.cache->peek(f.pool, compact_key(32)).value().version() == version);
  CHECK(f.cache->put(f.pool, compact_key(32), bytes("cas"),
      {.mode = nivo::WriteMode::compare_exchange, .expected_version = version}));
  f.clock->advance(4999ms);
  CHECK(f.cache->peek(f.pool, compact_key(64)));
  f.clock->advance(1ms);
  CHECK(f.cache->peek(f.pool, compact_key(64)).status() == Status::not_found);
  CHECK(f.cache->compact().value().expired_items == 13);
  CHECK(text(f.cache->peek(f.pool, compact_key(32)).value()) == "cas");
}

void compaction_preserves_eviction_order() {
  for (const nivo::EvictionPolicy policy : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    Fixture f(16 * 1024 * 1024);
    const nivo::PoolId pool = f.cache->add_pool({"ordered", 16 * 1024 * 1024, policy}).value();
    populate_sparse(*f.cache, pool);
    CHECK(f.cache->find(pool, compact_key(0)));
    CHECK(f.cache->compact(pool).value().moved_items == 16);
    const std::size_t used = f.cache->pool_stats(pool).value().allocated_bytes;
    CHECK(f.cache->resize_pool(pool, used - 15006) == Status::ok);
    const std::size_t victim = policy == nivo::EvictionPolicy::lru ? 32 : 0;
    CHECK(f.cache->peek(pool, compact_key(victim)).status() == Status::not_found);
    CHECK(f.cache->peek(pool, compact_key(victim == 0 ? 32 : 0)));
    CHECK(f.cache->pool_stats(pool).value().counters.evictions == 1);
  }
}

void compaction_pins_drafts_and_detached_versions() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  nivo::ReadHandle pin = std::move(f.cache->peek(f.pool, compact_key(0))).value();
  nivo::ReadHandle old = std::move(f.cache->peek(f.pool, compact_key(32))).value();
  f.put(compact_key(32), "replacement");
  nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "draft", 15000)).value();
  std::fill(draft.bytes().begin(), draft.bytes().end(), std::byte{42});
  const std::byte* pin_address = pin.bytes().data();
  const std::byte* old_address = old.bytes().data();
  const std::byte* draft_address = draft.bytes().data();
  const std::size_t used = f.cache->stats().allocated_bytes;
  const nivo::CompactionStats result = f.cache->compact(f.pool).value();
  CHECK(result.moved_items > 0 && result.skipped_items > 0);
  CHECK(result.moved_items + result.skipped_items == 16);
  CHECK(f.cache->stats().allocated_bytes == used);
  CHECK(pin.bytes().data() == pin_address && pin.charge() == 15006 && text(pin) == compact_value(0));
  CHECK(old.bytes().data() == old_address && text(old) == compact_value(32));
  CHECK(draft.bytes().data() == draft_address && draft.bytes().back() == std::byte{42});
  CHECK(f.cache->peek(f.pool, "draft").status() == Status::not_found);
  CHECK(f.cache->publish(std::move(draft)));
  CHECK(f.cache->peek(f.pool, "draft").value().bytes().back() == std::byte{42});
  pin.reset();
  old.reset();
  CHECK(f.cache->compact(f.pool));
  CHECK(f.cache->stats().allocated_bytes == f.cache->stats().indexed_bytes);
  nivo::ReadHandle survivor = std::move(f.cache->peek(f.pool, "draft")).value();
  f.cache.reset();
  CHECK(survivor.bytes().back() == std::byte{42});
}

void compaction_all_pinned_then_released() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  std::vector<nivo::ReadHandle> pins;
  for (std::size_t i = 0; i < 512; i += 32)
    pins.push_back(std::move(f.cache->peek(f.pool, compact_key(i))).value());
  const nivo::CompactionStats pinned = f.cache->compact(f.pool).value();
  CHECK(pinned.moved_items == 0 && pinned.skipped_items == 16);
  pins.clear();
  CHECK(f.cache->compact(f.pool).value().moved_items == 16);
  verify_sparse(*f.cache, f.pool);
}

void compaction_mixed_classes_and_large_values() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  const std::string binary_key("binary\0key", 10);
  const std::string long_key(500, 'k');
  const std::string medium(3000, 'm');
  const std::string large(100000, 'l');
  f.put(binary_key, {});
  f.put(long_key, medium);
  f.put("large", large);
  const nivo::AllocatorStats before = f.cache->stats().allocator;
  const nivo::CompactionStats result = f.cache->compact(f.pool).value();
  CHECK(result.moved_items == 18 && result.skipped_items == 1);
  CHECK(result.reserved_bytes_after < result.reserved_bytes_before);
  CHECK(f.cache->peek(f.pool, binary_key).value().bytes().empty());
  CHECK(f.cache->peek(f.pool, binary_key).value().key() == binary_key);
  CHECK(text(f.cache->peek(f.pool, long_key).value()) == medium);
  CHECK(text(f.cache->peek(f.pool, "large").value()) == large);
  CHECK(f.cache->stats().allocator.large_allocations == 1);
  CHECK(f.cache->stats().allocator.requested_bytes == before.requested_bytes);
  verify_sparse(*f.cache, f.pool);
}

void compaction_selection_and_errors() {
  Fixture f(16 * 1024 * 1024);
  const nivo::PoolId other = f.cache->add_pool({"other", 16 * 1024 * 1024}).value();
  CHECK(f.cache->compact(999).status() == Status::pool_not_found);
  const nivo::CompactionStats empty = f.cache->compact().value();
  CHECK(empty.pools == 2 && empty.moved_items == 0 && empty.reserved_bytes_after == 0);
  populate_sparse(*f.cache, f.pool);
  populate_sparse(*f.cache, other);
  const nivo::AllocatorStats before = f.cache->pool_stats(other).value().allocator;
  CHECK(f.cache->compact(f.pool).value().moved_items == 16);
  const nivo::AllocatorStats untouched = f.cache->pool_stats(other).value().allocator;
  CHECK(untouched.system_allocations == before.system_allocations && untouched.slabs == before.slabs);
  const nivo::CompactionStats all = f.cache->compact().value();
  CHECK(all.pools == 2 && all.moved_items == 16 && all.skipped_items == 16);
  verify_sparse(*f.cache, f.pool);
  verify_sparse(*f.cache, other);
  f.cache->close();
  CHECK(f.cache->compact().status() == Status::closed);
  CHECK(f.cache->compact(f.pool).status() == Status::closed);
  CHECK(f.cache->stats().allocated_bytes == 0 && f.cache->stats().allocator.regions == 0);
  nivo::Result<std::unique_ptr<nivo::Cache>> no_pools = nivo::Cache::create();
  CHECK(no_pools.value()->compact().value().pools == 0);
}

// Leave exactly one live allocation in each occupied system region, including
// eight-slab batches. This reproduces the worst retention pattern directly.
void retain_one_per_region(AllocatorProbe& probe, AllocatorBlocks& blocks) {
  const std::vector<AllocatorProbe::SystemBlock> regions = probe.live;
  std::vector<bool> retained(regions.size(), false);
  for (std::size_t i = 0; i < blocks.blocks.size(); ++i) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(blocks.blocks[i].data);
    bool found = false;
    for (std::size_t r = 0; r < regions.size(); ++r) {
      const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(regions[r].data);
      if (address < base || address - base >= regions[r].bytes) continue;
      if (retained[r]) blocks.release(i);
      else retained[r] = true;
      found = true;
      break;
    }
    CHECK(found);
  }
}

void allocator_compaction_failure_and_retry() {
  using Allocator = nivo::detail::MemoryAllocator;
  for (const std::size_t fail_after : {std::size_t{0}, std::size_t{4}}) {
    AllocatorProbe probe;
    Allocator allocator(64 * 1024 * 1024, probe.hooks());
    AllocatorBlocks blocks(allocator);
    for (std::size_t i = 0; i < 512; ++i) blocks.add();
    retain_one_per_region(probe, blocks);
    allocator.trim();
    const nivo::AllocatorStats before = allocator.stats();
    std::size_t moved = 0;
    bool failed = false;
    try {
      Allocator::Compaction compaction(allocator, blocks.blocks);
      for (Allocator::Allocation& block : blocks.blocks) {
        if (!block.data || !compaction.should_move(block)) continue;
        probe.fail_all = moved == fail_after;
        const Allocator::Allocation replacement = compaction.allocate(block);
        std::copy_n(static_cast<const std::byte*>(block.data), block.requested,
                    static_cast<std::byte*>(replacement.data));
        allocator.deallocate(block);
        block = replacement;
        ++moved;
      }
    } catch (const std::bad_alloc&) { failed = true; }
    CHECK(failed && moved == fail_after && !probe.invalid_delete);
    CHECK(allocator.stats().live_allocations == before.live_allocations);
    CHECK(allocator.stats().requested_bytes == before.requested_bytes);
    probe.fail_all = false;
    {
      Allocator::Compaction retry(allocator, blocks.blocks);
      for (Allocator::Allocation& block : blocks.blocks) {
        if (!block.data || !retry.should_move(block)) continue;
        const Allocator::Allocation replacement = retry.allocate(block);
        std::copy_n(static_cast<const std::byte*>(block.data), block.requested,
                    static_cast<std::byte*>(replacement.data));
        allocator.deallocate(block);
        block = replacement;
      }
    }
    CHECK(allocator.stats().reserved_bytes < before.reserved_bytes / 2);
    for (const Allocator::Allocation& block : blocks.blocks) {
      if (!block.data) continue;
      const std::byte* data = static_cast<const std::byte*>(block.data);
      CHECK(std::all_of(data, data + block.requested,
          [](std::byte value) { return value == std::byte{0x5a}; }));
    }
    blocks.clear();
    allocator.trim();
    CHECK(probe.live.empty() && !probe.invalid_delete);
  }
}

void allocator_compaction_skips_partly_pinned_regions() {
  using Allocator = nivo::detail::MemoryAllocator;
  AllocatorProbe probe;
  Allocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 256; ++i) blocks.add();
  CHECK(blocks.blocks[0].slab == blocks.blocks[1].slab);
  for (std::size_t i = 2; i < blocks.blocks.size(); ++i)
    if (i % 32 != 0) blocks.release(i);
  std::vector<Allocator::Allocation> candidates = blocks.blocks;
  candidates[0] = {};  // An external pin or unpublished draft is not movable.
  {
    Allocator::Compaction compaction(allocator, candidates);
    CHECK(!compaction.should_move(blocks.blocks[0]));
    CHECK(!compaction.should_move(blocks.blocks[1]));
    CHECK(compaction.should_move(blocks.blocks[32]));
  }
  CHECK(static_cast<const std::byte*>(blocks.blocks[0].data)[14999] == std::byte{0x5a});
  blocks.clear();
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
}

void compaction_concurrent_readers_and_writers() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  std::atomic<bool> failed{false};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < 3; ++worker) {
    threads.emplace_back([&, worker] {
      while (!start.load()) std::this_thread::yield();
      for (std::size_t i = 0; i < 300; ++i) {
        const std::size_t index = (i % 16) * 32;
        const nivo::Result<nivo::ReadHandle> found = f.cache->find(f.pool, compact_key(index));
        if (!found || text(found.value()) != compact_value(index)) { failed = true; continue; }
        nivo::ReadHandle copy = found.value();
        const std::byte* address = copy.bytes().data();
        std::this_thread::yield();
        if (copy.bytes().data() != address || copy.charge() != 15006 || text(copy) != compact_value(index))
          failed = true;
        const std::string key = "writer-" + std::to_string(worker);
        if (!f.cache->put(f.pool, key, bytes("temporary"))) failed = true;
        if (f.cache->erase(f.pool, key) != Status::ok) failed = true;
      }
    });
  }
  start = true;
  for (std::size_t i = 0; i < 100; ++i) if (!f.cache->compact()) failed = true;
  for (std::thread& thread : threads) thread.join();
  CHECK(!failed);
  CHECK(f.cache->compact(f.pool));
  verify_sparse(*f.cache, f.pool);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocated_bytes == 0 && f.cache->stats().allocator.regions == 0);
}

using MemoryField = std::size_t nivo::MemoryStats::*;
constexpr MemoryField memory_fields[] = {
  &nivo::MemoryStats::reserved_bytes, &nivo::MemoryStats::kv_bytes,
  &nivo::MemoryStats::indexed_kv_bytes, &nivo::MemoryStats::retained_kv_bytes,
  &nivo::MemoryStats::entry_metadata_bytes, &nivo::MemoryStats::allocator_metadata_bytes,
  &nivo::MemoryStats::internal_fragmentation_bytes, &nivo::MemoryStats::unused_bytes,
  &nivo::MemoryStats::reclaimable_bytes
};
bool valid_memory_breakdown(const nivo::MemoryStats& memory) {
  for (const MemoryField field : memory_fields)
    if (memory.*field > memory.reserved_bytes) return false;
  return memory.kv_bytes == memory.indexed_kv_bytes + memory.retained_kv_bytes &&
      memory.reserved_bytes == memory.kv_bytes + memory.entry_metadata_bytes +
          memory.allocator_metadata_bytes + memory.internal_fragmentation_bytes + memory.unused_bytes;
}

void memory_monitor_empty_and_process_rss() {
  Fixture f;
  const nivo::CacheMemoryStats sample = f.cache->memory_stats();
  CHECK(valid_memory_breakdown(sample.total) && sample.total.reserved_bytes == 0);
  CHECK(sample.pools.size() == 1 && sample.pools[0].id == f.pool && sample.pools[0].name == "default");
  CHECK(f.cache->memory_stats(f.pool).value().reserved_bytes == 0);
  CHECK(f.cache->memory_stats(999).status() == Status::pool_not_found);
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__linux__)
  CHECK(sample.process_rss_bytes.has_value() && *sample.process_rss_bytes > 0);
#else
  CHECK(!sample.process_rss_bytes.has_value());
#endif
  // A nonzero process RSS with zero cache storage must not be attributed to the cache.
  f.cache->close();
  CHECK(f.cache->memory_stats(f.pool).value().kv_bytes == 0);
  CHECK(f.cache->memory_stats().total.reserved_bytes == 0);
}

void memory_monitor_drafts_snapshots_and_reclamation() {
  Fixture f;
  nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "key", 5)).value();
  const nivo::MemoryStats unpublished = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(unpublished));
  CHECK(unpublished.kv_bytes == 8 && unpublished.retained_kv_bytes == 8 && unpublished.indexed_kv_bytes == 0);
  CHECK(unpublished.entry_metadata_bytes > 0 && unpublished.allocator_metadata_bytes > 0);
  CHECK(unpublished.unused_bytes > 0 && unpublished.reclaimable_bytes == 0);
  nivo::ReadHandle first = std::move(f.cache->publish(std::move(draft))).value();
  nivo::ReadHandle copy = first;
  const nivo::MemoryStats published = f.cache->memory_stats(f.pool).value();
  CHECK(published.kv_bytes == 8 && published.indexed_kv_bytes == 8 && published.retained_kv_bytes == 0);
  CHECK(published.reserved_bytes == unpublished.reserved_bytes);
  f.put("key", "new");
  const nivo::MemoryStats replaced = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(replaced));
  CHECK(replaced.kv_bytes == 14 && replaced.indexed_kv_bytes == 6 && replaced.retained_kv_bytes == 8);
  CHECK(replaced.entry_metadata_bytes == 2 * published.entry_metadata_bytes);
  first.reset();
  CHECK(f.cache->memory_stats(f.pool).value().kv_bytes == 14);
  copy.reset();
  CHECK(f.cache->memory_stats(f.pool).value().kv_bytes == 6);
  CHECK(f.cache->erase(f.pool, "key") == Status::ok);
  const nivo::MemoryStats idle = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(idle));
  CHECK(idle.kv_bytes == 0 && idle.entry_metadata_bytes == 0 && idle.internal_fragmentation_bytes == 0);
  CHECK(idle.reserved_bytes > 0 && idle.reclaimable_bytes == idle.reserved_bytes);
  CHECK(f.cache->compact(f.pool));
  const nivo::MemoryStats trimmed = f.cache->memory_stats(f.pool).value();
  for (const MemoryField field : memory_fields) CHECK(trimmed.*field == 0);
}

void memory_monitor_binary_empty_and_expiry() {
  Fixture f;
  const std::string key("k\0x", 3);
  nivo::ReadHandle pinned = std::move(f.cache->put(f.pool, key, {}, {.ttl = 1ms})).value();
  f.clock->advance(1ms);
  for (unsigned i = 0; i < 3; ++i) {
    const nivo::MemoryStats memory = f.cache->memory_stats(f.pool).value();
    CHECK(valid_memory_breakdown(memory));
    CHECK(memory.kv_bytes == 3 && memory.indexed_kv_bytes == 3 && memory.retained_kv_bytes == 0);
  }
  CHECK(f.cache->stats().counters.expirations == 0);
  CHECK(f.cache->prune_expired(f.pool).value() == 1);
  const nivo::MemoryStats expired = f.cache->memory_stats(f.pool).value();
  CHECK(expired.kv_bytes == 3 && expired.indexed_kv_bytes == 0 && expired.retained_kv_bytes == 3);
  f.cache->close();
  CHECK(f.cache->memory_stats().total.kv_bytes == 3);
  pinned.reset();
  CHECK(f.cache->memory_stats().total.reserved_bytes == 0);
}

void memory_monitor_large_allocations_and_close() {
  Fixture f(1024 * 1024);
  nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "large", 100000)).value();
  const nivo::MemoryStats memory = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(memory));
  CHECK(memory.kv_bytes == 100005 && memory.reserved_bytes == memory.kv_bytes + memory.entry_metadata_bytes);
  CHECK(memory.allocator_metadata_bytes == 0 && memory.internal_fragmentation_bytes == 0);
  CHECK(memory.unused_bytes == 0 && memory.reclaimable_bytes == 0);
  f.cache->close();
  CHECK(f.cache->memory_stats().total.retained_kv_bytes == 100005);
  draft.reset();
  CHECK(f.cache->memory_stats().total.reserved_bytes == 0);
}

void memory_monitor_pool_aggregation() {
  Fixture f(1024 * 1024);
  const nivo::PoolId second = f.cache->add_pool({"second", 1024 * 1024}).value();
  f.put("a", "one");
  CHECK(f.cache->put(second, "large", bytes(std::string(100000, 'x'))));
  nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "draft", 200)).value();
  const nivo::CacheMemoryStats sample = f.cache->memory_stats();
  CHECK(sample.pools.size() == 2 && valid_memory_breakdown(sample.total));
  CHECK(sample.total.kv_bytes == 4 + 100005 + 205);
  CHECK(sample.total.retained_kv_bytes == 205);
  for (const nivo::PoolMemoryStats& pool : sample.pools) {
    CHECK(valid_memory_breakdown(pool.memory));
    const nivo::MemoryStats individual = f.cache->memory_stats(pool.id).value();
    CHECK(pool.name == (pool.id == f.pool ? "default" : "second"));
    for (const MemoryField field : memory_fields) CHECK(individual.*field == pool.memory.*field);
  }
  for (const MemoryField field : memory_fields) {
    std::size_t sum = 0;
    for (const nivo::PoolMemoryStats& pool : sample.pools) sum += pool.memory.*field;
    CHECK(sample.total.*field == sum);
  }
  const nivo::AllocatorStats allocator = f.cache->stats().allocator;
  CHECK(allocator.metadata_bytes == sample.total.allocator_metadata_bytes);
  CHECK(allocator.reclaimable_bytes == sample.total.reclaimable_bytes);
}

void memory_monitor_compaction_effect() {
  Fixture f(16 * 1024 * 1024);
  populate_sparse(*f.cache, f.pool);
  const nivo::MemoryStats before = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(before) && before.unused_bytes > before.kv_bytes);
  const nivo::CompactionStats result = f.cache->compact(f.pool).value();
  const nivo::MemoryStats after = f.cache->memory_stats(f.pool).value();
  CHECK(valid_memory_breakdown(after));
  CHECK(result.moved_items == 16 && after.reserved_bytes < before.reserved_bytes / 2);
  CHECK(after.kv_bytes == before.kv_bytes && after.kv_bytes == 16 * 15006);
  CHECK(after.entry_metadata_bytes == before.entry_metadata_bytes);
  CHECK(after.internal_fragmentation_bytes == before.internal_fragmentation_bytes);
  CHECK(after.unused_bytes < before.unused_bytes && after.reclaimable_bytes == 0);
  verify_sparse(*f.cache, f.pool);
}

void memory_monitor_idle_and_partial_regions() {
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(64 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 512; ++i) blocks.add();
  retain_one_per_region(probe, blocks);
  allocator.trim();
  const nivo::AllocatorStats sparse = allocator.inspect();
  std::size_t requested_from_system = 0;
  for (const AllocatorProbe::SystemBlock& region : probe.live) requested_from_system += region.bytes;
  CHECK(sparse.reserved_bytes == requested_from_system && sparse.reclaimable_bytes == 0);
  CHECK(sparse.metadata_bytes > 0 && sparse.reserved_bytes > sparse.active_bytes + sparse.metadata_bytes);
  blocks.clear();
  // A final eight-slab region can exceed the one-slab reserve and be released
  // completely. Establish a single idle slab to test observation without trim.
  blocks.add();
  blocks.clear();
  const nivo::AllocatorStats idle = allocator.inspect();
  CHECK(idle.reserved_bytes > 0 && idle.reclaimable_bytes == idle.reserved_bytes);
  probe.milliseconds += 10000;
  const nivo::AllocatorStats observed = allocator.inspect();
  CHECK(observed.reserved_bytes == idle.reserved_bytes && observed.metadata_bytes == idle.metadata_bytes);
  allocator.trim();
  const nivo::AllocatorStats trimmed = allocator.inspect();
  CHECK(trimmed.reserved_bytes == 0 && trimmed.metadata_bytes == 0 && trimmed.reclaimable_bytes == 0);
  CHECK(probe.live.empty() && !probe.invalid_delete);
}

void memory_monitor_concurrent_handle_release() {
  Fixture f(16 * 1024 * 1024);
  std::vector<nivo::ReadHandle> handles;
  std::vector<nivo::WriteHandle> drafts;
  for (std::size_t i = 0; i < 1000; ++i) {
    handles.push_back(std::move(f.cache->put(f.pool, std::to_string(i), bytes("value"))).value());
    drafts.push_back(std::move(f.cache->allocate(f.pool, "draft", 300)).value());
  }
  CHECK(f.cache->clear(f.pool) == Status::ok);
  f.put("indexed", "live");
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      while (!start.load()) std::this_thread::yield();
      for (std::size_t i = worker; i < handles.size(); i += 4) {
        handles[i].reset();
        drafts[i].reset();
        std::this_thread::yield();
      }
    });
  }
  bool valid = true;
  start = true;
  for (unsigned i = 0; i < 1000; ++i) {
    const nivo::MemoryStats sample = f.cache->memory_stats(f.pool).value();
    valid = valid && valid_memory_breakdown(sample) && sample.indexed_kv_bytes == 11;
  }
  for (std::thread& thread : threads) thread.join();
  CHECK(valid);
  const nivo::MemoryStats released = f.cache->memory_stats().total;
  CHECK(released.kv_bytes == 11 && released.retained_kv_bytes == 0);
  f.cache->close();
  CHECK(f.cache->memory_stats().total.reserved_bytes == 0);
}

void memory_monitor_rss_parser() {
  CHECK(nivo::detail::rss_from_statm("100 25 10 0 0 0 0\n", 4096).value() == 102400);
  CHECK(nivo::detail::rss_from_statm(" \t100\t0\n", 16384).value() == 0);
  CHECK(!nivo::detail::rss_from_statm("", 4096));
  CHECK(!nivo::detail::rss_from_statm("100", 4096));
  CHECK(!nivo::detail::rss_from_statm("100 -1", 4096));
  CHECK(!nivo::detail::rss_from_statm("100 2x", 4096));
  CHECK(!nivo::detail::rss_from_statm("bad 2", 4096));
  CHECK(!nivo::detail::rss_from_statm("100 2", 0));
  CHECK(!nivo::detail::rss_from_statm("100 2", -1));
  CHECK(!nivo::detail::rss_from_statm("100 999999999999999999999999999999999999", 4096));
  const std::string overflow = "100 " + std::to_string(std::numeric_limits<std::size_t>::max());
  CHECK(!nivo::detail::rss_from_statm(overflow, 4096));
}

void allocator_compaction_skips_marginal_savings() {
  using Allocator = nivo::detail::MemoryAllocator;
  AllocatorProbe probe;
  Allocator allocator(128 * 1024 * 1024, probe.hooks());
  AllocatorBlocks blocks(allocator);
  for (std::size_t i = 0; i < 4000; ++i) blocks.add();
  const nivo::AllocatorStats before = allocator.inspect();
  CHECK(before.slabs > 1000 && before.batch_refills > 0);
  {
    Allocator::Compaction compaction(allocator, blocks.blocks);
    // Full slots with only a small batch tail must not trigger a copy of the
    // entire pool merely to recover that tail at substantial temporary cost.
    for (const Allocator::Allocation& block : blocks.blocks) CHECK(!compaction.should_move(block));
  }
  CHECK(allocator.inspect().system_allocations == before.system_allocations);
  CHECK(allocator.inspect().requested_bytes == before.requested_bytes);
  blocks.clear();
  allocator.trim();
  CHECK(probe.live.empty() && !probe.invalid_delete);
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
    {"concurrent_operations", concurrent_operations}, {"concurrent_close", concurrent_close},
    {"allocator_size_boundaries", allocator_size_boundaries},
    {"allocator_reuse_and_reclassification", allocator_reuse_and_reclassification},
    {"allocator_mixed_slabs", allocator_mixed_slabs},
    {"allocator_full_slab_reuse", allocator_full_slab_reuse},
    {"allocator_large_and_failure", allocator_large_and_failure},
    {"allocator_concurrent_remote_free", allocator_concurrent_remote_free},
    {"allocator_cache_storage_and_zeroing", allocator_cache_storage_and_zeroing},
    {"allocator_handle_copy_move_and_pins", allocator_handle_copy_move_and_pins},
    {"allocator_resize_and_pool_stats", allocator_resize_and_pool_stats},
    {"allocator_large_cache_value_and_overflow", allocator_large_cache_value_and_overflow},
    {"allocator_handles_outlive_cache_on_threads", allocator_handles_outlive_cache_on_threads},
    {"adaptive_refill_growth", adaptive_refill_growth},
    {"adaptive_refill_slow_and_recycled_traffic", adaptive_refill_slow_and_recycled_traffic},
    {"adaptive_refill_cooling", adaptive_refill_cooling},
    {"adaptive_refill_pinned_region_and_reclassification", adaptive_refill_pinned_region_and_reclassification},
    {"adaptive_refill_reserve_budget", adaptive_refill_reserve_budget},
    {"adaptive_refill_resize", adaptive_refill_resize},
    {"adaptive_refill_failure_fallback", adaptive_refill_failure_fallback},
    {"adaptive_refill_batch_and_fallback_failure", adaptive_refill_batch_and_fallback_failure},
    {"adaptive_refill_large_bypass", adaptive_refill_large_bypass},
    {"adaptive_refill_concurrent_remote_free", adaptive_refill_concurrent_remote_free},
    {"adaptive_cache_pool_limits_and_lifetime", adaptive_cache_pool_limits_and_lifetime},
    {"compaction_reclaims_sparse_regions", compaction_reclaims_sparse_regions},
    {"compaction_at_full_logical_capacity", compaction_at_full_logical_capacity},
    {"compaction_small_pool_and_idle_storage", compaction_small_pool_and_idle_storage},
    {"compaction_all_pools_share_expiry_cutoff", compaction_all_pools_share_expiry_cutoff},
    {"compaction_preserves_versions_and_ttl", compaction_preserves_versions_and_ttl},
    {"compaction_preserves_eviction_order", compaction_preserves_eviction_order},
    {"compaction_pins_drafts_and_detached_versions", compaction_pins_drafts_and_detached_versions},
    {"compaction_all_pinned_then_released", compaction_all_pinned_then_released},
    {"compaction_mixed_classes_and_large_values", compaction_mixed_classes_and_large_values},
    {"compaction_selection_and_errors", compaction_selection_and_errors},
    {"allocator_compaction_failure_and_retry", allocator_compaction_failure_and_retry},
    {"allocator_compaction_skips_partly_pinned_regions", allocator_compaction_skips_partly_pinned_regions},
    {"compaction_concurrent_readers_and_writers", compaction_concurrent_readers_and_writers},
    {"memory_monitor_empty_and_process_rss", memory_monitor_empty_and_process_rss},
    {"memory_monitor_drafts_snapshots_and_reclamation", memory_monitor_drafts_snapshots_and_reclamation},
    {"memory_monitor_binary_empty_and_expiry", memory_monitor_binary_empty_and_expiry},
    {"memory_monitor_large_allocations_and_close", memory_monitor_large_allocations_and_close},
    {"memory_monitor_pool_aggregation", memory_monitor_pool_aggregation},
    {"memory_monitor_compaction_effect", memory_monitor_compaction_effect},
    {"memory_monitor_idle_and_partial_regions", memory_monitor_idle_and_partial_regions},
    {"memory_monitor_concurrent_handle_release", memory_monitor_concurrent_handle_release},
    {"memory_monitor_rss_parser", memory_monitor_rss_parser},
    {"allocator_compaction_skips_marginal_savings", allocator_compaction_skips_marginal_savings}
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
