#include <nivo/nivo.h>
#include "../src/memory_allocator.h"
#include "../src/process_memory.h"
#include "../src/cache_index.h"
#include "../src/eviction_policy.h"
#include "../src/cache_pool.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <functional>
#include <future>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#if defined(__APPLE__)
#include <mach/mach_vm.h>
#endif

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
// The background reaper is disabled so expiry happens only where a test asks.
inline constexpr nivo::ReaperConfig no_reaper{.interval = nivo::Duration(0)};
struct Fixture {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache;
  nivo::PoolId pool;
  explicit Fixture(std::size_t capacity = 1024) {
    nivo::Result<std::unique_ptr<nivo::Cache>> result =
        nivo::Cache::create({.capacity_bytes = capacity * 2, .clock = clock, .reaper = no_reaper});
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

void page_memory_mapping_contract() {
  using Pages = nivo::detail::PageMemory;
  const std::size_t page = Pages::page_size();
  CHECK(page > 0);
  CHECK(Pages::mapping_size(1) == page);
  CHECK(Pages::mapping_size(page) == page);
  CHECK(Pages::mapping_size(page + 1) == 2 * page);
  for (const std::size_t size : {std::size_t{0}, std::numeric_limits<std::size_t>::max(),
                                 std::numeric_limits<std::size_t>::max() - page + 2}) {
    bool threw = false;
    try { (void)Pages::mapping_size(size); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(threw);
  }
  for (const std::size_t size : {std::size_t{0}, page + 1}) {
    bool threw = false;
    try { (void)Pages::map(size); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(threw);
  }
  void* memory = Pages::map(2 * page);
  const bool aligned = reinterpret_cast<std::uintptr_t>(memory) % page == 0;
  std::byte* payload = static_cast<std::byte*>(memory);
  const bool zeroed = std::all_of(payload, payload + 2 * page,
      [](std::byte value) { return value == std::byte{0}; });
  std::fill_n(payload, 2 * page, std::byte{0x5a});
  Pages::unmap(memory, 2 * page);
  // Inspect the address without dereferencing it. This distinguishes actual
  // unmapping from merely putting storage on an allocator free list.
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION info{};
  const SIZE_T result = VirtualQuery(memory, &info, sizeof(info));
  CHECK(result == sizeof(info) && info.State == MEM_FREE);
#elif defined(__APPLE__)
  mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(memory);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  const kern_return_t result = mach_vm_region(mach_task_self(), &address, &region_size,
      VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &info_count, &object);
  if (object != MACH_PORT_NULL) (void)mach_port_deallocate(mach_task_self(), object);
  CHECK(result == KERN_INVALID_ADDRESS || (result == KERN_SUCCESS &&
      address >= reinterpret_cast<mach_vm_address_t>(memory) + 2 * page));
#else
  unsigned char resident[2]{};
  errno = 0;
  const int result = mincore(memory, 2 * page, resident);
  CHECK(result == -1 && errno == ENOMEM);
#endif
  CHECK(aligned && zeroed);
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
  CHECK(allocator.stats().reserved_bytes == nivo::detail::PageMemory::mapping_size(block.requested));
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
  std::size_t fail_after = std::numeric_limits<std::size_t>::max();
  std::int64_t allocation_delay_ms = 0;
  bool fail_all = false;
  bool invalid_delete = false;

  static Allocator::TimePoint now(void* context) noexcept {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    return Allocator::TimePoint{} + std::chrono::milliseconds(probe.milliseconds.load());
  }
  static void* allocate(void* context, std::size_t bytes) {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    probe.requests.push_back(bytes);
    if (probe.fail_all || bytes > probe.fail_above || probe.requests.size() > probe.fail_after)
      throw std::bad_alloc();
    probe.milliseconds.fetch_add(probe.allocation_delay_ms);
    void* data = nivo::detail::PageMemory::map(bytes);
    try { probe.live.push_back({data, bytes}); }
    catch (...) { nivo::detail::PageMemory::unmap(data, bytes); throw; }
    return data;
  }
  static void deallocate(void* context, void* data, std::size_t bytes) noexcept {
    AllocatorProbe& probe = *static_cast<AllocatorProbe*>(context);
    std::vector<SystemBlock>::iterator it = std::find_if(probe.live.begin(), probe.live.end(),
        [data](const SystemBlock& block) { return block.data == data; });
    if (it == probe.live.end()) { probe.invalid_delete = true; return; }
    if (it->bytes != bytes) probe.invalid_delete = true;
    const std::size_t mapped_bytes = it->bytes;
    probe.live.erase(it);
    nivo::detail::PageMemory::unmap(data, mapped_bytes);
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
bool value_aligned(const nivo::ReadHandle& handle) {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(handle.bytes().data());
  return address % alignof(std::max_align_t) == 0;
}
void value_storage_alignment() {
  Fixture f(1024 * 1024);
  for (std::size_t length = 1; length <= 40; ++length) {
    const std::string key(length, static_cast<char>('a' + length % 26));
    const std::string value(length * 3 + 1, 'v');
    const nivo::Result<nivo::ReadHandle> published = f.cache->put(f.pool, key, bytes(value));
    CHECK(published && value_aligned(published.value()));
    nivo::Result<nivo::WriteHandle> draft = f.cache->allocate(f.pool, key + "#", length);
    CHECK(draft);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(draft.value().bytes().data());
    CHECK(address % alignof(std::max_align_t) == 0);
  }
  for (std::size_t length = 1; length <= 40; ++length) {
    const std::string key(length, static_cast<char>('a' + length % 26));
    const nivo::Result<nivo::ReadHandle> found = f.cache->find(f.pool, key);
    CHECK(found && found.value().key() == key && value_aligned(found.value()));
    CHECK(text(found.value()) == std::string(length * 3 + 1, 'v'));
  }
  // Compaction copies entries into new storage and must keep the guarantee.
  Fixture sparse(16 * 1024 * 1024);
  populate_sparse(*sparse.cache, sparse.pool);
  CHECK(sparse.cache->compact(sparse.pool).value().moved_items > 0);
  verify_sparse(*sparse.cache, sparse.pool);
  for (std::size_t i = 0; i < 512; i += 32)
    CHECK(value_aligned(sparse.cache->peek(sparse.pool, compact_key(i)).value()));
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
  std::unique_ptr<nivo::Cache> cache =
      std::move(nivo::Cache::create({.clock = clock, .reaper = no_reaper})).value();
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
#if defined(_WIN32) || (defined(__APPLE__) && defined(__MACH__)) || defined(__linux__)
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
  CHECK(memory.kv_bytes == 100005 && memory.reserved_bytes ==
      nivo::detail::PageMemory::mapping_size(memory.kv_bytes + memory.entry_metadata_bytes));
  CHECK(memory.allocator_metadata_bytes == 0 && memory.internal_fragmentation_bytes == 0);
  CHECK(memory.unused_bytes == memory.reserved_bytes - memory.kv_bytes - memory.entry_metadata_bytes);
  CHECK(memory.unused_bytes < nivo::detail::PageMemory::page_size() && memory.reclaimable_bytes == 0);
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

void initial_reserve_configuration_and_monitoring() {
  constexpr std::size_t mib = 1024 * 1024;
  Fixture f(8 * mib);
  CHECK(f.cache->memory_stats(f.pool).value().reserved_bytes == 0);
  CHECK(nivo::PoolConfig{}.initial_reserve_bytes == 0);
  CHECK(nivo::PoolConfig{}.initial_reserve_timeout == 30s);
  CHECK(f.cache->add_pool({.name = "invalid", .capacity_bytes = mib,
      .initial_reserve_bytes = mib + 1}).status() == Status::invalid_argument);
  CHECK(f.cache->add_pool({.name = "invalid", .capacity_bytes = mib,
      .initial_reserve_bytes = 1, .initial_reserve_timeout = 0ms}).status() == Status::invalid_argument);
  CHECK(f.cache->add_pool({.name = "invalid", .capacity_bytes = mib,
      .initial_reserve_timeout = -1ms}).status() == Status::invalid_argument);
  CHECK(f.cache->add_pool({.name = "invalid", .capacity_bytes = mib,
      .initial_reserve_bytes = 1, .initial_reserve_timeout = nivo::Duration::max()}).status() ==
      Status::invalid_argument);
  CHECK(f.cache->stats().pools == 1 && f.cache->stats().assigned_bytes == 8 * mib);
  const nivo::PoolId pool = f.cache->add_pool({.name = "reserved", .capacity_bytes = 8 * mib,
      .initial_reserve_bytes = mib + 1}).value();
  CHECK(pool == f.pool + 1);
  const nivo::AllocatorStats prepared = f.cache->pool_stats(pool).value().allocator;
  CHECK(prepared.slabs == 17 && prepared.empty_slabs == 17 && prepared.regions == 3);
  CHECK(prepared.reserved_bytes > mib + 65536 && prepared.live_allocations == 0);
  const nivo::CacheMemoryStats sample = f.cache->memory_stats();
  CHECK(valid_memory_breakdown(sample.total));
  CHECK(sample.total.reserved_bytes == prepared.reserved_bytes);
  CHECK(sample.total.reclaimable_bytes == sample.total.reserved_bytes);
  CHECK(sample.total.unused_bytes > mib && sample.total.kv_bytes == 0);
  CHECK(f.cache->put(pool, "key", bytes("value")));
  CHECK(f.cache->pool_stats(pool).value().allocator.system_allocations == prepared.system_allocations);
  CHECK(f.cache->memory_stats(f.pool).value().reserved_bytes == 0);
  CHECK(f.cache->clear(pool) == Status::ok);
  CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);
  // Clearing cancels startup retention; it never recreates the initial reserve.
  CHECK(f.cache->put(pool, "again", bytes("value")));
  CHECK(f.cache->memory_stats(pool).value().reserved_bytes < 2 * 65536);
}

void initial_reserve_deadline_and_failure() {
  using Allocator = nivo::detail::MemoryAllocator;
  constexpr std::size_t mib = 1024 * 1024;
  AllocatorProbe probe;
  probe.allocation_delay_ms = 200;
  Allocator allocator(8 * mib, probe.hooks());
  allocator.reserve_initial(mib, 1000ms);
  const nivo::AllocatorStats prepared = allocator.inspect();
  CHECK(probe.milliseconds == 400 && prepared.regions == 2);
  CHECK(allocator.maintain_initial_reserve().value() == Allocator::TimePoint{} + 1400ms);
  probe.milliseconds = 1399;
  AllocatorBlocks blocks(allocator);
  blocks.add(256);
  blocks.clear();
  CHECK(allocator.stats().reserved_bytes == prepared.reserved_bytes);
  CHECK(allocator.stats().system_allocations == prepared.system_allocations);
  CHECK(allocator.maintain_initial_reserve().has_value());
  probe.milliseconds = 1400;
  CHECK(!allocator.maintain_initial_reserve());
  CHECK(allocator.inspect().reserved_bytes == 0 && probe.live.empty());
  blocks.add(256);
  CHECK(allocator.inspect().slabs == 1);
  blocks.clear();
  allocator.trim();

  AllocatorProbe failed;
  failed.fail_after = 2;
  Allocator broken(8 * mib, failed.hooks());
  bool threw = false;
  try { broken.reserve_initial(2 * mib, 1s); }
  catch (const std::bad_alloc&) { threw = true; }
  CHECK(threw && failed.requests.size() == 3);
  CHECK(failed.live.empty() && !failed.invalid_delete);
  CHECK(broken.inspect().reserved_bytes == 0 && broken.inspect().metadata_bytes == 0);
  CHECK(!broken.maintain_initial_reserve());
}

void initial_reserve_partial_use_resize_and_trim() {
  constexpr std::size_t mib = 1024 * 1024;
  AllocatorProbe probe;
  nivo::detail::MemoryAllocator allocator(8 * mib, probe.hooks());
  allocator.reserve_initial(4 * mib, 100ms);
  AllocatorBlocks blocks(allocator);
  blocks.add(15000);
  const nivo::AllocatorStats before = allocator.inspect();
  CHECK(before.regions == 8 && before.slabs == 64);
  probe.milliseconds = 100;
  CHECK(!allocator.maintain_initial_reserve());
  const nivo::AllocatorStats after = allocator.inspect();
  CHECK(after.regions == 1 && after.slabs == 8 && after.live_allocations == 1);
  CHECK(after.reserved_bytes < before.reserved_bytes / 4);
  CHECK(static_cast<std::byte*>(blocks.blocks[0].data)[14999] == std::byte{0x5a});
  blocks.clear();
  CHECK(allocator.inspect().reserved_bytes == 0);

  Fixture f(8 * mib);
  const nivo::PoolId pool = f.cache->add_pool({.name = "resize", .capacity_bytes = 8 * mib,
      .initial_reserve_bytes = 4 * mib}).value();
  CHECK(f.cache->resize_pool(pool, mib) == Status::ok);
  CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);
  CHECK(f.cache->resize_pool(pool, 8 * mib) == Status::ok);
  CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);
}

bool wait_for_reserve_release(nivo::Cache& cache, nivo::PoolId pool, std::size_t below) {
  const std::chrono::steady_clock::time_point limit = std::chrono::steady_clock::now() + 5s;
  do {
    // Pure observation cannot trigger cooling or reclamation.
    if (cache.memory_stats(pool).value().reserved_bytes < below) return true;
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < limit);
  return false;
}

void initial_reserve_background_idle_expiry() {
  constexpr std::size_t mib = 1024 * 1024;
  Fixture f(8 * mib);
  const nivo::PoolId later = f.cache->add_pool({.name = "later", .capacity_bytes = 4 * mib,
      .initial_reserve_bytes = 2 * mib, .initial_reserve_timeout = 30s}).value();
  // Let the existing worker enter its long wait before adding an earlier timer.
  std::this_thread::sleep_for(20ms);
  const nivo::PoolId sooner = f.cache->add_pool({.name = "sooner", .capacity_bytes = 4 * mib,
      .initial_reserve_bytes = 2 * mib, .initial_reserve_timeout = 100ms}).value();
  CHECK(f.cache->memory_stats(sooner).value().reserved_bytes > 2 * mib);
  // The item TTL clock is frozen; reserve deadlines use elapsed real time.
  CHECK(wait_for_reserve_release(*f.cache, sooner, 1));
  CHECK(f.clock->now() == 0ms);
  CHECK(f.cache->memory_stats(later).value().reserved_bytes > 2 * mib);
  CHECK(f.cache->put(sooner, "after", bytes("expiry")));
  CHECK(f.cache->memory_stats(sooner).value().reserved_bytes < 2 * 65536);
  f.cache->close();
  CHECK(f.cache->memory_stats().total.reserved_bytes == 0);
}

void initial_reserve_background_live_data_and_lifetime() {
  constexpr std::size_t mib = 1024 * 1024;
  nivo::ReadHandle pinned;
  {
    Fixture f(8 * mib);
    const nivo::PoolId pool = f.cache->add_pool({.name = "live", .capacity_bytes = 8 * mib,
        .initial_reserve_bytes = 4 * mib, .initial_reserve_timeout = 100ms}).value();
    pinned = std::move(f.cache->put(pool, "key", bytes("value"))).value();
    CHECK(wait_for_reserve_release(*f.cache, pool, mib));
    CHECK(text(pinned) == "value");
    const nivo::MemoryStats memory = f.cache->memory_stats(pool).value();
    CHECK(valid_memory_breakdown(memory) && memory.kv_bytes == 8 && memory.indexed_kv_bytes == 8);
    f.cache->close();
    CHECK(f.cache->memory_stats(pool).value().retained_kv_bytes == 8);
  }
  CHECK(text(pinned) == "value");
  std::thread releaser([handle = std::move(pinned)]() mutable { handle.reset(); });
  releaser.join();
  // Destruction must wake a worker whose deadline is still far in the future.
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  {
    Fixture f(8 * mib);
    CHECK(f.cache->add_pool({.name = "long", .capacity_bytes = 8 * mib,
        .initial_reserve_bytes = mib, .initial_reserve_timeout = 1h}));
  }
  CHECK(std::chrono::steady_clock::now() - start < 5s);
}

void initial_reserve_concurrent_expiry_and_compaction() {
  constexpr std::size_t mib = 1024 * 1024;
  Fixture f(8 * mib);
  const nivo::PoolId pool = f.cache->add_pool({.name = "concurrent", .capacity_bytes = 8 * mib,
      .initial_reserve_bytes = 4 * mib, .initial_reserve_timeout = 20ms}).value();
  std::atomic<bool> valid{true};
  std::vector<std::thread> writers;
  for (unsigned worker = 0; worker < 4; ++worker) {
    writers.emplace_back([&, worker] {
      const std::string key = std::to_string(worker);
      for (unsigned i = 0; i < 100; ++i) {
        nivo::Result<nivo::ReadHandle> result = f.cache->put(pool, key, bytes("value"));
        if (!result || text(result.value()) != "value") valid = false;
        if (!valid_memory_breakdown(f.cache->memory_stats(pool).value())) valid = false;
        std::this_thread::sleep_for(1ms);
      }
    });
  }
  for (std::thread& writer : writers) writer.join();
  CHECK(valid && wait_for_reserve_release(*f.cache, pool, mib));
  CHECK(f.cache->compact(pool));
  for (unsigned worker = 0; worker < 4; ++worker)
    CHECK(text(f.cache->peek(pool, std::to_string(worker)).value()) == "value");
  CHECK(f.cache->clear(pool) == Status::ok);
  CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);

  Fixture empty(8 * mib);
  const nivo::PoolId reserved = empty.cache->add_pool({.name = "compact", .capacity_bytes = 8 * mib,
      .initial_reserve_bytes = 4 * mib}).value();
  CHECK(empty.cache->compact(reserved));
  CHECK(empty.cache->memory_stats(reserved).value().reserved_bytes == 0);
}

#if defined(__unix__) || defined(__APPLE__)
struct MappedReserveProbe {
  using Allocator = nivo::detail::MemoryAllocator;
  struct Block { void* address; std::size_t bytes; };
  std::vector<Block> blocks;
  std::size_t cold_pages = 0;
  bool invalid = false;
  static std::size_t resident_pages(void* data, std::size_t bytes) {
    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#if defined(__APPLE__)
    std::vector<char> resident((bytes - 1) / page_size + 1);
#else
    std::vector<unsigned char> resident((bytes - 1) / page_size + 1);
#endif
    CHECK(mincore(data, bytes, resident.data()) == 0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < resident.size(); ++i)
      if ((resident[i] & 1) != 0) ++count;
    return count;
  }
  static void* allocate(void* context, std::size_t bytes) {
    MappedReserveProbe& probe = *static_cast<MappedReserveProbe*>(context);
    void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (data == MAP_FAILED) throw std::bad_alloc();
    try {
      const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
      probe.cold_pages += (bytes - 1) / page_size + 1 - resident_pages(data, bytes);
      probe.blocks.push_back({data, bytes});
    } catch (...) {
      munmap(data, bytes);
      throw;
    }
    return data;
  }
  static void deallocate(void* context, void* data, std::size_t bytes) noexcept {
    MappedReserveProbe& probe = *static_cast<MappedReserveProbe*>(context);
    const std::vector<Block>::iterator found = std::find_if(probe.blocks.begin(), probe.blocks.end(),
        [data](const Block& block) { return block.address == data; });
    if (found == probe.blocks.end()) { probe.invalid = true; return; }
    if (found->bytes != bytes) probe.invalid = true;
    if (munmap(data, found->bytes) != 0) probe.invalid = true;
    probe.blocks.erase(found);
  }
};
#endif

void initial_reserve_pages_are_resident() {
#if defined(__unix__) || defined(__APPLE__)
  using Allocator = nivo::detail::MemoryAllocator;
  MappedReserveProbe probe;
  Allocator::Hooks hooks;
  hooks.context = &probe;
  hooks.allocate = MappedReserveProbe::allocate;
  hooks.deallocate = MappedReserveProbe::deallocate;
  Allocator allocator(8 * 1024 * 1024, hooks);
  allocator.reserve_initial(2 * 1024 * 1024, 30s);
  CHECK(probe.cold_pages > 0);
  const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  for (const MappedReserveProbe::Block& block : probe.blocks)
    CHECK(MappedReserveProbe::resident_pages(block.address, block.bytes) == (block.bytes - 1) / page_size + 1);
  allocator.trim();
  CHECK(probe.blocks.empty() && !probe.invalid);
#endif
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


void index_collision_chain_and_ownership() {
  std::array<nivo::detail::Entry, 5> entries;
  nivo::detail::CacheIndex index(8192);
  std::vector<std::string> keys;
  for (std::size_t i = 0; keys.size() < 3; ++i) {
    const std::string key = "collision-" + std::to_string(i);
    if (index.bucket_for(key) == 0) keys.push_back(key);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    entries[i].key = keys[i];
    entries[i].charge = i + 1;
    index.write(keys[i], [&](nivo::detail::CacheIndex::Slot& slot) {
      CHECK(!slot.get());
      CHECK(!slot.replace(entries[i]));
    });
  }
  CHECK(index.stats().items == 3 && index.stats().bytes == 6);
  nivo::detail::EntryPtr pinned = index.find(keys[1]);
  CHECK(&*pinned == &entries[1] && pinned.use_count() == 3);
  entries[3].key = keys[1];
  entries[3].charge = 9;
  index.write(keys[1], [&](nivo::detail::CacheIndex::Slot& slot) {
    nivo::detail::EntryPtr old = slot.replace(entries[3]);
    CHECK(&*old == &entries[1]);
  });
  CHECK(&*index.find(keys[1]) == &entries[3]);
  CHECK(&*pinned == &entries[1] && pinned.use_count() == 2);
  CHECK(index.stats().items == 3 && index.stats().bytes == 13);
  // Expiry followed by insertion in one transaction must not mistake the next
  // collision-chain node for the removed key or accidentally replace that node.
  entries[4].key = keys[0];
  entries[4].charge = 7;
  index.write(keys[0], [&](nivo::detail::CacheIndex::Slot& slot) {
    nivo::detail::EntryPtr old = slot.remove();
    CHECK(&*old == &entries[0] && !slot.get());
    CHECK(!slot.replace(entries[4]));
  });
  CHECK(&*index.find(keys[0]) == &entries[4]);
  CHECK(&*index.find(keys[1]) == &entries[3]);
  CHECK(&*index.find(keys[2]) == &entries[2]);
  CHECK(index.stats().items == 3 && index.stats().bytes == 19);
  for (const std::string& key : keys) {
    index.write(key, [](nivo::detail::CacheIndex::Slot& slot) { CHECK(slot.remove()); });
    CHECK(!index.find(key));
  }
  CHECK(index.stats().items == 0 && index.stats().bytes == 0);
  pinned.reset();
  for (const nivo::detail::Entry& entry : entries) CHECK(entry.references.load() == 1);
}

void index_stripe_parallelism() {
  nivo::detail::CacheIndex index(8192);
  const std::string first = "first";
  std::string other;
  std::string conflict;
  for (std::size_t i = 0; other.empty() || conflict.empty(); ++i) {
    const std::string key = "stripe-" + std::to_string(i);
    if (index.bucket_for(key) % index.stripe_count == index.bucket_for(first) % index.stripe_count)
      conflict = key;
    else other = key;
  }
  std::promise<void> entered;
  std::future<void> held = entered.get_future();
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::thread holder([&] {
    index.write(first, [&](nivo::detail::CacheIndex::Slot&) {
      entered.set_value();
      released.wait();
    });
  });
  held.wait();
  std::future<void> independent = std::async(std::launch::async, [&] {
    index.write(other, [](nivo::detail::CacheIndex::Slot&) {});
  });
  std::future<void> blocked = std::async(std::launch::async, [&] {
    index.write(conflict, [](nivo::detail::CacheIndex::Slot&) {});
  });
  std::future<void> reader = std::async(std::launch::async, [&] { (void)index.find(first); });
  const bool parallel = independent.wait_for(2s) == std::future_status::ready;
  const bool writer_blocked = blocked.wait_for(20ms) == std::future_status::timeout;
  const bool reader_blocked = reader.wait_for(20ms) == std::future_status::timeout;
  release.set_value();
  holder.join();
  independent.get();
  blocked.get();
  reader.get();
  CHECK(parallel && writer_blocked && reader_blocked);
}

void concurrent_quota_reservations() {
  Fixture f(1024);
  std::array<nivo::WriteHandle, 16> drafts;
  std::array<std::size_t, 16> charges{};
  std::barrier gate(16);
  std::atomic<std::size_t> successes{0}, exhausted{0}, errors{0};
  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < drafts.size(); ++i) {
    workers.emplace_back([&, i] {
      const std::string key = "draft-" + std::to_string(i);
      gate.arrive_and_wait();
      nivo::Result<nivo::WriteHandle> allocated = f.cache->allocate(f.pool, key, 128 - key.size());
      if (allocated) {
        charges[i] = allocated.value().charge();
        drafts[i] = std::move(allocated).value();
        ++successes;
      } else if (allocated.status() == Status::resource_exhausted) ++exhausted;
      else ++errors;
    });
  }
  for (std::thread& worker : workers) worker.join();
  CHECK(successes == 8 && exhausted == 8 && errors == 0);
  std::size_t total = 0;
  for (std::size_t charge : charges) total += charge;
  CHECK(total == 1024 && f.cache->stats().allocated_bytes == total);
  for (nivo::WriteHandle& draft : drafts) draft.reset();
  CHECK(f.cache->stats().allocated_bytes == 0);
}

void concurrent_insert_only_and_erase() {
  Fixture f(1024 * 1024);
  std::barrier gate(12);
  std::atomic<std::size_t> inserted{0}, duplicates{0}, erased{0}, missing{0}, errors{0};
  std::vector<std::thread> writers;
  for (std::size_t i = 0; i < 12; ++i) {
    writers.emplace_back([&] {
      gate.arrive_and_wait();
      nivo::Result<nivo::ReadHandle> result = f.cache->put(f.pool, "same", bytes("value"),
          {.mode = nivo::WriteMode::insert_only});
      if (result) ++inserted;
      else if (result.status() == Status::already_exists) ++duplicates;
      else ++errors;
    });
  }
  for (std::thread& writer : writers) writer.join();
  CHECK(inserted == 1 && duplicates == 11 && errors == 0);
  nivo::ReadHandle pin = std::move(f.cache->peek(f.pool, "same")).value();
  writers.clear();
  for (std::size_t i = 0; i < 12; ++i) {
    writers.emplace_back([&] {
      gate.arrive_and_wait();
      const Status status = f.cache->erase(f.pool, "same", pin.version());
      if (status == Status::ok) ++erased;
      else if (status == Status::not_found) ++missing;
      else ++errors;
    });
  }
  for (std::thread& writer : writers) writer.join();
  CHECK(erased == 1 && missing == 11 && errors == 0 && text(pin) == "value");
  const nivo::PoolStats stats = f.cache->pool_stats(f.pool).value();
  CHECK(stats.indexed_items == 0 && stats.counters.inserts == 1 && stats.counters.erases == 1);
  pin.reset();
  CHECK(f.cache->stats().allocated_bytes == 0);
}

void deferred_recency_and_relocation() {
  std::array<nivo::detail::Entry, 4> entries;
  nivo::detail::EvictionPolicyEngine queue(nivo::EvictionPolicy::lru);
  const std::array<std::string_view, 3> keys{"a", "b", "c"};
  for (std::size_t i = 0; i < 3; ++i) {
    entries[i].key = keys[i];
    entries[i].version = i + 1;
    queue.insert(entries[i]);
  }
  std::vector<std::thread> readers;
  for (std::size_t i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      for (std::size_t j = 0; j < 1000; ++j) queue.record_access(entries[0]);
    });
  }
  for (std::thread& reader : readers) reader.join();
  queue.record_access(entries[1]);
  entries[3].key = entries[0].key;
  entries[3].version = entries[0].version;
  queue.relocate(entries[0], entries[3]);
  nivo::detail::VictimSelection ordered = queue.select_victims();
  CHECK(ordered.next() == nivo::detail::CacheIndex::identity(entries[2]));
  CHECK(ordered.next() == nivo::detail::CacheIndex::identity(entries[3]));
  CHECK(ordered.next() == nivo::detail::CacheIndex::identity(entries[1]));
  CHECK(!ordered.next());
  queue.remove(entries[2]);
  queue.remove(entries[3]);
  queue.remove(entries[1]);
  CHECK(!queue.select_victims().next());
  nivo::detail::EvictionPolicyEngine fifo(nivo::EvictionPolicy::fifo);
  fifo.insert(entries[0]);
  fifo.insert(entries[1]);
  fifo.record_access(entries[0]);
  CHECK(fifo.select_victims().next() == nivo::detail::CacheIndex::identity(entries[0]));
  fifo.remove(entries[0]);
  fifo.remove(entries[1]);
}

void policy_configuration_and_failed_publication() {
  Fixture f(64);
  const std::size_t assigned = f.cache->stats().assigned_bytes;
  CHECK(f.cache->add_pool({"invalid", 64, static_cast<nivo::EvictionPolicy>(99)}).status() ==
        Status::invalid_argument);
  CHECK(f.cache->stats().assigned_bytes == assigned);
  for (const nivo::EvictionPolicy kind : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    CHECK(nivo::detail::EvictionPolicyEngine::validate(kind) == Status::ok);
    nivo::detail::EvictionPolicyEngine policy(kind);
    CHECK(!policy.select_victims().next());
    Fixture configured(64);
    const nivo::PoolId pool = configured.cache->add_pool({"configured", 64, kind}).value();
    CHECK(configured.cache->put(pool, "a", bytes("aaa")));
    CHECK(configured.cache->put(pool, "b", bytes("bbb")));
    CHECK(configured.cache->find(pool, "a"));
    CHECK(configured.cache->put(pool, "a", bytes("wrong"),
        {.mode = nivo::WriteMode::insert_only}).status() == Status::already_exists);
    CHECK(configured.cache->put(pool, "missing", bytes("wrong"),
        {.mode = nivo::WriteMode::replace_only}).status() == Status::not_found);
    CHECK(configured.cache->resize_pool(pool, 4) == Status::ok);
    const std::string_view victim = kind == nivo::EvictionPolicy::lru ? "b" : "a";
    CHECK(configured.cache->peek(pool, victim).status() == Status::not_found);
    const nivo::PoolStats stats = configured.cache->pool_stats(pool).value();
    CHECK(stats.indexed_items == 1 && stats.allocated_bytes == 4);
    CHECK(stats.counters.inserts == 2 && stats.counters.replacements == 0 && stats.counters.evictions == 1);
    CHECK(configured.cache->clear(pool) == Status::ok);
    CHECK(configured.cache->put(pool, "c", bytes("ccc")));
    CHECK(configured.cache->put(pool, "d", bytes("ddd")));
    CHECK(configured.cache->peek(pool, "c").status() == Status::not_found);
  }
}

void policy_stable_candidates_and_removal_events() {
  for (const nivo::EvictionPolicy kind : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    // Stack ownership supplies one reference in addition to index references.
    std::array<nivo::detail::Entry, 4> entries;
    nivo::detail::EvictionPolicyEngine policy(kind);
    nivo::detail::CacheIndex index(8192);
    std::vector<std::string> keys;
    for (std::size_t i = 0; keys.size() < 2; ++i) {
      const std::string key = "candidate-" + std::to_string(i);
      if (index.bucket_for(key) == 0) keys.push_back(key);
    }
    for (std::size_t i = 0; i < 2; ++i) {
      entries[i].key = keys[i];
      entries[i].version = i + 1;
      index.write(entries[i].key, [&](nivo::detail::CacheIndex::Slot& slot) {
        CHECK(!slot.replace(entries[i]));
        policy.insert(entries[i]);
      });
    }
    nivo::detail::VictimSelection saved = policy.select_victims();
    const nivo::detail::EntryId first = *saved.next();
    CHECK(first == nivo::detail::CacheIndex::identity(entries[0]));

    // Simulate a compaction commit at a different address, with the same ID.
    entries[2].key = entries[0].key;
    entries[2].version = entries[0].version;
    index.write(entries[0].key, [&](nivo::detail::CacheIndex::Slot& slot) {
      policy.relocate(entries[0], entries[2]);
      CHECK(&*slot.replace(entries[2]) == &entries[0]);
    });
    bool resolved_moved_entry = false;
    CHECK(!index.remove_if(first, [&](nivo::detail::Entry& entry) {
      resolved_moved_entry = &entry == &entries[2];
      return false;  // Core may reject an otherwise valid candidate.
    }));
    CHECK(resolved_moved_entry);

    // Replacement is a new object identity, even though its key is unchanged.
    entries[3].key = entries[2].key;
    entries[3].version = 3;
    index.write(entries[2].key, [&](nivo::detail::CacheIndex::Slot& slot) {
      policy.remove(*slot.get());
      CHECK(&*slot.replace(entries[3]) == &entries[2]);
      policy.insert(entries[3]);
    });
    bool stale_was_resolved = false;
    CHECK(!index.remove_if(first, [&](nivo::detail::Entry&) {
      stale_was_resolved = true;
      return true;
    }));
    CHECK(!stale_was_resolved);
    const std::uint64_t new_order = entries[3].access_sequence.load();
    policy.record_access(entries[2]);  // Late access to a retained old version.
    CHECK(entries[3].access_sequence.load() == new_order);
    CHECK(entries[2].policy_previous == nullptr && entries[2].policy_next == nullptr);

    const nivo::detail::EntryId second = *saved.next();
    CHECK(index.remove_if(second, [&](nivo::detail::Entry& entry) {
      policy.remove(entry);
      return true;
    }));
    // Reuse the exact address and key; stale IDs must still fail (no ABA).
    entries[1].version = 4;
    index.write(entries[1].key, [&](nivo::detail::CacheIndex::Slot& slot) {
      CHECK(!slot.replace(entries[1]));
      policy.insert(entries[1]);
    });
    CHECK(!index.remove_if(second, [](nivo::detail::Entry&) { return true; }));
    CHECK(!index.remove_if({}, [](nivo::detail::Entry&) { return true; }));
    CHECK(!saved.next() && !saved.next());

    std::size_t removals = 0;
    index.clear([&](nivo::detail::Entry& entry) noexcept {
      policy.remove(entry);
      ++removals;
    });
    CHECK(removals == 2 && index.entries().empty());
    CHECK(!policy.select_victims().next());
    for (const nivo::detail::Entry& entry : entries) CHECK(entry.references.load() == 1);
  }
}

void policy_selection_outlives_removed_storage() {
  nivo::detail::CacheIndex index(8192);
  std::optional<nivo::detail::VictimSelection> saved;
  {
    nivo::detail::EvictionPolicyEngine policy(nivo::EvictionPolicy::lru);
    std::unique_ptr<nivo::detail::Entry> entry = std::make_unique<nivo::detail::Entry>();
    const std::string key(256, 'k');
    entry->key = key;
    entry->version = 1;
    index.write(key, [&](nivo::detail::CacheIndex::Slot& slot) {
      CHECK(!slot.replace(*entry));
      policy.insert(*entry);
    });
    saved.emplace(policy.select_victims());
    index.clear([&](nivo::detail::Entry& removed) noexcept { policy.remove(removed); });
  }
  const std::optional<nivo::detail::EntryId> candidate = saved->next();
  CHECK(candidate.has_value());
  CHECK(!index.remove_if(*candidate, [](nivo::detail::Entry&) { return true; }));
  CHECK(!saved->next());
}

void policy_pins_exhaustion_and_reclamation() {
  for (const nivo::EvictionPolicy kind : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    Fixture f(12);
    const nivo::PoolId pool = f.cache->add_pool({"pinned", 12, kind}).value();
    nivo::ReadHandle old = std::move(f.cache->put(pool, "a", bytes("old"))).value();
    nivo::ReadHandle current = std::move(f.cache->put(pool, "a", bytes("new"))).value();
    nivo::ReadHandle other = std::move(f.cache->put(pool, "b", bytes("bbb"))).value();
    CHECK(old.version() != current.version());
    CHECK(f.cache->put(pool, "c", bytes("ccc")).status() == Status::resource_exhausted);
    CHECK(f.cache->erase(pool, "a") == Status::ok);
    CHECK(f.cache->memory_stats(pool).value().retained_kv_bytes == 8);
    CHECK(f.cache->put(pool, "c", bytes("ccc")).status() == Status::resource_exhausted);
    CHECK(f.cache->pool_stats(pool).value().counters.evictions == 0);
    CHECK(text(old) == "old" && text(current) == "new");
    old.reset();
    nivo::ReadHandle inserted = std::move(f.cache->put(pool, "c", bytes("ccc"))).value();
    CHECK(f.cache->resize_pool(pool, 4) == Status::resource_exhausted);
    CHECK(f.cache->pool_stats(pool).value().capacity_bytes == 12);
    f.cache->close();
    CHECK(f.cache->pool_stats(pool).value().indexed_items == 0);
    CHECK(f.cache->memory_stats(pool).value().retained_kv_bytes == 12);
    current.reset();
    other.reset();
    inserted.reset();
    CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);
  }
}

void policy_destruction_with_surviving_entry() {
  for (const nivo::EvictionPolicy kind : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    std::weak_ptr<nivo::detail::MemoryAllocator> storage;
    nivo::detail::EntryPtr survivor;
    {
      nivo::detail::CachePool pool(1, {"lifetime", 4096, kind},
          std::make_shared<nivo::detail::Owner>(), std::make_shared<ManualClock>());
      nivo::Result<nivo::detail::EntryPtr> draft = pool.allocate("a", 3);
      CHECK(draft);
      draft.value()->value[0] = std::byte{42};
      survivor = std::move(pool.publish(draft.value(), {})).value();
      storage = survivor->budget;
      CHECK(survivor.use_count() == 2);
    }
    CHECK(!storage.expired() && survivor.use_count() == 1);
    CHECK(survivor->value[0] == std::byte{42});
    CHECK(survivor->policy_previous == nullptr && survivor->policy_next == nullptr);
    std::thread release([entry = std::move(survivor)]() mutable { entry.reset(); });
    release.join();
    CHECK(storage.expired());
  }
}

void policy_compaction_with_concurrent_pressure() {
  for (const nivo::EvictionPolicy kind : {nivo::EvictionPolicy::lru, nivo::EvictionPolicy::fifo}) {
    Fixture f(1024 * 1024);
    const nivo::PoolId pool = f.cache->add_pool({"pressure", 1024 * 1024, kind}).value();
    nivo::ReadHandle pinned = std::move(f.cache->put(pool, "pinned", bytes(std::string(15000, 'p')))).value();
    const std::byte* address = pinned.bytes().data();
    std::atomic<bool> failed{false};
    std::barrier gate(4);
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < 4; ++worker) {
      workers.emplace_back([&, worker] {
        gate.arrive_and_wait();
        try {
          for (std::size_t i = 0; i < 160; ++i) {
            const std::string key = "key-" + std::to_string(i % 64);
            if (worker < 2) {
              const std::string value(15000, static_cast<char>('a' + worker));
              nivo::Result<nivo::ReadHandle> result = f.cache->put(pool, key, bytes(value));
              if (!result) { CHECK(result.status() == Status::resource_exhausted); continue; }
              std::this_thread::yield();
              CHECK(text(result.value()) == value);
            } else if (worker == 2) {
              nivo::Result<nivo::ReadHandle> result = f.cache->find(pool, key);
              if (!result) { CHECK(result.status() == Status::not_found); continue; }
              const std::string value = text(result.value());
              CHECK(value.size() == 15000 && value.find_first_not_of(value.front()) == std::string::npos);
              std::this_thread::yield();
              CHECK(text(result.value()) == value);
            } else {
              const Status resized = f.cache->resize_pool(pool, i % 2 ? 1024 * 1024 : 128 * 1024);
              CHECK(resized == Status::ok || resized == Status::resource_exhausted);
              CHECK(f.cache->compact(pool));
              CHECK(pinned.bytes().data() == address && pinned.bytes().front() == std::byte{'p'});
            }
          }
        } catch (...) { failed = true; }
      });
    }
    for (std::thread& worker : workers) worker.join();
    CHECK(!failed);
    // Force pressure independently of scheduling, preserving the pinned entry.
    CHECK(f.cache->resize_pool(pool, pinned.charge()) == Status::ok);
    const nivo::PoolStats stats = f.cache->pool_stats(pool).value();
    CHECK(stats.counters.evictions > 0 && stats.indexed_items == 1);
    CHECK(text(pinned) == std::string(15000, 'p'));
    CHECK(f.cache->erase(pool, "pinned") == Status::ok);
    pinned.reset();
    CHECK(f.cache->clear(pool) == Status::ok);
    CHECK(f.cache->memory_stats(pool).value().reserved_bytes == 0);
  }
}

struct BlockingClock final : nivo::Clock {
  mutable std::atomic<bool> armed{false};
  mutable std::mutex mutex;
  mutable std::condition_variable wakeup;
  mutable bool entered = false;
  mutable bool released = false;
  nivo::Duration now() const noexcept override {
    if (armed.exchange(false)) {
      std::unique_lock<std::mutex> lock(mutex);
      entered = true;
      wakeup.notify_all();
      wakeup.wait(lock, [&] { return released; });
    }
    return 0ms;
  }
  bool wait_until_entered() const {
    std::unique_lock<std::mutex> lock(mutex);
    return wakeup.wait_for(lock, 2s, [&] { return entered; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    wakeup.notify_all();
  }
};

void pool_maintenance_isolation() {
  std::shared_ptr<BlockingClock> clock = std::make_shared<BlockingClock>();
  std::unique_ptr<nivo::Cache> cache =
      std::move(nivo::Cache::create({.clock = clock, .reaper = no_reaper})).value();
  const nivo::PoolId first = cache->add_pool({"first", 4096}).value();
  const nivo::PoolId second = cache->add_pool({"second", 4096}).value();
  CHECK(cache->put(first, "key", bytes("first")));
  CHECK(cache->put(second, "key", bytes("second")));
  clock->armed = true;
  // resize_pool() reads the clock while holding the exclusive pool guard.
  std::future<nivo::Status> maintenance = std::async(std::launch::async, [&] {
    return cache->resize_pool(first, 4096);
  });
  const bool entered = clock->wait_until_entered();
  std::future<nivo::Result<nivo::ReadHandle>> independent = std::async(std::launch::async, [&] {
    return cache->find(second, "key");
  });
  std::future<nivo::Result<nivo::ReadHandle>> same_pool = std::async(std::launch::async, [&] {
    return cache->find(first, "key");
  });
  const bool parallel = independent.wait_for(2s) == std::future_status::ready;
  const bool blocked = same_pool.wait_for(20ms) == std::future_status::timeout;
  clock->release();
  CHECK(maintenance.get() == Status::ok);
  CHECK(text(independent.get().value()) == "second");
  CHECK(text(same_pool.get().value()) == "first");
  CHECK(entered && parallel && blocked);
}

void pool_read_does_not_serialize_writes() {
  std::shared_ptr<BlockingClock> clock = std::make_shared<BlockingClock>();
  std::unique_ptr<nivo::Cache> cache =
      std::move(nivo::Cache::create({.clock = clock, .reaper = no_reaper})).value();
  const nivo::PoolId pool = cache->add_pool({"pool", 4096}).value();
  CHECK(cache->put(pool, "reader", bytes("old")));
  clock->armed = true;
  std::future<nivo::Result<nivo::ReadHandle>> reader = std::async(std::launch::async, [&] {
    return cache->find(pool, "reader");
  });
  const bool entered = clock->wait_until_entered();
  std::future<nivo::Result<nivo::ReadHandle>> writer = std::async(std::launch::async, [&] {
    return cache->put(pool, "writer", bytes("new"));
  });
  const bool parallel = writer.wait_for(2s) == std::future_status::ready;
  clock->release();
  CHECK(text(reader.get().value()) == "old" && text(writer.get().value()) == "new");
  CHECK(entered && parallel);
}

void concurrent_same_key_snapshots() {
  Fixture f(1024 * 1024);
  std::barrier gate(6);
  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  for (std::size_t worker = 0; worker < 6; ++worker) {
    workers.emplace_back([&, worker] {
      gate.arrive_and_wait();
      try {
        for (std::size_t i = 0; i < 1000; ++i) {
          if (worker < 3) {
            const std::string value(128, static_cast<char>('a' + worker));
            nivo::Result<nivo::ReadHandle> result = f.cache->put(f.pool, "shared", bytes(value));
            CHECK(result && text(result.value()) == value);
            if (i % 7 == 0) {
              const Status status = f.cache->erase(f.pool, "shared", result.value().version());
              CHECK(status == Status::ok || status == Status::not_found || status == Status::version_mismatch);
            }
            std::this_thread::yield();
            CHECK(text(result.value()) == value);
          } else {
            nivo::Result<nivo::ReadHandle> result = f.cache->find(f.pool, "shared");
            if (!result) { CHECK(result.status() == Status::not_found); continue; }
            const std::string value = text(result.value());
            CHECK(value.size() == 128 && value.find_first_not_of(value.front()) == std::string::npos);
            std::this_thread::yield();
            CHECK(text(result.value()) == value);
          }
        }
      } catch (...) { failed = true; }
    });
  }
  for (std::thread& worker : workers) worker.join();
  CHECK(!failed);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocated_bytes == 0);
}


void shared_mutex_reader_writer_handoff() {
  nivo::detail::SharedMutex mutex;
  std::shared_lock<nivo::detail::SharedMutex> first_reader(mutex);
  std::future<void> second_reader = std::async(std::launch::async, [&] {
    std::shared_lock<nivo::detail::SharedMutex> lock(mutex);
  });
  const bool readers_parallel = second_reader.wait_for(2s) == std::future_status::ready;
  std::promise<void> writer_entered;
  std::future<void> entered = writer_entered.get_future();
  std::promise<void> release_writer;
  std::shared_future<void> released = release_writer.get_future().share();
  std::thread writer([&] {
    std::unique_lock<nivo::detail::SharedMutex> lock(mutex);
    writer_entered.set_value();
    released.wait();
  });
  const bool writer_blocked = entered.wait_for(20ms) == std::future_status::timeout;
  first_reader.unlock();
  const bool writer_progress = entered.wait_for(2s) == std::future_status::ready;
  std::future<void> late_reader = std::async(std::launch::async, [&] {
    std::shared_lock<nivo::detail::SharedMutex> lock(mutex);
  });
  const bool late_blocked = late_reader.wait_for(20ms) == std::future_status::timeout;
  release_writer.set_value();
  writer.join();
  second_reader.get();
  late_reader.get();
  CHECK(readers_parallel && writer_blocked && writer_progress && late_blocked);

  std::size_t left = 0, right = 0;
  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < 6; ++i) {
    workers.emplace_back([&, i] {
      for (std::size_t j = 0; j < 1000; ++j) {
        if (i < 2) {
          std::unique_lock<nivo::detail::SharedMutex> lock(mutex);
          ++left;
          std::this_thread::yield();
          ++right;
        } else {
          std::shared_lock<nivo::detail::SharedMutex> lock(mutex);
          if (left != right) failed = true;
        }
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  CHECK(!failed && left == 2000 && right == 2000);
}

void concurrent_eviction_and_pool_control() {
  Fixture f(4096);
  std::atomic<bool> failed{false};
  std::barrier gate(5);
  std::vector<std::thread> workers;
  for (std::size_t worker = 0; worker < 5; ++worker) {
    workers.emplace_back([&, worker] {
      gate.arrive_and_wait();
      try {
        for (std::size_t i = 0; i < 400; ++i) {
          const std::string key = "key-" + std::to_string(i % 64);
          if (worker < 2) {
            const std::string value(64, static_cast<char>('a' + worker));
            nivo::Result<nivo::ReadHandle> result = f.cache->put(f.pool, key, bytes(value), {.ttl = 20ms});
            if (!result) { CHECK(result.status() == Status::resource_exhausted); continue; }
            std::this_thread::yield();
            CHECK(text(result.value()) == value);
          } else if (worker < 4) {
            nivo::Result<nivo::ReadHandle> result = f.cache->find(f.pool, key);
            if (!result) { CHECK(result.status() == Status::not_found); continue; }
            const std::string value = text(result.value());
            CHECK(value.size() == 64 && value.find_first_not_of(value.front()) == std::string::npos);
            std::this_thread::yield();
            CHECK(text(result.value()) == value);
          } else {
            f.clock->advance(1ms);
            const Status resized = f.cache->resize_pool(f.pool, i % 2 ? 4096 : 512);
            CHECK(resized == Status::ok || resized == Status::resource_exhausted);
            if (i % 5 == 0) CHECK(f.cache->clear(f.pool) == Status::ok);
            else if (i % 3 == 0) CHECK(f.cache->compact(f.pool));
            else CHECK(f.cache->prune_expired(f.pool));
            const nivo::PoolStats stats = f.cache->pool_stats(f.pool).value();
            CHECK(stats.allocated_bytes <= stats.capacity_bytes && stats.indexed_bytes <= stats.allocated_bytes);
          }
        }
      } catch (...) { failed = true; }
    });
  }
  for (std::thread& worker : workers) worker.join();
  CHECK(!failed);
  CHECK(f.cache->clear(f.pool) == Status::ok);
  CHECK(f.cache->stats().allocated_bytes == 0);
}

void version_source_saturates() {
  nivo::detail::Owner owner;
  owner.version.store(std::numeric_limits<nivo::Version>::max() - 1);
  CHECK(owner.next_version() == std::numeric_limits<nivo::Version>::max());
  CHECK(owner.next_version() == nivo::invalid_version);
  CHECK(owner.next_version() == nivo::invalid_version);
}

}  // namespace

// Randomized check of the timing wheel against a reference set, including
// bounded draining, large jumps and the overflow horizon.
void expiry_wheel_matches_reference() {
  using nivo::detail::Entry;
  using nivo::detail::ExpiryWheel;
  for (unsigned seed = 1; seed <= 40; ++seed) {
    std::mt19937_64 random(seed);
    const std::int64_t base = seed % 4 == 0 ? (std::int64_t{1} << 36) - 5000
                                            : static_cast<std::int64_t>(random() % 100000);
    std::vector<Entry> entries(500);
    std::set<std::size_t> live;
    ExpiryWheel wheel{nivo::Duration(base)};
    std::int64_t now = base;
    for (int step = 0; step < 5000; ++step) {
      const std::uint64_t operation = random() % 10;
      if (operation < 5) {
        const std::size_t index = static_cast<std::size_t>(random() % entries.size());
        if (live.erase(index) != 0) {
          wheel.remove(entries[index]);
          CHECK(entries[index].expiry_slot == nivo::detail::expiry_unlinked);
          continue;
        }
        const std::int64_t spans[] = {64, 5000, 10000000, 1000};
        const std::uint64_t kind = random() % 5;
        std::int64_t ttl = static_cast<std::int64_t>(random() % 3) - 1;  // at or before wheel time
        if (kind < 3) ttl = 1 + static_cast<std::int64_t>(random() % static_cast<std::uint64_t>(spans[kind]));
        if (kind == 3) ttl = (std::int64_t{1} << 36) + static_cast<std::int64_t>(random() % 1000);
        entries[index].expires_at = nivo::Duration(now + ttl);
        wheel.insert(entries[index]);
        live.insert(index);
      } else if (operation < 9) {
        const std::uint64_t jumps[] = {3, 300, 100000, 40000000};
        now += static_cast<std::int64_t>(random() % jumps[random() % 4]);
        wheel.advance(nivo::Duration(now));
        std::size_t limit = random() % 2 ? std::numeric_limits<std::size_t>::max()
                                         : static_cast<std::size_t>(random() % 50);
        for (; limit > 0; --limit) {
          Entry* entry = wheel.pop_pending();
          if (!entry) break;
          const std::size_t index = static_cast<std::size_t>(entry - entries.data());
          CHECK(live.count(index) == 1);
          if (entry->expires_at.count() <= now) live.erase(index);
          else wheel.insert(*entry);
        }
        // Once pending is drained, nothing that is due may remain hidden.
        if (!wheel.has_pending())
          for (std::size_t index : live) CHECK(entries[index].expires_at.count() > now);
      } else if (!live.empty()) {
        std::set<std::size_t>::iterator victim = live.begin();
        std::advance(victim, static_cast<std::ptrdiff_t>(random() % live.size()));
        wheel.remove(entries[*victim]);
        live.erase(victim);
      }
      CHECK(wheel.size() == live.size());
    }
    now += std::int64_t{1} << 40;
    wheel.advance(nivo::Duration(now));
    while (Entry* entry = wheel.pop_pending()) {
      CHECK(entry->expires_at.count() <= now);
      live.erase(static_cast<std::size_t>(entry - entries.data()));
    }
    CHECK(live.empty() && wheel.size() == 0);
  }
}
bool eventually(const std::function<bool()>& condition) {
  for (int attempt = 0; attempt < 5000; ++attempt) {
    if (condition()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return false;
}
void reaper_reclaims_in_background() {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
      {.capacity_bytes = 1 << 20, .clock = clock, .reaper = {.interval = 1ms}})).value();
  const nivo::PoolId pool = cache->add_pool({"pool", 1 << 20}).value();
  for (int i = 0; i < 2000; ++i) {
    const nivo::Duration ttl = i % 2 ? nivo::Duration(0) : nivo::Duration(1 + i % 500);
    CHECK(cache->put(pool, "k" + std::to_string(i), bytes("value"), {.ttl = ttl}));
  }
  nivo::ReadHandle pinned = std::move(cache->find(pool, "k0")).value();
  CHECK(cache->pool_stats(pool).value().ttl_items == 1000);
  // Item TTLs follow the cache clock, which is frozen here.
  CHECK(eventually([&] { return cache->stats().reaper.passes >= 3; }));
  CHECK(cache->pool_stats(pool).value().counters.expirations == 0);
  clock->advance(250ms);
  CHECK(eventually([&] { return cache->pool_stats(pool).value().indexed_items == 1500; }));
  clock->advance(1s);
  CHECK(eventually([&] { return cache->pool_stats(pool).value().indexed_items == 1000; }));
  const nivo::PoolStats stats = cache->pool_stats(pool).value();
  CHECK(stats.ttl_items == 0 && stats.counters.expirations == 1000);
  CHECK(cache->stats().reaper.reclaimed == 1000);
  CHECK(text(pinned) == "value");
  CHECK(cache->memory_stats(pool).value().retained_kv_bytes == pinned.charge());
  pinned.reset();
  CHECK(cache->memory_stats(pool).value().retained_kv_bytes == 0);
  CHECK(cache->find(pool, "k1"));
  cache->close();
}
void prune_expired_bounded_and_all_pools() {
  CHECK(nivo::Cache::create({.reaper = {.batch_size = 0}}).status() == Status::invalid_argument);
  CHECK(nivo::Cache::create({.reaper = {.interval = -1ms}}).status() == Status::invalid_argument);
  CHECK(nivo::Cache::create({.reaper = {.sleep_slice = -1ms}}).status() == Status::invalid_argument);
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
      {.capacity_bytes = 1 << 22, .clock = clock, .reaper = no_reaper})).value();
  const nivo::PoolId a = cache->add_pool({"a", 1 << 21}).value();
  const nivo::PoolId b = cache->add_pool({"b", 1 << 21}).value();
  for (int i = 0; i < 3000; ++i) {
    CHECK(cache->put(a, std::to_string(i), bytes("x"), {.ttl = 10ms}));
    CHECK(cache->put(b, std::to_string(i), bytes("x"), {.ttl = i < 1000 ? nivo::Duration(10ms) : nivo::Duration(1h)}));
  }
  CHECK(cache->prune_expired(a).value() == 0);
  clock->advance(10ms);
  std::size_t total = 0;
  for (int call = 0; call < 100; ++call) {
    const std::size_t pruned = cache->prune_expired(a, 7).value();
    CHECK(pruned <= 7);
    total += pruned;
  }
  CHECK(total == 700);
  CHECK(cache->prune_expired(a).value() == 2300);
  CHECK(cache->prune_expired().value() == 1000);
  CHECK(cache->pool_stats(b).value().ttl_items == 2000);
  // Replacing with a permanent value or erasing keeps TTL tracking exact.
  CHECK(cache->put(b, "1500", bytes("y")));
  CHECK(cache->erase(b, "1501") == Status::ok);
  CHECK(cache->pool_stats(b).value().ttl_items == 1998);
  clock->advance(2h);
  CHECK(cache->prune_expired(b).value() == 1998);
  CHECK(cache->find(b, "1500"));
  CHECK(cache->put(b, "z", bytes("z"), {.ttl = 1ms}));
  CHECK(cache->clear(b) == Status::ok);
  CHECK(cache->pool_stats(b).value().ttl_items == 0);
  CHECK(cache->put(b, "z", bytes("z"), {.ttl = 1ms}));
  clock->advance(1ms);
  CHECK(cache->prune_expired(b).value() == 1);
  CHECK(cache->stats().reaper.passes == 0);
  cache->close();
  CHECK(cache->prune_expired().status() == Status::closed);
}
void ttl_tracking_survives_compaction_and_pressure() {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
      {.capacity_bytes = 1 << 22, .clock = clock, .reaper = no_reaper})).value();
  const nivo::PoolId pool = cache->add_pool({"pool", 1 << 22}).value();
  const std::string value(512, 'v');
  for (int i = 0; i < 6000; ++i)
    CHECK(cache->put(pool, std::to_string(i), bytes(value), {.ttl = nivo::Duration(100 + i)}));
  for (int i = 0; i < 6000; i += 2) CHECK(cache->erase(pool, std::to_string(i)) == Status::ok);
  CHECK(cache->compact(pool).value().moved_items > 0);
  nivo::PoolStats stats = cache->pool_stats(pool).value();
  CHECK(stats.ttl_items == stats.indexed_items);
  clock->advance(nivo::Duration(100 + 3000));
  CHECK(cache->prune_expired(pool).value() == 1500);  // odd keys up to 2999
  stats = cache->pool_stats(pool).value();
  CHECK(stats.ttl_items == stats.indexed_items && stats.indexed_items == 1500);
  // Allocation pressure prunes the remaining expired entries before evicting.
  clock->advance(1h);
  for (int i = 0; i < 9000; ++i) CHECK(cache->put(pool, "n" + std::to_string(i), bytes(value)));
  stats = cache->pool_stats(pool).value();
  CHECK(stats.ttl_items == 0 && stats.counters.expirations == 3000);
}
void concurrent_reaping() {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create({.capacity_bytes = 1 << 21,
      .clock = clock, .reaper = {.interval = 1ms, .batch_size = 16, .work_slice = 1ms,
                                 .sleep_slice = 1ms}})).value();
  const nivo::PoolId pool = cache->add_pool({"pool", 1 << 20}).value();
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  for (unsigned id = 0; id < 4; ++id) {
    threads.emplace_back([&, id] {
      std::mt19937 random(id);
      const std::string value(200, 'a');
      while (!stop.load()) {
        const std::string key = std::to_string(random() % 4000);
        const std::uint32_t operation = random() % 8;
        if (operation < 2) {
          const nivo::Duration ttl(random() % 3 ? random() % 50 : 0);
          (void)cache->put(pool, key, bytes(std::string_view(value).substr(0, random() % 200)), {.ttl = ttl});
        } else if (operation == 2) {
          (void)cache->erase(pool, key);
        } else if (operation == 3) {
          (void)cache->peek(pool, key);
        } else {
          (void)cache->find(pool, key);
        }
      }
    });
  }
  threads.emplace_back([&] {
    while (!stop.load()) {
      clock->advance(1ms);
      std::this_thread::sleep_for(100us);
    }
  });
  threads.emplace_back([&] {
    std::mt19937 random(99);
    while (!stop.load()) {
      const std::uint32_t operation = random() % 4;
      if (operation == 0) (void)cache->prune_expired(pool, random() % 20);
      else if (operation == 1) (void)cache->prune_expired();
      else if (operation == 2) (void)cache->compact(pool);
      else (void)cache->resize_pool(pool, (1 << 19) + random() % (1 << 19));
      std::this_thread::sleep_for(500us);
    }
  });
  std::this_thread::sleep_for(1s);
  stop = true;
  for (std::thread& thread : threads) thread.join();
  CHECK(cache->stats().reaper.reclaimed > 0);
  clock->advance(1h);
  (void)cache->prune_expired(pool).value();
  CHECK(cache->pool_stats(pool).value().ttl_items == 0);
}

void background_compaction_after_reaping() {
  CHECK(nivo::Cache::create({.reaper = {.compact_unused_percent = 101}}).status() == Status::invalid_argument);
  CHECK(nivo::Cache::create({.reaper = {.compact_cooldown = -1ms}}).status() == Status::invalid_argument);
  // The same interleaved workload, with background compaction off and on.
  const std::string value(200, 'v');
  nivo::MemoryStats results[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
    nivo::ReaperConfig reaper{.interval = 1ms, .compact_unused_percent = enabled ? 30u : 0u,
                              .compact_min_unused_bytes = 0, .compact_cooldown = nivo::Duration(0)};
    std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
        {.capacity_bytes = 1 << 23, .clock = clock, .reaper = reaper})).value();
    const nivo::PoolId pool = cache->add_pool({"pool", 1 << 23}).value();
    for (int i = 0; i < 20000; ++i) {
      const nivo::Duration ttl = i % 2 ? nivo::Duration(0) : nivo::Duration(10ms);
      CHECK(cache->put(pool, std::to_string(i), bytes(value), {.ttl = ttl}));
    }
    const nivo::MemoryStats loaded = cache->memory_stats(pool).value();
    clock->advance(10ms);
    CHECK(eventually([&] { return cache->pool_stats(pool).value().ttl_items == 0; }));
    if (enabled) CHECK(eventually([&] { return cache->stats().reaper.compactions >= 1; }));
    else std::this_thread::sleep_for(20ms);
    results[enabled] = cache->memory_stats(pool).value();
    CHECK(results[enabled].kv_bytes * 2 <= loaded.kv_bytes + 64);
    CHECK(cache->stats().reaper.compactions == (enabled ? 1u : 0u));
    for (int i = 1; i < 20000; i += 2) CHECK(text(cache->peek(pool, std::to_string(i)).value()) == value);
    if (!enabled) CHECK(results[enabled].reserved_bytes == loaded.reserved_bytes);
  }
  CHECK(results[1].reserved_bytes < results[0].reserved_bytes * 3 / 4);
  CHECK(results[1].kv_bytes == results[0].kv_bytes);
}

int main() {
  const std::pair<const char*, std::function<void()>> cases[] = {
    {"index_collision_chain_and_ownership", index_collision_chain_and_ownership},
    {"index_stripe_parallelism", index_stripe_parallelism},
    {"concurrent_quota_reservations", concurrent_quota_reservations},
    {"concurrent_insert_only_and_erase", concurrent_insert_only_and_erase},
    {"deferred_recency_and_relocation", deferred_recency_and_relocation},
    {"policy_configuration_and_failed_publication", policy_configuration_and_failed_publication},
    {"policy_stable_candidates_and_removal_events", policy_stable_candidates_and_removal_events},
    {"policy_selection_outlives_removed_storage", policy_selection_outlives_removed_storage},
    {"policy_pins_exhaustion_and_reclamation", policy_pins_exhaustion_and_reclamation},
    {"policy_destruction_with_surviving_entry", policy_destruction_with_surviving_entry},
    {"policy_compaction_with_concurrent_pressure", policy_compaction_with_concurrent_pressure},
    {"pool_maintenance_isolation", pool_maintenance_isolation},
    {"pool_read_does_not_serialize_writes", pool_read_does_not_serialize_writes},
    {"concurrent_same_key_snapshots", concurrent_same_key_snapshots},
    {"version_source_saturates", version_source_saturates},
    {"shared_mutex_reader_writer_handoff", shared_mutex_reader_writer_handoff},
    {"concurrent_eviction_and_pool_control", concurrent_eviction_and_pool_control},
    {"configuration", configuration}, {"unpublished_and_aborted", unpublished_and_aborted},
    {"publish_and_snapshot", publish_and_snapshot}, {"write_conditions_and_retry", write_conditions_and_retry},
    {"ttl_boundary_and_publish_time", ttl_boundary_and_publish_time}, {"expiry_maintenance", expiry_maintenance},
    {"allocation_prunes_only_under_pressure", allocation_prunes_only_under_pressure},
    {"expiry_wheel_matches_reference", expiry_wheel_matches_reference},
    {"reaper_reclaims_in_background", reaper_reclaims_in_background},
    {"prune_expired_bounded_and_all_pools", prune_expired_bounded_and_all_pools},
    {"ttl_tracking_survives_compaction_and_pressure", ttl_tracking_survives_compaction_and_pressure},
    {"concurrent_reaping", concurrent_reaping},
    {"background_compaction_after_reaping", background_compaction_after_reaping},
    {"lru_and_peek", lru_and_peek}, {"fifo_policy", fifo_policy},
    {"pins_and_capacity", pins_and_capacity}, {"update_protects_old_key", update_protects_old_key},
    {"eviction_skips_pins", eviction_skips_pins}, {"pool_isolation_resize", pool_isolation_resize},
    {"foreign_and_empty_handles", foreign_and_empty_handles}, {"binary_keys_and_validation", binary_keys_and_validation},
    {"close_and_lifetime", close_and_lifetime}, {"destructor_without_close", destructor_without_close},
    {"result_contract", result_contract}, {"concurrent_cas", concurrent_cas},
    {"concurrent_operations", concurrent_operations}, {"concurrent_close", concurrent_close},
    {"allocator_size_boundaries", allocator_size_boundaries},
    {"page_memory_mapping_contract", page_memory_mapping_contract},
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
    {"initial_reserve_configuration_and_monitoring", initial_reserve_configuration_and_monitoring},
    {"initial_reserve_deadline_and_failure", initial_reserve_deadline_and_failure},
    {"initial_reserve_partial_use_resize_and_trim", initial_reserve_partial_use_resize_and_trim},
    {"initial_reserve_background_idle_expiry", initial_reserve_background_idle_expiry},
    {"initial_reserve_background_live_data_and_lifetime", initial_reserve_background_live_data_and_lifetime},
    {"initial_reserve_concurrent_expiry_and_compaction", initial_reserve_concurrent_expiry_and_compaction},
    {"initial_reserve_pages_are_resident", initial_reserve_pages_are_resident},
    {"allocator_compaction_skips_marginal_savings", allocator_compaction_skips_marginal_savings},
    {"value_storage_alignment", value_storage_alignment}
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
