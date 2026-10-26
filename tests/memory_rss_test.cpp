#include <nivo/nivo.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t mib = 1024 * 1024;
constexpr std::size_t workload_bytes = 128 * mib;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Sample {
  std::size_t rss;
  std::size_t mapped;
};

Sample sample(nivo::Cache& cache, std::string_view stage, unsigned round) {
  const nivo::CacheMemoryStats memory = cache.memory_stats();
  require(memory.process_rss_bytes.has_value(), "Process RSS is unavailable");
  const Sample result{*memory.process_rss_bytes, memory.total.reserved_bytes};
  std::cout << "round=" << round << " stage=" << stage << std::fixed << std::setprecision(2)
            << " rss_mib=" << static_cast<double>(result.rss) / static_cast<double>(mib)
            << " mapped_mib=" << static_cast<double>(result.mapped) / static_cast<double>(mib)
            << std::endl;
  return result;
}

void verify_release(Sample baseline, Sample loaded, Sample released) {
  require(loaded.rss > baseline.rss + workload_bytes / 2,
          "The write-touched workload did not produce a measurable RSS increase");
  require(released.mapped == 0, "Item mappings remain after complete release");
  require(loaded.rss > released.rss &&
              loaded.rss - released.rss >= (loaded.rss - baseline.rss) * 4 / 5,
          "RSS did not shed at least 80 percent of the measured workload increase");
  // Heap-based index nodes, runtime initialization and RSS sampling are outside
  // item mappings. Allow bounded noise instead of requiring byte-exact RSS.
  require(released.rss <= baseline.rss + 16 * mib,
          "RSS remained more than 16 MiB above the pre-workload baseline");
}

std::unique_ptr<nivo::Cache> create_cache() {
  nivo::Result<std::unique_ptr<nivo::Cache>> result =
      nivo::Cache::create({.capacity_bytes = 256 * mib});
  require(result.has_value(), "Cache creation failed");
  return std::move(result).value();
}

void run_items(bool pinned) {
  std::unique_ptr<nivo::Cache> cache = create_cache();
  const nivo::PoolId pool = cache->add_pool({.name = "rss", .capacity_bytes = 256 * mib}).value();
  const std::size_t value_bytes = pinned ? mib + 1 : 8192;
  const std::size_t count = workload_bytes / value_bytes + 1;
  std::vector<nivo::ReadHandle> handles;
  if (pinned) handles.reserve(count);
  const Sample initial = sample(*cache, "initial", 0);
  for (unsigned round = 1; round <= 3; ++round) {
    const Sample baseline = sample(*cache, "baseline", round);
    for (std::size_t i = 0; i < count; ++i) {
      nivo::Result<nivo::WriteHandle> allocation = cache->allocate(pool, std::to_string(i), value_bytes);
      require(allocation.has_value(), "Workload allocation failed");
      nivo::WriteHandle draft = std::move(allocation).value();
      // Write the full value, not just its virtual reservation. Pattern changes
      // across items and rounds also make stale-data mistakes observable.
      const std::byte pattern{static_cast<unsigned char>((i + round) % 251 + 1)};
      std::fill(draft.bytes().begin(), draft.bytes().end(), pattern);
      nivo::Result<nivo::ReadHandle> published = cache->publish(std::move(draft));
      require(published.has_value(), "Workload publication failed");
      if (pinned) handles.push_back(std::move(published).value());
    }
    const Sample loaded = sample(*cache, "loaded", round);
    if (pinned) {
      require(cache->clear(pool) == nivo::Status::ok, "Clear failed");
      const Sample retained = sample(*cache, "cleared_with_handles", round);
      require(retained.mapped == loaded.mapped, "Pinned item storage was released early");
      for (std::size_t i = 0; i < handles.size(); ++i) {
        const std::byte expected{static_cast<unsigned char>((i + round) % 251 + 1)};
        require(handles[i].bytes().front() == expected && handles[i].bytes().back() == expected,
                "Pinned value changed after clear");
      }
      // Exercise the final releases from a thread other than the allocator.
      std::thread releaser([&handles] { handles.clear(); });
      releaser.join();
    } else {
      for (std::size_t i = 0; i < count; ++i)
        require(cache->erase(pool, std::to_string(i)) == nivo::Status::ok, "Erase failed");
      (void)sample(*cache, "erased_before_trim", round);
      require(cache->compact(pool).has_value(), "Idle-region trim failed");
    }
    const Sample released = sample(*cache, "released", round);
    verify_release(baseline, loaded, released);
    require(released.rss <= initial.rss + 16 * mib, "RSS accumulated across workload cycles");
  }
}

nivo::CompactionStats compact_and_measure(nivo::Cache& cache, nivo::PoolId pool,
                                         std::string_view stage, unsigned round) {
  const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  const nivo::Result<nivo::CompactionStats> result = cache.compact(pool);
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  require(result.has_value(), "Compaction failed");
  std::cout << "round=" << round << " stage=" << stage
            << " moved=" << result.value().moved_items
            << " skipped=" << result.value().skipped_items
            << " elapsed_ms=" << elapsed_ms << std::endl;
  return result.value();
}

void run_compaction(bool pin_survivors) {
  constexpr std::size_t value_bytes = 15000;
  constexpr std::size_t stride = 32;
  constexpr std::size_t count = (workload_bytes / value_bytes / stride + 1) * stride;
  constexpr std::size_t survivors = count / stride;
  std::unique_ptr<nivo::Cache> cache = create_cache();
  const nivo::PoolId pool = cache->add_pool({.name = "compaction", .capacity_bytes = 256 * mib}).value();
  std::vector<nivo::Version> versions(survivors);
  std::vector<nivo::ReadHandle> pins;
  std::vector<const std::byte*> addresses;
  pins.reserve(survivors);
  addresses.reserve(survivors);
  for (unsigned round = 1; round <= 3; ++round) {
    const Sample baseline = sample(*cache, "baseline", round);
    for (std::size_t i = 0; i < count; ++i) {
      nivo::WriteHandle draft = std::move(cache->allocate(pool, std::to_string(i), value_bytes)).value();
      const std::byte pattern{static_cast<unsigned char>((i + round) % 251 + 1)};
      std::fill(draft.bytes().begin(), draft.bytes().end(), pattern);
      nivo::ReadHandle published = std::move(cache->publish(std::move(draft))).value();
      if (i % stride == 0) versions[i / stride] = published.version();
    }
    const Sample loaded = sample(*cache, "loaded", round);
    const nivo::CompactionStats dense = compact_and_measure(*cache, pool, "dense_control", round);
    require(dense.moved_items == 0, "Dense data was moved without a worthwhile saving");

    // Retain scattered entries, keeping partially occupied regions alive.
    // This must exercise relocation, not just trimming wholly empty regions.
    for (std::size_t i = 0; i < count; ++i) {
      if (i % stride != 0)
        require(cache->erase(pool, std::to_string(i)) == nivo::Status::ok, "Sparse erase failed");
      else if (pin_survivors) {
        pins.push_back(std::move(cache->peek(pool, std::to_string(i))).value());
        addresses.push_back(pins.back().bytes().data());
      }
    }
    const Sample sparse = sample(*cache, "sparse_before_compaction", round);
    const nivo::MemoryStats live_before = cache->memory_stats(pool).value();
    require(sparse.mapped > loaded.mapped / 2, "Sparse workload did not retain fragmented regions");
    require(live_before.reclaimable_bytes < sparse.mapped / 10,
            "Too much storage was already reclaimable without relocation");
    require(cache->pool_stats(pool).value().indexed_items == survivors, "Unexpected survivor count");
    std::cout << "round=" << round << " survivors=" << survivors << " inserted=" << count
              << " surviving_value_mib=" << static_cast<double>(survivors * value_bytes) /
                  static_cast<double>(mib) << std::endl;

    if (pin_survivors) {
      const nivo::CompactionStats blocked = compact_and_measure(*cache, pool, "pinned_control", round);
      const Sample retained = sample(*cache, "compacted_while_pinned", round);
      require(blocked.moved_items == 0 && blocked.skipped_items == survivors,
              "Compaction did not skip all pinned entries");
      require(retained.mapped == sparse.mapped - live_before.reclaimable_bytes,
              "Pinned compaction released more than wholly idle regions");
      for (std::size_t i = 0; i < pins.size(); ++i)
        require(pins[i].bytes().data() == addresses[i] && pins[i].version() == versions[i],
                "Pinned address or version changed");
      pins.clear();
      addresses.clear();
    }

    const nivo::CompactionStats compacted = compact_and_measure(*cache, pool, "compact_survivors", round);
    const Sample packed = sample(*cache, "after_compaction", round);
    require(compacted.moved_items == survivors && compacted.skipped_items == 0,
            "Compaction did not relocate every movable survivor");
    require(packed.mapped < sparse.mapped / 4, "Compaction did not reclaim fragmented mappings");
    require(sparse.rss > packed.rss && sparse.rss - packed.rss >= sparse.mapped / 2,
            "RSS did not fall substantially after compaction");
    const nivo::MemoryStats live_after = cache->memory_stats(pool).value();
    require(live_after.kv_bytes == live_before.kv_bytes &&
                live_after.indexed_kv_bytes == live_before.indexed_kv_bytes,
            "Compaction changed live data accounting");
    for (std::size_t i = 0; i < count; ++i) {
      const nivo::Result<nivo::ReadHandle> found = cache->peek(pool, std::to_string(i));
      if (i % stride != 0) {
        require(found.status() == nivo::Status::not_found, "Compaction resurrected an erased key");
        continue;
      }
      require(found.has_value(), "Compaction lost a live key");
      const std::byte expected{static_cast<unsigned char>((i + round) % 251 + 1)};
      require(found.value().version() == versions[i / stride] && found.value().bytes().size() == value_bytes &&
                  std::all_of(found.value().bytes().begin(), found.value().bytes().end(),
                      [expected](std::byte value) { return value == expected; }),
              "Compaction changed the version or payload");
    }
    const nivo::CompactionStats repeated = compact_and_measure(*cache, pool, "already_packed", round);
    require(repeated.moved_items == 0 && cache->memory_stats(pool).value().reserved_bytes == packed.mapped,
            "Repeated compaction moved already packed data");
    require(cache->clear(pool) == nivo::Status::ok, "Final clear failed");
    verify_release(baseline, loaded, sample(*cache, "released", round));
  }
}

void run_reserve() {
  std::unique_ptr<nivo::Cache> cache = create_cache();
  const Sample baseline = sample(*cache, "baseline", 1);
  const nivo::PoolId pool = cache->add_pool({.name = "reserve", .capacity_bytes = 256 * mib,
      .initial_reserve_bytes = workload_bytes,
      .initial_reserve_timeout = std::chrono::milliseconds{1000}}).value();
  const Sample loaded = sample(*cache, "reserve_touched", 1);
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (cache->memory_stats(pool).value().reserved_bytes != 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  // memory_stats() only observes: the background expiry must do the unmapping.
  const Sample released = sample(*cache, "reserve_expired", 1);
  verify_release(baseline, loaded, released);
}

// Lets a scenario expire every TTL entry at once, independent of load speed.
struct ManualClock final : nivo::Clock {
  std::atomic<nivo::Duration::rep> ticks{0};
  nivo::Duration now() const noexcept override { return nivo::Duration(ticks.load()); }
  void advance(nivo::Duration delta) { ticks.fetch_add(delta.count()); }
};

std::unique_ptr<nivo::Cache> create_reaping_cache(std::uint32_t compact_unused_percent,
                                                  std::shared_ptr<const nivo::Clock> clock = nullptr) {
  nivo::Result<std::unique_ptr<nivo::Cache>> result = nivo::Cache::create({.capacity_bytes = 256 * mib,
      .clock = std::move(clock),
      .reaper = {.interval = std::chrono::milliseconds{50},
                 .compact_unused_percent = compact_unused_percent,
                 .compact_cooldown = std::chrono::milliseconds{0}}});
  require(result.has_value(), "Cache creation failed");
  return std::move(result).value();
}

bool wait_for(const std::function<bool()>& condition) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (!condition()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
  return true;
}

void put_pattern(nivo::Cache& cache, nivo::PoolId pool, std::size_t i, unsigned round,
                 std::size_t value_bytes, nivo::Duration ttl) {
  nivo::Result<nivo::WriteHandle> allocation = cache.allocate(pool, std::to_string(i), value_bytes);
  require(allocation.has_value(), "Workload allocation failed");
  nivo::WriteHandle draft = std::move(allocation).value();
  const std::byte pattern{static_cast<unsigned char>((i + round) % 251 + 1)};
  std::fill(draft.bytes().begin(), draft.bytes().end(), pattern);
  require(cache.publish(std::move(draft), {.ttl = ttl}).has_value(), "Workload publication failed");
}

// Every entry expires; the background reaper alone must unmap all item storage
// without any erase, clear, prune or compact call.
void run_ttl() {
  constexpr std::size_t value_bytes = 8192;
  constexpr std::size_t count = workload_bytes / value_bytes + 1;
  std::unique_ptr<nivo::Cache> cache = create_reaping_cache(0);
  const nivo::PoolId pool = cache->add_pool({.name = "ttl", .capacity_bytes = 256 * mib}).value();
  const Sample initial = sample(*cache, "initial", 0);
  for (unsigned round = 1; round <= 3; ++round) {
    const Sample baseline = sample(*cache, "baseline", round);
    for (std::size_t i = 0; i < count; ++i)
      put_pattern(*cache, pool, i, round, value_bytes, std::chrono::milliseconds{300});
    const Sample loaded = sample(*cache, "loaded", round);
    require(wait_for([&] { return cache->memory_stats(pool).value().reserved_bytes == 0; }),
            "Background reaping did not unmap expired storage");
    require(cache->pool_stats(pool).value().indexed_items == 0, "Expired entries remain indexed");
    const Sample released = sample(*cache, "reaped", round);
    verify_release(baseline, loaded, released);
    require(released.rss <= initial.rss + 16 * mib, "RSS accumulated across workload cycles");
  }
  require(cache->stats().reaper.reclaimed == 3 * count, "Reaper did not reclaim every entry");
}

// Expiring entries interleave with permanent ones, so reaping alone leaves
// every slab partially live. Background compaction must return the space.
// The cache clock stays frozen during the load, so all TTL entries expire
// together; otherwise slow builds compact before the last ones expire.
void run_ttl_compaction() {
  constexpr std::size_t value_bytes = 4000;
  constexpr std::size_t count = workload_bytes / value_bytes / 2 * 2;
  const std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache = create_reaping_cache(30, clock);
  const nivo::PoolId pool = cache->add_pool({.name = "ttl_compaction", .capacity_bytes = 256 * mib}).value();
  for (unsigned round = 1; round <= 3; ++round) {
    const Sample baseline = sample(*cache, "baseline", round);
    const std::uint64_t compactions = cache->stats().reaper.compactions;
    for (std::size_t i = 0; i < count; ++i)
      put_pattern(*cache, pool, i, round, value_bytes,
                  i % 2 == 0 ? nivo::Duration(std::chrono::milliseconds{300}) : nivo::Duration(0));
    const Sample loaded = sample(*cache, "loaded", round);
    clock->advance(std::chrono::milliseconds{300});
    require(wait_for([&] { return cache->stats().reaper.compactions > compactions; }),
            "Background compaction did not run after reaping");
    const Sample compacted = sample(*cache, "reaped_and_compacted", round);
    require(cache->pool_stats(pool).value().indexed_items == count / 2, "Unexpected survivor count");
    require(compacted.mapped < loaded.mapped * 2 / 3, "Compaction did not release fragmented mappings");
    // Background compaction must leave nothing a manual pass could still gain.
    const nivo::CompactionStats repeated = compact_and_measure(*cache, pool, "manual_after_background", round);
    require(repeated.moved_items == 0 && cache->memory_stats(pool).value().reserved_bytes == compacted.mapped,
            "Background compaction left work for a manual pass");
    require(loaded.rss > compacted.rss && loaded.rss - compacted.rss >= (loaded.rss - baseline.rss) / 3,
            "RSS did not fall after background compaction");
    for (std::size_t i = 1; i < count; i += 2) {
      const nivo::Result<nivo::ReadHandle> found = cache->peek(pool, std::to_string(i));
      const std::byte expected{static_cast<unsigned char>((i + round) % 251 + 1)};
      require(found.has_value() && found.value().bytes().size() == value_bytes &&
                  found.value().bytes().front() == expected && found.value().bytes().back() == expected,
              "Background compaction changed a surviving payload");
    }
    require(cache->clear(pool) == nivo::Status::ok, "Clear failed");
    verify_release(baseline, loaded, sample(*cache, "released", round));
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Expected one scenario: slabs, pinned, reserve, compaction, compaction_pinned, ttl, or ttl_compaction");
    const std::string_view scenario(argv[1]);
    if (scenario == "slabs") run_items(false);
    else if (scenario == "pinned") run_items(true);
    else if (scenario == "reserve") run_reserve();
    else if (scenario == "compaction") run_compaction(false);
    else if (scenario == "compaction_pinned") run_compaction(true);
    else if (scenario == "ttl") run_ttl();
    else if (scenario == "ttl_compaction") run_ttl_compaction();
    else throw std::runtime_error("Unknown RSS scenario");
    std::cout << "PASS " << scenario << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
