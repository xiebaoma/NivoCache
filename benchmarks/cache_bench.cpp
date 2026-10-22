#include <benchmark/benchmark.h>
#include <nivo/nivo.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if NIVO_BENCHMARK_WITH_CACHELIB
#include <cachelib/allocator/CacheAllocator.h>
#include <cachelib/allocator/CacheAllocatorConfig.h>
#endif

#ifndef NIVO_BENCHMARK_NIVO_REVISION
#define NIVO_BENCHMARK_NIVO_REVISION "unknown"
#endif

#ifndef NIVO_BENCHMARK_CACHELIB_REVISION
#define NIVO_BENCHMARK_CACHELIB_REVISION "not-built"
#endif

#ifndef NIVO_BENCHMARK_COMPILER
#define NIVO_BENCHMARK_COMPILER "unknown"
#endif

namespace {
constexpr std::size_t kCacheCapacityBytes = 64U * 1024U * 1024U;
constexpr std::size_t kWorkingSetSize = 16384U;
constexpr std::size_t kValueSize = 128U;
constexpr std::size_t kWorkingSetMask = kWorkingSetSize - 1U;
constexpr std::size_t kWriteEvery = 20U;
static_assert((kWorkingSetSize & kWorkingSetMask) == 0U,
              "The working set size must be a power of two");

struct WorkloadData {
  std::vector<std::string> hit_keys;
  std::vector<std::string> miss_keys;
  std::array<std::byte, kValueSize> value{};

  WorkloadData() {
    hit_keys.reserve(kWorkingSetSize);
    miss_keys.reserve(kWorkingSetSize);
    for (std::size_t index = 0; index < kWorkingSetSize; ++index) {
      hit_keys.push_back("hit-" + std::to_string(index));
      miss_keys.push_back("miss-" + std::to_string(index));
    }
    value.fill(std::byte{0x5a});
  }
};

const WorkloadData &workload_data() {
  static const WorkloadData data;
  return data;
}

class NivoAdapter final {
 public:
  NivoAdapter() {
    nivo::CacheConfig config;
    config.capacity_bytes = kCacheCapacityBytes;
    nivo::Result<std::unique_ptr<nivo::Cache>> created =
        nivo::Cache::create(config);
    if (!created) {
      throw std::runtime_error("NivoCache creation failed");
    }
    cache_ = std::move(created).value();
    nivo::Result<nivo::PoolId> pool = cache_->add_pool(
        {"benchmark", kCacheCapacityBytes, nivo::EvictionPolicy::lru});
    if (!pool) {
      throw std::runtime_error("NivoCache pool creation failed");
    }
    pool_ = pool.value();
    populate();
  }

  bool find(std::string_view key, std::size_t &checksum) {
    nivo::Result<nivo::ReadHandle> result = cache_->find(pool_, key);
    if (!result) {
      return false;
    }
    const std::span<const std::byte> bytes = result.value().bytes();
    checksum += bytes.size();
    benchmark::DoNotOptimize(bytes.data());
    return true;
  }

  bool miss(std::string_view key) {
    nivo::Result<nivo::ReadHandle> result = cache_->find(pool_, key);
    benchmark::DoNotOptimize(result.status());
    return result.status() == nivo::Status::not_found;
  }

  bool write(std::string_view key) {
    const WorkloadData &data = workload_data();
    nivo::Result<nivo::ReadHandle> result =
        cache_->put(pool_, key, std::span<const std::byte>(data.value));
    if (!result) {
      return false;
    }
    benchmark::DoNotOptimize(result.value().bytes().data());
    return true;
  }

 private:
  void populate() {
    const WorkloadData &data = workload_data();
    for (const std::string &key : data.hit_keys) {
      if (!write(key)) {
        throw std::runtime_error("NivoCache population failed");
      }
    }
  }

  std::unique_ptr<nivo::Cache> cache_;
  nivo::PoolId pool_ = nivo::invalid_pool;
};

#if NIVO_BENCHMARK_WITH_CACHELIB
class CacheLibAdapter final {
 public:
  using Cache = facebook::cachelib::LruAllocator;

  CacheLibAdapter() : cache_(make_config()) {
    Cache::MMConfig mm_config(0, true, true);
    pool_ = cache_.addPool("benchmark", kCacheCapacityBytes, {}, mm_config);
    populate();
  }

  bool find(std::string_view key, std::size_t &checksum) {
    Cache::ReadHandle handle = cache_.find(make_key(key));
    if (!handle) {
      return false;
    }
    checksum += handle->getSize();
    benchmark::DoNotOptimize(handle->getMemory());
    return true;
  }

  bool miss(std::string_view key) {
    Cache::ReadHandle handle = cache_.find(make_key(key));
    benchmark::DoNotOptimize(handle.get());
    return !handle;
  }

  bool write(std::string_view key) {
    const WorkloadData &data = workload_data();
    Cache::WriteHandle handle = cache_.allocate(
        pool_, make_key(key), static_cast<std::uint32_t>(data.value.size()));
    if (!handle) {
      return false;
    }
    std::memcpy(handle->getMemory(), data.value.data(), data.value.size());
    Cache::WriteHandle replaced = cache_.insertOrReplace(handle);
    benchmark::DoNotOptimize(replaced.get());
    return true;
  }

 private:
  static Cache::Config make_config() {
    Cache::Config config;
    config.setCacheName("nivocache-comparison");
    config.setCacheSize(kCacheCapacityBytes);
    config.setAccessConfig(kWorkingSetSize * 2U);
    return config;
  }

  static Cache::Key make_key(std::string_view key) {
    return Cache::Key(key.data(), key.size());
  }

  void populate() {
    const WorkloadData &data = workload_data();
    for (const std::string &key : data.hit_keys) {
      if (!write(key)) {
        throw std::runtime_error("CacheLib population failed");
      }
    }
  }

  Cache cache_;
  facebook::cachelib::PoolId pool_{};
};
#endif

template <typename Adapter>
Adapter &adapter() {
  static Adapter cache_adapter;
  return cache_adapter;
}

void finish_benchmark(benchmark::State &state, std::size_t checksum) {
  state.counters["ops_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
  state.SetItemsProcessed(state.iterations());
  benchmark::DoNotOptimize(checksum);
}

template <typename Adapter>
void find_hit(benchmark::State &state) {
  Adapter &cache_adapter = adapter<Adapter>();
  const WorkloadData &data = workload_data();
  std::size_t operation = static_cast<std::size_t>(state.thread_index());
  std::size_t checksum = 0;
  for ([[maybe_unused]] benchmark::State::StateIterator::Value ignored :
       state) {
    const std::size_t index = operation & kWorkingSetMask;
    if (!cache_adapter.find(data.hit_keys[index], checksum)) {
      state.SkipWithError("Expected cache hit");
      break;
    }
    operation += static_cast<std::size_t>(state.threads());
  }
  finish_benchmark(state, checksum);
}

template <typename Adapter>
void find_miss(benchmark::State &state) {
  Adapter &cache_adapter = adapter<Adapter>();
  const WorkloadData &data = workload_data();
  std::size_t operation = static_cast<std::size_t>(state.thread_index());
  for ([[maybe_unused]] benchmark::State::StateIterator::Value ignored :
       state) {
    const std::size_t index = operation & kWorkingSetMask;
    if (!cache_adapter.miss(data.miss_keys[index])) {
      state.SkipWithError("Expected cache miss");
      break;
    }
    operation += static_cast<std::size_t>(state.threads());
  }
  finish_benchmark(state, operation);
}

template <typename Adapter>
void insert_replace(benchmark::State &state) {
  Adapter &cache_adapter = adapter<Adapter>();
  const WorkloadData &data = workload_data();
  std::size_t operation = static_cast<std::size_t>(state.thread_index());
  for ([[maybe_unused]] benchmark::State::StateIterator::Value ignored :
       state) {
    const std::size_t index = operation & kWorkingSetMask;
    if (!cache_adapter.write(data.hit_keys[index])) {
      state.SkipWithError("Cache write failed");
      break;
    }
    operation += static_cast<std::size_t>(state.threads());
  }
  finish_benchmark(state, operation);
}

template <typename Adapter>
void read_95_write_5(benchmark::State &state) {
  Adapter &cache_adapter = adapter<Adapter>();
  const WorkloadData &data = workload_data();
  std::size_t operation = static_cast<std::size_t>(state.thread_index());
  std::size_t checksum = 0;
  for ([[maybe_unused]] benchmark::State::StateIterator::Value ignored :
       state) {
    const std::size_t index = operation & kWorkingSetMask;
    const bool succeeded =
        operation % kWriteEvery == 0U
            ? cache_adapter.write(data.hit_keys[index])
            : cache_adapter.find(data.hit_keys[index], checksum);
    if (!succeeded) {
      state.SkipWithError("Mixed workload operation failed");
      break;
    }
    operation += static_cast<std::size_t>(state.threads());
  }
  finish_benchmark(state, checksum);
}

template <typename Adapter>
void register_adapter(const char *name) {
  const std::string prefix(name);
  const std::array<int, 4> thread_counts{1, 2, 4, 8};
  for (const int thread_count : thread_counts) {
    benchmark::RegisterBenchmark((prefix + "/FindHit").c_str(),
                                 &find_hit<Adapter>)
        ->Threads(thread_count)
        ->UseRealTime();
    benchmark::RegisterBenchmark((prefix + "/FindMiss").c_str(),
                                 &find_miss<Adapter>)
        ->Threads(thread_count)
        ->UseRealTime();
    benchmark::RegisterBenchmark((prefix + "/InsertReplace").c_str(),
                                 &insert_replace<Adapter>)
        ->Threads(thread_count)
        ->UseRealTime();
    benchmark::RegisterBenchmark((prefix + "/Read95Write5").c_str(),
                                 &read_95_write_5<Adapter>)
        ->Threads(thread_count)
        ->UseRealTime();
  }
}

void register_benchmarks() {
  register_adapter<NivoAdapter>("NivoCache");
#if NIVO_BENCHMARK_WITH_CACHELIB
  register_adapter<CacheLibAdapter>("CacheLib");
#endif
}
}  // namespace

int main(int argc, char **argv) {
  register_benchmarks();
  benchmark::AddCustomContext("nivo_revision", NIVO_BENCHMARK_NIVO_REVISION);
  benchmark::AddCustomContext("cachelib_revision",
                              NIVO_BENCHMARK_CACHELIB_REVISION);
  benchmark::AddCustomContext("compiler", NIVO_BENCHMARK_COMPILER);
  benchmark::AddCustomContext("cache_capacity_bytes",
                              std::to_string(kCacheCapacityBytes));
  benchmark::AddCustomContext("working_set_keys",
                              std::to_string(kWorkingSetSize));
  benchmark::AddCustomContext("value_bytes", std::to_string(kValueSize));
  benchmark::AddCustomContext("lru_refresh", "every hit");
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
