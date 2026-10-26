// Hit-rate and throughput comparison of eviction policies on cache-aside
// workloads with variable value sizes. Every policy serves exactly the same
// request stream with the same capacity: a run processes a fixed number of
// requests, and thread t takes requests t, t + T, t + 2T, ... so the global
// request multiset is identical. Thread scheduling can reorder requests;
// deterministic hit-rate comparisons therefore use one thread. Warmup is a
// fixed request count too, so hit rates do not depend on throughput.
//
// A get that misses is followed by a put of that key (a cache-aside fill).
// Every hit is checked against the size and id written for its key, and a run
// with an unexpected status, rejected write or capacity overrun is reported
// as INVALID instead of being measured.

#include <nivo/nivo.h>

#include "bench_support.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using bench::Histogram;
using bench::Key;
using bench::SplitMix64;
using bench::Zipfian;

std::size_t kCapacityBytes = 64U * 1024U * 1024U;
std::uint32_t kUniverse = 1000000U;
std::size_t kMinValueBytes = 100U;
std::size_t kMaxValueBytes = 1100U;
double kZipfTheta = 0.99;
// Scan: each period is kScanBaseOps Zipfian requests followed by a one-pass
// sequential scan over kScanLength keys that are never requested again.
std::uint64_t kScanBaseOps = 200000U;
std::uint64_t kScanLength = 100000U;
// Shift: the Zipfian popularity ranking moves to different keys every
// kShiftEvery requests.
std::uint64_t kShiftEvery = 1000000U;
std::uint32_t kShiftStep = kUniverse / 8U;
constexpr std::uint32_t kMixedWritePercent = 20U;
constexpr std::uint64_t kSampleEvery = 16U;
constexpr std::uint64_t kPolicySeed = 42U;
nivo::WTinyLfuConfig tiny_config;

// Request encoding: key id in the low 40 bits, then flag bits.
constexpr std::uint64_t kIdMask = (std::uint64_t{1} << 40U) - 1U;
constexpr std::uint64_t kWriteFlag = std::uint64_t{1} << 40U;
constexpr std::uint64_t kScanFlag = std::uint64_t{1} << 41U;
constexpr std::uint64_t kScanIdBase = std::uint64_t{1} << 32U;

enum class Workload { zipf, uniform, scan, shift, burst, mixed };

constexpr std::array<Workload, 6> kAllWorkloads{Workload::zipf, Workload::uniform, Workload::scan,
                                                Workload::shift, Workload::burst, Workload::mixed};
constexpr std::array<nivo::EvictionPolicy, 7> kDefaultPolicies{
    nivo::EvictionPolicy::lru, nivo::EvictionPolicy::lfu,
    nivo::EvictionPolicy::sieve, nivo::EvictionPolicy::s3_fifo, nivo::EvictionPolicy::slru,
    nivo::EvictionPolicy::two_q, nivo::EvictionPolicy::w_tinylfu};
constexpr std::array<nivo::EvictionPolicy, 11> kAllPolicies{
    nivo::EvictionPolicy::lru,    nivo::EvictionPolicy::fifo,  nivo::EvictionPolicy::lfu,
    nivo::EvictionPolicy::clock,  nivo::EvictionPolicy::random, nivo::EvictionPolicy::no_eviction,
    nivo::EvictionPolicy::sieve,  nivo::EvictionPolicy::s3_fifo, nivo::EvictionPolicy::slru,
    nivo::EvictionPolicy::two_q, nivo::EvictionPolicy::w_tinylfu};

const char *policy_name(nivo::EvictionPolicy policy) {
  switch (policy) {
    case nivo::EvictionPolicy::slru: return "slru";
    case nivo::EvictionPolicy::two_q: return "two_q";
    case nivo::EvictionPolicy::w_tinylfu: return "w_tinylfu";
    case nivo::EvictionPolicy::lru: return "lru";
    case nivo::EvictionPolicy::fifo: return "fifo";
    case nivo::EvictionPolicy::lfu: return "lfu";
    case nivo::EvictionPolicy::clock: return "clock";
    case nivo::EvictionPolicy::random: return "random";
    case nivo::EvictionPolicy::no_eviction: return "no_eviction";
    case nivo::EvictionPolicy::sieve: return "sieve";
    case nivo::EvictionPolicy::s3_fifo: return "s3_fifo";
  }
  return "unknown";
}

const char *workload_name(Workload workload) {
  switch (workload) {
    case Workload::zipf: return "Zipf";
    case Workload::uniform: return "Uniform";
    case Workload::scan: return "Scan";
    case Workload::shift: return "Shift";
    case Workload::burst: return "Burst";
    case Workload::mixed: return "Mixed";
  }
  return "unknown";
}

std::uint64_t mix(std::uint64_t value) {
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

// Value sizes are uniform in [kMinValueBytes, kMaxValueBytes] and fixed per key.
std::size_t value_size(std::uint64_t id) {
  return kMinValueBytes + static_cast<std::size_t>(mix(id) % (kMaxValueBytes - kMinValueBytes + 1U));
}

std::vector<std::uint64_t> make_requests(Workload workload, std::size_t count) {
  std::vector<std::uint64_t> requests(count);
  SplitMix64 random{0x776f726b6c6f6164ULL + static_cast<std::uint64_t>(workload)};
  const Zipfian zipfian(kUniverse, kZipfTheta);
  std::uint64_t next_scan_id = kScanIdBase;
  for (std::size_t index = 0; index < count; ++index) {
    std::uint64_t request = 0;
    switch (workload) {
      case Workload::zipf:
        request = zipfian.next(random);
        break;
      case Workload::uniform:
        request = random.below(kUniverse);
        break;
      case Workload::scan:
        if (index % (kScanBaseOps + kScanLength) < kScanBaseOps) request = zipfian.next(random);
        else request = next_scan_id++ | kScanFlag;
        break;
      case Workload::shift: {
        const std::uint64_t offset = index / kShiftEvery * kShiftStep;
        request = (zipfian.next(random) + offset) % kUniverse;
        break;
      }
      case Workload::burst:
        // A short, intense burst on a temporary 1% hot set, then return to base.
        request = index % kShiftEvery < kShiftEvery / 4
            ? kUniverse + (index / kShiftEvery) * kUniverse + random.below(std::max(1U, kUniverse / 100))
            : zipfian.next(random);
        break;
      case Workload::mixed:
        request = zipfian.next(random);
        if (random.below(100U) < kMixedWritePercent) request |= kWriteFlag;
        break;
    }
    requests[index] = request;
  }
  return requests;
}

struct ThreadResult {
  std::uint64_t gets = 0;
  std::uint64_t hits = 0;
  std::uint64_t requested_bytes = 0;
  std::uint64_t hit_bytes = 0;
  std::uint64_t base_gets = 0;  // gets outside scans
  std::uint64_t base_hits = 0;
  std::uint64_t puts = 0;
  std::uint64_t rejected = 0;
  std::uint64_t mismatches = 0;
  std::uint64_t errors = 0;
  Histogram get_latency;
  Histogram put_latency;
};

struct RunConfig {
  nivo::EvictionPolicy policy;
  Workload workload;
  std::size_t threads;
};

struct RunResult {
  RunConfig config;
  std::size_t repetition = 0;
  bool valid = true;
  std::string error;
  double seconds = 0.0;
  std::uint64_t requests = 0;
  ThreadResult totals;
  nivo::PoolStats before;
  nivo::PoolStats after;
};

std::uint64_t elapsed_nanoseconds(Clock::time_point begin) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

class Run {
 public:
  Run(RunConfig config, const std::vector<std::uint64_t> &requests)
      : config_(config), requests_(requests) {
    nivo::CacheConfig cache_config;
    cache_config.capacity_bytes = kCapacityBytes;
    cache_config.max_value_bytes = kMaxValueBytes;
    cache_config.reaper.interval = nivo::Duration{0};
    cache_ = nivo::Cache::create(cache_config).value();
    nivo::PoolConfig pool_config;
    pool_config.name = "workload";
    pool_config.capacity_bytes = kCapacityBytes;
    pool_config.eviction = config.policy;
    pool_config.eviction_seed = kPolicySeed;
    pool_config.w_tinylfu = tiny_config;
    pool_ = cache_->add_pool(pool_config).value();
  }

  RunResult execute(std::size_t warmup, std::size_t measure) {
    RunResult result;
    result.config = config_;
    const std::vector<ThreadResult> warm = phase(0, warmup);
    for (const ThreadResult &thread : warm)
      if (thread.mismatches || thread.errors || thread.rejected) fail(result, "warmup failure");
    result.before = cache_->pool_stats(pool_).value();
    const Clock::time_point started = Clock::now();
    const std::vector<ThreadResult> measured = phase(warmup, warmup + measure);
    result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    result.requests = measure;
    for (const ThreadResult &thread : measured) accumulate(result.totals, thread);
    result.after = cache_->pool_stats(pool_).value();
    if (result.totals.mismatches) fail(result, "value mismatch");
    if (result.totals.errors) fail(result, "unexpected status");
    if (result.totals.rejected) fail(result, "write rejected");
    if (result.after.allocated_bytes > result.after.capacity_bytes) fail(result, "capacity exceeded");
    return result;
  }

 private:
  static void fail(RunResult &result, const char *reason) {
    if (result.valid) result.error = reason;
    result.valid = false;
  }

  static void accumulate(ThreadResult &total, const ThreadResult &thread) {
    total.gets += thread.gets;
    total.hits += thread.hits;
    total.requested_bytes += thread.requested_bytes;
    total.hit_bytes += thread.hit_bytes;
    total.base_gets += thread.base_gets;
    total.base_hits += thread.base_hits;
    total.puts += thread.puts;
    total.rejected += thread.rejected;
    total.mismatches += thread.mismatches;
    total.errors += thread.errors;
    total.get_latency.merge(thread.get_latency);
    total.put_latency.merge(thread.put_latency);
  }

  std::vector<ThreadResult> phase(std::size_t begin, std::size_t end) {
    std::vector<ThreadResult> results(config_.threads);
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(config_.threads);
    for (std::size_t thread = 0; thread < config_.threads; ++thread) {
      workers.emplace_back([this, thread, begin, end, &results, &ready, &start] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        work(begin + thread, end, results[thread]);
      });
    }
    while (ready.load(std::memory_order_acquire) != config_.threads) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) worker.join();
    return results;
  }

  void work(std::size_t first, std::size_t end, ThreadResult &out) {
    std::vector<std::byte> value(kMaxValueBytes, std::byte{0x5a});
    std::uint64_t operation = 0;
    for (std::size_t index = first; index < end; index += config_.threads, ++operation) {
      const std::uint64_t request = requests_[index];
      const std::uint64_t id = request & kIdMask;
      const bool sample = operation % kSampleEvery == 0U;
      if (request & kWriteFlag) {
        put(id, value, out, sample);
      } else if (!get(id, (request & kScanFlag) != 0U, out, sample)) {
        put(id, value, out, sample);
      }
    }
  }

  bool get(std::uint64_t id, bool scan, ThreadResult &out, bool sample) {
    const Key key(id);
    const std::size_t size = value_size(id);
    const Clock::time_point begin = sample ? Clock::now() : Clock::time_point{};
    bool hit = false;
    {
      nivo::Result<nivo::ReadHandle> found = cache_->find(pool_, key.view());
      if (found) {
        hit = true;
        const std::span<const std::byte> bytes = found.value().bytes();
        std::uint64_t stored = 0;
        if (bytes.size() == size) std::memcpy(&stored, bytes.data(), sizeof(stored));
        if (bytes.size() != size || stored != id) ++out.mismatches;
      } else if (found.status() != nivo::Status::not_found) {
        ++out.errors;
      }
    }
    if (sample) out.get_latency.record(elapsed_nanoseconds(begin));
    ++out.gets;
    out.requested_bytes += size;
    if (!scan) ++out.base_gets;
    if (hit) {
      ++out.hits;
      out.hit_bytes += size;
      if (!scan) ++out.base_hits;
    }
    return hit;
  }

  void put(std::uint64_t id, std::vector<std::byte> &value, ThreadResult &out, bool sample) {
    const Key key(id);
    const std::size_t size = value_size(id);
    std::memcpy(value.data(), &id, sizeof(id));
    const Clock::time_point begin = sample ? Clock::now() : Clock::time_point{};
    nivo::Status status = nivo::Status::ok;
    {
      nivo::Result<nivo::ReadHandle> written =
          cache_->put(pool_, key.view(), std::span<const std::byte>(value.data(), size));
      status = written.status();
    }
    if (sample) out.put_latency.record(elapsed_nanoseconds(begin));
    ++out.puts;
    if (status == nivo::Status::resource_exhausted) ++out.rejected;
    else if (status != nivo::Status::ok) ++out.errors;
  }

  RunConfig config_;
  const std::vector<std::uint64_t> &requests_;
  std::unique_ptr<nivo::Cache> cache_;
  nivo::PoolId pool_ = nivo::invalid_pool;
};

struct Options {
  std::vector<std::size_t> threads;
  std::vector<std::size_t> skipped_threads;
  std::vector<nivo::EvictionPolicy> policies{kDefaultPolicies.begin(), kDefaultPolicies.end()};
  std::vector<Workload> workloads{kAllWorkloads.begin(), kAllWorkloads.end()};
  std::size_t repetitions = 3;
  std::size_t warmup = 3000000;
  std::size_t measure = 6000000;
};

std::vector<std::string> split(std::string_view text) {
  std::vector<std::string> parts;
  while (!text.empty()) {
    const std::size_t comma = text.find(',');
    parts.emplace_back(text.substr(0, comma));
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1U);
  }
  return parts;
}

bool parse(int argc, char **argv, Options &options) {
  std::vector<std::size_t> requested{1, 2, 4, 8, 16};
  bool force_threads = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const std::size_t equals = argument.find('=');
    const std::string_view name = argument.substr(0, equals);
    const std::string value(equals == std::string_view::npos ? std::string_view{}
                                                             : argument.substr(equals + 1U));
    if (name == "--threads") {
      requested.clear();
      for (const std::string &part : split(value)) requested.push_back(std::stoul(part));
      force_threads = true;
    } else if (name == "--policies") {
      options.policies.clear();
      for (const std::string &part : split(value))
        for (const nivo::EvictionPolicy policy : kAllPolicies)
          if (part == policy_name(policy)) options.policies.push_back(policy);
    } else if (name == "--workloads") {
      options.workloads.clear();
      for (const std::string &part : split(value))
        for (const Workload workload : kAllWorkloads)
          if (part == workload_name(workload)) options.workloads.push_back(workload);
    } else if (name == "--repetitions") {
      options.repetitions = std::stoul(value);
    } else if (name == "--sketch-width") {
      tiny_config.sketch_width = std::stoull(value);
    } else if (name == "--sample-size") {
      tiny_config.sample_size = std::stoull(value);
    } else if (name == "--capacity-bytes") {
      kCapacityBytes = std::stoull(value);
    } else if (name == "--universe") {
      kUniverse = static_cast<std::uint32_t>(std::stoul(value));
      kShiftStep = kUniverse / 8;
    } else if (name == "--min-value-bytes") {
      kMinValueBytes = std::stoull(value);
    } else if (name == "--max-value-bytes") {
      kMaxValueBytes = std::stoull(value);
    } else if (name == "--zipf-theta") {
      kZipfTheta = std::stod(value);
    } else if (name == "--phase-ops") {
      kShiftEvery = std::stoull(value);
      kScanBaseOps = kShiftEvery * 2 / 3;
      kScanLength = kShiftEvery - kScanBaseOps;
    } else if (name == "--warmup-ops") {
      options.warmup = std::stoul(value);
    } else if (name == "--measure-ops") {
      options.measure = std::stoul(value);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--threads=1,2,4] [--policies=lru,sieve,s3_fifo] "
                   "[--workloads=Zipf,Uniform,Scan,Shift,Burst,Mixed] [--repetitions=3] "
                   "[--warmup-ops=3000000] [--measure-ops=6000000] "
                   "[--capacity-bytes=67108864] [--universe=1000000] [--zipf-theta=0.99] "
                   "[--min-value-bytes=100] [--max-value-bytes=1100] [--phase-ops=1000000] "
                   "[--sketch-width=16384] [--sample-size=65536]\n",
                   argv[0]);
      return false;
    }
  }
  const std::size_t hardware = std::thread::hardware_concurrency();
  for (const std::size_t threads : requested) {
    if (threads == 0U) continue;
    if (!force_threads && hardware != 0U && threads > hardware) options.skipped_threads.push_back(threads);
    else options.threads.push_back(threads);
  }
  return !options.threads.empty() && !options.policies.empty() && !options.workloads.empty() &&
         options.repetitions != 0U && options.measure != 0U &&
         kUniverse >= 8 && kMinValueBytes >= sizeof(std::uint64_t) &&
         kMaxValueBytes >= kMinValueBytes && kCapacityBytes >= kMaxValueBytes + bench::kKeyBytes &&
         std::isfinite(kZipfTheta) && kZipfTheta >= 0 && kZipfTheta < 1 && kShiftEvery >= 4 && tiny_config.sketch_width >= 64 &&
         tiny_config.sketch_width <= (1U << 20) && std::has_single_bit(tiny_config.sketch_width) &&
         tiny_config.sample_size >= 8192;
}

double ratio(std::uint64_t part, std::uint64_t whole) {
  return whole ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}

void print_row(const RunResult &result) {
  const ThreadResult &totals = result.totals;
  const std::uint64_t evictions = result.after.counters.evictions - result.before.counters.evictions;
  const std::int64_t selection_ns =
      (result.after.eviction.selection_time - result.before.eviction.selection_time).count();
  const double items = static_cast<double>(result.after.indexed_items);
  std::printf(
      "row,%s,%s,%zu,%zu,%s,%.0f,%.4f,%.4f,%.4f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%zu,%zu,"
      "%.2f,%llu,%lld,%s\n",
      workload_name(result.config.workload), policy_name(result.config.policy), result.config.threads,
      result.repetition, result.valid ? "ok" : "INVALID",
      result.seconds > 0.0 ? static_cast<double>(result.requests) / result.seconds : 0.0,
      ratio(totals.hits, totals.gets), ratio(totals.hit_bytes, totals.requested_bytes),
      ratio(totals.base_hits, totals.base_gets), static_cast<unsigned long long>(totals.gets),
      static_cast<unsigned long long>(totals.puts),
      static_cast<unsigned long long>(totals.get_latency.percentile(0.50)),
      static_cast<unsigned long long>(totals.get_latency.percentile(0.99)),
      static_cast<unsigned long long>(totals.get_latency.percentile(0.999)),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.50)),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.99)),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.999)),
      result.after.indexed_items, result.after.eviction.policy_metadata_bytes,
      items > 0.0 ? static_cast<double>(result.after.eviction.policy_metadata_bytes) / items : 0.0,
      static_cast<unsigned long long>(evictions), static_cast<long long>(selection_ns),
      result.error.c_str());
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse(argc, argv, options)) return 2;
  std::printf("# capacity_bytes=%zu universe=%u value_bytes=%zu-%zu zipf_theta=%.2f "
              "scan=%llu+%llu shift_every=%llu shift_step=%u mixed_write_percent=%u\n",
              kCapacityBytes, kUniverse, kMinValueBytes, kMaxValueBytes, kZipfTheta,
              static_cast<unsigned long long>(kScanBaseOps), static_cast<unsigned long long>(kScanLength),
              static_cast<unsigned long long>(kShiftEvery), kShiftStep, kMixedWritePercent);
  std::printf("# sketch_width=%zu sample_size=%zu\n", tiny_config.sketch_width, tiny_config.sample_size);
  std::printf("# warmup_ops=%zu measure_ops=%zu sample_every=%llu hardware_concurrency=%u\n",
              options.warmup, options.measure, static_cast<unsigned long long>(kSampleEvery),
              std::thread::hardware_concurrency());
  for (const std::size_t threads : options.skipped_threads)
    std::printf("# skipped threads=%zu (exceeds hardware concurrency)\n", threads);
  std::printf("columns,workload,policy,threads,repetition,status,ops_per_second,hit_rate,"
              "byte_hit_rate,base_hit_rate,gets,puts,get_p50_ns,get_p99_ns,get_p999_ns,put_p50_ns,"
              "put_p99_ns,put_p999_ns,items,policy_metadata_bytes,policy_metadata_per_item,"
              "evictions,selection_ns,error\n");
  for (const Workload workload : options.workloads) {
    const std::vector<std::uint64_t> requests = make_requests(workload, options.warmup + options.measure);
    // Rotate the policy order per repetition so drift is not tied to a policy.
    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
      for (const std::size_t threads : options.threads) {
        for (std::size_t offset = 0; offset < options.policies.size(); ++offset) {
          const nivo::EvictionPolicy policy = options.policies[(offset + repetition) % options.policies.size()];
          Run run({policy, workload, threads}, requests);
          RunResult result = run.execute(options.warmup, options.measure);
          result.repetition = repetition;
          print_row(result);
        }
      }
    }
  }
  return 0;
}
