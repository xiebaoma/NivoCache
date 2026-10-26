// Eviction policy comparison. Every policy runs the same key traces, cache
// capacity, value size and thread counts. Unlike cache_bench.cpp, the Insert
// and Mixed workloads keep the cache full so the eviction path is exercised,
// and per-operation latency is sampled into histograms.
//
// Every hit is checked against the value written for its key, and a run whose
// counters are inconsistent is reported as INVALID instead of being measured.

#include <nivo/nivo.h>

#include "bench_support.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

constexpr std::size_t kCapacityBytes = 64U * 1024U * 1024U;
constexpr std::size_t kKeyBytes = bench::kKeyBytes;
constexpr std::size_t kValueBytes = 128U;
constexpr std::size_t kChargeBytes = kKeyBytes + kValueBytes;
constexpr std::size_t kCapacityEntries = kCapacityBytes / kChargeBytes;
// GetHit and GetMiss read a dataset that fits without eviction.
constexpr std::uint32_t kResidentKeys = 400000U;
static_assert(kResidentKeys < kCapacityEntries);
// Mixed draws from a Zipfian universe a little over four times the capacity.
constexpr std::uint32_t kKeyUniverse = 2000000U;
constexpr double kZipfTheta = 0.99;
constexpr std::uint32_t kWritePercent = 10U;
// Uniform streams may wrap: every resident id stays a hit. The Mixed stream
// must not wrap, since replaying a finite trace hides most of the universe and
// inflates hit rates; a run that exhausts its share is reported as INVALID.
constexpr std::size_t kUniformTraceLength = std::size_t{1} << 22U;
constexpr std::size_t kMixedTraceLength = std::size_t{1} << 26U;
constexpr std::uint32_t kTraceWriteBit = 1U << 31U;
constexpr std::uint64_t kSampleEvery = 16U;
constexpr std::uint64_t kStopCheckEvery = 64U;
constexpr std::uint64_t kMissKeyBase = std::uint64_t{1} << 40U;
constexpr std::uint64_t kInsertKeyBase = std::uint64_t{1} << 41U;
constexpr std::uint64_t kPolicySeed = 42U;

enum class Workload { get_hit, get_miss, insert, mixed };

constexpr std::array<nivo::EvictionPolicy, 11> kAllPolicies{
    nivo::EvictionPolicy::lru,    nivo::EvictionPolicy::fifo,
    nivo::EvictionPolicy::lfu,    nivo::EvictionPolicy::clock,
    nivo::EvictionPolicy::random, nivo::EvictionPolicy::no_eviction,
    nivo::EvictionPolicy::sieve,  nivo::EvictionPolicy::s3_fifo, nivo::EvictionPolicy::slru,
    nivo::EvictionPolicy::two_q, nivo::EvictionPolicy::w_tinylfu};
constexpr std::array<Workload, 4> kAllWorkloads{
    Workload::get_hit, Workload::get_miss, Workload::insert, Workload::mixed};

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
    case Workload::get_hit: return "GetHit";
    case Workload::get_miss: return "GetMiss";
    case Workload::insert: return "Insert";
    case Workload::mixed: return "Mixed";
  }
  return "unknown";
}

using bench::Histogram;
using bench::Key;
using bench::SplitMix64;
using bench::Zipfian;

// Operation streams, identical for every policy. Each entry is a key id, with
// kTraceWriteBit marking Mixed writes. GetHit and GetMiss give each thread its
// own uniform stream over the resident ids; Mixed threads take disjoint
// segments of one long Zipfian stream.
class Traces {
 public:
  const std::vector<std::uint32_t> &uniform(std::size_t thread) {
    // Owned through pointers so returned references survive later growth.
    while (uniform_.size() <= thread)
      uniform_.push_back(std::make_unique<std::vector<std::uint32_t>>());
    std::vector<std::uint32_t> &trace = *uniform_[thread];
    if (trace.empty()) {
      trace.resize(kUniformTraceLength);
      SplitMix64 random{0x756e69666f726dULL + thread};
      for (std::uint32_t &entry : trace) entry = random.below(kResidentKeys);
    }
    return trace;
  }
  const std::vector<std::uint32_t> &mixed() {
    if (mixed_.empty()) {
      const Zipfian zipfian(kKeyUniverse, kZipfTheta);
      mixed_.resize(kMixedTraceLength);
      SplitMix64 random{0x6d69786564ULL};
      for (std::uint32_t &entry : mixed_) {
        entry = zipfian.next(random);
        if (random.below(100U) < kWritePercent) entry |= kTraceWriteBit;
      }
    }
    return mixed_;
  }

 private:
  std::vector<std::unique_ptr<std::vector<std::uint32_t>>> uniform_;
  std::vector<std::uint32_t> mixed_;
};

struct ThreadCursor {
  const std::vector<std::uint32_t> *trace = nullptr;
  std::size_t begin = 0;
  std::size_t end = 0;
  std::size_t position = 0;
  std::uint64_t next_insert = 0;
};

struct ThreadResult {
  std::uint64_t operations = 0;
  std::uint64_t gets = 0;
  std::uint64_t hits = 0;
  std::uint64_t puts = 0;
  std::uint64_t fills = 0;  // Mixed puts that follow a miss
  std::uint64_t rejected = 0;
  std::uint64_t mismatches = 0;
  std::uint64_t errors = 0;
  std::uint64_t trace_wraps = 0;
  Histogram get_latency;
  Histogram put_latency;
  Clock::time_point finished;
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
  ThreadResult totals;
  nivo::PoolStats before;
  nivo::PoolStats after;

  double throughput() const {
    return seconds > 0.0 ? static_cast<double>(totals.operations) / seconds : 0.0;
  }
  double hit_rate() const {
    return totals.gets ? static_cast<double>(totals.hits) /
                             static_cast<double>(totals.gets) : 0.0;
  }
};

class Bench {
 public:
  Bench(RunConfig config, Traces &traces) : config_(config) {
    nivo::CacheConfig cache_config;
    cache_config.capacity_bytes = kCapacityBytes;
    cache_config.reaper.interval = nivo::Duration{0};
    nivo::Result<std::unique_ptr<nivo::Cache>> created =
        nivo::Cache::create(cache_config);
    cache_ = std::move(created).value();
    nivo::PoolConfig pool_config;
    pool_config.name = "policy";
    pool_config.capacity_bytes = kCapacityBytes;
    pool_config.eviction = config.policy;
    pool_config.eviction_seed = kPolicySeed;
    pool_ = cache_->add_pool(pool_config).value();
    cursors_.resize(config.threads);
    for (std::size_t thread = 0; thread < config.threads; ++thread) {
      cursors_[thread].next_insert = kInsertKeyBase + thread;
      // Traces are generated here, before any worker starts.
      ThreadCursor &cursor = cursors_[thread];
      if (config.workload == Workload::mixed) {
        cursor.trace = &traces.mixed();
        const std::size_t share = kMixedTraceLength / config.threads;
        cursor.begin = thread * share;
        cursor.end = cursor.begin + share;
      } else if (config.workload != Workload::insert) {
        cursor.trace = &traces.uniform(thread);
        cursor.end = kUniformTraceLength;
      }
      cursor.position = cursor.begin;
    }
  }

  RunResult run(std::chrono::milliseconds warmup, std::chrono::milliseconds measure) {
    RunResult result;
    result.config = config_;
    if (!prefill(result)) return result;
    std::vector<ThreadResult> ignored = phase(warmup);
    for (const ThreadResult &thread : ignored)
      if (thread.mismatches || thread.errors || (config_.workload == Workload::mixed && thread.trace_wraps))
        return invalid(result, "warmup failure");
    result.before = cache_->pool_stats(pool_).value();
    const Clock::time_point started = Clock::now();
    std::vector<ThreadResult> measured = phase(measure);
    Clock::time_point finished = started;
    for (const ThreadResult &thread : measured) {
      finished = std::max(finished, thread.finished);
      accumulate(result.totals, thread);
    }
    result.seconds = std::chrono::duration<double>(finished - started).count();
    result.after = cache_->pool_stats(pool_).value();
    check(result);
    return result;
  }

 private:
  static RunResult &invalid(RunResult &result, std::string reason) {
    if (result.valid) result.error = std::move(reason);
    result.valid = false;
    return result;
  }

  static void accumulate(ThreadResult &total, const ThreadResult &thread) {
    total.operations += thread.operations;
    total.gets += thread.gets;
    total.hits += thread.hits;
    total.puts += thread.puts;
    total.fills += thread.fills;
    total.rejected += thread.rejected;
    total.mismatches += thread.mismatches;
    total.errors += thread.errors;
    total.trace_wraps += thread.trace_wraps;
    total.get_latency.merge(thread.get_latency);
    total.put_latency.merge(thread.put_latency);
  }

  static void fill_value(std::array<std::byte, kValueBytes> &value, std::uint64_t id) {
    std::memcpy(value.data(), &id, sizeof(id));
  }

  bool put(std::uint64_t id, std::array<std::byte, kValueBytes> &value, ThreadResult &out,
           bool sample) {
    fill_value(value, id);
    const Key key(id);
    const Clock::time_point begin = sample ? Clock::now() : Clock::time_point{};
    nivo::Status status = nivo::Status::ok;
    {
      nivo::Result<nivo::ReadHandle> written =
          cache_->put(pool_, key.view(), std::span<const std::byte>(value));
      status = written.status();
    }
    if (sample) out.put_latency.record(elapsed_nanoseconds(begin));
    ++out.puts;
    if (status == nivo::Status::ok) return true;
    if (status == nivo::Status::resource_exhausted) ++out.rejected;
    else ++out.errors;
    return false;
  }

  bool get(std::uint64_t id, ThreadResult &out, bool sample) {
    const Key key(id);
    const Clock::time_point begin = sample ? Clock::now() : Clock::time_point{};
    bool hit = false;
    {
      nivo::Result<nivo::ReadHandle> found = cache_->find(pool_, key.view());
      if (found) {
        hit = true;
        const std::span<const std::byte> bytes = found.value().bytes();
        std::uint64_t stored = 0;
        if (bytes.size() == kValueBytes) std::memcpy(&stored, bytes.data(), sizeof(stored));
        if (bytes.size() != kValueBytes || stored != id) ++out.mismatches;
      } else if (found.status() != nivo::Status::not_found) {
        ++out.errors;
      }
    }
    if (sample) out.get_latency.record(elapsed_nanoseconds(begin));
    ++out.gets;
    if (hit) ++out.hits;
    return hit;
  }

  static std::uint64_t elapsed_nanoseconds(Clock::time_point begin) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
  }

  bool prefill(RunResult &result) {
    std::array<std::byte, kValueBytes> value{};
    value.fill(std::byte{0x5a});
    ThreadResult scratch;
    if (config_.workload == Workload::get_hit || config_.workload == Workload::get_miss) {
      for (std::uint32_t id = 0; id < kResidentKeys; ++id)
        if (!put(id, value, scratch, false)) {
          invalid(result, "prefill failed");
          return false;
        }
      if (cache_->pool_stats(pool_).value().counters.evictions != 0U) {
        invalid(result, "prefill evicted");
        return false;
      }
    } else if (config_.workload == Workload::insert) {
      // Overfill so the measured inserts start from a full cache.
      const std::uint64_t count = kCapacityEntries + kCapacityEntries / 10U;
      for (std::uint64_t id = 0; id < count; ++id) put(id, value, scratch, false);
      if (scratch.errors) {
        invalid(result, "prefill failed");
        return false;
      }
    }
    // Mixed starts empty; its warmup phase brings the cache to steady state.
    return true;
  }

  std::vector<ThreadResult> phase(std::chrono::milliseconds duration) {
    std::vector<ThreadResult> results(config_.threads);
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    workers.reserve(config_.threads);
    for (std::size_t thread = 0; thread < config_.threads; ++thread) {
      workers.emplace_back([this, thread, &results, &ready, &start, &stop] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        work(cursors_[thread], results[thread], stop);
      });
    }
    while (ready.load(std::memory_order_acquire) != config_.threads)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_relaxed);
    for (std::thread &worker : workers) worker.join();
    return results;
  }

  void work(ThreadCursor &cursor, ThreadResult &out,
            const std::atomic<bool> &stop) {
    std::array<std::byte, kValueBytes> value{};
    value.fill(std::byte{0x5a});
    std::uint64_t operation = 0;
    for (;; ++operation) {
      if (operation % kStopCheckEvery == 0U && stop.load(std::memory_order_relaxed)) break;
      const bool sample = operation % kSampleEvery == 0U;
      if (config_.workload == Workload::insert) {
        put(cursor.next_insert, value, out, sample);
        cursor.next_insert += config_.threads;
        continue;
      }
      const std::uint32_t entry = (*cursor.trace)[cursor.position];
      if (++cursor.position == cursor.end) {
        cursor.position = cursor.begin;
        ++out.trace_wraps;
      }
      const std::uint32_t id = entry & ~kTraceWriteBit;
      switch (config_.workload) {
        case Workload::get_hit:
          get(id, out, sample);
          break;
        case Workload::get_miss:
          get(kMissKeyBase + id, out, sample);
          break;
        case Workload::mixed:
          if (entry & kTraceWriteBit) {
            put(id, value, out, sample);
          } else if (!get(id, out, sample)) {
            ++out.fills;
            put(id, value, out, sample);
          }
          break;
        case Workload::insert:
          break;
      }
    }
    out.operations = operation;
    out.finished = Clock::now();
  }

  void check(RunResult &result) const {
    const ThreadResult &totals = result.totals;
    if (totals.mismatches) invalid(result, "value mismatch");
    if (totals.errors) invalid(result, "unexpected status");
    if (config_.workload == Workload::mixed && totals.trace_wraps)
      invalid(result, "trace segment exhausted");
    if (result.after.allocated_bytes > result.after.capacity_bytes)
      invalid(result, "capacity exceeded");
    switch (config_.workload) {
      case Workload::get_hit:
        if (totals.hits != totals.gets) invalid(result, "missed a resident key");
        break;
      case Workload::get_miss:
        if (totals.hits != 0U) invalid(result, "hit an absent key");
        break;
      case Workload::insert:
      case Workload::mixed:
        if (config_.policy != nivo::EvictionPolicy::no_eviction && totals.rejected)
          invalid(result, "evicting policy rejected a write");
        break;
    }
    const std::uint64_t evictions =
        result.after.counters.evictions - result.before.counters.evictions;
    if (config_.policy == nivo::EvictionPolicy::no_eviction && evictions)
      invalid(result, "no_eviction evicted");
  }

  RunConfig config_;
  std::unique_ptr<nivo::Cache> cache_;
  nivo::PoolId pool_ = nivo::invalid_pool;
  std::vector<ThreadCursor> cursors_;
};

struct Options {
  std::vector<std::size_t> threads;
  std::vector<std::size_t> skipped_threads;
  std::vector<nivo::EvictionPolicy> policies{kAllPolicies.begin(), kAllPolicies.end()};
  std::vector<Workload> workloads{kAllWorkloads.begin(), kAllWorkloads.end()};
  std::size_t repetitions = 3;
  std::chrono::milliseconds measure{2000};
  std::chrono::milliseconds warmup{500};
  std::chrono::milliseconds mixed_warmup{1500};
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
    } else if (name == "--measure-ms") {
      options.measure = std::chrono::milliseconds(std::stol(value));
    } else if (name == "--warmup-ms") {
      options.warmup = std::chrono::milliseconds(std::stol(value));
    } else if (name == "--mixed-warmup-ms") {
      options.mixed_warmup = std::chrono::milliseconds(std::stol(value));
    } else {
      std::fprintf(stderr,
                   "usage: %s [--threads=1,2,4] [--policies=lru,clock] "
                   "[--workloads=GetHit,Mixed] [--repetitions=3] [--measure-ms=2000] "
                   "[--warmup-ms=500] [--mixed-warmup-ms=1500]\n",
                   argv[0]);
      return false;
    }
  }
  // Unless thread counts are given explicitly, skip counts above the
  // hardware concurrency rather than report oversubscribed runs.
  const std::size_t hardware = std::thread::hardware_concurrency();
  for (const std::size_t threads : requested) {
    if (threads == 0U) continue;
    if (!force_threads && hardware != 0U && threads > hardware)
      options.skipped_threads.push_back(threads);
    else
      options.threads.push_back(threads);
  }
  return !options.threads.empty() && !options.policies.empty() &&
         !options.workloads.empty() && options.repetitions != 0U;
}

// Resolution and cost of the clock used for latency samples.
void report_clock() {
  std::uint64_t resolution = UINT64_MAX;
  Clock::time_point previous = Clock::now();
  for (int index = 0; index < 1000000; ++index) {
    const Clock::time_point now = Clock::now();
    const std::uint64_t delta = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous).count());
    if (delta) resolution = std::min(resolution, delta);
    previous = now;
  }
  constexpr int kCalls = 10000000;
  const Clock::time_point begin = Clock::now();
  std::int64_t sink = 0;
  for (int index = 0; index < kCalls; ++index)
    sink += Clock::now().time_since_epoch().count() & 1;
  const double cost =
      std::chrono::duration<double, std::nano>(Clock::now() - begin).count() / kCalls;
  std::printf("# clock_resolution_ns=%llu clock_call_ns=%.1f (sink %lld)\n",
              static_cast<unsigned long long>(resolution), cost, static_cast<long long>(sink));
}

void print_row(const RunResult &result) {
  const ThreadResult &totals = result.totals;
  const Histogram &primary = result.config.workload == Workload::insert
                                 ? totals.put_latency : totals.get_latency;
  const std::uint64_t evictions =
      result.after.counters.evictions - result.before.counters.evictions;
  const std::uint64_t attempts =
      result.after.eviction.attempts - result.before.eviction.attempts;
  const std::int64_t selection_ns =
      (result.after.eviction.selection_time - result.before.eviction.selection_time).count();
  const double items = static_cast<double>(result.after.indexed_items);
  std::printf(
      "row,%s,%s,%zu,%zu,%s,%.0f,%.4f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%zu,%zu,%.2f,%llu,%llu,%lld,%s\n",
      workload_name(result.config.workload), policy_name(result.config.policy),
      result.config.threads, result.repetition, result.valid ? "ok" : "INVALID",
      result.throughput(), result.hit_rate(),
      static_cast<unsigned long long>(totals.operations),
      static_cast<unsigned long long>(totals.gets),
      static_cast<unsigned long long>(totals.hits),
      static_cast<unsigned long long>(totals.puts),
      static_cast<unsigned long long>(totals.rejected),
      static_cast<unsigned long long>(primary.percentile(0.50)),
      static_cast<unsigned long long>(primary.percentile(0.99)),
      static_cast<unsigned long long>(primary.percentile(0.999)),
      static_cast<unsigned long long>(primary.max()),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.50)),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.99)),
      static_cast<unsigned long long>(totals.put_latency.percentile(0.999)),
      static_cast<unsigned long long>(totals.fills),
      result.after.indexed_items, result.after.eviction.policy_metadata_bytes,
      items > 0.0 ? static_cast<double>(result.after.eviction.policy_metadata_bytes) / items : 0.0,
      static_cast<unsigned long long>(evictions), static_cast<unsigned long long>(attempts),
      static_cast<long long>(selection_ns), result.error.c_str());
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse(argc, argv, options)) return 2;
  std::printf("# capacity_bytes=%zu key_bytes=%zu value_bytes=%zu capacity_entries=%zu\n",
              kCapacityBytes, kKeyBytes, kValueBytes, kCapacityEntries);
  std::printf("# resident_keys=%u key_universe=%u zipf_theta=%.2f write_percent=%u "
              "sample_every=%llu hardware_concurrency=%u\n",
              kResidentKeys, kKeyUniverse, kZipfTheta, kWritePercent,
              static_cast<unsigned long long>(kSampleEvery),
              std::thread::hardware_concurrency());
  for (const std::size_t threads : options.skipped_threads)
    std::printf("# skipped threads=%zu (exceeds hardware concurrency)\n", threads);
  report_clock();
  std::printf("columns,workload,policy,threads,repetition,status,ops_per_second,hit_rate,"
              "operations,gets,hits,puts,rejected,p50_ns,p99_ns,p999_ns,max_ns,put_p50_ns,"
              "put_p99_ns,put_p999_ns,fills,items,policy_metadata_bytes,"
              "policy_metadata_per_item,evictions,eviction_attempts,selection_ns,error\n");
  Traces traces;
  // Rotate the policy order per repetition so drift is not tied to a policy.
  for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
    for (const Workload workload : options.workloads) {
      for (const std::size_t threads : options.threads) {
        for (std::size_t offset = 0; offset < options.policies.size(); ++offset) {
          const nivo::EvictionPolicy policy =
              options.policies[(offset + repetition) % options.policies.size()];
          Bench bench({policy, workload, threads}, traces);
          RunResult result = bench.run(
              workload == Workload::mixed ? options.mixed_warmup : options.warmup,
              options.measure);
          result.repetition = repetition;
          print_row(result);
        }
      }
    }
  }
  return 0;
}
