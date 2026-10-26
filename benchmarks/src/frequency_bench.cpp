// Isolate estimator CPU cost from cache/index/LRU work. Keys are generated
// outside the timed region; all workers share one bounded sketch.
#include "../../src/eviction_policy.h"
#include "bench_support.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>

int main() {
  using Clock = std::chrono::steady_clock;
  constexpr std::size_t operations = 4000000;
  std::vector<std::size_t> keys(operations);
  bench::SplitMix64 random{42};
  for (std::size_t& key : keys) key = random.next();
  std::printf("mode,threads,sketch_width,metadata_bytes,ns_per_op,cpu_ns_per_op,p99_ns,checksum\n");
  for (const std::size_t threads : {1, 4, 8}) {
    for (const std::size_t width : {1024, 16384}) {
      for (int mode = 0; mode < 3; ++mode) {
        nivo::detail::TinyLfuSketch sketch({.sketch_width = width});
        std::vector<bench::Histogram> latencies(threads);
        std::vector<std::size_t> checksums(threads);
        std::vector<std::thread> workers;
        std::atomic<std::size_t> ready{0};
        std::atomic<bool> start{false};
        for (std::size_t worker = 0; worker < threads; ++worker) {
          workers.emplace_back([&, worker] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            std::size_t checksum = 0;
            for (std::size_t i = worker; i < operations; i += threads) {
              const bool sample = (i / threads) % 64 == 0;
              const std::size_t key = mode == 2 ? 42 : keys[i];
              const Clock::time_point begin = sample ? Clock::now() : Clock::time_point{};
              if (mode != 0) sketch.record(key);
              checksum ^= key;
              if (sample) latencies[worker].record(static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count()));
            }
            checksums[worker] = checksum;
          });
        }
        while (ready.load(std::memory_order_acquire) < threads) std::this_thread::yield();
        const std::clock_t cpu_begin = std::clock();
        const Clock::time_point begin = Clock::now();
        start.store(true, std::memory_order_release);
        for (std::thread& worker : workers) worker.join();
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - begin).count();
        const double cpu_ns = static_cast<double>(std::clock() - cpu_begin) * 1e9 / CLOCKS_PER_SEC;
        bench::Histogram total;
        std::size_t checksum = 0;
        for (std::size_t worker = 0; worker < threads; ++worker) {
          total.merge(latencies[worker]); checksum ^= checksums[worker];
        }
        std::printf("%s,%zu,%zu,%zu,%.2f,%.2f,%llu,%zu\n", mode == 0 ? "baseline" : mode == 1 ? "uniform" : "hot_key",
                    threads, width, sketch.bytes(), ns / operations, cpu_ns / operations,
                    static_cast<unsigned long long>(total.percentile(0.99)), checksum);
        std::fflush(stdout);
      }
    }
  }
}
