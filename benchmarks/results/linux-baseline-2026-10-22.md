# Linux Baseline Report: NivoCache versus CacheLib

## Executive summary

This report records the first complete paired Linux baseline for NivoCache and
CacheLib. The two engines were measured in the same Google Benchmark executable
with identical keys, values, operation order, thread counts, and measurement
settings.

- NivoCache won 1 of 16 cases: single-thread `InsertReplace`, by 22.4%.
- CacheLib won the other 15 cases.
- The geometric mean NivoCache-to-CacheLib throughput ratio was `0.251x`, or
  approximately `3.98x` in CacheLib's favor when every case is weighted equally.
- NivoCache's principal weakness is concurrency. Its throughput at eight threads
  was only 19.8% to 26.6% of its own single-thread throughput, depending on the
  workload.
- The most severe gap was the eight-thread miss path: NivoCache reached
  1.374 M operations/s and CacheLib reached 68.445 M operations/s. This result
  had high CacheLib run-to-run variance and should be repeated on isolated CPUs,
  but the size of the gap is still actionable.

## Run identity and environment

| Field | Value |
|---|---|
| Run ID | `benchmark-20261022-234905-963759` |
| Start time | `2026-10-22T23:49:05+08:00` |
| NivoCache revision | `2e2c271094b4efbfb32ce55f698b860c0ee764f5` |
| CacheLib revision | `f4012d692d849d1ae9927dd17ea186fb45f6ef00` |
| Host | Alibaba Cloud KVM virtual machine |
| OS | Ubuntu, Linux `6.8.0-139-generic`, x86-64 |
| CPU | 16 logical CPUs, Intel Xeon Platinum, 8 cores / 16 threads, 2.5 GHz reported |
| NUMA | One node |
| Memory | 30 GiB, no swap |
| Compiler | GCC 13.3.0, Release build |
| Google Benchmark | v1.9.5, Release build |
| Load average at start | `0.226, 0.182, 1.155` |
| CPU scaling | Reported disabled by Google Benchmark; no cpufreq governor exposed by the VM |
| ASLR | Enabled |

The NivoCache implementation corresponds to the recorded revision. The benchmark
harness also contains the uncommitted integration fixes listed in this report:
CacheLib's usable pool size is used after its metadata reservation, missing
exported CMake dependencies are discovered explicitly, and the installed
CacheLib include root is supplied to the consumer target.

## Methodology

- Both engines used a configured total cache capacity of 64 MiB.
- CacheLib reserves part of that capacity for allocator metadata. Its benchmark
  pool therefore uses `getCacheMemoryStats().ramCacheSize`, following CacheLib's
  own examples, rather than incorrectly requesting another full 64 MiB pool.
- The working set contained 16,384 keys with 128-byte values.
- Keys were deterministic strings. Hit keys were populated before measurement;
  miss keys were never inserted.
- Both engines used LRU and refreshed recency on every successful hit.
- Thread counts were 1, 2, 4, and 8.
- Every case used wall-clock time, a one-second minimum measurement time, five
  repetitions, random interleaving, and the median result.
- Cache construction and initial population were outside the measured region.

The live working set is far below 64 MiB. This run measures lookup, replacement,
and synchronization overhead; it is not an eviction-throughput or capacity test.

## Complete median results

`Ratio` is NivoCache throughput divided by CacheLib throughput. Values above
`1.0x` favor NivoCache.

| Workload | Threads | Nivo ops/s | CacheLib ops/s | Ratio | Nivo ns/op | CacheLib ns/op | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| FindHit | 1 | 3.641 M | 5.702 M | 0.639x | 274.6 | 175.4 | CacheLib |
| FindHit | 2 | 1.498 M | 3.760 M | 0.399x | 1,334.7 | 532.0 | CacheLib |
| FindHit | 4 | 855.417 K | 3.575 M | 0.239x | 4,676.1 | 1,118.8 | CacheLib |
| FindHit | 8 | 769.046 K | 4.729 M | 0.163x | 10,402.5 | 1,691.5 | CacheLib |
| FindMiss | 1 | 5.175 M | 14.301 M | 0.362x | 193.2 | 69.9 | CacheLib |
| FindMiss | 2 | 2.383 M | 19.786 M | 0.120x | 839.1 | 101.1 | CacheLib |
| FindMiss | 4 | 1.663 M | 36.245 M | 0.046x | 2,404.7 | 110.4 | CacheLib |
| FindMiss | 8 | 1.374 M | 68.445 M | 0.020x | 5,820.4 | 116.9 | CacheLib |
| InsertReplace | 1 | 2.188 M | 1.787 M | 1.224x | 457.1 | 559.5 | NivoCache |
| InsertReplace | 2 | 922.964 K | 1.364 M | 0.677x | 2,166.9 | 1,466.5 | CacheLib |
| InsertReplace | 4 | 502.918 K | 1.417 M | 0.355x | 7,953.6 | 2,822.6 | CacheLib |
| InsertReplace | 8 | 432.823 K | 1.981 M | 0.218x | 18,483.3 | 4,038.2 | CacheLib |
| Read95Write5 | 1 | 3.281 M | 5.246 M | 0.625x | 304.7 | 190.6 | CacheLib |
| Read95Write5 | 2 | 1.464 M | 3.522 M | 0.416x | 1,366.3 | 567.8 | CacheLib |
| Read95Write5 | 4 | 835.011 K | 3.541 M | 0.236x | 4,790.4 | 1,129.7 | CacheLib |
| Read95Write5 | 8 | 750.118 K | 4.696 M | 0.160x | 10,665.0 | 1,703.7 | CacheLib |

## Scaling from one to eight threads

The following values divide eight-thread throughput by single-thread throughput.
A value below `1.0x` means aggregate throughput fell as concurrency increased.

| Workload | NivoCache scaling | CacheLib scaling |
|---|---:|---:|
| FindHit | 0.211x | 0.829x |
| FindMiss | 0.266x | 4.786x |
| InsertReplace | 0.198x | 1.108x |
| Read95Write5 | 0.229x | 0.895x |

NivoCache loses aggregate throughput on every workload as threads are added.
CacheLib remains near its single-thread rate on hit-heavy workloads, slightly
improves replacement throughput, and scales the read-only miss path strongly.

## Measurement stability

Most cases had a real-time coefficient of variation below 4.3%. NivoCache's
maximum was 3.01%. CacheLib's hit, replacement, and mixed cases were also below
4.3%, but its very short miss benchmark was noisier: 5.03%, 16.04%, 25.30%, and
31.22% at 1, 2, 4, and 8 threads respectively.

The CacheLib miss results should be rerun with CPU affinity, isolated cores, ASLR
disabled, and a longer minimum time before using small percentage changes as a
regression threshold. This caveat does not reverse the current conclusion: even
the eight-thread median would need to fall by roughly 98% to match NivoCache.

## Code-level interpretation

The following points are hypotheses supported by the current implementation and
the scaling shape. They should be confirmed with `perf` profiles before a large
redesign.

1. Every public operation is serialized by one cache-wide `std::mutex`.
   `find`, a read miss, allocation, publication, statistics, and pool management
   all contend on the same lock.
2. The index is a `std::map`, so lookup is pointer-heavy and logarithmic rather
   than a cache-friendly hash lookup.
3. A successful LRU hit updates the shared linked list while holding the global
   lock. This makes a read hit a write to shared synchronization state.
4. `put` takes the global lock once in `allocate` and again in `publish`, while
   also allocating an entry and value storage. The strong single-thread
   replacement result shows that the local path is competitive, but its lock
   structure does not scale.
5. Misses still acquire the global lock, read the clock, check expiry, update a
   shared counter, and traverse the ordered index. CacheLib avoids most of this
   shared serialization on its miss path.

## Recommended optimization order

1. Capture `perf stat` and `perf record` data for the four one-thread and
   eight-thread workloads to quantify lock wait, cycles, cache misses, and
   allocator cost.
2. Shard the index and synchronization by key hash. Keep independent eviction
   state per shard initially so unrelated keys do not share one lock.
3. Replace the ordered map with a hash-based index designed for heterogeneous
   `string_view` lookup.
4. Remove cache-wide counter contention through per-shard or per-thread counters
   aggregated by the statistics API.
5. Make LRU maintenance less synchronous, for example with per-shard queues or
   buffered/sampled recency updates, while preserving the documented policy.
6. Add a same-size replace fast path that avoids redundant index work and
   minimizes allocation, without violating active-handle lifetime guarantees.
7. Extend the benchmark matrix with eviction pressure, value sizes, working-set
   sizes, Zipfian access, TTL expiry, and tail-latency measurements.

The first milestone should be stopping throughput from decreasing between one
and eight threads. After that, use `1.0x` in every row as the initial parity gate
before expanding the workload matrix.

## Artifacts and reproduction

- Machine-readable ledger: `benchmarks/results/results.csv`
- Generated history table: `benchmarks/results/history.md`
- Raw JSON: `build/benchmark-results/benchmark-20261022-234905-963759.json`
- Raw JSON SHA-256: `cdff322184cd31180259a97edbbe626cd699cfd9a38370e5597d0f448e564005`

Reproduce the paired run on Linux with:

```sh
./benchmarks/scripts/prepare_cachelib.sh
./benchmarks/scripts/run_comparison.sh
```

Results from different machines, compiler versions, virtualization settings, or
benchmark library versions should be treated as separate series.
