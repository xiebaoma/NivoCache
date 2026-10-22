# NivoCache versus CacheLib benchmark

This suite compares the two engines in the same Google Benchmark executable. It
uses the same keys, 128-byte values, 64 MiB cache capacity, 16,384-key working
set, thread counts, operation order, and timing process for both engines.

The workloads are:

- `FindHit`: read an existing key.
- `FindMiss`: read a key that has never been inserted.
- `InsertReplace`: allocate, copy, and replace an existing value.
- `Read95Write5`: deterministic 95% hit / 5% replacement traffic.

Both engines use LRU and refresh recency on every hit. CacheLib's default
60-second refresh suppression is explicitly disabled because NivoCache currently
updates recency on every successful `find`. Cache setup and initial population
are outside the measured region. Multi-thread rates use wall-clock time.

## Reproduce the comparison

Use a dedicated Linux host with an idle system, a fixed CPU governor, and enough
free memory and disk for CacheLib's dependency build. Python 3.10 or newer,
CMake, Git, and a C++20 compiler are required.

```sh
./benchmarks/prepare_cachelib.sh
./benchmarks/run_comparison.sh
```

The first command builds CacheLib and its compatible dependency set at pinned
revision `f4012d692d849d1ae9927dd17ea186fb45f6ef00`. Google Benchmark is pinned to
revision `31c66f47f60d06790a4885a261d682ba339bd100` (v1.9.4). The second command
builds the paired executable, runs five randomly interleaved repetitions per
case, keeps the median, appends detailed data to `benchmarks/results.csv`, and
regenerates `benchmarks/RESULTS.md`. Raw Google Benchmark JSON is kept under
`build/benchmark-results/`.

CacheLib uses its official `getdeps.py` build, so the initial dependency build
can take substantial time. If required system packages are missing, use
CacheLib's documented command before preparing the dependency:

```sh
cd build/cachelib-dependency/source
sudo python3 ./build/fbcode_builder/getdeps.py install-system-deps --recursive cachelib
```

Do not compare rows from different machines, power modes, compiler toolchains,
or CacheLib revisions as if they were a regression series. Use the CSV metadata
to isolate a consistent series. A ratio above `1.0x` means NivoCache has higher
median throughput for that workload and thread count.

## Run NivoCache alone

The CacheLib adapter is optional so the normal macOS and Linux build stays
lightweight:

```sh
cmake -S . -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DNIVO_BUILD_BENCHMARKS=ON
cmake --build build/benchmark --parallel
./build/benchmark/nivo_cache_bench
```

This mode can also be appended to the shared result ledger for before/after
analysis:

```sh
python3 benchmarks/run_and_record.py \
  --binary build/benchmark/nivo_cache_bench \
  --nivo-only \
  --label experiment-name
```

Paired comparison remains the default. `--nivo-only` must be explicit so an
accidentally incomplete CacheLib comparison is not silently recorded.
