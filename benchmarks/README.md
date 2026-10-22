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

## Layout

- `src/` contains the Google Benchmark executable source.
- `scripts/` contains CacheLib setup, comparison, and result-recording tools.
- `results/results.csv` is the machine-readable append-only result ledger.
- `results/history.md` is generated from the ledger.
- Dated detailed reports are stored in `results/`.
- Raw Google Benchmark JSON is stored outside this directory under
  `build/benchmark-results/`.

## Reproduce the comparison

Use a dedicated Linux host with an idle system, a fixed CPU governor, and enough
free memory and disk for CacheLib's dependency build. Python 3.10 or newer,
CMake, Git, and a C++20 compiler are required.

```sh
./benchmarks/scripts/prepare_cachelib.sh
./benchmarks/scripts/run_comparison.sh
```

The first command builds CacheLib and its compatible dependency set at pinned
revision `f4012d692d849d1ae9927dd17ea186fb45f6ef00`. The paired comparison reuses
the Google Benchmark version from that pinned dependency graph. The second
command builds the paired executable, runs five randomly interleaved repetitions
per case, keeps the median, appends detailed data to
`benchmarks/results/results.csv`, and regenerates
`benchmarks/results/history.md`. Raw Google Benchmark JSON is kept under
`build/benchmark-results/`. Standalone NivoCache benchmark builds fetch Google
Benchmark at revision `31c66f47f60d06790a4885a261d682ba339bd100` (v1.9.4).

CacheLib uses its official `getdeps.py` build, so the initial dependency build
can take substantial time. If required system packages are missing, use
CacheLib's documented command before preparing the dependency:

```sh
cd ../cachelib-dependency/source
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
python3 benchmarks/scripts/run_and_record.py \
  --binary build/benchmark/nivo_cache_bench \
  --nivo-only \
  --label experiment-name
```

Paired comparison remains the default. `--nivo-only` must be explicit so an
accidentally incomplete CacheLib comparison is not silently recorded.

## Reuse dependencies on disposable Linux hosts

Export the installed CacheLib SDK before releasing a benchmark server:

```sh
python3 benchmarks/scripts/cachelib_bundle.py pack \
  /root/cachelib-bundles/cachelib-ubuntu24.04-x86_64-gcc13-f4012d6.tar.gz
```

Copy the archive and its `.sha256` file to permanent storage. The bundle contains
`install/`, the pinned CacheLib `source/` checkout, `environment.sh`, and a
manifest recording the OS, architecture, compiler, and system package versions.
It excludes dependency build scratch, virtual environments, and NivoCache build
outputs. Keep the NivoCache checkout separately; it can evolve without rebuilding
the dependency bundle. The bundle is a Linux SDK, even when stored on a Mac.

Use **Ubuntu 24.04 x86_64 with GCC 13** for the current bundle. Some dependencies
contain architecture-specific optimized code; prefer a comparable Xeon CPU and
run a smoke test when changing CPU families. This is not a macOS or ARM bundle.
Installed CMake files contain absolute paths, so restoration uses the original
`/root/cachelib-dependency` path. Existing SDK directories are never overwritten.

On the replacement server, upload both files to `/root/cachelib-bundles/`, copy
or clone this NivoCache checkout, and run from the checkout:

```sh
sudo apt-get update
sudo apt-get install -y python3
sudo python3 benchmarks/scripts/cachelib_bundle.py restore \
  /root/cachelib-bundles/cachelib-ubuntu24.04-x86_64-gcc13-f4012d6.tar.gz \
  --install-system-packages
sudo env CC=gcc-13 CXX=g++-13 bash benchmarks/scripts/run_comparison.sh \
  /root/cachelib-dependency
```

The restore command verifies SHA-256 and OS/architecture, installs the required
Ubuntu development packages with apt, and unpacks the SDK. It does not download
or compile CacheLib, Folly, FBThrift, Boost, or Google Benchmark. Apt still needs
network access; this is not a fully offline operating-system image. System package
updates are allowed but reported, so a new package set must be treated as a new
environment when comparing results. The only C++ build in the final command is
NivoCache and its benchmark adapter.

Mac-side checksum verification after download:

```sh
shasum -a 256 -c cachelib-ubuntu24.04-x86_64-gcc13-f4012d6.tar.gz.sha256
```

Changing the server can change CPU performance even with the same SDK. Always run
both engines again on each replacement server; retain the host and compiler
metadata in the result ledger rather than treating old absolute rates as a
same-machine regression series.
