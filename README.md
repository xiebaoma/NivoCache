# NivoCache

A C++20 in-process cache engine inspired by [CacheLib](https://github.com/facebook/CacheLib), built with the long-term goal of improving performance, memory efficiency, and ease of integration.

NivoCache is in early development. It provides a working in-memory reference implementation and defined API contracts; performance optimization and SSD support are planned.

## Features

- Isolated memory pools with configurable capacity.
- Immutable read handles and two-phase allocation and publication.
- Versioned updates, compare-and-exchange, and TTL expiration.
- LRU/FIFO eviction, statistics, and thread-safe operations.
- No third-party runtime dependencies.

## Build

Requires CMake 3.20+ and a C++20 compiler. Targets Linux and macOS.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/nivo_basic
```

## Integration

```cmake
add_subdirectory(path/to/NivoCache)
target_link_libraries(my_service PRIVATE Nivo::cache)
```

Include `<nivo/nivo.h>` to get started. See the [basic example](examples/basic.cpp) for usage.

## Documentation

See [API semantics](docs/api-semantics.md) for handle lifetimes, writes, expiration, and capacity rules.
