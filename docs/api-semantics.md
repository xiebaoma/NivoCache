# API Semantics

Include `<nivo/nivo.h>`. The current implementation is synchronous and memory-only.
Public declarations are in [include/nivo](../include/nivo); see [basic.cpp](../examples/basic.cpp) for usage.

## Pools and values

- Create a cache with `Cache::create`, then add pools with `add_pool`. There is no default pool.
- Each pool has its own key namespace and capacity. Pool names must be unique; quotas cannot exceed the cache capacity in total.
- Keys are nonempty binary strings; values are raw bytes and may be empty. Both must respect configured size limits.
- Input views must remain valid during the call. The cache owns its stored keys and values.

## Handles and publication

`allocate → fill WriteHandle → publish → ReadHandle`

- `allocate` returns a move-only, zero-initialized draft that is not yet visible to readers.
- `publish` consumes the draft only on success. On failure, it remains writable and retryable. Drafts belong to their originating cache.
- After publication, stop using any mutable views obtained from the draft. Published values are immutable.
- `ReadHandle` is a copyable snapshot. Replacement, removal, expiration, and cache destruction do not invalidate existing handles.
- Views borrow memory from handles; they do not extend its lifetime. Release handles promptly to avoid retaining capacity.

## Writes and versions

| Mode | Key missing or expired | Key present |
| --- | --- | --- |
| `upsert` | Insert | Replace |
| `insert_only` | Insert | `already_exists` |
| `replace_only` | `not_found` | Replace |
| `compare_exchange` | `not_found` | Replace if the version matches; otherwise `version_mismatch` |

CAS requires a nonzero `expected_version`; other write modes require zero. Successful publication assigns a new, increasing version scoped to the cache instance.

`erase(pool, key, version)` conditionally deletes that version; zero means unconditional deletion.
`put` combines allocation, copying, and publication. Conditions are checked atomically at publication, but the full sequence is not a transaction.

## Expiration and eviction

- TTL is measured in milliseconds from publication using a monotonic clock. Zero means no expiration; negative TTL is invalid.
- An entry expires when `now >= publication_time + ttl`. Reads do not extend TTL.
- Expiration is discovered during operations or through `prune_expired`; there is no background worker.
- `find` updates hit/miss counters and LRU order. `peek` updates neither, but still checks expiration. FIFO reads do not change order.
- Custom clocks must be thread-safe, nonnegative, nondecreasing, nonthrowing, and must not call back into the cache.

## Capacity

Capacity is charged as `key.size() + value.size()`, including drafts and old versions retained by handles. This is a logical byte budget, not an RSS limit; metadata and allocator overhead are excluded.

Read handles prevent capacity-driven eviction of their versions. Explicit removal and expiration can unlink pinned entries, but memory is reclaimed only after the last reference is released.

Replacement needs room for both the old value and the new draft. Allocation never evicts its own target key, but may evict other keys even if the write later fails. Failed pool shrinking keeps the old quota; any completed evictions remain.

## Concurrency and errors

Cache operations are thread-safe; individual lookups, publications, and removals are linearizable. Multi-call sequences are not atomic. Synchronize mutation of a shared handle object, and stop member calls before destroying the cache.

`close` is idempotent: it clears the index and rejects further data operations. Existing handles and statistics remain usable. `put` validates write options before checking closure.

`Result<T>` carries a value or a `Status`; check it before calling `.value()`. Expected failures include misses, conflicts, invalid arguments, unknown pools, capacity exhaustion, and closure. Standard allocation/system exceptions may still propagate.

Statistics do not trigger expiration cleanup. Indexed counts may include expired entries, and allocated-byte totals are observational snapshots because handles can be released concurrently.

SSD, persistence, and asynchronous APIs are not implemented; requesting hybrid storage returns `unsupported`. API and ABI compatibility are not guaranteed before 1.0.
