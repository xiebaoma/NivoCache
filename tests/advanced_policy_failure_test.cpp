#include <nivo/nivo.h>
#include "../src/eviction_policy.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <new>
#include <span>

// Fail exactly one C++ allocation on this thread. All policy tables use these
// allocation paths; the core's mmap allocator is covered by its existing tests.
namespace {
thread_local long allocations_left = -1;
void fail_if_due() {
  if (allocations_left < 0) return;
  if (allocations_left-- == 0) {
    allocations_left = -1;
    throw std::bad_alloc();
  }
}
}
void* operator new(std::size_t size) {
  fail_if_due();
  if (void* pointer = std::malloc(size ? size : 1)) return pointer;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  fail_if_due();
  void* pointer = nullptr;
  if (posix_memalign(&pointer, static_cast<std::size_t>(alignment), size ? size : 1) == 0) return pointer;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }

int main() {
  for (const nivo::EvictionPolicy policy : {nivo::EvictionPolicy::slru, nivo::EvictionPolicy::two_q,
                                           nivo::EvictionPolicy::w_tinylfu}) {
    std::size_t failures = 0;
    bool succeeded = false;
    for (long position = 0; position < 64; ++position) {
      std::unique_ptr<nivo::Cache> cache = std::move(nivo::Cache::create(
          {.capacity_bytes = 1024, .reaper = {.interval = nivo::Duration(0)}})).value();
      nivo::PoolConfig config{.name = "failure", .capacity_bytes = 512, .eviction = policy};
      allocations_left = position;
      nivo::Result<nivo::PoolId> added = cache->add_pool(std::move(config));
      const bool consumed = allocations_left < 0;
      allocations_left = -1;
      if (!added) {
        if (added.status() != nivo::Status::resource_exhausted) return 1;
        ++failures;
        // A failed constructor must not leave a registered pool or quota.
        added = cache->add_pool({.name = "failure", .capacity_bytes = 512, .eviction = policy});
        if (!added) return 2;
      }
      if (!cache->put(added.value(), "a", {})) return 3;
      if (cache->clear(added.value()) != nivo::Status::ok) return 4;
      if (!consumed) { succeeded = true; break; }
    }
    if (!succeeded || failures == 0) return 5;

    // Once constructed, publication, reads, selection, replacement, relocation,
    // removal and history maintenance must not allocate or throw.
    nivo::detail::EvictionPolicyEngine engine(policy);
    std::array<nivo::detail::Entry, 4> entries;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      entries[i].key = "key";
      entries[i].charge = 4;
      entries[i].version = i + 1;
    }
    allocations_left = 0;
    bool valid = engine.publish(entries[0], nullptr) && engine.publish(entries[1], nullptr);
    engine.record_access(entries[0]);
    engine.record_miss("absent");
    valid = valid && engine.publish(entries[2], &entries[0]);
    engine.relocate(entries[2], entries[3]);
    {
      nivo::detail::VictimSelection selection = engine.select_victims();
      valid = valid && selection.next().has_value();
    }
    engine.remove(entries[1]);
    engine.remove(entries[3]);
    engine.reset_history();
    valid = valid && allocations_left == 0;
    allocations_left = -1;
    if (!valid) return 6;
    std::cout << "PASS allocation failures policy=" << static_cast<int>(policy)
              << " injected=" << failures << '\n';
  }
}
