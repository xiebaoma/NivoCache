#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "nivo/stats.h"
#include "page_memory.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NIVO_ALLOCATOR_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(NIVO_ALLOCATOR_ASAN)
#define NIVO_ALLOCATOR_ASAN 1
#endif
#if defined(NIVO_ALLOCATOR_ASAN)
#include <sanitizer/asan_interface.h>
#endif

namespace nivo::detail {

// Per-pool segregated storage. Only the owning cache allocates, but the last
// handle may return a block on any thread, including after cache destruction.
class MemoryAllocator final {
  static constexpr std::size_t alignment = alignof(std::max_align_t);
  static constexpr std::size_t class_count = 128;
  struct FreeBlock { FreeBlock* next; };
  struct Region;

  struct alignas(std::max_align_t) Slab {
    Region* region = nullptr;
    Slab* previous = nullptr;
    Slab* next = nullptr;
    FreeBlock* free = nullptr;
    std::size_t carved = 0;
    std::size_t live = 0;
    std::size_t slot_size = 0;
    std::size_t class_id = 0;
  };

  // One OS mapping contains this header and all of its slabs. Regions
  // are released only when every slab is empty. Only explicit compaction moves
  // eligible entries; normal allocation and reclamation preserve addresses.
  struct alignas(std::max_align_t) Region {
    Region* available_previous = nullptr;
    Region* available_next = nullptr;
    Region* idle_previous = nullptr;
    Region* idle_next = nullptr;
    Slab* empty = nullptr;
    std::size_t slab_count = 0;
    std::size_t active_slabs = 0;
    std::size_t mapped_bytes = 0;
    bool initial_reserve = false;
  };

 public:
  using TimePoint = std::chrono::steady_clock::time_point;
  // Internal dependency seam for deterministic timing and allocation failures.
  // The context must outlive the allocator. Returned memory must have the same
  // alignment as max_align_t; allocation may throw bad_alloc. Allocation and
  // release receive the same page-rounded mapping length.
  struct Hooks {
    void* context = nullptr;
    TimePoint (*now)(void*) noexcept = [](void*) noexcept -> TimePoint {
      return std::chrono::steady_clock::now();
    };
    void* (*allocate)(void*, std::size_t) = [](void*, std::size_t size) -> void* {
      return PageMemory::map(size);
    };
    void (*deallocate)(void*, void*, std::size_t) noexcept =
        [](void*, void* data, std::size_t size) noexcept {
      PageMemory::unmap(data, size);
    };
  };

  struct Allocation {
    void* data = nullptr;
    Slab* slab = nullptr;  // null identifies a direct large allocation
    std::size_t requested = 0;
    std::size_t mapped_bytes = 0;  // direct allocations only; slabs use their region
  };

  // Candidates are unique live allocations from this allocator. The caller
  // keeps them alive until moved and prevents new allocations or references to
  // its movable objects for this scope (Cache holds its pool maintenance guard). Other
  // threads may return non-candidate allocations. Regions are kept alive until
  // the scope ends, including on failure after partially completed work.
  class Compaction final {
   public:
    Compaction(MemoryAllocator& allocator, std::span<const Allocation> candidates)
        : allocator_(allocator) {
      std::lock_guard<std::mutex> lock(allocator_.mutex_);
      assert(!allocator_.compacting_);
      std::vector<Region*> sorted;
      sorted.reserve(candidates.size());
      for (const Allocation& allocation : candidates) {
        if (allocation.slab) sorted.push_back(allocation.slab->region);
      }
      std::sort(sorted.begin(), sorted.end(), std::less<Region*>{});
      std::size_t source_bytes = 0;
      for (std::size_t first = 0; first < sorted.size();) {
        std::size_t end = first + 1;
        while (end < sorted.size() && sorted[end] == sorted[first]) ++end;
        Region* region = sorted[first];
        std::size_t live = 0;
        for (std::size_t i = 0; i < region->slab_count; ++i) {
          const Slab* slab = reinterpret_cast<const Slab*>(reinterpret_cast<const std::byte*>(region) +
              sizeof(Region) + i * allocator_.slab_bytes_);
          live += slab->live;
        }
        // Unindexed old versions, unpublished drafts and pinned entries also
        // contribute to live. Evacuate only regions whose entire contents are
        // represented by movable candidates; moving part would strand storage.
        if (live == end - first) {
          regions_.push_back(region);
          source_bytes += region->mapped_bytes;
        }
        first = end;
      }
      std::array<std::size_t, class_count> counts{};
      std::array<std::size_t, class_count> sizes{};
      for (const Allocation& allocation : candidates) {
        if (should_move(allocation)) {
          ++counts[allocation.slab->class_id];
          sizes[allocation.slab->class_id] = allocation.slab->slot_size;
        }
      }
      std::size_t destination_slabs = 0;
      for (std::size_t id = 0; id < class_count; ++id) {
        if (counts[id] == 0) continue;
        const std::size_t per_slab = (allocator_.slab_bytes_ - sizeof(Slab)) / sizes[id];
        destination_slabs += (counts[id] - 1) / per_slab + 1;
      }
      const std::size_t region_bytes = allocator_.region_mapping_size(1);
      // One tightly filled slab per new region avoids speculative storage.
      // Compare before multiplying to keep planning safe for very large pools.
      // Small tail/header savings do not justify copying an otherwise dense
      // pool under its maintenance guard. Require a slab and at least one percent.
      const std::size_t minimum_savings = std::max(allocator_.slab_bytes_,
          source_bytes / 100 + (source_bytes % 100 != 0 ? 1 : 0));
      if (destination_slabs > source_bytes / region_bytes ||
          source_bytes - destination_slabs * region_bytes < minimum_savings) {
        regions_.clear();
      } else {
        active_ = true;
        allocator_.compacting_ = true;
      }
    }
    Compaction(const Compaction&) = delete;
    Compaction& operator=(const Compaction&) = delete;
    ~Compaction() {
      std::lock_guard<std::mutex> lock(allocator_.mutex_);
      if (active_) allocator_.compacting_ = false;
      allocator_.reset_policy();
      allocator_.cancel_initial_reserve();
      allocator_.release_idle_regions(0);
    }

    [[nodiscard]] bool should_move(const Allocation& allocation) const noexcept {
      return allocation.slab && std::binary_search(regions_.begin(), regions_.end(),
          allocation.slab->region, std::less<Region*>{});
    }

    [[nodiscard]] Allocation allocate(const Allocation& source) {
      std::lock_guard<std::mutex> lock(allocator_.mutex_);
      assert(active_ && should_move(source));
      const std::size_t id = source.slab->class_id;
      Slab* slab = destinations_[id];
      if (!slab) {
        Region* region = allocator_.create_region(1);
        slab = allocator_.take_empty_slab(region);
        slab->slot_size = source.slab->slot_size;
        slab->class_id = id;
        allocator_.link(slab);
        destinations_[id] = slab;
      }
      const Allocation allocation = allocator_.allocate_from_slab(slab, source.requested);
      if (!slab->free && slab->carved + slab->slot_size > allocator_.slab_bytes_ - sizeof(Slab))
        destinations_[id] = nullptr;
      return allocation;
    }

   private:
    MemoryAllocator& allocator_;
    std::vector<Region*> regions_;
    std::array<Slab*, class_count> destinations_{};
    bool active_ = false;
  };

  explicit MemoryAllocator(std::size_t capacity) : MemoryAllocator(capacity, Hooks{}) {}
  MemoryAllocator(std::size_t capacity, Hooks hooks)
      : slab_bytes_(std::clamp(std::bit_floor(capacity),
                              std::size_t{4096}, std::size_t{65536})),
        capacity_(capacity), hooks_(hooks) {
    update_reserve_limit();
  }
  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;
  ~MemoryAllocator() {
    assert(stats_.live_allocations == 0);
    cancel_initial_reserve();
    release_idle_regions(0);
    assert(stats_.regions == 0);
  }

  std::atomic<std::size_t> used{0};  // logical key + value budget

  // Called once on a new allocator, before publication to the cache. Small
  // independent regions bound the storage pinned by a sparsely used reserve.
  void reserve_initial(std::size_t bytes, Duration timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(stats_.system_allocations == 0);
    if (bytes > capacity_ || timeout.count() <= 0)
      throw std::invalid_argument("Invalid initial reserve configuration");
    if (bytes == 0) return;
    if (timeout > std::chrono::duration_cast<Duration>(TimePoint::max() - hooks_.now(hooks_.context)))
      throw std::invalid_argument("Initial reserve timeout is too large");
    try {
      std::size_t slabs = (bytes - 1) / slab_bytes_ + 1;
      while (slabs != 0) {
        const std::size_t count = std::min(slabs, std::size_t{8});
        create_region(count, true);
        slabs -= count;
      }
      const TimePoint ready = hooks_.now(hooks_.context);
      if (timeout > std::chrono::duration_cast<Duration>(TimePoint::max() - ready))
        throw std::invalid_argument("Initial reserve timeout is too large");
      initial_reserve_deadline_ = ready + timeout;
      initial_idle_slabs_ = stats_.empty_slabs;
    } catch (...) {
      cancel_initial_reserve();
      release_idle_regions(0);
      throw;
    }
  }

  // Cache owns the timer; allocator hooks keep deadline behavior testable.
  [[nodiscard]] std::optional<TimePoint> maintain_initial_reserve() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initial_reserve_deadline_) cool(hooks_.now(hooks_.context));
    return initial_reserve_deadline_;
  }

  [[nodiscard]] Allocation allocate(std::size_t size) {
    if (size == 0 || size > std::numeric_limits<std::size_t>::max() - alignment)
      throw std::bad_alloc();
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t slot_size = rounded_size(size);
    // Large values bypass classes, avoiding slab-sized rounding and any 4 MiB
    // item limit. Small pools use smaller slabs without reserving their quota.
    if (slot_size > (slab_bytes_ - sizeof(Slab)) / 4) {
      cool(hooks_.now(hooks_.context));
      const std::size_t mapped_bytes = PageMemory::mapping_size(size);
      if (mapped_bytes > std::numeric_limits<std::size_t>::max() - stats_.reserved_bytes)
        throw std::bad_alloc();
      void* data = system_allocate(mapped_bytes);
      stats_.reserved_bytes += mapped_bytes;
      stats_.active_bytes += size;
      stats_.requested_bytes += size;
      ++stats_.live_allocations;
      ++stats_.large_allocations;
      ++stats_.system_allocations;
      return {data, nullptr, size, mapped_bytes};
    }
    const std::size_t id = class_index(slot_size);
    Slab* slab = available_[id];
    if (!slab) {
      const TimePoint now = hooks_.now(hooks_.context);
      cool(now);
      if (!available_regions_) refill(now);
      else ++stats_.reused_allocations;
      slab = take_empty_slab(available_regions_);
      slab->slot_size = slot_size;
      slab->class_id = id;
      link(slab);
    }
    return allocate_from_slab(slab, size);
  }

  // The exact allocation must be returned once, after its object is destroyed.
  void deallocate(Allocation allocation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.requested_bytes -= allocation.requested;
    --stats_.live_allocations;
    Slab* slab = allocation.slab;
    if (!slab) {
      stats_.active_bytes -= allocation.requested;
      stats_.reserved_bytes -= allocation.mapped_bytes;
      --stats_.large_allocations;
      hooks_.deallocate(hooks_.context, allocation.data, allocation.mapped_bytes);
      return;
    }
    stats_.active_bytes -= slab->slot_size;
    const bool was_full = !slab->free &&
        slab->carved + slab->slot_size > slab_bytes_ - sizeof(Slab);
    unpoison(allocation.data, sizeof(FreeBlock));
    slab->free = new (allocation.data) FreeBlock{slab->free};
    poison(allocation.data, slab->slot_size);
    --slab->live;
    if (slab->live == 0) {
      if (!was_full) unlink(slab);
      return_empty_slab(slab);
      cool(hooks_.now(hooks_.context));
    } else if (was_full) {
      link(slab);
    }
  }

  void trim(bool release_future_empty_slabs = false) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (release_future_empty_slabs) retain_spares_ = false;
    cancel_initial_reserve();
    reset_policy();
    release_idle_regions(0);
  }

  void resize(std::size_t capacity) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    cancel_initial_reserve();
    update_reserve_limit();
    reset_policy();
    release_idle_regions(0);
  }

  // Cooling is opportunistic: slab acquisition/return and statistics sampling
  // perform maintenance. The cache separately schedules startup-reserve expiry.
  [[nodiscard]] AllocatorStats stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    cool(hooks_.now(hooks_.context));
    return stats_;
  }

  // Pure observation for monitoring: unlike stats(), do not cool or trim.
  [[nodiscard]] AllocatorStats inspect() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  Allocation allocate_from_slab(Slab* slab, std::size_t size) noexcept {
    void* data;
    if (slab->free) {
      FreeBlock* block = slab->free;
      unpoison(block, sizeof(FreeBlock));
      slab->free = block->next;
      poison(block, sizeof(FreeBlock));
      data = block;
      ++stats_.reused_allocations;
    } else {
      data = reinterpret_cast<std::byte*>(slab) + sizeof(Slab) + slab->carved;
      slab->carved += slab->slot_size;
    }
    ++slab->live;
    if (!slab->free && slab->carved + slab->slot_size > slab_bytes_ - sizeof(Slab)) unlink(slab);
    unpoison(data, size);
    stats_.requested_bytes += size;
    stats_.active_bytes += slab->slot_size;
    ++stats_.live_allocations;
    return {data, slab, size};
  }
  // Eight classes per doubling above 128 bytes: <= 12.5% rounding there.
  static std::size_t rounded_size(std::size_t size) noexcept {
    if (size > 16384) return size;  // always on the direct allocation path
    const std::size_t step = std::max(alignment, std::bit_floor(size - 1) / 8);
    return (size + step - 1) & ~(step - 1);
  }
  static std::size_t class_index(std::size_t size) noexcept {
    if (size <= 128) return (size - 1) / alignment;
    const std::size_t power = static_cast<std::size_t>(std::bit_width(size - 1) - 1);
    return 128 / alignment + (power - 7) * 8 + (size - (std::size_t{1} << power) - 1) /
        (std::size_t{1} << (power - 3));
  }
  void link(Slab* slab) noexcept {
    slab->previous = nullptr;
    slab->next = available_[slab->class_id];
    if (slab->next) slab->next->previous = slab;
    available_[slab->class_id] = slab;
  }
  void unlink(Slab* slab) noexcept {
    if (slab->previous) slab->previous->next = slab->next;
    else available_[slab->class_id] = slab->next;
    if (slab->next) slab->next->previous = slab->previous;
    slab->previous = slab->next = nullptr;
  }
  void* system_allocate(std::size_t size) {
    void* data = hooks_.allocate(hooks_.context, size);
    if (!data) throw std::bad_alloc();
    return data;
  }

  void update_reserve_limit() noexcept {
    // Bound speculative empty storage to 1/16 of the logical pool capacity,
    // with a one-slab floor and an eight-slab ceiling. This is NOT an RSS cap.
    stats_.reserve_limit_bytes = std::clamp(capacity_ / 16, slab_bytes_, 8 * slab_bytes_);
  }
  void reset_policy() noexcept {
    stats_.refill_batch_size = 1;
    rapid_refills_ = 0;
    have_refill_ = false;
  }
  void cool(TimePoint now) noexcept {
    if (compacting_) return;
    if (initial_reserve_deadline_ && now >= *initial_reserve_deadline_)
      cancel_initial_reserve();
    constexpr std::chrono::milliseconds interval{500};
    if (have_refill_ && now >= last_decay_ && now - last_decay_ >= interval) {
      const std::chrono::steady_clock::duration elapsed = now - last_decay_;
      const std::chrono::steady_clock::duration::rep steps = elapsed / interval;
      if (steps >= 3) stats_.refill_batch_size = 1;
      else stats_.refill_batch_size = std::max(std::size_t{1},
          stats_.refill_batch_size >> static_cast<unsigned>(steps));
      last_decay_ = now;
      rapid_refills_ = 0;
    }
    const std::size_t keep = retain_spares_ ? std::min(stats_.refill_batch_size,
        stats_.reserve_limit_bytes / slab_bytes_) : 0;
    release_idle_regions(keep);
  }

  void refill(TimePoint now) {
    // Only exhaustion of the common slab reserve drives growth. Recycling an
    // empty slab or repeatedly replacing one key cannot by itself ramp up.
    constexpr std::chrono::milliseconds fast_slab_interval{50};
    if (have_refill_ && now >= last_refill_ &&
        now - last_refill_ <= fast_slab_interval *
            static_cast<std::chrono::milliseconds::rep>(last_refill_slabs_)) {
      if (++rapid_refills_ >= 2) {
        stats_.refill_batch_size = std::min(std::size_t{8}, stats_.refill_batch_size * 2);
        rapid_refills_ = 0;
      }
    } else {
      rapid_refills_ = 0;
    }
    const std::size_t spare_slots = stats_.reserve_limit_bytes / slab_bytes_;
    const std::size_t spare_room = spare_slots > stats_.empty_slabs ?
        spare_slots - stats_.empty_slabs : 0;
    // Existing quota counts key+value, so mandatory storage may exceed it.
    // Prefetch never makes that worse: extras must fit in actual byte headroom.
    const std::size_t headroom = capacity_ > stats_.reserved_bytes ?
        capacity_ - stats_.reserved_bytes : 0;
    const std::size_t affordable = headroom > sizeof(Region) ?
        (headroom - sizeof(Region)) / slab_bytes_ : 0;
    std::size_t count = std::min({stats_.refill_batch_size, spare_room + 1,
                                 std::max(std::size_t{1}, affordable)});
    // Page rounding also consumes headroom. Never prefetch extra slabs beyond
    // the logical capacity merely because their unrounded layout would fit.
    while (count > 1 && region_mapping_size(count) > headroom) --count;
    try {
      create_region(count);
    } catch (const std::bad_alloc&) {
      if (count == 1) throw;
      ++stats_.batch_fallbacks;
      stats_.refill_batch_size = 1;
      rapid_refills_ = 0;
      count = 1;
      create_region(count);
    }
    if (count > 1) ++stats_.batch_refills;
    have_refill_ = true;
    last_refill_ = last_decay_ = now;
    last_refill_slabs_ = count;
  }

  static void prefault(void* data, std::size_t bytes) noexcept {
    const std::size_t page_size = PageMemory::page_size();
    // Volatile writes cannot be optimized away or satisfied by a shared zero
    // page. The last byte covers the final page of an unaligned allocation.
    volatile unsigned char* memory = static_cast<volatile unsigned char*>(data);
    for (std::size_t offset = 0; offset < bytes; offset += page_size) memory[offset] = 0;
    memory[bytes - 1] = 0;
  }

  void cancel_initial_reserve() noexcept {
    initial_reserve_deadline_.reset();
    initial_idle_slabs_ = 0;
  }

  std::size_t region_mapping_size(std::size_t count) const {
    if (count == 0 || count > (std::numeric_limits<std::size_t>::max() - sizeof(Region)) / slab_bytes_)
      throw std::bad_alloc();
    return PageMemory::mapping_size(sizeof(Region) + count * slab_bytes_);
  }

  Region* create_region(std::size_t count, bool initial_reserve = false) {
    const std::size_t bytes = region_mapping_size(count);
    if (bytes > std::numeric_limits<std::size_t>::max() - stats_.reserved_bytes)
      throw std::bad_alloc();
    void* memory = system_allocate(bytes);
    if (initial_reserve) prefault(memory, bytes);
    Region* region = new (memory) Region{};
    region->slab_count = count;
    region->mapped_bytes = bytes;
    region->initial_reserve = initial_reserve;
    for (std::size_t i = count; i > 0; --i) {
      void* slab_memory = reinterpret_cast<std::byte*>(region) + sizeof(Region) + (i - 1) * slab_bytes_;
      Slab* slab = new (slab_memory) Slab{};
      slab->region = region;
      slab->next = region->empty;
      region->empty = slab;
      poison(reinterpret_cast<std::byte*>(slab) + sizeof(Slab), slab_bytes_ - sizeof(Slab));
    }
    stats_.reserved_bytes += bytes;
    stats_.metadata_bytes += sizeof(Region) + count * sizeof(Slab);
    stats_.slabs += count;
    stats_.empty_slabs += count;
    ++stats_.regions;
    ++stats_.system_allocations;
    link_region(region, false);
    link_idle(region);
    return region;
  }

  Slab* take_empty_slab(Region* region) noexcept {
    if (region->active_slabs == 0) unlink_idle(region);
    Slab* slab = region->empty;
    region->empty = slab->next;
    if (!region->empty) unlink_region(region);
    ++region->active_slabs;
    --stats_.empty_slabs;
    *slab = Slab{};
    slab->region = region;
    return slab;
  }
  void return_empty_slab(Slab* slab) noexcept {
    Region* region = slab->region;
    const bool was_full = region->empty == nullptr;
    slab->next = region->empty;
    slab->previous = nullptr;
    region->empty = slab;
    ++stats_.empty_slabs;
    --region->active_slabs;
    if (region->active_slabs == 0) {
      if (!was_full) unlink_region(region);
      // Reuse partially occupied regions first so other regions can drain.
      link_region(region, false);
      link_idle(region);
    } else if (was_full) {
      link_region(region, true);
    }
  }

  void link_region(Region* region, bool front) noexcept {
    region->available_previous = front ? nullptr : available_tail_;
    region->available_next = front ? available_regions_ : nullptr;
    if (region->available_previous) region->available_previous->available_next = region;
    else available_regions_ = region;
    if (region->available_next) region->available_next->available_previous = region;
    else available_tail_ = region;
  }
  void unlink_region(Region* region) noexcept {
    if (region->available_previous)
      region->available_previous->available_next = region->available_next;
    else available_regions_ = region->available_next;
    if (region->available_next)
      region->available_next->available_previous = region->available_previous;
    else available_tail_ = region->available_previous;
    region->available_previous = region->available_next = nullptr;
  }
  void link_idle(Region* region) noexcept {
    if (initial_reserve_deadline_ && region->initial_reserve)
      initial_idle_slabs_ += region->slab_count;
    stats_.reclaimable_bytes += region->mapped_bytes;
    region->idle_previous = idle_tail_;
    region->idle_next = nullptr;
    if (idle_tail_) idle_tail_->idle_next = region;
    else idle_regions_ = region;
    idle_tail_ = region;
  }
  void unlink_idle(Region* region) noexcept {
    if (initial_reserve_deadline_ && region->initial_reserve)
      initial_idle_slabs_ -= region->slab_count;
    stats_.reclaimable_bytes -= region->mapped_bytes;
    if (region->idle_previous) region->idle_previous->idle_next = region->idle_next;
    else idle_regions_ = region->idle_next;
    if (region->idle_next) region->idle_next->idle_previous = region->idle_previous;
    else idle_tail_ = region->idle_previous;
    region->idle_previous = region->idle_next = nullptr;
  }
  void release_idle_regions(std::size_t keep_slabs) noexcept {
    if (compacting_) return;
    // Empty slabs in a partly occupied region cannot be individually unmapped.
    // Include them in the reserve count, but release only wholly idle regions.
    Region* candidate = idle_regions_;
    while (candidate && stats_.empty_slabs - initial_idle_slabs_ > keep_slabs) {
      Region* region = candidate;
      candidate = region->idle_next;
      if (initial_reserve_deadline_ && region->initial_reserve) continue;
      unlink_idle(region);
      unlink_region(region);
      const std::size_t count = region->slab_count;
      const std::size_t bytes = region->mapped_bytes;
      unpoison(region, bytes);
      for (std::size_t i = 0; i < count; ++i) {
        Slab* slab = reinterpret_cast<Slab*>(reinterpret_cast<std::byte*>(region) +
                                           sizeof(Region) + i * slab_bytes_);
        slab->~Slab();
      }
      region->~Region();
      hooks_.deallocate(hooks_.context, region, bytes);
      stats_.reserved_bytes -= bytes;
      stats_.metadata_bytes -= sizeof(Region) + count * sizeof(Slab);
      stats_.slabs -= count;
      stats_.empty_slabs -= count;
      --stats_.regions;
    }
  }
  static void poison(void* data, std::size_t size) noexcept {
#if defined(NIVO_ALLOCATOR_ASAN)
    __asan_poison_memory_region(data, size);
#else
    (void)data;
    (void)size;
#endif
  }
  static void unpoison(void* data, std::size_t size) noexcept {
#if defined(NIVO_ALLOCATOR_ASAN)
    __asan_unpoison_memory_region(data, size);
#else
    (void)data;
    (void)size;
#endif
  }
  const std::size_t slab_bytes_;
  mutable std::mutex mutex_;
  std::array<Slab*, class_count> available_{};
  std::size_t capacity_;
  const Hooks hooks_;
  Region* available_regions_ = nullptr;
  Region* available_tail_ = nullptr;
  Region* idle_regions_ = nullptr;
  Region* idle_tail_ = nullptr;
  bool retain_spares_ = true;
  bool compacting_ = false;
  bool have_refill_ = false;
  unsigned rapid_refills_ = 0;
  std::size_t last_refill_slabs_ = 1;
  TimePoint last_refill_{};
  TimePoint last_decay_{};
  std::optional<TimePoint> initial_reserve_deadline_;
  std::size_t initial_idle_slabs_ = 0;
  AllocatorStats stats_;
};
}  // namespace nivo::detail
