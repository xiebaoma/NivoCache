#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>

namespace nivo::detail {
// Reader fast paths use only atomics. Contended callers park with atomic::wait
// instead of spinning. Pending writers stop new readers so pool maintenance can
// finish even under sustained lookup traffic. Compatible with shared_lock and
// unique_lock; no recursion or lock upgrades are supported.
class SharedMutex final {
 public:
  void lock_shared() noexcept {
    std::uint64_t state = state_.load(std::memory_order_relaxed);
    for (;;) {
      if ((state & ~reader_mask) != 0) {
        state_.wait(state, std::memory_order_relaxed);
        state = state_.load(std::memory_order_relaxed);
        continue;
      }
      assert(state < reader_mask);
      if (state_.compare_exchange_weak(state, state + 1, std::memory_order_acquire,
                                      std::memory_order_relaxed)) return;
    }
  }
  void unlock_shared() noexcept {
    const std::uint64_t previous = state_.fetch_sub(1, std::memory_order_release);
    assert((previous & reader_mask) != 0 && (previous & writer_bit) == 0);
    // Waiting-writer registration shares the same atomic word. The last reader
    // can safely skip notification when nobody waits, without a lost-wakeup race.
    if ((previous & reader_mask) == 1 && previous >= waiting_writer) state_.notify_all();
  }
  void lock() noexcept {
    std::uint64_t state = 0;
    if (state_.compare_exchange_strong(state, writer_bit, std::memory_order_acquire,
                                      std::memory_order_relaxed)) return;
    state = state_.fetch_add(waiting_writer, std::memory_order_relaxed) + waiting_writer;
    for (;;) {
      if ((state & (reader_mask | writer_bit)) == 0) {
        if (state_.compare_exchange_weak(state, state - waiting_writer + writer_bit,
                                        std::memory_order_acquire, std::memory_order_relaxed)) return;
      } else {
        state_.wait(state, std::memory_order_relaxed);
        state = state_.load(std::memory_order_relaxed);
      }
    }
  }
  void unlock() noexcept {
    [[maybe_unused]] const std::uint64_t previous = state_.fetch_sub(writer_bit, std::memory_order_release);
    assert((previous & writer_bit) != 0 && (previous & reader_mask) == 0);
    // Readers may be waiting even if no other writer is registered.
    state_.notify_all();
  }

 private:
  static constexpr std::uint64_t writer_bit = std::uint64_t{1} << 32;
  static constexpr std::uint64_t reader_mask = writer_bit - 1;
  static constexpr std::uint64_t waiting_writer = writer_bit << 1;
  std::atomic<std::uint64_t> state_{0};
};
}  // namespace nivo::detail
