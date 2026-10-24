#pragma once

#include <cstddef>
#include <exception>
#include <limits>
#include <new>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#else
#error "NivoCache page memory requires macOS or Linux"
#endif

namespace nivo::detail {
// Owns the OS boundary only. Callers own mapping lifetimes and reuse policy.
// Mapping lengths include page rounding; no general-purpose heap is involved.
class PageMemory final {
 public:
  [[nodiscard]] static std::size_t page_size() {
    static const std::size_t size = []() -> std::size_t {
      const long native_size = sysconf(_SC_PAGESIZE);
      if (native_size <= 0) throw std::bad_alloc();
      return static_cast<std::size_t>(native_size);
    }();
    return size;
  }

  [[nodiscard]] static std::size_t mapping_size(std::size_t bytes) {
    const std::size_t page = page_size();
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (page - 1))
      throw std::bad_alloc();
    return ((bytes - 1) / page + 1) * page;
  }

  // The caller passes a length obtained from mapping_size(), and retains that
  // length for release(). Failed mappings never escape as usable pointers.
  [[nodiscard]] static void* map(std::size_t bytes) {
    if (bytes == 0 || bytes % page_size() != 0) throw std::bad_alloc();
#if defined(__APPLE__)
    constexpr int anonymous_flag = MAP_ANON;
#else
    constexpr int anonymous_flag = MAP_ANONYMOUS;
#endif
    void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | anonymous_flag, -1, 0);
    if (memory == MAP_FAILED) throw std::bad_alloc();
    return memory;
  }

  static void unmap(void* memory, std::size_t bytes) noexcept {
    // A failure indicates a broken ownership/length invariant. Do not silently
    // report storage as released while leaving its mapping behind.
    if (munmap(memory, bytes) != 0) std::terminate();
  }
};
}  // namespace nivo::detail
