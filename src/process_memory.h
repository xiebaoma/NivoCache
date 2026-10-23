#pragma once

#include <charconv>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <cstdio>
#include <unistd.h>
#endif

namespace nivo::detail {

// /proc/self/statm reports virtual pages followed by resident pages. Keep the
// parser platform-independent so malformed and overflowing input is testable.
inline std::optional<std::size_t> rss_from_statm(std::string_view input, long page_size) noexcept {
  if (page_size <= 0) return std::nullopt;
  std::size_t pages = 0;
  for (unsigned field = 0; field < 2; ++field) {
    const std::size_t first = input.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return std::nullopt;
    input.remove_prefix(first);
    const char* const end = input.data() + input.size();
    const std::from_chars_result parsed = std::from_chars(input.data(), end, pages);
    if (parsed.ec != std::errc{}) return std::nullopt;
    if (parsed.ptr != end && *parsed.ptr != ' ' && *parsed.ptr != '\t' &&
        *parsed.ptr != '\r' && *parsed.ptr != '\n') return std::nullopt;
    input.remove_prefix(static_cast<std::size_t>(parsed.ptr - input.data()));
  }
  const std::size_t bytes_per_page = static_cast<std::size_t>(page_size);
  if (pages > std::numeric_limits<std::size_t>::max() / bytes_per_page) return std::nullopt;
  return pages * bytes_per_page;
}

inline std::optional<std::size_t> process_rss_bytes() noexcept {
#if defined(__APPLE__) && defined(__MACH__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  const kern_return_t status = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
      reinterpret_cast<task_info_t>(&info), &count);
  if (status != KERN_SUCCESS || info.resident_size > std::numeric_limits<std::size_t>::max())
    return std::nullopt;
  return static_cast<std::size_t>(info.resident_size);
#elif defined(__linux__)
  std::FILE* file = std::fopen("/proc/self/statm", "r");
  if (!file) return std::nullopt;
  char buffer[256]{};
  const char* read = std::fgets(buffer, sizeof(buffer), file);
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  if (!read || failed) return std::nullopt;
  return rss_from_statm(buffer, sysconf(_SC_PAGESIZE));
#else
  return std::nullopt;
#endif
}
}  // namespace nivo::detail
