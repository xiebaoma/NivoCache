#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "nivo/config.h"

namespace nivo {
namespace detail { struct Entry; }
class Cache;

// Copyable immutable snapshot. Pins memory, including after erase/replace/close
// or Cache destruction. Empty and moved-from handles have empty views/version 0.
class ReadHandle {
 public:
  ReadHandle() = default;
  [[nodiscard]] explicit operator bool() const noexcept { return !!entry_; }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] std::string_view key() const noexcept;
  [[nodiscard]] PoolId pool() const noexcept;
  [[nodiscard]] Version version() const noexcept;
  [[nodiscard]] std::size_t charge() const noexcept;
  void reset() noexcept { entry_.reset(); }

 private:
  friend class Cache;
  explicit ReadHandle(std::shared_ptr<const detail::Entry> entry) : entry_(std::move(entry)) {}
  std::shared_ptr<const detail::Entry> entry_;
};

// Move-only, unpublished allocation. Do not share one mutable handle across
// threads. All mutable views must be abandoned before successful publish().
class WriteHandle {
 public:
  WriteHandle() = default;
  WriteHandle(WriteHandle&&) noexcept = default;
  WriteHandle& operator=(WriteHandle&&) noexcept = default;
  WriteHandle(const WriteHandle&) = delete;
  WriteHandle& operator=(const WriteHandle&) = delete;
  [[nodiscard]] explicit operator bool() const noexcept { return !!entry_; }
  [[nodiscard]] std::span<std::byte> bytes() noexcept;
  [[nodiscard]] std::string_view key() const noexcept;
  [[nodiscard]] PoolId pool() const noexcept;
  [[nodiscard]] std::size_t charge() const noexcept;
  void reset() noexcept { entry_.reset(); }

 private:
  friend class Cache;
  explicit WriteHandle(std::shared_ptr<detail::Entry> entry) : entry_(std::move(entry)) {}
  std::shared_ptr<detail::Entry> entry_;
};
}  // namespace nivo
