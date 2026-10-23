#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "nivo/config.h"

namespace nivo {
namespace detail {
struct Entry;
// Intrusive ownership keeps the reference count in the allocator-owned entry.
class EntryPtr {
 public:
  EntryPtr() noexcept = default;
  explicit EntryPtr(Entry* entry) noexcept : entry_(entry) {}
  EntryPtr(const EntryPtr& other) noexcept;
  EntryPtr(EntryPtr&& other) noexcept : entry_(std::exchange(other.entry_, nullptr)) {}
  EntryPtr& operator=(const EntryPtr& other) noexcept;
  EntryPtr& operator=(EntryPtr&& other) noexcept;
  ~EntryPtr();
  [[nodiscard]] explicit operator bool() const noexcept { return entry_ != nullptr; }
  [[nodiscard]] Entry* operator->() const noexcept { return entry_; }
  [[nodiscard]] Entry& operator*() const noexcept { return *entry_; }
  [[nodiscard]] std::size_t use_count() const noexcept;
  void reset() noexcept;

 private:
  Entry* entry_ = nullptr;
};
}
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
  explicit ReadHandle(detail::EntryPtr entry) : entry_(std::move(entry)) {}
  detail::EntryPtr entry_;
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
  explicit WriteHandle(detail::EntryPtr entry) : entry_(std::move(entry)) {}
  detail::EntryPtr entry_;
};
}  // namespace nivo
