#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "cache_entry.h"

namespace nivo::detail {
// Hierarchical timing wheel over millisecond deadlines. Each index stripe owns
// one and guards it with the stripe lock, so TTL bookkeeping adds no lock to
// publication or removal. Insert and remove are O(1). Advancing is O(levels):
// occupancy bitmaps find passed slots, which are spliced onto a pending list
// without touching their entries. Callers then examine pending entries in
// bounded batches, expiring or cascading each one, so a burst of simultaneous
// deadlines never forces one long lock hold. Entries without a TTL are never
// visited, and each TTL entry cascades at most once per level.
//
// Lists are doubly linked with the tail stored in the head's previous link.
// Only a list's head and tail carry an accurate slot id, so a splice relabels
// two entries instead of the whole list.
class ExpiryWheel final {
 public:
  using Tick = Duration::rep;
  static constexpr unsigned slot_bits = 6;
  static constexpr std::size_t slots_per_level = std::size_t{1} << slot_bits;
  static constexpr std::size_t levels = 6;
  // Deadlines whose bits above this differ from the wheel time wait in the
  // overflow list until the top level turns over (about every 795 days).
  static constexpr unsigned horizon_bits = static_cast<unsigned>(levels) * slot_bits;
  static constexpr std::uint16_t overflow = static_cast<std::uint16_t>(levels * slots_per_level);
  static constexpr std::uint16_t pending = overflow + 1;

  explicit ExpiryWheel(Duration now) noexcept : current_(now.count()) {}
  ExpiryWheel(const ExpiryWheel&) = delete;
  ExpiryWheel& operator=(const ExpiryWheel&) = delete;

  Duration current() const noexcept { return Duration(current_); }
  std::size_t size() const noexcept { return size_; }
  bool has_pending() const noexcept { return heads_[pending] != nullptr; }

  void insert(Entry& entry) noexcept {
    link(entry, slot_for(entry.expires_at.count()));
    ++size_;
  }
  void remove(Entry& entry) noexcept {
    if (entry.expiry_slot == expiry_unlinked) return;
    unlink(entry);
    --size_;
  }
  // Unlinks one entry whose slot the wheel time has passed. The caller either
  // expires it or inserts it again, which places it on a lower level.
  Entry* pop_pending() noexcept {
    Entry* entry = heads_[pending];
    if (entry) remove(*entry);
    return entry;
  }
  // Moves the wheel time forward to now. Never moves it backward.
  void advance(Duration now) noexcept {
    const Tick target = now.count();
    if (target <= current_) return;
    for (std::size_t level = 0; level < levels; ++level) {
      const unsigned shift = static_cast<unsigned>(level) * slot_bits;
      const Tick from = current_ >> shift;
      const Tick to = target >> shift;
      if (from == to) break;  // Higher levels cannot have moved either.
      // Slots (from, to] at this level, or all of them after a full turn.
      std::uint64_t crossed = ~std::uint64_t{0};
      if (to - from < static_cast<Tick>(slots_per_level)) {
        crossed = (std::uint64_t{1} << static_cast<unsigned>(to - from)) - 1;
        crossed = std::rotl(crossed, static_cast<int>((from + 1) & slot_mask));
      }
      std::uint64_t due = occupied_[level] & crossed;
      while (due != 0) {
        const std::size_t slot = static_cast<std::size_t>(std::countr_zero(due));
        due &= due - 1;
        splice_pending(static_cast<std::uint16_t>(level * slots_per_level + slot));
      }
    }
    if ((current_ >> horizon_bits) != (target >> horizon_bits)) splice_pending(overflow);
    current_ = target;
  }
  // Forgets every entry without visiting it; used when the index drops them all.
  void reset() noexcept {
    heads_.fill(nullptr);
    occupied_.fill(0);
    size_ = 0;
  }

 private:
  static constexpr Tick slot_mask = static_cast<Tick>(slots_per_level - 1);

  // The level is chosen by the highest bit group where the deadline differs
  // from the wheel time, so its slot is reached exactly when that group of
  // the wheel time catches up, and is then cascaded to a finer level.
  std::uint16_t slot_for(Tick deadline) const noexcept {
    if (deadline <= current_) return pending;
    const std::uint64_t difference = static_cast<std::uint64_t>(deadline ^ current_);
    const unsigned level = (static_cast<unsigned>(std::bit_width(difference)) - 1) / slot_bits;
    if (level >= levels) return overflow;
    const Tick slot = (deadline >> (level * slot_bits)) & slot_mask;
    return static_cast<std::uint16_t>(level * slots_per_level + static_cast<std::size_t>(slot));
  }
  void mark(std::uint16_t slot) noexcept {
    if (slot < overflow) occupied_[slot / slots_per_level] |= std::uint64_t{1} << (slot % slots_per_level);
  }
  void clear(std::uint16_t slot) noexcept {
    if (slot < overflow) occupied_[slot / slots_per_level] &= ~(std::uint64_t{1} << (slot % slots_per_level));
  }
  void link(Entry& entry, std::uint16_t slot) noexcept {
    Entry* head = heads_[slot];
    entry.expiry_slot = slot;
    entry.expiry_next = head;
    if (head) {
      // The old head keeps an accurate id if it is also the tail.
      entry.expiry_previous = head->expiry_previous;
      head->expiry_previous = &entry;
    } else {
      entry.expiry_previous = &entry;
      mark(slot);
    }
    heads_[slot] = &entry;
  }
  void unlink(Entry& entry) noexcept {
    Entry* previous = entry.expiry_previous;
    Entry* next = entry.expiry_next;
    // A head's previous link is the tail, whose next link is null.
    if (previous->expiry_next != &entry) {
      const std::uint16_t slot = entry.expiry_slot;
      heads_[slot] = next;
      if (next) {
        next->expiry_previous = previous;
        next->expiry_slot = slot;
      } else {
        clear(slot);
      }
    } else {
      previous->expiry_next = next;
      if (next) {
        next->expiry_previous = previous;
      } else {
        // Removing the tail: the previous entry becomes the tail.
        previous->expiry_slot = entry.expiry_slot;
        heads_[entry.expiry_slot]->expiry_previous = previous;
      }
    }
    entry.expiry_previous = nullptr;
    entry.expiry_next = nullptr;
    entry.expiry_slot = expiry_unlinked;
  }
  void splice_pending(std::uint16_t slot) noexcept {
    Entry* head = heads_[slot];
    if (!head) return;
    heads_[slot] = nullptr;
    clear(slot);
    Entry* tail = head->expiry_previous;
    Entry* pending_head = heads_[pending];
    tail->expiry_slot = pending;
    if (!pending_head) {
      head->expiry_slot = pending;
      heads_[pending] = head;
      return;
    }
    Entry* pending_tail = pending_head->expiry_previous;
    pending_tail->expiry_next = head;
    head->expiry_previous = pending_tail;
    pending_head->expiry_previous = tail;
  }

  std::array<Entry*, pending + 1> heads_{};
  std::array<std::uint64_t, levels> occupied_{};
  Tick current_;
  std::size_t size_ = 0;
};
}  // namespace nivo::detail
