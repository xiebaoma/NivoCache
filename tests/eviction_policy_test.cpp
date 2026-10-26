#include <nivo/nivo.h>
#include "../src/eviction_policy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using nivo::EvictionPolicy;
using nivo::Status;
#define CHECK(...) do { if (!(__VA_ARGS__)) throw std::runtime_error( \
  std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + #__VA_ARGS__); } while (false)

namespace {
struct ManualClock final : nivo::Clock {
  std::atomic<nivo::Duration::rep> ticks{0};
  nivo::Duration now() const noexcept override { return nivo::Duration(ticks.load()); }
  void advance(nivo::Duration delta) { ticks.fetch_add(delta.count()); }
};
std::span<const std::byte> bytes(std::string_view s) {
  return std::as_bytes(std::span(s.data(), s.size()));
}
std::string text(const nivo::ReadHandle& handle) {
  std::span<const std::byte> b = handle.bytes();
  if (b.empty()) return {};
  return {reinterpret_cast<const char*>(b.data()), b.size()};
}
inline constexpr nivo::ReaperConfig no_reaper{.interval = nivo::Duration(0)};

constexpr std::array<EvictionPolicy, 11> all_policies{
    EvictionPolicy::lru, EvictionPolicy::fifo, EvictionPolicy::lfu,
    EvictionPolicy::clock, EvictionPolicy::random, EvictionPolicy::no_eviction,
    EvictionPolicy::sieve, EvictionPolicy::s3_fifo, EvictionPolicy::slru, EvictionPolicy::two_q, EvictionPolicy::w_tinylfu};
constexpr std::array<EvictionPolicy, 10> evicting_policies{
    EvictionPolicy::lru, EvictionPolicy::fifo, EvictionPolicy::lfu,
    EvictionPolicy::clock, EvictionPolicy::random, EvictionPolicy::sieve,
    EvictionPolicy::s3_fifo, EvictionPolicy::slru, EvictionPolicy::two_q, EvictionPolicy::w_tinylfu};

const char* policy_name(EvictionPolicy policy) {
  switch (policy) {
    case EvictionPolicy::w_tinylfu: return "w_tinylfu";
    case EvictionPolicy::two_q: return "two_q";
    case EvictionPolicy::slru: return "slru";
    case EvictionPolicy::lru: return "lru";
    case EvictionPolicy::fifo: return "fifo";
    case EvictionPolicy::lfu: return "lfu";
    case EvictionPolicy::clock: return "clock";
    case EvictionPolicy::random: return "random";
    case EvictionPolicy::no_eviction: return "no_eviction";
    case EvictionPolicy::sieve: return "sieve";
    case EvictionPolicy::s3_fifo: return "s3_fifo";
  }
  return "unknown";
}
// Policy fields per tracked entry; S3-FIFO adds its ghost table and Random
// its slot array on top.
bool exact_metadata(EvictionPolicy policy) {
  return policy != EvictionPolicy::w_tinylfu && policy != EvictionPolicy::two_q && policy != EvictionPolicy::random && policy != EvictionPolicy::s3_fifo &&
         policy != EvictionPolicy::no_eviction;
}
std::size_t fixed_metadata(EvictionPolicy policy) {
  nivo::detail::EvictionPolicyEngine engine(policy);
  return engine.metadata_bytes();
}
// Runs a scenario per policy and names the policy in any failure.
void for_each_policy(std::span<const EvictionPolicy> policies,
                     const std::function<void(EvictionPolicy)>& scenario) {
  for (const EvictionPolicy policy : policies) {
    try {
      scenario(policy);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(policy_name(policy)) + ": " + error.what());
    }
  }
}

// Single-character keys with single-character values charge two bytes each,
// so a capacity of 6 holds exactly three entries.
struct PolicyFixture {
  std::shared_ptr<ManualClock> clock = std::make_shared<ManualClock>();
  std::unique_ptr<nivo::Cache> cache;
  nivo::PoolId pool = nivo::invalid_pool;
  PolicyFixture(EvictionPolicy policy, std::size_t capacity, std::uint64_t seed = 1,
                nivo::S3FifoConfig s3_fifo = {}) {
    nivo::Result<std::unique_ptr<nivo::Cache>> created =
        nivo::Cache::create({.capacity_bytes = capacity * 4, .clock = clock, .reaper = no_reaper});
    CHECK(created);
    cache = std::move(created).value();
    nivo::Result<nivo::PoolId> added = cache->add_pool(
        {.name = "policy", .capacity_bytes = capacity, .eviction = policy, .eviction_seed = seed,
         .s3_fifo = s3_fifo});
    CHECK(added);
    pool = added.value();
  }
  Status put(std::string_view key, std::string_view value, nivo::WriteOptions options = {}) {
    return cache->put(pool, key, bytes(value), options).status();
  }
  nivo::ReadHandle pin(std::string_view key, std::string_view value) {
    nivo::Result<nivo::ReadHandle> stored = cache->put(pool, key, bytes(value));
    CHECK(stored);
    return std::move(stored).value();
  }
  bool touch(std::string_view key) { return cache->find(pool, key).has_value(); }
  // Which of the given single-character keys are present; peek() leaves
  // every policy's state untouched.
  std::string resident(std::string_view keys) {
    std::string result;
    for (const char key : keys)
      if (cache->peek(pool, std::string_view(&key, 1))) result.push_back(key);
    return result;
  }
  nivo::PoolStats stats() { return cache->pool_stats(pool).value(); }
};

// --- Algorithm semantics ---------------------------------------------------

void lru_policy() {
  PolicyFixture f(EvictionPolicy::lru, 6);
  for (const std::string_view key : {"A", "B", "C"}) CHECK(f.put(key, "1") == Status::ok);
  CHECK(f.touch("A"));
  CHECK(f.put("D", "1") == Status::ok);
  CHECK(f.resident("ABCD") == "ACD");
  // peek() does not refresh: C stays the least recently used entry.
  CHECK(f.cache->peek(f.pool, "C"));
  CHECK(f.put("E", "1") == Status::ok);
  CHECK(f.resident("ACDE") == "ADE");
  // A replacement is a write, so it becomes the most recently used entry.
  // The key being written is never its own victim; D goes instead.
  CHECK(f.put("A", "2") == Status::ok);
  CHECK(f.resident("ADE") == "AE");
  CHECK(f.put("F", "1") == Status::ok);
  CHECK(f.put("G", "1") == Status::ok);
  CHECK(f.resident("AEFG") == "AFG");
  const nivo::PoolStats stats = f.stats();
  CHECK(stats.counters.evictions == 4 && stats.counters.hits == 1);
  CHECK(stats.eviction.attempts == 4 && stats.eviction.successes == 4 && stats.eviction.failures == 0);
}

void fifo_policy_order() {
  PolicyFixture f(EvictionPolicy::fifo, 6);
  for (const std::string_view key : {"A", "B", "C"}) CHECK(f.put(key, "1") == Status::ok);
  CHECK(f.touch("A"));  // Reads never change publication order.
  CHECK(f.put("D", "1") == Status::ok);
  CHECK(f.resident("ABCD") == "BCD");
  // Replacing B publishes a new version: C is evicted to make room (B is the
  // protected key) and the new B becomes the newest entry.
  CHECK(f.put("B", "2") == Status::ok);
  CHECK(f.resident("BCD") == "BD");
  CHECK(text(f.cache->peek(f.pool, "B").value()) == "2");
  CHECK(f.put("E", "1") == Status::ok);
  CHECK(f.put("F", "1") == Status::ok);
  CHECK(f.resident("BDEF") == "BEF");
  CHECK(f.stats().counters.evictions == 3);
}

void lfu_policy() {
  PolicyFixture f(EvictionPolicy::lfu, 6);
  for (const std::string_view key : {"A", "B", "C"}) CHECK(f.put(key, "1") == Status::ok);
  CHECK(f.touch("A") && f.touch("A") && f.touch("C"));  // A=3, B=1, C=2
  CHECK(f.put("D", "1") == Status::ok);
  CHECK(f.resident("ABCD") == "ACD");
  // A new entry starts at 1 and is the least frequent.
  CHECK(f.put("E", "1") == Status::ok);
  CHECK(f.resident("ACDE") == "ACE");
  // C and E now share frequency 2; C reached it first, so it goes first.
  CHECK(f.touch("E"));
  CHECK(f.put("F", "1") == Status::ok);
  CHECK(f.resident("ACEF") == "AEF");
  // A replacement keeps the key's count (3). Had it restarted at 1, the
  // older A would lose the tie with G below.
  CHECK(f.put("A", "2") == Status::ok);
  CHECK(f.resident("AEF") == "AE");
  CHECK(f.put("G", "1") == Status::ok);
  CHECK(f.put("H", "1") == Status::ok);
  CHECK(f.resident("AEGH") == "AEH");
  CHECK(f.stats().counters.evictions == 5);
}

void lfu_saturation_and_aging() {
  using Lfu = nivo::detail::LfuEvictionPolicy;
  std::array<nivo::detail::Entry, 4> entries;
  const std::array<std::string_view, 4> keys{"a", "b", "c", "d"};
  nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::lfu);
  Lfu& lfu = *engine.get_if<Lfu>();
  for (std::size_t i = 0; i < 3; ++i) {
    entries[i].key = keys[i];
    entries[i].version = i + 1;
    CHECK(engine.publish(entries[i], nullptr));
  }
  const std::size_t window = Lfu::aging_window(3);
  CHECK(window == Lfu::min_aging_window);
  for (std::size_t i = 0; i < 20; ++i) engine.record_access(entries[2]);
  CHECK(lfu.frequency(entries[2]) == Lfu::max_frequency);  // saturated, not 21
  for (std::size_t i = 0; i < 13; ++i) engine.record_access(entries[0]);
  for (std::size_t i = 0; i < 4; ++i) engine.record_access(entries[1]);
  CHECK(lfu.frequency(entries[0]) == 14 && lfu.frequency(entries[1]) == 5);
  CHECK(lfu.aging_epochs() == 0);
  for (std::size_t i = 37; i < window; ++i) engine.record_access(entries[2]);
  // The window-th access halves every count in O(levels).
  CHECK(lfu.aging_epochs() == 1);
  CHECK(lfu.frequency(entries[0]) == 7 && lfu.frequency(entries[1]) == 2 &&
        lfu.frequency(entries[2]) == 7);
  {
    // Levels 14 (a) and 15 (c) merged into 7; the lower half goes first.
    nivo::detail::VictimSelection order = engine.select_victims();
    CHECK(order.next() == nivo::detail::CacheIndex::identity(entries[1]));
    CHECK(order.next() == nivo::detail::CacheIndex::identity(entries[0]));
    CHECK(order.next() == nivo::detail::CacheIndex::identity(entries[2]));
    CHECK(!order.next());
  }
  // Untouched counts keep decaying lazily until popularity is forgotten.
  for (std::size_t epoch = 0; epoch < 4; ++epoch)
    for (std::size_t i = 0; i < window; ++i) engine.record_access(entries[2]);
  CHECK(lfu.aging_epochs() == 5);
  CHECK(lfu.frequency(entries[0]) == 0 && lfu.frequency(entries[1]) == 0);
  CHECK(lfu.frequency(entries[2]) == 7);
  // A new entry (count 1) now outranks the formerly hot but stale a.
  entries[3].key = keys[3];
  entries[3].version = 4;
  CHECK(engine.publish(entries[3], nullptr));
  {
    nivo::detail::VictimSelection order = engine.select_victims();
    std::set<nivo::Version> stale{order.next()->version, order.next()->version};
    CHECK(stale == std::set<nivo::Version>{1, 2});
    CHECK(order.next() == nivo::detail::CacheIndex::identity(entries[3]));
    CHECK(order.next() == nivo::detail::CacheIndex::identity(entries[2]));
    CHECK(!order.next());
  }
  for (nivo::detail::Entry& entry : entries) engine.remove(entry);
  CHECK(engine.metadata_bytes() == 0);
}

void clock_policy() {
  {
    PolicyFixture f(EvictionPolicy::clock, 6);
    for (const std::string_view key : {"A", "B", "C"}) CHECK(f.put(key, "1") == Status::ok);
    CHECK(f.touch("A"));
    // The hand gives A a second chance (clearing its bit) and evicts B.
    CHECK(f.put("D", "1") == Status::ok);
    CHECK(f.resident("ABCD") == "ACD");
    // D was placed behind the hand, so C is examined before it.
    CHECK(f.put("E", "1") == Status::ok);
    CHECK(f.resident("ACDE") == "ADE");
    // A's chance was used up by the first sweep.
    CHECK(f.put("F", "1") == Status::ok);
    CHECK(f.resident("ADEF") == "DEF");
    CHECK(f.stats().counters.evictions == 3);
  }
  {
    // All referenced: one sweep clears every bit, then the hand's entry goes.
    PolicyFixture f(EvictionPolicy::clock, 6);
    for (const std::string_view key : {"A", "B", "C"}) CHECK(f.put(key, "1") == Status::ok);
    CHECK(f.touch("A") && f.touch("B") && f.touch("C"));
    CHECK(f.put("D", "1") == Status::ok);
    CHECK(f.resident("ABCD") == "BCD");
    CHECK(f.touch("B"));
    CHECK(f.put("E", "1") == Status::ok);
    CHECK(f.resident("BCDE") == "BDE");
  }
}

std::vector<std::string> random_history(std::uint64_t seed) {
  PolicyFixture f(EvictionPolicy::random, 16, seed);
  const std::string_view alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::vector<std::string> history;
  for (const char key : alphabet) {
    CHECK(f.put(std::string_view(&key, 1), "v") == Status::ok);
    const std::string resident = f.resident(alphabet);
    CHECK(resident.size() == std::min<std::size_t>(history.size() + 1, 8));
    CHECK(resident.find(key) != std::string::npos);
    history.push_back(resident);
  }
  CHECK(f.stats().counters.evictions == alphabet.size() - 8);
  return history;
}

void random_policy() {
  // A fixed seed reproduces the exact victims; another seed differs.
  CHECK(random_history(42) == random_history(42));
  CHECK(random_history(42) != random_history(43));
  // Pinned entries and the key being written are never chosen.
  PolicyFixture f(EvictionPolicy::random, 16, 7);
  nivo::ReadHandle pinned = f.pin("a", "p");
  for (char key = 'b'; key <= 'z'; ++key) {
    CHECK(f.put(std::string_view(&key, 1), "v") == Status::ok);
    CHECK(f.cache->peek(f.pool, "a"));
    if (key > 'i') {
      // The pool is full; rewriting the newest key must not evict that key.
      CHECK(f.put(std::string_view(&key, 1), "w") == Status::ok);
      CHECK(text(f.cache->peek(f.pool, std::string_view(&key, 1)).value()) == "w");
    }
  }
  CHECK(text(pinned) == "p");
}

void no_eviction_policy() {
  PolicyFixture f(EvictionPolicy::no_eviction, 6);
  CHECK(f.put("A", "1") == Status::ok && f.put("B", "1") == Status::ok);
  CHECK(f.put("C", "1", {.ttl = std::chrono::milliseconds{10}}) == Status::ok);
  // Full: the write fails explicitly and nothing is evicted.
  CHECK(f.put("D", "1") == Status::resource_exhausted);
  CHECK(f.resident("ABCD") == "ABC");
  // Failed allocations roll back their reservation.
  CHECK(f.cache->allocate(f.pool, "X", 1).status() == Status::resource_exhausted);
  nivo::PoolStats stats = f.stats();
  CHECK(stats.allocated_bytes == 6 && stats.counters.evictions == 0);
  CHECK(stats.counters.allocation_failures == 2);
  CHECK(stats.eviction.attempts == 2 && stats.eviction.failures == 2 && stats.eviction.successes == 0);
  CHECK(stats.eviction.policy_metadata_bytes == 0);
  // A replacement needs room for both versions; the old value survives.
  CHECK(f.put("A", "2") == Status::resource_exhausted);
  CHECK(text(f.cache->peek(f.pool, "A").value()) == "1");
  // Shrinking below the used bytes fails and keeps the old capacity.
  CHECK(f.cache->resize_pool(f.pool, 4) == Status::resource_exhausted);
  CHECK(f.stats().capacity_bytes == 6 && f.resident("ABC") == "ABC");
  // Expired entries are not live keys: pressure still reclaims them.
  f.clock->advance(std::chrono::milliseconds{10});
  CHECK(f.put("D", "1") == Status::ok);
  CHECK(f.resident("ABCD") == "ABD");
  stats = f.stats();
  CHECK(stats.counters.expirations == 1 && stats.counters.evictions == 0);
  // Explicit deletion and handle release work as usual.
  nivo::ReadHandle held = std::move(f.cache->find(f.pool, "B")).value();
  CHECK(f.cache->erase(f.pool, "B") == Status::ok);
  CHECK(f.put("E", "1") == Status::resource_exhausted);  // B's bytes are still pinned
  held.reset();
  CHECK(f.put("E", "1") == Status::ok);
  CHECK(f.resident("ABDE") == "ADE");
  CHECK(f.cache->erase(f.pool, "A") == Status::ok);
  CHECK(f.put("A", "3") == Status::ok);
  CHECK(f.stats().counters.evictions == 0);
}

// White-box entries for driving one policy directly. Keys are distinct,
// versions are unique and charges are explicit, as Core would set them.
struct EntrySet {
  std::vector<std::string> keys;
  std::unique_ptr<nivo::detail::Entry[]> entries;
  explicit EntrySet(std::size_t count, std::size_t charge = 10)
      : entries(new nivo::detail::Entry[count]) {
    for (std::size_t i = 0; i < count; ++i) keys.push_back("key-" + std::to_string(i));
    for (std::size_t i = 0; i < count; ++i) {
      entries[i].key = keys[i];
      entries[i].version = i + 1;
      entries[i].charge = charge;
    }
  }
  nivo::detail::Entry& operator[](std::size_t i) { return entries[i]; }
};
nivo::detail::EntryId id_of(const nivo::detail::Entry& entry) {
  return nivo::detail::CacheIndex::identity(entry);
}
// Candidate versions in order, without removing any.
std::vector<nivo::Version> candidates(nivo::detail::EvictionPolicyEngine& engine) {
  std::vector<nivo::Version> order;
  nivo::detail::VictimSelection selection = engine.select_victims();
  while (const std::optional<nivo::detail::EntryId> candidate = selection.next())
    order.push_back(candidate->version);
  return order;
}

void sieve_policy() {
  PolicyFixture f(EvictionPolicy::sieve, 8);  // four entries
  for (const std::string_view key : {"A", "B", "C", "D"}) CHECK(f.put(key, "1") == Status::ok);
  CHECK(f.touch("A") && f.touch("C"));
  // The hand starts at the tail: A is visited (cleared, kept in place), B goes.
  CHECK(f.put("E", "1") == Status::ok);
  CHECK(f.resident("ABCDE") == "ACDE");
  // The hand resumes at C, not at the tail: C is cleared and D goes.
  CHECK(f.put("F", "1") == Status::ok);
  CHECK(f.resident("ACDEF") == "ACEF");
  // E entered at the head after the hand had passed A and C, so the unvisited
  // newcomer is evicted before them: quick demotion of new entries.
  CHECK(f.put("G", "1") == Status::ok);
  CHECK(f.resident("ACEFG") == "ACFG");
  // The hand keeps meeting unvisited newcomers before it can wrap, so the
  // retained A and C survive although they are the oldest entries.
  CHECK(f.put("H", "1") == Status::ok);
  CHECK(f.put("I", "1") == Status::ok);
  CHECK(f.resident("ACFGHI") == "ACHI");
  CHECK(f.stats().counters.evictions == 5);
}

void sieve_hits_do_not_move_entries() {
  using Sieve = nivo::detail::SieveEvictionPolicy;
  EntrySet e(5);
  nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::sieve);
  Sieve& sieve = *engine.get_if<Sieve>();
  for (std::size_t i = 0; i < 5; ++i) CHECK(engine.publish(e[i], nullptr));
  engine.record_access(e[1]);
  engine.record_access(e[3]);
  engine.record_access(e[3]);  // already set: no second write
  CHECK(sieve.is_visited(e[1]) && sieve.is_visited(e[3]) && !sieve.is_visited(e[0]));
  // Tail to head, skipping (and clearing) visited entries, then wrapping to
  // them: the queue itself never changed order.
  CHECK(candidates(engine) == std::vector<nivo::Version>({1, 3, 5, 2, 4}));
  CHECK(!sieve.is_visited(e[1]) && !sieve.is_visited(e[3]));
  // The hand persists between selections: the last candidate was e3, so the
  // search resumes at e4 rather than at the tail.
  CHECK(sieve.hand() == &e[4]);
  {
    nivo::detail::VictimSelection selection = engine.select_victims();
    CHECK(selection.next() == id_of(e[4]));
    engine.remove(e[4]);
  }
  CHECK(sieve.hand() == nullptr);  // e4 was the head: wrap to the tail
  engine.record_access(e[1]);
  {
    nivo::detail::VictimSelection selection = engine.select_victims();
    CHECK(selection.next() == id_of(e[0]));
    engine.remove(e[0]);
  }
  CHECK(sieve.hand() == &e[1]);
  // A replacement keeps the position, the visited bit and the hand.
  nivo::detail::Entry replacement;
  replacement.key = e.keys[1];
  replacement.version = 10;
  CHECK(engine.publish(replacement, &e[1]));
  CHECK(sieve.is_visited(replacement) && sieve.hand() == &replacement);
  CHECK(candidates(engine) == std::vector<nivo::Version>({3, 4, 10}));
  CHECK(sieve.hand() == &e[2]);
  // Removing the hand's entry moves the hand towards the head.
  engine.remove(e[2]);
  CHECK(sieve.hand() == &e[3]);
  engine.remove(e[3]);
  engine.remove(replacement);
  CHECK(engine.metadata_bytes() == 0 && sieve.hand() == nullptr);
}

void s3fifo_queues_and_ghost() {
  using S3 = nivo::detail::S3FifoEvictionPolicy;
  EntrySet e(12);
  nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::s3_fifo);
  S3& s3 = *engine.get_if<S3>();
  for (std::size_t i = 0; i < 10; ++i) CHECK(engine.publish(e[i], nullptr));
  CHECK(s3.snapshot().small_entries == 10 && s3.snapshot().small_bytes == 100);
  // Counters saturate at three; most hits on a popular entry write nothing.
  for (std::size_t i = 0; i < 10; ++i) engine.record_access(e[0]);
  CHECK(s3.frequency(e[0]) == 3);
  engine.record_access(e[1]);  // accessed once: not enough to promote
  {
    nivo::detail::VictimSelection selection = engine.select_victims();
    // The small tail e0 (freq 3 > 1) moves to main with its counter cleared;
    // e1 (freq 1) is the first victim.
    CHECK(selection.next() == id_of(e[1]));
    CHECK(s3.in_main_queue(e[0]) && s3.frequency(e[0]) == 0);
    engine.remove(e[1]);  // accepted by Core: evicted into the ghost
    // A rejected candidate (e2) is neither evicted nor remembered.
    CHECK(selection.next() == id_of(e[2]));
  }
  S3::Snapshot snapshot = s3.snapshot();
  CHECK(snapshot.small_entries == 8 && snapshot.main_entries == 1 && snapshot.promotions == 1);
  CHECK(snapshot.ghost_entries == 1 && snapshot.ghost_limit == 8);  // 90% of 9 entries
  // An explicit removal is not an eviction and is not remembered either.
  engine.remove(e[9]);
  CHECK(s3.snapshot().ghost_entries == 1);
  // The evicted key comes back: a ghost hit admits it straight into main.
  nivo::detail::Entry returning;
  returning.key = e.keys[1];
  returning.version = 20;
  returning.charge = 10;
  CHECK(engine.publish(returning, nullptr));
  snapshot = s3.snapshot();
  CHECK(s3.in_main_queue(returning) && snapshot.ghost_hits == 1 && snapshot.ghost_entries == 0);
  // A key that was never evicted goes to small.
  CHECK(engine.publish(e[10], nullptr) && !s3.in_main_queue(e[10]));
  // A replacement keeps the queue and the counter.
  engine.record_access(returning);
  engine.record_access(returning);
  nivo::detail::Entry rewritten;
  rewritten.key = e.keys[1];
  rewritten.version = 21;
  rewritten.charge = 30;
  CHECK(engine.publish(rewritten, &returning));
  CHECK(s3.in_main_queue(rewritten) && s3.frequency(rewritten) == 2);
  CHECK(s3.snapshot().main_bytes == 40);  // e0 (10) + rewritten (30)
  // Drain: small entries go first while small holds >= 10% of the bytes;
  // then main reinserts rewritten twice (2 -> 1 -> 0) before evicting it.
  std::vector<nivo::Version> order;
  {
    nivo::detail::VictimSelection selection = engine.select_victims();
    while (const std::optional<nivo::detail::EntryId> candidate = selection.next()) {
      order.push_back(candidate->version);
      for (std::size_t i = 0; i < 11; ++i)
        if (e[i].version == candidate->version) engine.remove(e[i]);
      if (candidate->version == rewritten.version) engine.remove(rewritten);
    }
  }
  CHECK(order == std::vector<nivo::Version>({3, 4, 5, 6, 7, 8, 9, 11, 1, 21}));
  CHECK(engine.metadata_bytes() == 0);  // the ghost is released with the last entry
}

void s3fifo_byte_shares() {
  using S3 = nivo::detail::S3FifoEvictionPolicy;
  EntrySet e(10);
  e[9].charge = 5;
  nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::s3_fifo);
  S3& s3 = *engine.get_if<S3>();
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK(engine.publish(e[i], nullptr));
    if (i < 9) for (std::size_t hit = 0; hit < 2; ++hit) engine.record_access(e[i]);
  }
  // e0..e8 are promoted. Small then holds 1 of 10 entries but only 5 of 95
  // bytes (< 10%), so main's tail is offered before e9: shares are measured
  // in bytes, not entries. Nothing is removed, so all ten are offered once.
  CHECK(candidates(engine) == std::vector<nivo::Version>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  S3::Snapshot snapshot = s3.snapshot();
  CHECK(snapshot.main_entries == 9 && snapshot.small_bytes == 5 && snapshot.main_bytes == 90);
  CHECK(snapshot.promotions == 9);
  // A rejected candidate reached again in the same selection is rotated to
  // its queue's head, so after offering all of main, e8 is main's tail.
  {
    nivo::detail::VictimSelection selection = engine.select_victims();
    CHECK(selection.next() == id_of(e[8]));
  }
  for (std::size_t i = 0; i < 10; ++i) engine.remove(e[i]);
}

void s3fifo_ghost_is_bounded() {
  using S3 = nivo::detail::S3FifoEvictionPolicy;
  constexpr std::size_t resident = 100;
  constexpr std::size_t total = 5000;
  // Values are never retained: ghost memory depends on entry counts only.
  for (const std::size_t charge : {std::size_t{16}, std::size_t{1} << 20}) {
    EntrySet e(total, charge);
    nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::s3_fifo);
    S3& s3 = *engine.get_if<S3>();
    std::size_t oldest = 0;
    for (std::size_t i = 0; i < total; ++i) {
      CHECK(engine.publish(e[i], nullptr));
      if (i - oldest + 1 > resident) {
        nivo::detail::VictimSelection selection = engine.select_victims();
        const std::optional<nivo::detail::EntryId> victim = selection.next();
        CHECK(victim == id_of(e[oldest]));  // one-hit wonders leave in FIFO order
        engine.remove(e[oldest++]);
      }
      const S3::Snapshot snapshot = s3.snapshot();
      CHECK(snapshot.ghost_entries <= snapshot.ghost_limit);
      CHECK(snapshot.ghost_bytes <= 22 * snapshot.ghost_limit + 128);
    }
    const S3::Snapshot snapshot = s3.snapshot();
    CHECK(snapshot.ghost_limit == 90 && snapshot.ghost_entries > 80);
    CHECK(snapshot.ghost_bytes == 1024);  // 16 buckets of one cache line
    CHECK(engine.metadata_bytes() == resident * nivo::detail::policy_entry_bytes + 1024);
    // The newest evictions are remembered, the oldest have aged out.
    nivo::detail::Entry recent;
    recent.key = e.keys[oldest - 1];
    recent.version = total + 1;
    CHECK(engine.publish(recent, nullptr) && s3.in_main_queue(recent));
    nivo::detail::Entry ancient;
    ancient.key = e.keys[0];
    ancient.version = total + 2;
    CHECK(engine.publish(ancient, nullptr) && !s3.in_main_queue(ancient));
    engine.remove(recent);
    engine.remove(ancient);
    for (std::size_t i = oldest; i < total; ++i) engine.remove(e[i]);
    CHECK(engine.metadata_bytes() == 0);
  }
}

void s3fifo_configuration() {
  using S3 = nivo::detail::S3FifoEvictionPolicy;
  {
    // Promotion after one access, counters capped at five, ghost disabled.
    EntrySet e(4);
    nivo::detail::EvictionPolicyEngine engine(
        EvictionPolicy::s3_fifo, 0,
        {.ghost_queue_percent = 0, .max_frequency = 5, .promotion_frequency = 1});
    S3& s3 = *engine.get_if<S3>();
    for (std::size_t i = 0; i < 3; ++i) CHECK(engine.publish(e[i], nullptr));
    for (std::size_t i = 0; i < 9; ++i) engine.record_access(e[0]);
    CHECK(s3.frequency(e[0]) == 5);
    engine.record_access(e[1]);
    {
      nivo::detail::VictimSelection selection = engine.select_victims();
      CHECK(selection.next() == id_of(e[2]));  // e0 and e1 were promoted
      engine.remove(e[2]);
    }
    CHECK(s3.snapshot().main_entries == 2 && s3.snapshot().ghost_bytes == 0);
    CHECK(engine.publish(e[3], nullptr) && !s3.in_main_queue(e[3]));
    nivo::detail::Entry again;
    again.key = e.keys[2];
    again.version = 9;
    CHECK(engine.publish(again, nullptr) && !s3.in_main_queue(again));  // no ghost
    for (nivo::detail::Entry* entry : {&e[0], &e[1], &e[3], &again}) engine.remove(*entry);
  }
  {
    // A 50% small queue keeps evicting from small until it is half the bytes.
    EntrySet e(4);
    nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::s3_fifo, 0, {.small_queue_percent = 50});
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(engine.publish(e[i], nullptr));
      if (i < 2) for (std::size_t hit = 0; hit < 2; ++hit) engine.record_access(e[i]);
    }
    std::vector<nivo::Version> order;
    {
      nivo::detail::VictimSelection selection = engine.select_victims();
      while (const std::optional<nivo::detail::EntryId> candidate = selection.next()) {
        order.push_back(candidate->version);
        engine.remove(e[candidate->version - 1]);
      }
    }
    // e0, e1 promoted; e2 evicted (small 20 of 40 bytes, still due);
    // then small (10 of 30) is under half, so main's e0 goes before e3.
    CHECK(order == std::vector<nivo::Version>({3, 1, 4, 2}));
  }
  // Invalid parameters are rejected when the pool is created.
  PolicyFixture f(EvictionPolicy::lru, 64);
  const std::array<nivo::S3FifoConfig, 6> invalid{{
      {.small_queue_percent = 0}, {.small_queue_percent = 100}, {.ghost_queue_percent = 401},
      {.max_frequency = 0}, {.max_frequency = 8}, {.max_frequency = 2, .promotion_frequency = 3}}};
  for (const nivo::S3FifoConfig& config : invalid)
    CHECK(f.cache->add_pool({.name = "bad", .capacity_bytes = 8, .eviction = EvictionPolicy::s3_fifo,
                             .s3_fifo = config}).status() == Status::invalid_argument);
  // Other policies ignore the S3-FIFO parameters.
  CHECK(f.cache->add_pool({.name = "lru", .capacity_bytes = 8, .eviction = EvictionPolicy::lru,
                           .s3_fifo = invalid[0]}));
  CHECK(f.cache->add_pool({.name = "s3", .capacity_bytes = 8, .eviction = EvictionPolicy::s3_fifo,
                           .s3_fifo = {.small_queue_percent = 25, .ghost_queue_percent = 400}}));
}

// Twenty hot keys are each read twice, then 300 one-off keys stream through
// a 100-entry pool. LRU loses the hot set; S3-FIFO has promoted it to main and
// SIEVE's hand keeps evicting the newcomers just ahead of it.
void scan_resistance() {
  for (const EvictionPolicy policy : {EvictionPolicy::lru, EvictionPolicy::sieve, EvictionPolicy::s3_fifo, EvictionPolicy::slru, EvictionPolicy::w_tinylfu}) {
    PolicyFixture f(policy, 500);  // four-byte keys and one-byte values: 100 entries
    std::vector<std::string> hot;
    for (std::size_t i = 0; i < 20; ++i) hot.push_back("h" + std::to_string(100 + i));
    for (const std::string& key : hot) CHECK(f.put(key, "v") == Status::ok);
    for (const std::string& key : hot) CHECK(f.touch(key) && f.touch(key));
    for (std::size_t i = 0; i < 300; ++i) CHECK(f.put("s" + std::to_string(100 + i), "v") == Status::ok);
    std::size_t kept = 0;
    for (const std::string& key : hot) kept += f.cache->peek(f.pool, key).has_value() ? 1 : 0;
    try {
      CHECK(kept == (policy == EvictionPolicy::lru ? 0u : 20u));
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(policy_name(policy)) + ": " + error.what());
    }
  }
}

// Readers mark entries while a writer replaces them, the way Core serializes
// writers under stripe locks but never blocks find(). Late marks on detached
// versions must neither corrupt the queues nor resurrect anything. Selection
// excludes readers, as the exclusive pool guard does.
void access_races_replacement() {
  for_each_policy(std::array<EvictionPolicy, 5>{EvictionPolicy::sieve, EvictionPolicy::s3_fifo,
                                                 EvictionPolicy::slru, EvictionPolicy::two_q, EvictionPolicy::w_tinylfu},
                  [](EvictionPolicy policy) {
    constexpr std::size_t keys = 8;
    EntrySet versions(keys * 2);
    nivo::detail::EvictionPolicyEngine engine(policy);
    std::array<std::atomic<std::size_t>, keys> live{};
    for (std::size_t k = 0; k < keys; ++k) {
      versions[k + keys].key = versions.keys[k];
      CHECK(engine.publish(versions[k], nullptr));
      live[k] = k;
    }
    std::atomic<bool> stop{false};
    std::shared_mutex pool_guard;
    std::vector<std::thread> readers;
    for (std::size_t r = 0; r < 4; ++r) {
      readers.emplace_back([&, r] {
        std::size_t i = r;
        while (!stop.load(std::memory_order_relaxed)) {
          // Either version of a key, live or retired, as a pinned reader may hold.
          std::shared_lock<std::shared_mutex> shared(pool_guard);
          engine.record_access(versions[(i * 7) % (keys * 2)]);
          ++i;
        }
      });
    }
    std::uint64_t next_version = keys * 2 + 1;
    // Readers must be joined before a failed check unwinds.
    struct JoinReaders {
      std::atomic<bool>& stop;
      std::vector<std::thread>& readers;
      ~JoinReaders() {
        stop = true;
        for (std::thread& reader : readers) if (reader.joinable()) reader.join();
      }
    } join_readers{stop, readers};
    for (std::size_t round = 0; round < 20000; ++round) {
      const std::size_t k = round % keys;
      const std::size_t current = live[k];
      const std::size_t spare = current < keys ? current + keys : current - keys;
      versions[spare].version = next_version++;
      CHECK(engine.publish(versions[spare], &versions[current]));
      live[k] = spare;
      if (round % 97 == 0) {
        // Evict one entry under the exclusive guard, then readmit it.
        std::unique_lock<std::shared_mutex> exclusive(pool_guard);
        std::optional<nivo::detail::EntryId> victim;
        {
          nivo::detail::VictimSelection selection = engine.select_victims();
          victim = selection.next();
          CHECK(victim.has_value());
          for (std::size_t j = 0; j < keys; ++j)
            if (versions[live[j]].version == victim->version) {
              engine.remove(versions[live[j]]);
              CHECK(engine.publish(versions[live[j]], nullptr));
            }
        }
      }
    }
    stop = true;
    for (std::thread& reader : readers) reader.join();
    // Exactly the live versions remain, each offered once.
    std::set<nivo::Version> expected;
    for (std::size_t k = 0; k < keys; ++k) expected.insert(versions[live[k]].version);
    const std::vector<nivo::Version> order = candidates(engine);
    CHECK(std::set<nivo::Version>(order.begin(), order.end()) == expected && order.size() == keys);
    for (std::size_t k = 0; k < keys; ++k) engine.remove(versions[live[k]]);
    CHECK(engine.metadata_bytes() == fixed_metadata(policy));
  });
}

// --- Framework properties --------------------------------------------------

// Every policy offers each member exactly once and then stops, whether or not
// Core removes candidates, so pressure over pinned entries always terminates.
void selection_is_bounded_and_unique() {
  constexpr std::size_t count = 50;
  std::vector<std::string> keys;
  for (std::size_t i = 0; i < count; ++i) keys.push_back("key-" + std::to_string(i));
  for_each_policy(all_policies, [&](EvictionPolicy policy) {
    std::unique_ptr<nivo::detail::Entry[]> entries(new nivo::detail::Entry[count]);
    nivo::detail::EvictionPolicyEngine engine(policy, 9);
    for (std::size_t i = 0; i < count; ++i) {
      entries[i].key = keys[i];
      entries[i].version = i + 1;
      CHECK(engine.publish(entries[i], nullptr));
      if (i % 3 == 0) engine.record_access(entries[i]);
    }
    const bool tracks = policy != EvictionPolicy::no_eviction;
    // Nothing removed (all candidates rejected).
    {
      std::set<nivo::Version> seen;
      nivo::detail::VictimSelection selection = engine.select_victims();
      while (const std::optional<nivo::detail::EntryId> candidate = selection.next())
        CHECK(seen.insert(candidate->version).second);
      CHECK(seen.size() == (tracks ? count : 0));
      CHECK(!selection.next());
    }
    // Remove every other returned candidate and one not yet returned.
    std::vector<bool> present(count, true);
    {
      std::set<nivo::Version> seen;
      nivo::detail::VictimSelection selection = engine.select_victims();
      std::optional<std::size_t> skipped;
      while (const std::optional<nivo::detail::EntryId> candidate = selection.next()) {
        const std::size_t index = candidate->version - 1;
        CHECK(present[index] && seen.insert(candidate->version).second);
        if (seen.size() % 2 == 0) {
          engine.remove(entries[index]);
          present[index] = false;
        }
        if (!skipped) {
          for (std::size_t other = 0; other < count; ++other) {
            if (present[other] && !seen.count(other + 1)) {
              engine.remove(entries[other]);
              present[other] = false;
              skipped = other;
              break;
            }
          }
        }
      }
      if (tracks) CHECK(skipped && seen.size() == count - 1 && !seen.count(*skipped + 1));
    }
    std::size_t remaining = 0;
    {
      nivo::detail::VictimSelection selection = engine.select_victims();
      while (const std::optional<nivo::detail::EntryId> candidate = selection.next()) {
        CHECK(present[candidate->version - 1]);
        ++remaining;
      }
    }
    const std::size_t expected = static_cast<std::size_t>(std::count(present.begin(), present.end(), true));
    CHECK(remaining == (tracks ? expected : 0));
    for (std::size_t i = 0; i < count; ++i)
      if (present[i] && tracks) engine.remove(entries[i]);
    CHECK(engine.metadata_bytes() == fixed_metadata(policy));
  });
}

void variable_size_capacity() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    PolicyFixture f(policy, 20);
    CHECK(f.put("a", "aaaa") == Status::ok);       // charge 5
    CHECK(f.put("b", "bbbb") == Status::ok);       // charge 5
    CHECK(f.put("c", "ccccccccc") == Status::ok);  // charge 10
    // Larger than the pool: rejected up front, nothing evicted.
    CHECK(f.put("x", std::string(20, 'x')) == Status::resource_exhausted);
    CHECK(f.resident("abc") == "abc" && f.stats().eviction.attempts == 0);
    const Status status = f.put("d", "ddddddddd");  // charge 10, needs two small or one large
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(status == Status::resource_exhausted && f.resident("abcd") == "abc");
      return;
    }
    CHECK(status == Status::ok);
    const std::string resident = f.resident("abcd");
    if (policy == EvictionPolicy::random) {
      CHECK(resident == "cd" || resident == "ad" || resident == "bd" || resident == "abd");
    } else if (policy == EvictionPolicy::w_tinylfu) {
      CHECK(resident == "abd");  // window candidate loses a frequency tie
    } else {
      CHECK(resident == "cd");  // the two oldest small entries made room
    }
    nivo::PoolStats stats = f.stats();
    CHECK(stats.allocated_bytes <= 20 && stats.allocated_bytes == stats.indexed_bytes);
    // A value exactly filling the pool evicts everything else.
    CHECK(f.put("y", std::string(19, 'y')) == Status::ok);
    CHECK(f.resident("abcdy") == "y");
    stats = f.stats();
    CHECK(stats.allocated_bytes == 20 && stats.indexed_items == 1);
    CHECK(stats.eviction.failures == 0 && stats.eviction.successes == 2);
  });
}

void insert_erase_overwrite() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    PolicyFixture f(policy, 64);
    CHECK(f.put("a", "1") == Status::ok && f.put("b", "1") == Status::ok && f.put("c", "1") == Status::ok);
    CHECK(f.cache->erase(f.pool, "b") == Status::ok);
    CHECK(f.cache->erase(f.pool, "b") == Status::not_found);
    CHECK(f.put("c", "22") == Status::ok);
    nivo::PoolStats stats = f.stats();
    CHECK(stats.counters.inserts == 3 && stats.counters.replacements == 1 && stats.counters.erases == 1);
    CHECK(stats.indexed_items == 2);
    const std::size_t tracked = stats.eviction.policy_metadata_bytes;
    if (policy == EvictionPolicy::no_eviction) CHECK(tracked == 0);
    else if (!exact_metadata(policy)) CHECK(tracked >= 2 * nivo::detail::policy_entry_bytes);
    else CHECK(tracked == 2 * nivo::detail::policy_entry_bytes);
    // Shrinking to one entry evicts exactly one; erased and replaced versions
    // are no longer candidates.
    const Status resized = f.cache->resize_pool(f.pool, 3);
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(resized == Status::resource_exhausted && f.resident("abc") == "ac");
    } else {
      CHECK(resized == Status::ok);
      const std::string resident = f.resident("abc");
      CHECK(policy == EvictionPolicy::random ? resident.size() == 1 : resident == "c");
      CHECK(f.stats().counters.evictions == 1);
    }
    CHECK(f.cache->clear(f.pool) == Status::ok);
    stats = f.stats();
    CHECK(stats.allocated_bytes == 0 && stats.eviction.policy_metadata_bytes == fixed_metadata(policy));
  });
}

void all_entries_pinned() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    PolicyFixture f(policy, 6);
    nivo::ReadHandle a = f.pin("a", "1");
    nivo::ReadHandle b = f.pin("b", "1");
    nivo::ReadHandle c = f.pin("c", "1");
    // Nothing is reclaimable: the request fails instead of spinning.
    CHECK(f.put("d", "1") == Status::resource_exhausted);
    CHECK(f.resident("abcd") == "abc");
    nivo::PoolStats stats = f.stats();
    CHECK(stats.counters.evictions == 0 && stats.eviction.failures == 1);
    b.reset();
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(f.cache->allocate(f.pool, "d", 1).status() == Status::resource_exhausted);
      CHECK(f.resident("abc") == "abc");
      return;
    }
    nivo::WriteHandle draft = std::move(f.cache->allocate(f.pool, "d", 1)).value();
    CHECK(f.resident("abc") == "ac");  // b was the only reclaimable entry
    // An unpublished draft is not a candidate either.
    CHECK(f.cache->allocate(f.pool, "e", 1).status() == Status::resource_exhausted);
    CHECK(f.cache->publish(std::move(draft)));
    a.reset();
    CHECK(f.put("e", "1") == Status::ok);
    const std::string resident = f.resident("acde");
    if (policy == EvictionPolicy::random) CHECK(resident == "cde" || resident == "ace");
    // SIEVE's hand stopped after c (rejected while pinned), so it reaches the
    // newer d before wrapping back to a.
    else if (policy == EvictionPolicy::sieve || policy == EvictionPolicy::w_tinylfu) CHECK(resident == "ace");
    else CHECK(resident == "cde");
    stats = f.stats();
    CHECK(stats.counters.evictions == 2 && stats.eviction.failures == 2 && stats.eviction.successes == 2);
  });
}

void handles_survive_logical_removal() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    PolicyFixture f(policy, 12);
    nivo::ReadHandle erased = f.pin("a", "aaaa");
    CHECK(f.cache->erase(f.pool, "a") == Status::ok);
    nivo::ReadHandle replaced = f.pin("b", "bb");
    CHECK(f.put("b", "BB") == Status::ok);
    // Logically gone, physically alive: readable, still charged, untracked.
    CHECK(text(erased) == "aaaa" && text(replaced) == "bb");
    CHECK(text(f.cache->find(f.pool, "b").value()) == "BB");
    nivo::PoolStats stats = f.stats();
    CHECK(stats.allocated_bytes == 11 && stats.indexed_items == 1);
    if (exact_metadata(policy))
      CHECK(stats.eviction.policy_metadata_bytes == nivo::detail::policy_entry_bytes);
    // Pressure can only evict the indexed version; retained bytes stay charged
    // until the last handle goes away.
    const Status status = f.put("c", "c");
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(status == Status::resource_exhausted);
    } else {
      CHECK(status == Status::ok && f.resident("abc") == "c");
      CHECK(text(erased) == "aaaa" && text(replaced) == "bb");
      CHECK(f.stats().allocated_bytes == 10);
    }
    erased.reset();
    replaced.reset();
    stats = f.stats();
    CHECK(stats.allocated_bytes == stats.indexed_bytes);
  });
}

void eviction_stats_and_metadata() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    PolicyFixture f(policy, 6);
    CHECK(f.stats().eviction.policy_metadata_bytes == fixed_metadata(policy));
    CHECK(f.put("a", "1") == Status::ok && f.put("b", "1") == Status::ok && f.put("c", "1") == Status::ok);
    const std::size_t metadata = f.stats().eviction.policy_metadata_bytes;
    if (policy == EvictionPolicy::no_eviction) CHECK(metadata == 0);
    else if (policy == EvictionPolicy::random)
      CHECK(metadata >= 3 * (nivo::detail::policy_entry_bytes + sizeof(void*)));
    else if ((policy == EvictionPolicy::s3_fifo || policy == EvictionPolicy::two_q || policy == EvictionPolicy::w_tinylfu)) CHECK(metadata > 3 * nivo::detail::policy_entry_bytes);
    else CHECK(metadata == 3 * nivo::detail::policy_entry_bytes);
    // Reads are not attempts and do not touch the counters.
    for (std::size_t i = 0; i < 100; ++i) CHECK(f.touch("a"));
    CHECK(f.stats().eviction.attempts == 0);
    const bool evicts = policy != EvictionPolicy::no_eviction;
    CHECK(f.put("d", "1") == (evicts ? Status::ok : Status::resource_exhausted));
    std::vector<nivo::ReadHandle> pins;
    for (const std::string_view key : {"a", "b", "c", "d"}) {
      nivo::Result<nivo::ReadHandle> found = f.cache->find(f.pool, key);
      if (found) pins.push_back(std::move(found).value());
    }
    CHECK(pins.size() == 3);
    CHECK(f.put("e", "1") == Status::resource_exhausted);
    const nivo::EvictionStats stats = f.stats().eviction;
    CHECK(stats.attempts == 2);
    CHECK(stats.successes == (evicts ? 1u : 0u) && stats.failures == (evicts ? 1u : 2u));
    CHECK(stats.selection_time >= stats.max_selection_time && stats.max_selection_time.count() >= 0);
    // The cache-wide view sums pools.
    const nivo::PoolId other = f.cache->add_pool({.name = "other", .capacity_bytes = 1, .eviction = policy}).value();
    CHECK(f.cache->put(other, "z", {}).has_value());
    CHECK(f.cache->put(other, "y", {}).has_value() == evicts);
    const nivo::EvictionStats total = f.cache->stats().eviction;
    const nivo::EvictionStats second = f.cache->pool_stats(other).value().eviction;
    CHECK(total.attempts == stats.attempts + second.attempts && second.attempts == 1);
    CHECK(total.failures == stats.failures + second.failures);
    CHECK(total.policy_metadata_bytes == stats.policy_metadata_bytes + second.policy_metadata_bytes);
  });
}

void concurrent_reads_and_writes() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    constexpr std::size_t value_size = 64;
    PolicyFixture f(policy, 40 * 70);
    std::atomic<bool> failed{false};
    std::barrier gate(7);
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < 7; ++worker) {
      workers.emplace_back([&, worker] {
        gate.arrive_and_wait();
        try {
          for (std::size_t i = 0; i < 1500; ++i) {
            const std::string key = "k" + std::to_string((i * 7 + worker * 13) % 96);
            if (worker < 3) {
              const std::string value(value_size, static_cast<char>('a' + (i + worker) % 26));
              nivo::Result<nivo::ReadHandle> stored = f.cache->put(f.pool, key, bytes(value));
              if (!stored) { CHECK(stored.status() == Status::resource_exhausted); continue; }
              CHECK(text(stored.value()) == value);
            } else if (worker < 6) {
              nivo::Result<nivo::ReadHandle> found =
                  worker % 2 ? f.cache->find(f.pool, key) : f.cache->peek(f.pool, key);
              if (!found) { CHECK(found.status() == Status::not_found); continue; }
              const std::string value = text(found.value());
              CHECK(value.size() == value_size && value.find_first_not_of(value.front()) == std::string::npos);
              std::this_thread::yield();
              CHECK(text(found.value()) == value);
            } else {
              const Status erased = f.cache->erase(f.pool, key);
              CHECK(erased == Status::ok || erased == Status::not_found);
            }
          }
        } catch (...) { failed = true; }
      });
    }
    for (std::thread& worker : workers) worker.join();
    CHECK(!failed);
    nivo::PoolStats stats = f.stats();
    CHECK(stats.allocated_bytes <= stats.capacity_bytes && stats.allocated_bytes == stats.indexed_bytes);
    CHECK(stats.eviction.attempts == stats.eviction.successes + stats.eviction.failures);
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(stats.counters.evictions == 0);
      CHECK(f.cache->clear(f.pool) == Status::ok);
    } else {
      CHECK(stats.counters.evictions > 0);
      // Every indexed entry is still a policy member: shrinking to one byte
      // can evict them all.
      CHECK(f.cache->resize_pool(f.pool, 1) == Status::ok);
    }
    stats = f.stats();
    CHECK(stats.indexed_items == 0 && stats.allocated_bytes == 0);
    CHECK(stats.eviction.policy_metadata_bytes == fixed_metadata(policy));
  });
}

std::string sparse_key(std::size_t i) { return std::to_string(100000 + i); }
std::string sparse_value(std::size_t i) { return std::string(15000, static_cast<char>(i % 251)); }

// Fills slabs and leaves every 32nd entry, so compaction has entries to move.
void populate_sparse(nivo::Cache& cache, nivo::PoolId pool) {
  for (std::size_t i = 0; i < 512; ++i) CHECK(cache.put(pool, sparse_key(i), bytes(sparse_value(i))));
  for (std::size_t i = 0; i < 512; ++i)
    if (i % 32 != 0) CHECK(cache.erase(pool, sparse_key(i)) == Status::ok);
}

// Shrinks one entry at a time and records which key leaves each step.
std::vector<std::size_t> eviction_order(nivo::Cache& cache, nivo::PoolId pool) {
  std::vector<std::size_t> order;
  std::set<std::size_t> survivors;
  for (std::size_t i = 0; i < 512; i += 32) survivors.insert(i);
  while (survivors.size() > 1) {
    const std::size_t used = cache.pool_stats(pool).value().allocated_bytes;
    CHECK(cache.resize_pool(pool, used - 1) == Status::ok);
    std::vector<std::size_t> gone;
    for (const std::size_t i : survivors)
      if (!cache.peek(pool, sparse_key(i))) gone.push_back(i);
    CHECK(gone.size() == 1);
    order.push_back(gone.front());
    survivors.erase(gone.front());
  }
  return order;
}

void compaction_preserves_policy_state() {
  for_each_policy(evicting_policies, [](EvictionPolicy policy) {
    std::array<PolicyFixture, 2> twins{PolicyFixture(policy, 16 * 1024 * 1024, 5),
                                       PolicyFixture(policy, 16 * 1024 * 1024, 5)};
    for (PolicyFixture& f : twins) {
      populate_sparse(*f.cache, f.pool);
      for (const std::size_t i : {64, 64, 64, 128, 128, 0, 256, 480})
        CHECK(f.cache->find(f.pool, sparse_key(i)));
    }
    CHECK(twins[1].cache->compact(twins[1].pool).value().moved_items == 16);
    // Relocation keeps recency, frequency, reference bits and random slots,
    // so the compacted pool evicts in exactly the same order.
    CHECK(eviction_order(*twins[0].cache, twins[0].pool) == eviction_order(*twins[1].cache, twins[1].pool));
  });
}

void concurrent_compaction_and_pressure() {
  for_each_policy(all_policies, [](EvictionPolicy policy) {
    constexpr std::size_t capacity = 1024 * 1024;
    PolicyFixture f(policy, capacity);
    nivo::ReadHandle pinned = f.pin("pinned", std::string(15000, 'p'));
    const std::byte* address = pinned.bytes().data();
    std::atomic<bool> failed{false};
    std::barrier gate(4);
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < 4; ++worker) {
      workers.emplace_back([&, worker] {
        gate.arrive_and_wait();
        try {
          for (std::size_t i = 0; i < 120; ++i) {
            const std::string key = "key-" + std::to_string(i % 64);
            if (worker < 2) {
              const std::string value(15000, static_cast<char>('a' + worker));
              nivo::Result<nivo::ReadHandle> stored = f.cache->put(f.pool, key, bytes(value));
              if (!stored) { CHECK(stored.status() == Status::resource_exhausted); continue; }
              std::this_thread::yield();
              CHECK(text(stored.value()) == value);
            } else if (worker == 2) {
              nivo::Result<nivo::ReadHandle> found = f.cache->find(f.pool, key);
              if (!found) { CHECK(found.status() == Status::not_found); continue; }
              const std::string value = text(found.value());
              CHECK(value.size() == 15000 && value.find_first_not_of(value.front()) == std::string::npos);
            } else {
              const Status resized = f.cache->resize_pool(f.pool, i % 2 ? capacity : capacity / 8);
              CHECK(resized == Status::ok || resized == Status::resource_exhausted);
              CHECK(f.cache->compact(f.pool));
              CHECK(pinned.bytes().data() == address && pinned.bytes().front() == std::byte{'p'});
            }
          }
        } catch (...) { failed = true; }
      });
    }
    for (std::thread& worker : workers) worker.join();
    CHECK(!failed);
    const Status shrunk = f.cache->resize_pool(f.pool, pinned.charge());
    if (policy == EvictionPolicy::no_eviction) {
      CHECK(f.stats().counters.evictions == 0);
      CHECK(shrunk == (f.stats().indexed_items == 1 ? Status::ok : Status::resource_exhausted));
    } else {
      CHECK(shrunk == Status::ok);
      const nivo::PoolStats stats = f.stats();
      CHECK(stats.counters.evictions > 0 && stats.indexed_items == 1);
    }
    CHECK(text(pinned) == std::string(15000, 'p'));
    pinned.reset();
    CHECK(f.cache->clear(f.pool) == Status::ok);
    CHECK(f.cache->memory_stats(f.pool).value().reserved_bytes == 0);
  });
}

void slru_segments() {
  nivo::detail::SlruEvictionPolicy policy({.protected_percent = 50}, 40);
  std::array<nivo::detail::Entry, 5> entries;
  for (std::size_t i = 0; i < 4; ++i) {
    entries[i].version = i + 1;
    entries[i].charge = 10;
    CHECK(policy.publish(entries[i], nullptr));
  }
  policy.record_access(entries[0]);
  policy.record_access(entries[1]);
  policy.record_access(entries[2]);
  CHECK(policy.snapshot().bytes == (std::array<std::size_t, 3>{20, 20, 0}));
  policy.begin_selection();
  CHECK(policy.next_victim()->version == 4);
  CHECK(policy.next_victim()->version == 1);  // protected LRU was demoted to probation MRU
  policy.end_selection();
  entries[4].version = 5;
  entries[4].charge = 30;
  CHECK(policy.publish(entries[4], &entries[2]));
  policy.record_access(entries[2]);  // stale pinned version cannot rejoin
  CHECK(entries[2].policy_state.load() == 0);
  CHECK(policy.snapshot().bytes[1] <= 30);
  policy.remove(entries[4]);
  CHECK(policy.snapshot().bytes[1] <= 15);
  PolicyFixture f(EvictionPolicy::slru, 8);
  for (const std::string_view key : {"a", "b", "c", "d"}) CHECK(f.put(key, "1") == Status::ok);
  CHECK(f.touch("a") && f.touch("b"));
  for (char key = 'e'; key <= 'z'; ++key) CHECK(f.put(std::string(1, key), "1") == Status::ok);
  CHECK(f.resident("ab") == "ab");
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::slru, {}, {0}) == Status::invalid_argument);
}

void two_q_history() {
  nivo::detail::TwoQEvictionPolicy policy({.a1in_percent = 25, .ghost_capacity = 2});
  std::array<nivo::detail::Entry, 6> entries;
  const std::array<std::string_view, 6> keys{"a", "b", "c", "a", "d", "a"};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].key = keys[i]; entries[i].version = i + 1; entries[i].charge = 10;
  }
  for (std::size_t i = 0; i < 3; ++i) CHECK(policy.publish(entries[i], nullptr));
  policy.record_access(entries[0]);  // correlated hits do not promote A1in
  CHECK(policy.snapshot().entries[1] == 0);
  policy.begin_selection();
  CHECK(policy.next_victim()->version == 1);
  policy.remove(entries[0]);
  policy.end_selection();
  CHECK(policy.ghost_size() == 1);
  CHECK(policy.publish(entries[3], nullptr));
  CHECK(policy.ghost_size() == 0 && policy.snapshot().entries[1] == 1);
  policy.remove(entries[1]);  // explicit removal creates no history
  CHECK(policy.ghost_size() == 0);
  CHECK(policy.publish(entries[4], nullptr));
  CHECK(policy.publish(entries[5], &entries[3]));
  policy.record_access(entries[3]);
  CHECK(entries[3].policy_state.load() == 0 && policy.snapshot().entries[1] == 1);
  nivo::detail::TwoQGhost ghost(2);
  ghost.insert(0); ghost.insert(1); ghost.insert(2);
  CHECK(ghost.size() == 2 && !ghost.take(0) && ghost.take(1));
  for (std::size_t i = 0; i < 10000; ++i) { ghost.insert(i); CHECK(ghost.size() <= 2); }
  CHECK(ghost.take(9999) && ghost.take(9998) && ghost.size() == 0);
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::two_q, {}, {}, {0, 1}) == Status::invalid_argument);
}

void window_tiny_lfu_admission() {
  const nivo::WTinyLfuConfig config{.window_percent = 20, .protected_percent = 50,
                                  .sketch_width = 64, .sample_size = 8192};
  nivo::detail::WTinyLfuEvictionPolicy policy(config, 50);
  std::array<nivo::detail::Entry, 7> entries;
  const std::array<std::string_view, 7> keys{"a", "b", "c", "d", "e", "f", "g"};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].key = keys[i]; entries[i].charge = 10; entries[i].version = i + 1;
  }
  for (std::size_t i = 0; i < 5; ++i) CHECK(policy.publish(entries[i], nullptr));
  CHECK(policy.snapshot().bytes == (std::array<std::size_t, 3>{10, 40, 0}));
  policy.record_access(entries[4]);
  policy.record_access(entries[4]);
  policy.begin_selection();
  CHECK(policy.next_victim()->version == 1);  // candidate e defeats a
  CHECK(policy.admissions() == 0);  // offering a victim is not an admission
  policy.remove(entries[0]);
  policy.end_selection();
  CHECK(policy.admissions() == 1 && policy.snapshot().entries[0] == 0);
  CHECK(policy.publish(entries[5], nullptr));
  policy.record_access(entries[1]);  // protect b, then c
  policy.record_access(entries[2]);
  CHECK(policy.snapshot().bytes[2] == 20);
  policy.begin_selection();
  CHECK(policy.next_victim()->version == 6);  // cold f loses ties to main
  policy.remove(entries[5]);
  policy.end_selection();
  CHECK(policy.publish(entries[6], nullptr));
  policy.record_request("g"); policy.record_request("g");
  policy.begin_selection();
  CHECK(policy.next_victim()->version == 4);
  policy.end_selection();  // pinned/rejected victim: g stays window
  CHECK(policy.snapshot().entries[0] == 1 && policy.admissions() == 1);
  CHECK(policy.frequency("g") == 2);
  policy.set_capacity(20);
  CHECK(policy.snapshot().bytes[2] <= 8);
  nivo::detail::TinyLfuSketch sketch(config);
  for (std::size_t i = 0; i < 100; ++i) sketch.record(123);
  CHECK(sketch.estimate(123) == 100);
  for (std::size_t i = 0; i < 8192; ++i) sketch.record(456);
  CHECK(sketch.estimate(123) == 50 && sketch.estimate(456) <= 255);
  for (std::size_t i = 0; i < 8192 * 8; ++i) sketch.record(456);
  CHECK(sketch.estimate(123) == 0);
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::w_tinylfu, {}, {}, {},
        {.sketch_width = 65}) == Status::invalid_argument);
}

void advanced_policy_boundaries() {
  // The one-slot history must survive evicting b to allocate a's draft.
  nivo::Result<std::unique_ptr<nivo::Cache>> created = nivo::Cache::create(
      {.capacity_bytes = 64, .reaper = no_reaper});
  CHECK(created);
  std::unique_ptr<nivo::Cache> cache = std::move(created).value();
  const nivo::PoolId pool = cache->add_pool({.name = "two-q", .capacity_bytes = 4,
      .eviction = EvictionPolicy::two_q, .two_q = {.ghost_capacity = 1}}).value();
  CHECK(cache->put(pool, "a", bytes("1")));
  CHECK(cache->put(pool, "b", bytes("1")));
  CHECK(cache->put(pool, "c", bytes("1")));  // a in ghost
  CHECK(cache->put(pool, "a", bytes("1")));  // b evicted, a must enter Am
  for (const std::string_view key : {"d", "e", "f", "g"}) CHECK(cache->put(pool, key, bytes("1")));
  CHECK(cache->peek(pool, "a"));
  CHECK(cache->clear(pool) == Status::ok);
  CHECK(cache->put(pool, "g", bytes("1")));
  CHECK(cache->put(pool, "h", bytes("1")));
  CHECK(cache->put(pool, "i", bytes("1")));
  CHECK(!cache->peek(pool, "g"));  // clear removed ghost history

  // A variable-size window candidate requires two actual victims before
  // promotion. Refusing the first offered victim must not move the candidate.
  nivo::detail::WTinyLfuEvictionPolicy weighted({.window_percent = 20, .sketch_width = 64}, 50);
  std::array<nivo::detail::Entry, 6> entries;
  const std::array<std::string_view, 6> keys{"a", "b", "c", "d", "e", "large"};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].key = keys[i]; entries[i].version = i + 1; entries[i].charge = i == 5 ? 20 : 10;
  }
  for (std::size_t i = 0; i < 5; ++i) CHECK(weighted.publish(entries[i], nullptr));
  CHECK(weighted.publish(entries[5], &entries[4]));
  weighted.record_request("large");
  weighted.begin_selection();
  CHECK(weighted.next_victim()->version == 1);
  CHECK(weighted.snapshot().bytes[0] == 20);
  weighted.remove(entries[0]);
  CHECK(weighted.admissions() == 0);
  CHECK(weighted.next_victim()->version == 2);
  weighted.remove(entries[1]);
  CHECK(weighted.admissions() == 1 && weighted.snapshot().bytes[0] == 0);
  weighted.end_selection();
  weighted.reset_history();
  CHECK(weighted.frequency("large") == 0);

  for (const EvictionPolicy policy : {EvictionPolicy::slru, EvictionPolicy::two_q, EvictionPolicy::w_tinylfu}) {
    PolicyFixture f(policy, 32);
    CHECK(f.put("a", "1", {.ttl = nivo::Duration(1)}) == Status::ok);
    CHECK(f.put("b", "1", {.ttl = nivo::Duration(1)}) == Status::ok);
    CHECK(f.touch("a"));
    f.clock->advance(nivo::Duration(2));
    CHECK(f.cache->prune_expired(f.pool).value() == 2);
    CHECK(f.stats().indexed_items == 0);
    CHECK(f.put("a", "1") == Status::ok);
    CHECK(f.put("a", "2", {.mode = nivo::WriteMode::insert_only}) == Status::already_exists);
    CHECK(f.put("a", "2", {.mode = nivo::WriteMode::compare_exchange, .expected_version = 999}) == Status::version_mismatch);
    CHECK(text(f.cache->peek(f.pool, "a").value()) == "1");
  }
}

void advanced_segment_invariants() {
  for (const EvictionPolicy kind : {EvictionPolicy::slru, EvictionPolicy::two_q, EvictionPolicy::w_tinylfu}) {
    nivo::detail::EvictionPolicyEngine engine(kind, 1, {}, {}, {.ghost_capacity = 8},
        {.window_percent = 20, .sketch_width = 64}, 128);
    std::array<nivo::detail::Entry, 32> entries;
    std::array<std::string, 32> keys;
    std::array<bool, 32> present{};
    std::uint64_t random = 42;
    for (std::size_t step = 0; step < 10000; ++step) {
      random = random * 6364136223846793005ULL + 1442695040888963407ULL;
      const std::size_t i = (random >> 32) % entries.size();
      if (!present[i]) {
        keys[i] = "key-" + std::to_string(i);
        entries[i].key = keys[i]; entries[i].charge = i % 13 + 1; entries[i].version = step + 1;
        CHECK(engine.publish(entries[i], nullptr)); present[i] = true;
      } else if (step % 3 == 0) {
        engine.remove(entries[i]); present[i] = false;
      } else engine.record_access(entries[i]);
      if (step % 11 == 0) {
        nivo::detail::VictimSelection selection = engine.select_victims();
        const std::optional<nivo::detail::EntryId> victim = selection.next();
        if (victim && step % 2) {
          for (std::size_t j = 0; j < entries.size(); ++j)
            if (present[j] && entries[j].version == victim->version) { engine.remove(entries[j]); present[j] = false; break; }
        }
      }
      std::array<std::size_t, 3> counts{}, charges{}, heads{};
      for (std::size_t j = 0; j < entries.size(); ++j) {
        if (!present[j]) { CHECK(entries[j].policy_state.load() == 0); continue; }
        const std::size_t queue = (entries[j].policy_state.load() & 3) - 1;
        CHECK(queue < 3);
        ++counts[queue]; charges[queue] += entries[j].charge;
        if (!entries[j].policy_previous) ++heads[queue];
        else CHECK(entries[j].policy_previous->policy_next == &entries[j]);
        if (entries[j].policy_next) CHECK(entries[j].policy_next->policy_previous == &entries[j]);
      }
      const nivo::detail::PolicySegments::Snapshot snapshot = kind == EvictionPolicy::slru
          ? engine.get_if<nivo::detail::SlruEvictionPolicy>()->snapshot()
          : kind == EvictionPolicy::two_q ? engine.get_if<nivo::detail::TwoQEvictionPolicy>()->snapshot()
          : engine.get_if<nivo::detail::WTinyLfuEvictionPolicy>()->snapshot();
      CHECK(snapshot.entries == counts && snapshot.bytes == charges);
      for (std::size_t q = 0; q < 3; ++q) CHECK(heads[q] == (counts[q] ? 1 : 0));
      if (kind == EvictionPolicy::slru) CHECK(charges[1] <= 128 * 80 / 100);
      if (kind == EvictionPolicy::two_q) CHECK(engine.get_if<nivo::detail::TwoQEvictionPolicy>()->ghost_size() <= 8);
      if (kind == EvictionPolicy::w_tinylfu) CHECK(charges[2] <= 103 * 80 / 100);
    }
  }
}

void advanced_configuration_and_misses() {
  for (const std::uint32_t invalid : {0U, 100U}) {
    CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::slru, {}, {invalid}) == Status::invalid_argument);
    CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::two_q, {}, {}, {invalid}) == Status::invalid_argument);
    CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::w_tinylfu, {}, {}, {},
        {.window_percent = invalid}) == Status::invalid_argument);
    CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::w_tinylfu, {}, {}, {},
        {.protected_percent = invalid}) == Status::invalid_argument);
  }
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::two_q, {}, {},
      {.ghost_capacity = (1U << 20) + 1}) == Status::invalid_argument);
  for (const std::size_t width : {0U, 63U, 65U, (1U << 20) + 1})
    CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::w_tinylfu, {}, {}, {},
        {.sketch_width = width}) == Status::invalid_argument);
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::w_tinylfu, {}, {}, {},
      {.sample_size = 8191}) == Status::invalid_argument);
  CHECK(nivo::detail::EvictionPolicyEngine::validate(EvictionPolicy::lru, {}, {0}, {0},
      {.window_percent = 0}) == Status::ok);
  for (const std::uint32_t percent : {1U, 99U}) {
    PolicyFixture f(EvictionPolicy::lru, 256);
    const nivo::PoolId pool = f.cache->add_pool({.name = "window", .capacity_bytes = 128,
        .eviction = EvictionPolicy::w_tinylfu, .w_tinylfu = {.window_percent = percent,
        .protected_percent = percent, .sketch_width = 64, .sample_size = 8192}}).value();
    for (std::size_t i = 0; i < 200; ++i) {
      const std::string key = std::to_string(i);
      CHECK(f.cache->put(pool, key, bytes("value")));
      CHECK(f.cache->find(pool, key));
    }
    CHECK(f.cache->resize_pool(pool, 1) == Status::ok);
    CHECK(f.cache->pool_stats(pool).value().indexed_items == 0);
  }
  nivo::detail::EvictionPolicyEngine engine(EvictionPolicy::w_tinylfu);
  engine.record_miss("miss");
  CHECK(engine.get_if<nivo::detail::WTinyLfuEvictionPolicy>()->frequency("miss") == 1);
  engine.reset_history();
  CHECK(engine.get_if<nivo::detail::WTinyLfuEvictionPolicy>()->frequency("miss") == 0);
}

void policy_configuration() {
  PolicyFixture f(EvictionPolicy::lru, 64);
  for (const EvictionPolicy policy : all_policies) {
    CHECK(nivo::detail::EvictionPolicyEngine::validate(policy) == Status::ok);
    CHECK(f.cache->add_pool({.name = policy_name(policy), .capacity_bytes = 8, .eviction = policy}));
  }
  CHECK(f.cache->add_pool({.name = "bad", .capacity_bytes = 8,
                           .eviction = static_cast<EvictionPolicy>(255)}).status() == Status::invalid_argument);
}
}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> cases = {
    {"advanced_configuration_and_misses", advanced_configuration_and_misses},
    {"advanced_segment_invariants", advanced_segment_invariants},
    {"advanced_policy_boundaries", advanced_policy_boundaries},
    {"window_tiny_lfu_admission", window_tiny_lfu_admission},
    {"two_q_history", two_q_history},
    {"slru_segments", slru_segments},
    {"lru_policy", lru_policy},
    {"fifo_policy_order", fifo_policy_order},
    {"lfu_policy", lfu_policy},
    {"lfu_saturation_and_aging", lfu_saturation_and_aging},
    {"clock_policy", clock_policy},
    {"random_policy", random_policy},
    {"no_eviction_policy", no_eviction_policy},
    {"sieve_policy", sieve_policy},
    {"sieve_hits_do_not_move_entries", sieve_hits_do_not_move_entries},
    {"s3fifo_queues_and_ghost", s3fifo_queues_and_ghost},
    {"s3fifo_byte_shares", s3fifo_byte_shares},
    {"s3fifo_ghost_is_bounded", s3fifo_ghost_is_bounded},
    {"s3fifo_configuration", s3fifo_configuration},
    {"scan_resistance", scan_resistance},
    {"access_races_replacement", access_races_replacement},
    {"selection_is_bounded_and_unique", selection_is_bounded_and_unique},
    {"variable_size_capacity", variable_size_capacity},
    {"insert_erase_overwrite", insert_erase_overwrite},
    {"all_entries_pinned", all_entries_pinned},
    {"handles_survive_logical_removal", handles_survive_logical_removal},
    {"eviction_stats_and_metadata", eviction_stats_and_metadata},
    {"concurrent_reads_and_writes", concurrent_reads_and_writes},
    {"compaction_preserves_policy_state", compaction_preserves_policy_state},
    {"concurrent_compaction_and_pressure", concurrent_compaction_and_pressure},
    {"policy_configuration", policy_configuration},
  };
  unsigned failures = 0;
  for (const std::pair<const char*, std::function<void()>>& test_case : cases) {
    try {
      test_case.second();
      std::cout << "PASS " << test_case.first << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << test_case.first << ": " << error.what() << '\n';
    }
  }
  return failures == 0 ? 0 : 1;
}
