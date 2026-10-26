// Helpers shared by the NivoCache benchmark executables.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace bench {
constexpr std::size_t kKeyBytes = 16U;

// Fixed-width keys keep every operation's charge identical: 'k' followed by
// fifteen hexadecimal digits of the key id.
struct Key {
  std::array<char, kKeyBytes> bytes{};

  explicit Key(std::uint64_t id) {
    bytes[0] = 'k';
    for (std::size_t index = kKeyBytes - 1U; index > 0U; --index) {
      bytes[index] = "0123456789abcdef"[id & 0xfU];
      id >>= 4U;
    }
  }
  std::string_view view() const { return {bytes.data(), bytes.size()}; }
};

struct SplitMix64 {
  std::uint64_t state;

  std::uint64_t next() {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }
  double unit() { return static_cast<double>(next() >> 11U) * 0x1.0p-53; }
  std::uint32_t below(std::uint32_t bound) {
    return static_cast<std::uint32_t>((next() >> 32U) * bound >> 32U);
  }
};

// Gray et al., "Quickly Generating Billion-Record Synthetic Databases"; the
// same generator YCSB uses. Rank 0 is the most popular key.
class Zipfian {
 public:
  Zipfian(std::uint32_t items, double theta) : items_(items), theta_(theta) {
    for (std::uint32_t rank = 1; rank <= items; ++rank)
      zeta_n_ += 1.0 / std::pow(static_cast<double>(rank), theta);
    const double zeta_2 = 1.0 + std::pow(0.5, theta);
    alpha_ = 1.0 / (1.0 - theta);
    eta_ = (1.0 - std::pow(2.0 / static_cast<double>(items), 1.0 - theta)) /
           (1.0 - zeta_2 / zeta_n_);
  }

  std::uint32_t next(SplitMix64 &random) const {
    const double unit = random.unit();
    const double scaled = unit * zeta_n_;
    if (scaled < 1.0) return 0U;
    if (scaled < 1.0 + std::pow(0.5, theta_)) return 1U;
    const double rank = static_cast<double>(items_) *
                        std::pow(eta_ * unit - eta_ + 1.0, alpha_);
    return std::min(static_cast<std::uint32_t>(rank), items_ - 1U);
  }

 private:
  std::uint32_t items_;
  double theta_;
  double zeta_n_ = 0.0;
  double alpha_ = 0.0;
  double eta_ = 0.0;
};

// Log-linear buckets: exact below 128 ns, then 64 buckets per power of two
// (at most 1.6% relative error).
class Histogram {
 public:
  Histogram() : counts_(kBuckets, 0U) {}

  void record(std::uint64_t nanoseconds) {
    ++counts_[bucket(nanoseconds)];
    ++count_;
    max_ = std::max(max_, nanoseconds);
  }
  void merge(const Histogram &other) {
    for (std::size_t index = 0; index < kBuckets; ++index)
      counts_[index] += other.counts_[index];
    count_ += other.count_;
    max_ = std::max(max_, other.max_);
  }
  std::uint64_t count() const { return count_; }
  std::uint64_t max() const { return max_; }
  // Lower bound of the bucket holding the requested quantile.
  std::uint64_t percentile(double quantile) const {
    if (count_ == 0U) return 0U;
    const std::uint64_t rank = static_cast<std::uint64_t>(
        std::ceil(quantile * static_cast<double>(count_)));
    std::uint64_t seen = 0;
    for (std::size_t index = 0; index < kBuckets; ++index) {
      seen += counts_[index];
      if (seen >= std::max<std::uint64_t>(rank, 1U)) return lower_bound(index);
    }
    return max_;
  }

 private:
  static constexpr std::size_t kSubBuckets = 64U;
  static constexpr std::size_t kBuckets = kSubBuckets * 59U + kSubBuckets;

  static std::size_t bucket(std::uint64_t value) {
    if (value < 2U * kSubBuckets) return static_cast<std::size_t>(value);
    const std::size_t shift = static_cast<std::size_t>(std::bit_width(value)) - 7U;
    return kSubBuckets * shift + static_cast<std::size_t>(value >> shift);
  }
  static std::uint64_t lower_bound(std::size_t index) {
    if (index < 2U * kSubBuckets) return index;
    const std::size_t shift = index / kSubBuckets - 1U;
    return static_cast<std::uint64_t>(index - kSubBuckets * shift) << shift;
  }

  std::vector<std::uint64_t> counts_;
  std::uint64_t count_ = 0;
  std::uint64_t max_ = 0;
};
}  // namespace bench
