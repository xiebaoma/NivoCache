#pragma once

#include <stdexcept>
#include <utility>
#include <variant>

namespace nivo {
enum class Status {
  ok,
  not_found,
  already_exists,
  version_mismatch,
  invalid_argument,
  pool_not_found,
  resource_exhausted,
  closed,
  unsupported,
};

[[nodiscard]] constexpr const char* to_string(Status status) noexcept {
  switch (status) {
    case Status::ok: return "ok";
    case Status::not_found: return "not_found";
    case Status::already_exists: return "already_exists";
    case Status::version_mismatch: return "version_mismatch";
    case Status::invalid_argument: return "invalid_argument";
    case Status::pool_not_found: return "pool_not_found";
    case Status::resource_exhausted: return "resource_exhausted";
    case Status::closed: return "closed";
    case Status::unsupported: return "unsupported";
  }
  return "unknown";
}

// Operational errors are values. Host allocation failures may throw bad_alloc.
// Accessing an error result throws logic_error; there is no default success value.
template <class T>
class [[nodiscard]] Result {
 public:
  Result(T value) : state_(std::in_place_index<0>, std::move(value)) {}
  Result(Status error) : state_(std::in_place_index<1>, error) {
    if (error == Status::ok) throw std::invalid_argument("Result needs a value");
  }
  [[nodiscard]] bool has_value() const noexcept { return state_.index() == 0; }
  explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] Status status() const noexcept {
    return has_value() ? Status::ok : std::get<1>(state_);
  }
  T& value() & { check(); return std::get<0>(state_); }
  const T& value() const& { check(); return std::get<0>(state_); }
  T&& value() && { check(); return std::move(std::get<0>(state_)); }

 private:
  void check() const {
    if (!has_value()) throw std::logic_error(to_string(status()));
  }
  std::variant<T, Status> state_;
};
}  // namespace nivo
