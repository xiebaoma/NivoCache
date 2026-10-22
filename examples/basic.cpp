#include <nivo/nivo.h>

#include <algorithm>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>

int main() {
  auto created = nivo::Cache::create();
  if (!created) return 1;
  auto cache = std::move(created).value();
  auto pool = cache->add_pool({.name = "objects", .capacity_bytes = 1024 * 1024});
  if (!pool) return 1;
  const std::string_view value = "hello, Nivo";
  auto draft = cache->allocate(pool.value(), "greeting", value.size());
  if (!draft) return 1;
  auto input = std::as_bytes(std::span(value.data(), value.size()));
  std::copy(input.begin(), input.end(), draft.value().bytes().begin());
  auto published = cache->publish(std::move(draft).value(),
      {.mode = nivo::WriteMode::insert_only, .ttl = std::chrono::seconds(30)});
  if (!published) return 1;
  auto found = cache->find(pool.value(), "greeting");
  if (!found) return 1;
  auto bytes = found.value().bytes();
  std::cout.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  std::cout << " (version " << found.value().version() << ")\n";
  if (cache->erase(pool.value(), "greeting", found.value().version()) != nivo::Status::ok) return 1;
  // Existing snapshots still own the value after erase and after Cache destruction.
  cache.reset();
  return found.value().bytes().size() == value.size() ? 0 : 1;
}
