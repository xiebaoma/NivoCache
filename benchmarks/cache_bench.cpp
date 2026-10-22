#include <nivo/nivo.h>

#include <charconv>
#include <chrono>
#include <iostream>
#include <span>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::size_t iterations = 100000;
  if (argc > 2) return 1;
  if (argc == 2) {
    const std::string_view arg(argv[1]);
    const std::from_chars_result result =
        std::from_chars(arg.data(), arg.data() + arg.size(), iterations);
    if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() || iterations == 0) return 1;
  }
  nivo::Result<std::unique_ptr<nivo::Cache>> created = nivo::Cache::create();
  if (!created) return 1;
  std::unique_ptr<nivo::Cache> cache = std::move(created).value();
  nivo::Result<nivo::PoolId> pool = cache->add_pool({"bench", 32 * 1024 * 1024});
  if (!pool) return 1;
  std::vector<std::string> keys;
  const std::string payload(128, 'x');
  for (unsigned i = 0; i < 10000; ++i) {
    keys.push_back(std::to_string(i));
    nivo::Result<nivo::ReadHandle> result =
        cache->put(pool.value(), keys.back(),
                   std::as_bytes(std::span(payload.data(), payload.size())));
    if (!result) return 1;
  }
  std::size_t checksum = 0;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    nivo::Result<nivo::ReadHandle> result = cache->find(pool.value(), keys[i % keys.size()]);
    if (!result) return 1;
    checksum += result.value().bytes().size();
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "engine=reference policy=lru threads=1 keys=10000 value_bytes=128 hit_ratio=1\n"
            << "operations=" << iterations << " seconds=" << seconds
            << " ops_per_second=" << static_cast<double>(iterations) / seconds
            << " checksum=" << checksum << '\n';
}
