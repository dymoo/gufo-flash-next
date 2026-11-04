#ifndef GUFO_CORE_PARALLEL_HPP_
#define GUFO_CORE_PARALLEL_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace gufo {

/// Runs f(0..n-1) on up to every hardware thread, eight items or more per
/// thread; f handles its own errors. Used for independent per-session host
/// work (sampling, logit copies) that a wide decode step would otherwise
/// spend serially on one core while the GPU waits for it.
// ponytail: threads per call (tens of microseconds against 100 ms steps); a
// persistent pool if the spawn cost ever shows in a profile.
template<class F>
void ParallelFor(std::size_t n, F&& f) {
  const std::size_t threads = std::min<std::size_t>(
      n / 8, std::max(1U, std::thread::hardware_concurrency()));
  if (threads <= 1) {
    for (std::size_t i = 0; i < n; ++i)
      f(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i; (i = next.fetch_add(1)) < n;)
      f(i);
  };
  std::vector<std::jthread> pool;
  pool.reserve(threads - 1);
  for (std::size_t t = 1; t < threads; ++t)
    pool.emplace_back(work);
  work();
}

}  // namespace gufo

#endif  // GUFO_CORE_PARALLEL_HPP_
