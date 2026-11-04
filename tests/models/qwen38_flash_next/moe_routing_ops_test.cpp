#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

// RoutePolicy against a host model of the same float32 rules, over ragged
// batches, plus the per-call cost of RouterTopK and RoutePolicy.
namespace {

namespace qfn = gufo::models::qwen38_flash_next;
using Mode = qfn::MoeRouting::Mode;
constexpr std::uint32_t kExperts = 512, kStride = kExperts + 1, kUsed = 10;

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}

void Require(bool ok, const std::string& what) {
  if (!ok)
    throw std::runtime_error(what);
}

/// Skewed router logits: shared per-expert popularity plus per-row noise;
/// `coarse` rounds them to quarter steps so ties exercise the index order.
std::vector<float> MakeLogits(std::uint32_t rows, std::uint32_t seed,
                              bool coarse) {
  std::uint32_t s = seed;
  const auto uniform = [&s] {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (static_cast<float>(s >> 8) + 0.5F) / 16777216.0F;
  };
  const auto normal = [&] {
    return std::sqrt(-2.0F * std::log(uniform())) *
           std::cos(6.2831853F * uniform());
  };
  std::vector<float> bias(kExperts);
  for (auto& b : bias)
    b = 0.75F * normal();
  std::vector<float> logits(std::size_t{rows} * kStride);
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t e = 0; e < kStride; ++e) {
      float v = e < kExperts ? bias[e] + 1.5F * normal() : normal();
      logits[std::size_t{r} * kStride + e] =
          coarse ? std::round(v * 4.0F) / 4.0F : v;
    }
  return logits;
}

/// Host model of RoutePolicyKernel over RouterTopK's exact ids/weights.
void Expected(const qfn::MoeRouting& p, const std::vector<float>& logits,
              std::uint32_t rows, std::vector<std::int32_t>& ids,
              std::vector<float>& w) {
  std::vector<bool> fetched(kExperts, false);
  if (p.mode != Mode::kMass) {
    std::vector<std::uint32_t> demand(kExperts, 0);
    for (std::uint32_t i = 0; i < rows * kUsed; ++i) {
      if (i % kUsed < p.keep)
        fetched[ids[i]] = true;
      else if (p.mode == Mode::kBudget)
        demand[ids[i]] += static_cast<std::uint32_t>(w[i] * 65536.0F);
    }
    if (p.mode == Mode::kBudget) {
      const auto floor = static_cast<std::uint32_t>(
          std::count(fetched.begin(), fetched.end(), true));
      std::vector<std::uint32_t> order;
      for (std::uint32_t e = 0; e < kExperts; ++e)
        if (!fetched[e] && demand[e] > 0)
          order.push_back(e);
      std::stable_sort(order.begin(), order.end(),
                       [&](auto a, auto b) { return demand[a] > demand[b]; });
      for (std::uint32_t i = 0; floor + i < p.budget && i < order.size(); ++i)
        fetched[order[i]] = true;
    }
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    const float* src = logits.data() + std::size_t{r} * kStride;
    std::int32_t* row_ids = ids.data() + r * kUsed;
    float* row_w = w.data() + r * kUsed;
    std::vector<std::int32_t> next(row_ids, row_ids + kUsed);
    bool changed = false;
    if (p.mode == Mode::kMass) {
      float before = 0.0F;
      for (std::uint32_t s = 0; s < kUsed; ++s) {
        if (s >= p.keep && before >= p.value) {
          next[s] = -1;
          changed = true;
        }
        before += row_w[s];
      }
    } else {
      std::vector<float> score(kExperts);
      for (std::uint32_t e = 0; e < kExperts; ++e)
        score[e] = p.mode == Mode::kAffinity
                       ? (fetched[e] ? src[e] + p.value : src[e])
                       : (fetched[e] ? src[e] : -INFINITY);
      for (std::uint32_t s = 0; s < p.keep; ++s)
        score[row_ids[s]] = -INFINITY;
      for (std::uint32_t s = p.keep; s < kUsed; ++s) {
        const auto best = static_cast<std::int32_t>(
            std::max_element(score.begin(), score.end()) - score.begin());
        next[s] = score[best] == -INFINITY ? -1 : best;
        score[best] = -INFINITY;
        changed |= std::find(row_ids + p.keep, row_ids + kUsed, next[s]) ==
                   row_ids + kUsed;
      }
    }
    if (!changed)
      continue;
    float m = -INFINITY, sum = 0.0F;
    for (const auto e : next)
      m = e >= 0 ? std::max(m, src[e]) : m;
    for (const auto e : next)
      sum += e >= 0 ? std::exp(src[e] - m) : 0.0F;
    for (std::uint32_t s = 0; s < kUsed; ++s) {
      row_w[s] = next[s] >= 0 ? std::exp(src[next[s]] - m) / sum : 0.0F;
      row_ids[s] = next[s] >= 0 ? next[s] : next[0];
    }
  }
}

struct Device {
  float* logits{};
  std::int32_t* ids{};
  float* w{};
  explicit Device(std::uint32_t rows) {
    CheckHip(hipMalloc(&logits, std::size_t{rows} * kStride * 4), "malloc");
    CheckHip(hipMalloc(&ids, std::size_t{rows} * kUsed * 4), "malloc");
    CheckHip(hipMalloc(&w, std::size_t{rows} * kUsed * 4), "malloc");
  }
  ~Device() {
    (void)hipFree(logits);
    (void)hipFree(ids);
    (void)hipFree(w);
  }
  void Route(std::uint32_t rows, const qfn::MoeRouting* p) const {
    gufo::models::qwen38_flash_next::rocm::RouterTopK(
        logits, kStride, ids, w, rows, kExperts, kUsed, nullptr);
    if (p != nullptr)
      gufo::models::qwen38_flash_next::rocm::RoutePolicy(
          logits, kStride, ids, w, rows, kExperts, kUsed,
          static_cast<std::uint32_t>(p->mode), p->keep, p->value, p->budget,
          nullptr);
  }
  template<typename T>
  std::vector<T> Get(const T* src, std::size_t n) const {
    std::vector<T> out(n);
    CheckHip(hipMemcpy(out.data(), src, n * sizeof(T), hipMemcpyDeviceToHost),
             "download");
    return out;
  }
};

qfn::MoeRouting Policy(const char* spec) {
  std::string error;
  const auto p = qfn::MoeRouting::Parse(spec, kUsed, &error);
  Require(p.has_value(), error);
  return *p;
}

void Check(const char* spec, std::uint32_t rows, bool coarse) {
  const auto p = Policy(spec);
  const auto logits = MakeLogits(rows, 0x9E3779B9U ^ rows, coarse);
  Device d(rows);
  CheckHip(hipMemcpy(d.logits, logits.data(), logits.size() * 4,
                     hipMemcpyHostToDevice),
           "upload");
  d.Route(rows, nullptr);
  auto ids = d.Get(d.ids, std::size_t{rows} * kUsed);
  auto w = d.Get(d.w, std::size_t{rows} * kUsed);
  const auto exact_ids = ids;
  const auto exact_w = w;
  Expected(p, logits, rows, ids, w);
  d.Route(rows, &p);
  const auto got_ids = d.Get(d.ids, ids.size());
  const auto got_w = d.Get(d.w, w.size());
  std::vector<bool> used(kExperts, false), exact_used(kExperts, false);
  for (std::uint32_t r = 0; r < rows; ++r) {
    bool ok = true, same = true;
    for (std::uint32_t s = 0; s < kUsed; ++s) {
      const std::size_t i = std::size_t{r} * kUsed + s;
      ok &= got_ids[i] == ids[i] && std::abs(got_w[i] - w[i]) <= 2e-6F;
      same &= ids[i] == exact_ids[i] && w[i] == exact_w[i];
      exact_used[exact_ids[i]] = true;
      used[got_ids[i]] = used[got_ids[i]] || got_w[i] > 0.0F;
    }
    for (std::uint32_t s = 0; same && s < kUsed; ++s)
      ok &= got_w[std::size_t{r} * kUsed + s] ==
            exact_w[std::size_t{r} * kUsed + s];
    if (ok)
      continue;
    std::cerr << spec << " rows " << rows << (coarse ? " ties" : "") << " row "
              << r << " (slot: exact id/w logit | host | gpu)\n";
    for (std::uint32_t s = 0; s < kUsed; ++s) {
      const std::size_t i = std::size_t{r} * kUsed + s;
      const auto logit = [&](std::int32_t e) {
        return logits[std::size_t{r} * kStride + e];
      };
      std::cerr.precision(9);
      std::cerr << "  " << s << ": " << exact_ids[i] << '/' << exact_w[i] << ' '
                << logit(exact_ids[i]) << " | " << ids[i] << '/' << w[i] << ' '
                << logit(ids[i]) << " | " << got_ids[i] << '/' << got_w[i]
                << ' ' << logit(got_ids[i]) << '\n';
    }
    throw std::runtime_error("routing mismatch");
  }
  std::cout << spec << " rows " << rows << (coarse ? " ties" : "") << ": union "
            << std::count(used.begin(), used.end(), true) << " (exact "
            << std::count(exact_used.begin(), exact_used.end(), true) << ")\n";
}

void Time(std::uint32_t rows) {
  const auto logits = MakeLogits(rows, 7U, false);
  Device d(rows);
  CheckHip(hipMemcpy(d.logits, logits.data(), logits.size() * 4,
                     hipMemcpyHostToDevice),
           "upload");
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  const auto us = [&](const qfn::MoeRouting* p) {
    constexpr int kIters = 200;
    d.Route(rows, p);
    CheckHip(hipEventRecord(start, nullptr), "record");
    for (int i = 0; i < kIters; ++i)
      d.Route(rows, p);
    CheckHip(hipEventRecord(stop, nullptr), "record");
    CheckHip(hipEventSynchronize(stop), "sync");
    float ms = 0.0F;
    CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
    return ms * 1000.0F / kIters;
  };
  const float topk = us(nullptr);
  std::cout << "timing rows " << rows << ": RouterTopK " << topk << " us";
  for (const char* spec : {"affinity:6:1", "mass:2:0.9", "budget:2:128"}) {
    const auto p = Policy(spec);
    std::cout << ", +" << spec << " " << us(&p) - topk << " us";
  }
  std::cout << '\n';
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
  try {
    for (const std::uint32_t rows : {1U, 7U, 8U, 9U, 31U, 33U, 64U, 65U, 512U})
      for (const char* spec :
           {"affinity:6:1", "affinity:4:2", "affinity:10:1", "mass:2:0.9",
            "mass:6:0.5", "budget:2:128", "budget:6:60", "budget:1:4"})
        for (const bool coarse : {false, true})
          Check(spec, rows, coarse);
    for (const std::uint32_t rows : {1U, 8U, 32U, 64U, 128U, 256U, 512U})
      Time(rows);
    std::cout << "MoE routing policy checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
