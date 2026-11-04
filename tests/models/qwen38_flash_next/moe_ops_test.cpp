// RoutedExperts against an FP64 reference over synthetic GGUF blocks:
// ragged buckets, an expert split across several work items, an unrouted
// slot, and bitwise independence of a row from its batch and position.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"
#include "tests/models/qwen38_flash_next/moe_reference.hpp"

namespace {

namespace r = gufo::models::qwen38_flash_next::rocm;
namespace ref = moe_reference;
using gufo::core::GgmlType;

constexpr int kExperts = 24;
constexpr int kUsed = 10;

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

std::uint16_t F16(float v) {
  const _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

int Elements(GgmlType t) {
  return t == GgmlType::kQ5_1 || t == GgmlType::kQ8_0 || t == GgmlType::kIQ4_NL
             ? 32
             : 256;
}

/// Random blocks of one format with realistic F16 scales.
std::vector<std::uint8_t> Weights(GgmlType type, int rows, int k,
                                  std::mt19937& rng) {
  std::vector<std::uint8_t> w(ref::RowBytes(type, k) * rows);
  for (auto& b : w)
    b = static_cast<std::uint8_t>(rng());
  std::uniform_real_distribution<float> scale(0.002f, 0.02f);
  const auto put = [&](std::size_t at, float v) {
    const auto h = F16(v);
    std::memcpy(w.data() + at, &h, 2);
  };
  const std::size_t block = ref::RowBytes(type, 256) / (256 / Elements(type));
  // The IQ formats carry only `d` (any other bytes are valid codes/scales).
  const bool offset = type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
                      type == GgmlType::kQ5_1;
  for (std::size_t at = 0; at < w.size(); at += block) {
    put(at, scale(rng));
    if (offset)
      put(at + 2, (type == GgmlType::kQ5_1 ? -16.0f : 1.0f) * scale(rng));
  }
  return w;
}

void* Upload(const std::vector<std::uint8_t>& v) {
  void* p = nullptr;
  Hip(hipMalloc(&p, v.size() + 4096), "malloc");
  Hip(hipMemcpy(p, v.data(), v.size(), hipMemcpyHostToDevice), "copy");
  return p;
}

/// WeightType values are the GGUF type ids.
r::WeightType Type(GgmlType t) {
  return static_cast<r::WeightType>(t);
}

bool Run(GgmlType gt, GgmlType dt, std::mt19937& rng) {
  const auto gate = Weights(gt, kExperts * ref::kFf, ref::kHidden, rng);
  const auto up = Weights(gt, kExperts * ref::kFf, ref::kHidden, rng);
  const auto down = Weights(dt, kExperts * ref::kHidden, ref::kFf, rng);
  void* dg = Upload(gate);
  void* du = Upload(up);
  void* dd = Upload(down);
  if (r::RoutedPacked(Type(gt), Type(dt)) &&
      !(r::PackRoutedExperts(dg, Type(gt), kExperts * ref::kFf, ref::kHidden,
                             nullptr) &&
        r::PackRoutedExperts(du, Type(gt), kExperts * ref::kFf, ref::kHidden,
                             nullptr) &&
        r::PackRoutedExperts(dd, Type(dt), kExperts * ref::kHidden, ref::kFf,
                             nullptr)))
    throw std::runtime_error("PackRoutedExperts failed");
  constexpr int kRows = 70;
  std::vector<float> x(std::size_t(kRows) * ref::kHidden);
  std::normal_distribution<float> normal;
  for (auto& v : x)
    v = normal(rng);
  // Distinct experts per row; expert 3 is in every row (5 work items).
  std::vector<std::int32_t> ids(kRows * kUsed);
  for (int t = 0; t < kRows; ++t) {
    std::vector<int> pool(kExperts);
    for (int e = 0; e < kExperts; ++e)
      pool[e] = e;
    std::shuffle(pool.begin() + 1, pool.end(), rng);
    std::swap(pool[0], pool[3]);
    for (int s = 0; s < kUsed; ++s)
      ids[t * kUsed + s] = pool[s];
  }
  ids[7 * kUsed + 4] = -1;
  std::copy_n(x.begin() + 2 * ref::kHidden, ref::kHidden,
              x.begin() + 50 * ref::kHidden);
  std::copy_n(ids.begin() + 2 * kUsed, kUsed, ids.begin() + 50 * kUsed);
  float *dx, *dy;
  std::int32_t* di;
  void* scratch;
  const std::size_t y_floats = std::size_t(kRows) * kUsed * ref::kHidden;
  Hip(hipMalloc(&dx, x.size() * 4), "malloc");
  Hip(hipMalloc(&di, ids.size() * 4), "malloc");
  Hip(hipMalloc(&dy, y_floats * 4), "malloc");
  Hip(hipMalloc(&scratch, r::RoutedExpertsScratchBytes(kRows, kUsed)),
      "malloc");
  Hip(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  const auto run = [&](int first, int rows) {
    Hip(hipMemcpy(di, ids.data() + first * kUsed, rows * kUsed * 4,
                  hipMemcpyHostToDevice),
        "ids");
    Hip(hipMemset(dy, 0xFF, y_floats * 4), "poison");
    if (!r::RoutedExperts(dg, du, Type(gt), dd, Type(dt), kExperts,
                          dx + std::size_t(first) * ref::kHidden, di, dy, rows,
                          kUsed, scratch, nullptr))
      throw std::runtime_error("RoutedExperts rejected the shapes");
    std::vector<float> y(std::size_t(rows) * kUsed * ref::kHidden);
    Hip(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost), "y");
    return y;
  };
  bool ok = true;
  const auto all = run(0, kRows);
  double worst = 0;
  for (const int t : {0, 7, 33, 69}) {
    for (int s = 0; s < kUsed; s += 3) {
      const float* got =
          all.data() + (std::size_t(t) * kUsed + s) * ref::kHidden;
      const int e = ids[t * kUsed + s];
      const auto want =
          ref::Expert(gt, gate.data(), up.data(), dt, down.data(), e,
                      x.data() + std::size_t(t) * ref::kHidden, true);
      double num = 0, den = 0;
      for (int i = 0; i < ref::kHidden; ++i) {
        num = std::max(num, std::fabs(got[i] - want[i]));
        den = std::max(den, std::fabs(want[i]));
      }
      worst = std::max(worst, num / den);
    }
  }
  const float* dropped =
      all.data() + (7 * kUsed + 4) * std::size_t(ref::kHidden);
  for (int i = 0; i < ref::kHidden; ++i)
    ok &= dropped[i] == 0.0f;
  const std::size_t row = std::size_t(kUsed) * ref::kHidden;
  ok &= std::memcmp(all.data() + 2 * row, all.data() + 50 * row, row * 4) == 0;
  for (const int first : {0, 9, 33, 69}) {
    for (const int rows : {1, 3}) {
      if (first + rows > kRows)
        continue;
      const auto part = run(first, rows);
      ok &= std::memcmp(part.data(), all.data() + first * row,
                        part.size() * 4) == 0;
    }
  }
  ok &= worst < 1e-3;
  std::printf("%d/%d: max relative error %.3g, batch independence %s\n",
              int(gt), int(dt), worst, ok ? "ok" : "FAILED");
  for (void* p : {dg, du, dd, static_cast<void*>(dx), static_cast<void*>(di),
                  static_cast<void*>(dy), scratch})
    (void)hipFree(p);
  return ok;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  bool ok = Run(GgmlType::kQ4_K, GgmlType::kQ5_1, rng);
  ok &= Run(GgmlType::kQ5_K, GgmlType::kQ8_0, rng);
  ok &= Run(GgmlType::kQ8_0, GgmlType::kQ8_0, rng);
  // UD-IQ4_XS and UD-Q3_K_XL layer formats.
  ok &= Run(GgmlType::kIQ3_S, GgmlType::kIQ4_NL, rng);
  ok &= Run(GgmlType::kIQ4_XS, GgmlType::kQ8_0, rng);
  ok &= Run(GgmlType::kIQ3_XXS, GgmlType::kIQ4_NL, rng);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
