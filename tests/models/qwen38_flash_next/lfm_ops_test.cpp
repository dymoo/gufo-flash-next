#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/lfm.hpp"

// LFM2.5 kernels. Prompt attention (WMMA, F16 scores and probabilities)
// against the F32 decode path on the same cache: each prompt row is
// recomputed as a one-token decode at its position, for a cold prompt and a
// chunk after a prefix, partial query blocks and tiles, and every KV format.
// The fused router against a double-precision host reference.
namespace q = gufo::models::qwen38_flash_next::rocm;
using gufo::models::qwen38_flash_next::KvCacheFormat;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

template<typename T>
T* Alloc(std::size_t count) {
  T* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(T)), "hipMalloc");
  CheckHip(hipMemset(p, 0, count * sizeof(T)), "hipMemset");
  return p;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* p = Alloc<T>(host.size());
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "upload");
  return p;
}

const char* Name(KvCacheFormat f) {
  return f == KvCacheFormat::kF16   ? "f16"
         : f == KvCacheFormat::kTq4 ? "tq4"
         : f == KvCacheFormat::kTq8 ? "tq8"
                                    : "tq3";
}

std::vector<float> Check(KvCacheFormat format, std::uint32_t prefix,
                         std::uint32_t n) {
  constexpr std::uint32_t kRow = q::kLfmHeads * q::kLfmHeadDim;
  constexpr std::uint32_t kKvRow = q::kLfmKvHeads * q::kLfmHeadDim;
  const std::uint32_t total = prefix + n;
  std::uint32_t seed = 0x9E3779B9U ^ total;
  std::vector<float> qkv(static_cast<std::size_t>(total) * q::kLfmQkvRow);
  for (float& x : qkv) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    x = static_cast<float>(static_cast<int>(seed & 0xFFFFU) - 32768) / 16384.0F;
  }
  std::vector<float> gamma(q::kLfmHeadDim, 1.0F);
  std::vector<std::uint32_t> positions(total);
  for (std::uint32_t i = 0; i < total; ++i)
    positions[i] = i;

  float* d_qkv = Upload(qkv);
  float* d_gamma = Upload(gamma);
  std::uint32_t* d_pos = Upload(positions);
  float* d_q = Alloc<float>(static_cast<std::size_t>(total) * kRow);
  float* d_out = Alloc<float>(static_cast<std::size_t>(n) * kRow);
  float* d_ref = Alloc<float>(static_cast<std::size_t>(n) * kRow);
  float* d_partials =
      Alloc<float>(std::size_t{kRow / q::kLfmHeadDim} * q::kMaxAttentionSplits *
                   (q::kLfmHeadDim + 2));
  q::KvView kv;
  kv.format = format;
  const std::size_t slots = static_cast<std::size_t>(total) * q::kLfmKvHeads;
  if (format == KvCacheFormat::kF16) {
    kv.k = Alloc<__half>(static_cast<std::size_t>(total) * kKvRow);
    kv.v = Alloc<__half>(static_cast<std::size_t>(total) * kKvRow);
  } else {
    const std::size_t bytes =
        slots * q::KvCodeBytes(format) * q::kLfmHeadDim / 256;
    kv.k = Alloc<std::uint8_t>(bytes);
    kv.v = Alloc<std::uint8_t>(bytes);
    kv.k_scale = Alloc<__half>(slots);
    kv.v_scale = Alloc<__half>(slots);
  }

  const auto prepare = [&](std::uint32_t start, std::uint32_t rows) {
    q::LfmPrepareAttention(
        d_qkv + static_cast<std::size_t>(start) * q::kLfmQkvRow, d_gamma,
        d_gamma, d_q + static_cast<std::size_t>(start) * kRow, kv, rows,
        d_pos + start, 1.0e6F, 1.0e-5F, nullptr);
  };
  if (prefix != 0)
    prepare(0, prefix);
  prepare(prefix, n);
  const float* chunk_q = d_q + static_cast<std::size_t>(prefix) * kRow;
  q::LfmAttention(chunk_q, kv, d_out, nullptr, n, d_pos + prefix, nullptr);
  for (std::uint32_t r = 0; r < n; ++r)
    q::LfmAttention(chunk_q + static_cast<std::size_t>(r) * kRow, kv,
                    d_ref + static_cast<std::size_t>(r) * kRow, d_partials, 1,
                    d_pos + prefix + r, nullptr);
  CheckHip(hipDeviceSynchronize(), "attention");

  std::vector<float> out(static_cast<std::size_t>(n) * kRow);
  std::vector<float> ref(out.size());
  CheckHip(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(ref.data(), d_ref, ref.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  double worst = 0.0, sum = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i]))
      throw std::runtime_error("non-finite WMMA attention output");
    const double e = std::fabs(static_cast<double>(out[i]) - ref[i]);
    worst = std::max(worst, e);
    sum += e;
    norm += std::fabs(static_cast<double>(ref[i]));
  }
  const double relative = sum / std::max(norm, 1e-30);
  std::cout << Name(format) << " prefix " << prefix << " rows " << n
            << ": worst |WMMA - F32| " << worst << ", relative L1 " << relative
            << "\n";
  // F16 queries and probabilities against F32: 3e-4 relative measured.
  if (worst > 5e-3 || relative > 1.5e-3)
    throw std::runtime_error("WMMA prompt attention drifted from F32 decode");

  for (void* p :
       {static_cast<void*>(d_qkv), static_cast<void*>(d_gamma),
        static_cast<void*>(d_pos), static_cast<void*>(d_q),
        static_cast<void*>(d_out), static_cast<void*>(d_ref),
        static_cast<void*>(d_partials), kv.k, kv.v,
        static_cast<void*>(kv.k_scale), static_cast<void*>(kv.v_scale)})
    if (p != nullptr)
      CheckHip(hipFree(p), "hipFree");
  return out;
}

/// Relative L1 distance of a quantized cache's attention from F16's.
double Distance(const std::vector<float>& a, const std::vector<float>& f16) {
  double diff = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff += std::fabs(static_cast<double>(a[i]) - f16[i]);
    norm += std::fabs(static_cast<double>(f16[i]));
  }
  return diff / std::max(norm, 1e-30);
}

void CheckRouter(std::uint32_t n) {
  constexpr std::uint32_t kHidden = 2048, kExperts = 64, kUsed = 4;
  std::uint32_t seed = 0x2545F491U ^ n;
  const auto next = [&seed] {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return static_cast<float>(static_cast<int>(seed & 0xFFFFU) - 32768) /
           32768.0F;
  };
  std::vector<float> x(static_cast<std::size_t>(n) * kHidden);
  std::vector<float> w(static_cast<std::size_t>(kExperts) * kHidden);
  std::vector<float> bias(kExperts);
  for (float& v : x)
    v = next();
  for (float& v : w)
    v = next() * 0.05F;
  for (float& v : bias)
    v = next() * 0.1F;
  float* d_x = Upload(x);
  float* d_w = Upload(w);
  float* d_bias = Upload(bias);
  auto* d_ids = Alloc<std::int32_t>(std::size_t{n} * kUsed);
  float* d_weights = Alloc<float>(std::size_t{n} * kUsed);
  q::LfmRouter(d_x, d_w, d_bias, d_ids, d_weights, n, kUsed, nullptr);
  CheckHip(hipDeviceSynchronize(), "router");
  std::vector<std::int32_t> ids(std::size_t{n} * kUsed);
  std::vector<float> weights(ids.size());
  CheckHip(hipMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t),
                     hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(weights.data(), d_weights, weights.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  for (std::uint32_t t = 0; t < n; ++t) {
    std::vector<double> score(kExperts), sel(kExperts);
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      double dot = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i)
        dot += static_cast<double>(x[std::size_t{t} * kHidden + i]) *
               w[std::size_t{e} * kHidden + i];
      score[e] = 1.0 / (1.0 + std::exp(-dot));
      sel[e] = score[e] + bias[e];
    }
    double sum = 0.0;
    std::vector<std::uint32_t> chosen;
    for (std::uint32_t slot = 0; slot < kUsed; ++slot) {
      std::uint32_t best = 0;
      for (std::uint32_t e = 1; e < kExperts; ++e)
        if (sel[e] > sel[best])
          best = e;
      chosen.push_back(best);
      sum += score[best];
      sel[best] = -1e30;
    }
    for (std::uint32_t slot = 0; slot < kUsed; ++slot) {
      const std::size_t i = std::size_t{t} * kUsed + slot;
      if (ids[i] != static_cast<std::int32_t>(chosen[slot]) ||
          std::fabs(weights[i] - score[chosen[slot]] / sum) > 1e-5)
        throw std::runtime_error("router mismatch at token " +
                                 std::to_string(t));
    }
  }
  std::cout << "router " << n << " tokens: ids and weights match\n";
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_w),
                  static_cast<void*>(d_bias), static_cast<void*>(d_ids),
                  static_cast<void*>(d_weights)})
    CheckHip(hipFree(p), "hipFree");
}

}  // namespace

int main() {
  try {
    const auto f16 = Check(KvCacheFormat::kF16, 77, 300);
    Check(KvCacheFormat::kF16, 0, 40);
    for (const auto format :
         {KvCacheFormat::kTq8, KvCacheFormat::kTq4, KvCacheFormat::kTq3}) {
      Check(format, 0, 40);
      const double d = Distance(Check(format, 77, 300), f16);
      std::cout << Name(format) << " vs f16 cache: relative L1 " << d << "\n";
      // tq8 keeps attention within 1% of F16; the 4/3-bit formats are lossy.
      if (format == KvCacheFormat::kTq8 && d > 1e-2)
        throw std::runtime_error("tq8 attention drifted from the F16 cache");
    }
    for (const std::uint32_t n : {1U, 7U, 256U})
      CheckRouter(n);
    std::cout << "PASS lfm ops\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
