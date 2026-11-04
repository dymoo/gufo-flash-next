// FP64 reference for one routed expert (gate/up SwiGLU, then down) over the
// GGUF encodings RoutedExperts reads, shared by moe_ops_test and moe_bench.
#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_MOE_REFERENCE_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_MOE_REFERENCE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace moe_reference {

using gufo::core::GgmlType;

constexpr int kHidden = 2560;
constexpr int kFf = 640;

inline float Half(std::uint16_t h) {
  _Float16 f;
  std::memcpy(&f, &h, 2);
  return static_cast<float>(f);
}
inline float RoundHalf(float v) {
  return static_cast<float>(static_cast<_Float16>(v));
}

/// Q8 values of one 32-element block as the kernels quantize them
/// (correctly rounded divisions, dequantized with the F16 scale); `sum` is
/// the dequantized sum that multiplies the weight offsets.
inline void QuantizeBlock(const float* v, float* deq, double* sum) {
  float amax = 0;
  for (int i = 0; i < 32; ++i)
    amax = std::max(amax, std::fabs(v[i]));
  const float d = amax / 127.0f;
  const float dh = RoundHalf(d);
  *sum = 0;
  for (int i = 0; i < 32; ++i) {
    deq[i] = amax == 0 ? 0.0f : std::round(v[i] / d) * dh;
    *sum += deq[i];
  }
}

inline std::size_t RowBytes(GgmlType type, int k) {
  switch (type) {
    case GgmlType::kQ4_K:
      return k / 256 * 144;
    case GgmlType::kQ5_K:
      return k / 256 * 176;
    case GgmlType::kQ5_1:
      return k / 32 * 24;
    case GgmlType::kQ8_0:
      return k / 32 * 34;
    case GgmlType::kIQ4_NL:
      return k / 32 * 18;
    case GgmlType::kIQ4_XS:
      return k / 256 * 136;
    case GgmlType::kIQ3_S:
      return k / 256 * 110;
    case GgmlType::kIQ3_XXS:
      return k / 256 * 98;
    default:
      throw std::runtime_error("reference row bytes");
  }
}

/// Dequantizes one row of `k` weights as w[i] + bias[i / 32]: the scaled
/// code and the block offset (the min for K-quants and Q5_1).
inline void DequantRow(GgmlType type, const std::uint8_t* p, int k, double* w,
                       double* bias) {
  if (type == GgmlType::kQ4_K || type == GgmlType::kQ5_K) {
    const bool q5 = type == GgmlType::kQ5_K;
    const int bytes = q5 ? 176 : 144;
    for (int sb = 0; sb < k / 256; ++sb, p += bytes) {
      std::uint16_t dh, mh;
      std::memcpy(&dh, p, 2);
      std::memcpy(&mh, p + 2, 2);
      const double d = Half(dh), dmin = Half(mh);
      const std::uint8_t* sc = p + 4;
      const std::uint8_t* qh = p + 16;
      const std::uint8_t* qs = p + (q5 ? 48 : 16);
      for (int j = 0; j < 8; ++j) {
        int sv, m;
        if (j < 4) {
          sv = sc[j] & 63;
          m = sc[j + 4] & 63;
        } else {
          sv = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
          m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
        bias[sb * 8 + j] = -dmin * m;
        for (int e = 0; e < 32; ++e) {
          const int byte = qs[(j / 2) * 32 + e];
          int q = j % 2 ? byte >> 4 : byte & 15;
          if (q5)
            q += ((qh[e] >> j) & 1) << 4;
          w[sb * 256 + j * 32 + e] = d * sv * q;
        }
      }
    }
  } else if (type == GgmlType::kQ5_1) {
    for (int b = 0; b < k / 32; ++b, p += 24) {
      std::uint16_t dh, mh;
      std::uint32_t qh;
      std::memcpy(&dh, p, 2);
      std::memcpy(&mh, p + 2, 2);
      std::memcpy(&qh, p + 4, 4);
      bias[b] = Half(mh);
      for (int e = 0; e < 32; ++e) {
        const int nib = e < 16 ? p[8 + e] & 15 : p[8 + e - 16] >> 4;
        w[b * 32 + e] = Half(dh) * double(nib | (((qh >> e) & 1) << 4));
      }
    }
  } else if (type == GgmlType::kQ8_0) {
    for (int b = 0; b < k / 32; ++b, p += 34) {
      std::uint16_t dh;
      std::memcpy(&dh, p, 2);
      bias[b] = 0;
      for (int e = 0; e < 32; ++e)
        w[b * 32 + e] = Half(dh) * double(static_cast<std::int8_t>(p[2 + e]));
    }
  } else {
    // The IQ codebook formats have no offsets; the core CPU dequantizers
    // (ggml-quants.c ports) are exact in FP32 (at most 24 significant bits).
    std::vector<float> f(k);
    namespace q = gufo::quant;
    switch (type) {
      case GgmlType::kIQ4_NL:
        q::DequantizeIQ4_NL(p, f.data(), k);
        break;
      case GgmlType::kIQ4_XS:
        q::DequantizeIQ4_XS(p, f.data(), k);
        break;
      case GgmlType::kIQ3_S:
        q::DequantizeIQ3_S(p, f.data(), k);
        break;
      case GgmlType::kIQ3_XXS:
        q::DequantizeIQ3_XXS(p, f.data(), k);
        break;
      default:
        throw std::runtime_error("reference type");
    }
    std::copy(f.begin(), f.end(), w);
    std::fill(bias, bias + k / 32, 0.0);
  }
}

/// One matrix-vector product of weight row `row` against `x` with its
/// per-block sums `xs` (the offsets multiply the block sums).
inline double Dot(GgmlType type, const std::uint8_t* matrix, std::size_t row,
                  int k, const float* x, const double* xs) {
  std::vector<double> w(k), bias(k / 32);
  DequantRow(type, matrix + row * RowBytes(type, k), k, w.data(), bias.data());
  double acc = 0;
  for (int i = 0; i < k; ++i)
    acc += w[i] * x[i];
  for (int b = 0; b < k / 32; ++b)
    acc += bias[b] * xs[b];
  return acc;
}

/// y[kHidden] of expert e for token row x in FP64, over the Q8-rounded
/// activations when `quantized` (otherwise the exact operator).
inline std::vector<double> Expert(GgmlType gate_type, const void* gate,
                                  const void* up, GgmlType down_type,
                                  const void* down, int e, const float* x,
                                  bool quantized) {
  const auto quantize = [&](const float* v, int n, std::vector<float>& q,
                            std::vector<double>& sums) {
    q.assign(v, v + n);
    sums.assign(n / 32, 0.0);
    for (int b = 0; b < n / 32; ++b) {
      for (int i = 0; i < 32; ++i)
        sums[b] += v[b * 32 + i];
      if (quantized)
        QuantizeBlock(v + b * 32, q.data() + b * 32, &sums[b]);
    }
  };
  std::vector<float> xq, hq;
  std::vector<double> xs, hs;
  quantize(x, kHidden, xq, xs);
  const auto* g = static_cast<const std::uint8_t*>(gate);
  const auto* u = static_cast<const std::uint8_t*>(up);
  std::vector<float> h(kFf);
  for (int j = 0; j < kFf; ++j) {
    const std::size_t row = std::size_t(e) * kFf + j;
    const double gs = Dot(gate_type, g, row, kHidden, xq.data(), xs.data());
    const double us = Dot(gate_type, u, row, kHidden, xq.data(), xs.data());
    h[j] = static_cast<float>(gs / (1.0 + std::exp(-gs)) * us);
  }
  quantize(h.data(), kFf, hq, hs);
  std::vector<double> y(kHidden);
  const auto* d = static_cast<const std::uint8_t*>(down);
  for (int i = 0; i < kHidden; ++i)
    y[i] = Dot(down_type, d, std::size_t(e) * kHidden + i, kFf, hq.data(),
               hs.data());
  return y;
}

}  // namespace moe_reference

#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_MOE_REFERENCE_HPP_
