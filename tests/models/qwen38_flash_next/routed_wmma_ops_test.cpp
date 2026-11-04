#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host, std::size_t extra_bytes = 0) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + extra_bytes + 4096),
           "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

/// Packed expert weights plus their dequantized values [e][m][k].
struct Experts {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
  std::vector<std::uint8_t> mmq;  ///< the MMQ route's copy when it differs
};

/// Random Q4_K blocks with the real 6-bit scale/min packing.
Experts MakeQ4K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 144);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 144;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qs = blk + 16;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code = (qs[base] >> shift) & 0xF;
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_K blocks: the Q4_K header, 32 high-bit bytes, 128 nibble bytes.
Experts MakeQ5K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 176);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 176;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qh = blk + 16;
    for (int i = 0; i < 32; ++i) {
      qh[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    std::uint8_t* qs = blk + 48;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code =
          ((qs[base] >> shift) & 0xF) | (((qh[j % 32] >> sb32) & 1) << 4);
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_1 blocks: d, m, 32 high bits, 16 packed low nibbles.
Experts MakeQ5_1(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 24);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 24;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half mn = __float2half(Uniform(&seed, 0.5F));
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &mn, 2);
    const std::uint32_t qh = NextRandom(&seed);
    std::memcpy(blk + 4, &qh, 4);
    std::uint8_t* qs = blk + 8;
    for (int i = 0; i < 16; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(mn);
    for (int j = 0; j < 32; ++j) {
      const int low = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
      const int code = low | (static_cast<int>((qh >> j) & 1U) << 4);
      w.values[b * 32 + j] = df * static_cast<float>(code) + mf;
    }
  }
  return w;
}

/// Random Q4_0 blocks (d, element j in the low and j + 16 in the high
/// nibble of byte j, value d * (q - 8)); MMQ reads the same values as the
/// exactly equal Q8_0 blocks (codes q - 8).
Experts MakeQ4_0(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 18);
  w.mmq.resize(blocks * 34);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 18;
    std::uint8_t* q8 = w.mmq.data() + b * 34;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    std::memcpy(q8, &d, 2);
    const float df = __half2float(d);
    for (int j = 0; j < 16; ++j)
      blk[2 + j] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    for (int j = 0; j < 32; ++j) {
      const int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
      q8[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q - 8));
      w.values[b * 32 + j] = df * static_cast<float>(q - 8);
    }
  }
  return w;
}

/// Random Q8_0 blocks: d and 32 signed codes.
Experts MakeQ8_0(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 34);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 34;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    const float df = __half2float(d);
    for (int j = 0; j < 32; ++j) {
      const auto q = static_cast<std::int8_t>(NextRandom(&seed) & 0xFF);
      blk[2 + j] = static_cast<std::uint8_t>(q);
      w.values[b * 32 + j] = df * static_cast<float>(q);
    }
  }
  return w;
}

struct Result {
  double vs_mmq;
  double vs_ref;
  double mmq_vs_ref;
  double scale;
};

/// Runs the MMQ tier and the F16 route over the same routing and reports
/// the worst absolute errors of the F16 route against the tier and against
/// an F64 reference over the dequantized weights (the tier's own F64 gap is
/// its activation quantization).
Result Run(q::WeightType type, std::size_t n_tokens, std::size_t used,
           std::size_t experts, std::size_t m, std::size_t k,
           std::uint32_t seed, std::uint32_t tile_rows = 48) {
  const Experts w =
      type == q::WeightType::kQ4_K   ? MakeQ4K(experts, m, k, seed)
      : type == q::WeightType::kQ5_K ? MakeQ5K(experts, m, k, seed)
      : type == q::WeightType::kQ5_1 ? MakeQ5_1(experts, m, k, seed)
      : type == q::WeightType::kQ4_0 ? MakeQ4_0(experts, m, k, seed)
                                     : MakeQ8_0(experts, m, k, seed);
  const std::size_t slots = n_tokens * used;
  // Skewed top-k without replacement: the low experts see most rows.
  std::vector<std::int32_t> ids(slots);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (std::size_t t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < used) {
      const std::uint32_t r = NextRandom(&state);
      const auto e =
          static_cast<std::int32_t>((r % 3 == 0 ? r / 3 % 4 : r / 3) % experts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * used);
  }
  std::vector<float> x(n_tokens * k);
  for (float& v : x) {
    v = Uniform(&state, 2.0F);
  }
  std::vector<std::uint32_t> counts(experts, 0);
  for (std::int32_t e : ids) {
    ++counts[e];
  }
  std::uint32_t max_bucket = 0;
  for (std::uint32_t c : counts) {
    max_bucket = std::max(max_bucket, (c + 15u) / 16u * 16u);
  }

  // GGUF, for the MMQ route.
  std::uint8_t* d_w = Upload(w.mmq.empty() ? w.packed : w.mmq);
  // The F16 route reads the lane-interleaved expert layout (moe.hpp).
  std::uint8_t* d_wp = Upload(w.packed);
  if (!q::PackRoutedExperts(d_wp, type, experts * m, k, nullptr)) {
    throw std::runtime_error("PackRoutedExperts failed");
  }
  float* d_x = Upload(x);
  std::int32_t* d_ids = Upload(ids);
  std::uint32_t* d_counts = Upload(counts);
  const std::size_t compact = q::RoutedCompactRows(slots, experts);
  std::vector<std::int32_t> zeros(compact + experts + 1, 0);
  std::int32_t* d_bounds = Upload(std::vector<std::int32_t>(experts + 1, 0));
  std::int32_t* d_cursors = Upload(std::vector<std::int32_t>(experts, 0));
  std::int32_t* d_rows_token = Upload(zeros);
  std::int32_t* d_rows_slot = Upload(zeros);
  std::vector<float> zero_out(slots * m, 0.0F);
  float* d_mmq = Upload(zero_out);

  const auto mmq_raw = type == q::WeightType::kQ4_K   ? qfn_mmq_q4_K_moe_raw
                       : type == q::WeightType::kQ5_K ? qfn_mmq_q5_K_moe_raw
                       : type == q::WeightType::kQ5_1 ? qfn_mmq_q5_1_moe_raw
                                                      : qfn_mmq_q8_0_moe_raw;
  // Q4_0 (above) and Q8_0 both run MMQ on Q8_0 blocks.
  const int rc =
      mmq_raw(d_w, d_x, d_ids, d_mmq, static_cast<int>(m), static_cast<int>(k),
              static_cast<int>(n_tokens), static_cast<int>(experts),
              static_cast<int>(used), nullptr);
  if (rc != 0) {
    throw std::runtime_error("MMQ routed GEMM failed");
  }
  // The F16 route over the same assignments: F16 activation rows and the
  // host-built tile map. `used == 1` (the down projection's view) gathers
  // slot rows; the gate/up view gathers tokens.
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, static_cast<std::uint32_t>(n_tokens),
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(experts), nullptr);
  std::vector<__half> x_half(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x_half[i] = __float2half(x[i]);
  }
  std::vector<std::int32_t> tiles;
  for (std::size_t e = 0; e < experts; ++e) {
    const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
    for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j) {
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
  }
  __half* d_x_half = Upload(x_half);
  std::int32_t* d_tiles = Upload(tiles);
  float* d_f16 = Upload(zero_out);
  if (!q::RoutedF16Gemm(d_wp, type, d_x_half, d_tiles,
                        static_cast<std::uint32_t>(tiles.size()), tile_rows,
                        d_bounds, d_rows_token, d_rows_slot, nullptr, d_f16,
                        nullptr, m, k, nullptr)) {
    throw std::runtime_error("routed F16 GEMM rejected the shape");
  }
  // The F16-output route (the up and down projections' rows) must match
  // the F32 output to F16 rounding.
  constexpr std::size_t kHalfGuard = 16;
  const __half half_guard = __float2half(123.0F);
  __half* d_f16_half =
      Upload(std::vector<__half>(slots * m + kHalfGuard, half_guard));
  if (!q::RoutedF16Gemm(d_wp, type, d_x_half, d_tiles,
                        static_cast<std::uint32_t>(tiles.size()), tile_rows,
                        d_bounds, d_rows_token, d_rows_slot, nullptr, nullptr,
                        d_f16_half, m, k, nullptr)) {
    throw std::runtime_error("routed F16 GEMM (F16 out) rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "routed GEMMs");
  const auto mmq = Download(d_mmq, slots * m);
  const auto f16 = Download(d_f16, slots * m);
  const auto f16_half = Download(d_f16_half, slots * m);
  if (tile_rows == 64) {
    std::vector<std::int32_t> narrow_tiles;
    for (std::size_t e = 0; e < experts; ++e) {
      const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 47u) / 48u; ++j) {
        narrow_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
      }
    }
    auto* d_narrow_tiles = Upload(narrow_tiles);
    if (!q::RoutedF16Gemm(d_wp, type, d_x_half, d_narrow_tiles,
                          static_cast<std::uint32_t>(narrow_tiles.size()), 48,
                          d_bounds, d_rows_token, d_rows_slot, nullptr, d_f16,
                          nullptr, m, k, nullptr)) {
      throw std::runtime_error("routed reference tile rejected");
    }
    CheckHip(hipDeviceSynchronize(), "routed tile comparison");
    const auto narrow = Download(d_f16, slots * m);
    if (std::memcmp(narrow.data(), f16.data(), f16.size() * sizeof(float)) !=
        0) {
      throw std::runtime_error("routed tile widths change output bits");
    }
    CheckHip(hipFree(d_narrow_tiles), "free reference tiles");
  }
  for (std::size_t i = 0; i < slots * m; ++i) {
    const __half expected = __float2half(f16[i]);
    if (std::memcmp(&expected, &f16_half[i], sizeof(__half)) != 0) {
      throw std::runtime_error("routed F16 output changed rounding");
    }
  }
  const auto half_tail = Download(d_f16_half + slots * m, kHalfGuard);
  for (__half value : half_tail) {
    if (std::memcmp(&value, &half_guard, sizeof(__half)) != 0)
      throw std::runtime_error("routed F16 output overwrote its guard");
  }

  if (tile_rows == 48 &&
      (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K)) {
    // Different gate/up weights, with the up scale small enough that these
    // deliberately large synthetic weights do not overflow the F16 output.
    auto up = w.packed;
    const std::size_t block_bytes = type == q::WeightType::kQ4_K ? 144 : 176;
    for (std::size_t b = 0; b < up.size(); b += block_bytes) {
      for (std::size_t offset : {0U, 2U}) {
        __half scale;
        std::memcpy(&scale, up.data() + b + offset, sizeof(scale));
        scale = __float2half(__half2float(scale) / 1024.0F);
        std::memcpy(up.data() + b + offset, &scale, sizeof(scale));
      }
    }
    auto* d_up = Upload(up);
    if (!q::PackRoutedExperts(d_up, type, experts * m, k, nullptr)) {
      throw std::runtime_error("PackRoutedExperts failed");
    }
    auto* d_pair = Upload(std::vector<__half>(slots * m, __float2half(0.0F)));
    if (!q::RoutedF16Gemm(d_up, type, d_x_half, d_tiles,
                          static_cast<std::uint32_t>(tiles.size()), tile_rows,
                          d_bounds, d_rows_token, d_rows_slot, d_f16, nullptr,
                          d_f16_half, m, k, nullptr)) {
      throw std::runtime_error("separate routed SwiGLU launch failed");
    }
    const auto separate = Download(d_f16_half, slots * m);
    for (std::uint32_t pair_rows : {64U, 128U}) {
      std::vector<std::int32_t> gate_tiles;
      for (std::size_t e = 0; e < experts; ++e) {
        for (std::uint32_t j = 0; j < (counts[e] + pair_rows - 1) / pair_rows;
             ++j) {
          gate_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
        }
      }
      auto* d_gate_tiles = Upload(gate_tiles);
      auto launch_pair = [&] {
        if (!q::RoutedGatedF16Gemm(
                d_wp, d_up, type, d_x_half, d_gate_tiles,
                static_cast<std::uint32_t>(gate_tiles.size()), pair_rows,
                d_bounds, d_rows_token, d_rows_slot, d_pair, m, k, nullptr)) {
          throw std::runtime_error("paired routed SwiGLU launch failed");
        }
      };
      launch_pair();
      const auto paired = Download(d_pair, slots * m);
      for (std::size_t i = 0; i < paired.size(); ++i) {
        const float a = __half2float(separate[i]);
        const float b = __half2float(paired[i]);
        // The independent epilogue permits either sign of zero; every
        // numerical value must match, including subnormal F16 values.
        if (!std::isfinite(a) || !std::isfinite(b) || a != b) {
          throw std::runtime_error(
              "paired routed SwiGLU differs at " + std::to_string(i) + ": " +
              std::to_string(a) + " vs " + std::to_string(b));
        }
      }
      launch_pair();
      const auto replay = Download(d_pair, slots * m);
      if (std::memcmp(paired.data(), replay.data(),
                      paired.size() * sizeof(__half)) != 0) {
        throw std::runtime_error("paired routed SwiGLU replay changed bits");
      }
      CheckHip(hipFree(d_gate_tiles), "paired tile map free");
    }
    CheckHip(hipFree(d_up), "paired up free");
    CheckHip(hipFree(d_pair), "paired output free");
  }

  Result r{0.0, 0.0, 0.0, 0.0};
  for (std::size_t s = 0; s < slots; ++s) {
    const std::size_t t = s / used;
    const std::int32_t e = ids[s];
    for (std::size_t row = 0; row < m; ++row) {
      double ref = 0.0;
      const float* wr = w.values.data() + (e * m + row) * k;
      const float* xr = x.data() + t * k;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(wr[i]) * xr[i];
      }
      const float f = f16[s * m + row];
      if (!std::isfinite(f)) {
        throw std::runtime_error("routed F16 output is not finite");
      }
      const float v = mmq[s * m + row];
      r.vs_mmq = std::max(r.vs_mmq, std::abs(static_cast<double>(v - f)));
      r.vs_ref = std::max(r.vs_ref, std::abs(ref - f));
      r.mmq_vs_ref = std::max(r.mmq_vs_ref, std::abs(ref - v));
      r.scale = std::max(r.scale, std::abs(ref));
    }
  }
  std::cout << (type == q::WeightType::kQ4_K   ? "Q4_K"
                : type == q::WeightType::kQ5_K ? "Q5_K"
                : type == q::WeightType::kQ5_1 ? "Q5_1"
                : type == q::WeightType::kQ4_0 ? "Q4_0"
                                               : "Q8_0")
            << " routed n=" << n_tokens << " used=" << used << " E=" << experts
            << " m=" << m << " k=" << k << ": worst |F16 - MMQ| " << r.vs_mmq
            << ", worst |F16 - F64| " << r.vs_ref << ", worst |MMQ - F64| "
            << r.mmq_vs_ref << " (scale " << r.scale << ", widest bucket "
            << max_bucket << ")\n";
  for (void* p :
       {static_cast<void*>(d_w), static_cast<void*>(d_wp),
        static_cast<void*>(d_x), static_cast<void*>(d_ids),
        static_cast<void*>(d_counts), static_cast<void*>(d_bounds),
        static_cast<void*>(d_cursors), static_cast<void*>(d_rows_token),
        static_cast<void*>(d_rows_slot), static_cast<void*>(d_mmq),
        static_cast<void*>(d_x_half), static_cast<void*>(d_tiles),
        static_cast<void*>(d_f16), static_cast<void*>(d_f16_half)}) {
    (void)hipFree(p);
  }
  return r;
}

bool Ok(const Result& r) {
  // The tier quantizes the activations to 8 bits per 32-wide block, the F16
  // route rounds them (and the dequantized weights) to F16, so the two
  // agree to the tier's quantization; the F16 route sits well inside it.
  return r.vs_mmq < 1e-2 * r.scale && r.vs_ref < 2e-3 * r.scale &&
         r.mmq_vs_ref < 1e-2 * r.scale;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    // Gate/up view: 64 experts, top-10, 640 x 2560 Q4_K.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 10, 64, 640, 2560, 0x1234ABCDU)) &&
         ok;
    // Down view: single-expert slot rows, 2560 x 640 Q5_1.
    ok = Ok(Run(q::WeightType::kQ5_1, 3000, 1, 64, 2560, 640, 0x0BADF00DU,
                64)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 97, 1, 8, 160, 640, 0x51516464U, 64)) &&
         ok;
    // Q5_K gate/up view (one layer).
    ok = Ok(Run(q::WeightType::kQ5_K, 300, 10, 64, 640, 2560, 0x5A5A0001U)) &&
         ok;
    // Q8_0 down view (five layers keep Q8_0 down projections).
    ok = Ok(Run(q::WeightType::kQ8_0, 3000, 1, 64, 2560, 640, 0x0C0FFEE0U,
                64)) &&
         ok;
    // Ragged rows against the 128-row tile and a tiny batch.
    ok = Ok(Run(q::WeightType::kQ4_K, 40, 4, 8, 224, 512, 0xDEADBEEFU)) && ok;
    // The 16-row tile (small buckets) on every type.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 10, 64, 640, 2560, 0x16161616U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_K, 300, 10, 64, 640, 2560, 0x16160002U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 3000, 1, 64, 2560, 640, 0x16160003U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ8_0, 3000, 1, 64, 2560, 640, 0x16160004U,
                16)) &&
         ok;
    // lfm2moe (LFM2.5-24B-A2B): 64 experts, top-4, 1536 x 2048 Q4_K
    // gate/up and 2048 x 1536 Q5_1 down.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 4, 64, 1536, 2048, 0x1F2A0001U)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 4, 64, 1536, 2048, 0x1F2A0002U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 1200, 1, 64, 2048, 1536, 0x1F2A0003U,
                64)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 1200, 1, 64, 2048, 1536, 0x1F2A0004U,
                16)) &&
         ok;
    // Liquid's QAD file: Q4_0 gate/up in the Q4_K step layout.
    ok = Ok(Run(q::WeightType::kQ4_0, 300, 4, 64, 1536, 2048, 0x1F2A0005U)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ4_0, 300, 4, 64, 1536, 2048, 0x1F2A0006U,
                16)) &&
         ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
