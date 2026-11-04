#include "src/models/qwen38_flash_next/kernels/rocm/dense_i4.hpp"

#include <cmath>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm::dense_i4 {
namespace {

constexpr unsigned kThreads = 256;
constexpr int kMaxK = 6144;
constexpr int kMaxCross = 48;
constexpr int kMaxQuads = kMaxK / 4 / static_cast<int>(kThreads);

constexpr bool HadamardOrder(std::uint32_t n) {
  if (n >= 1 && (n & (n - 1)) == 0)
    return true;
  const std::uint32_t q = n - 1;
  if (q < 3 || q % 4 != 3)
    return false;
  for (std::uint32_t d = 2; d * d <= q; ++d)
    if (q % d == 0)
      return false;
  return true;
}

/// Entry (r, c) of the Hadamard matrix of order n (Sylvester / Paley I),
/// the spike's HadamardEntry.
int HadamardEntry(std::uint32_t n, std::uint32_t r, std::uint32_t c) {
  if ((n & (n - 1)) == 0)
    return (__builtin_popcount(r & c) & 1) != 0 ? -1 : 1;
  if (r == 0)
    return 1;
  if (c == 0)
    return -1;
  if (r == c)
    return 1;
  const std::uint32_t q = n - 1;
  const std::uint32_t a = (c + q - r) % q;
  std::uint32_t p = 1;
  for (std::uint32_t e = 0; e < (q - 1) / 2; ++e)
    p = (p * a) % q;
  return p == 1 ? 1 : -1;
}

/// The rotation as kernel arguments: inner block b and its normalization,
/// nb cross blocks (1: none) with one sign mask per output block (bit i set:
/// entry -1) and their normalization.
struct Plan {
  std::uint64_t row[kMaxCross];
  int b;
  int nb;
  float inner_norm;
  float cross_norm;
};

Plan MakePlan(std::uint32_t k, Rotation rotation) {
  Plan p{};
  p.b = rotation == Rotation::kFull ? static_cast<int>(FullBlock(k)) : 128;
  p.nb = rotation == Rotation::kFull ? static_cast<int>(k) / p.b : 1;
  p.inner_norm = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.b)));
  p.cross_norm = static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.nb)));
  for (int r = 0; r < p.nb && p.nb > 1; ++r)
    for (int i = 0; i < p.nb; ++i)
      if (HadamardEntry(static_cast<std::uint32_t>(p.nb),
                        static_cast<std::uint32_t>(r),
                        static_cast<std::uint32_t>(i)) < 0)
        p.row[r] |= std::uint64_t{1} << i;
  return p;
}

/// The spike's fixed sign vector D.
__device__ __forceinline__ float Sign(int i) {
  const unsigned h = static_cast<unsigned>(i) * 2654435761U;
  return ((h >> 29) & 1U) != 0U ? -1.0F : 1.0F;
}

__device__ __forceinline__ float BlockMax(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  if ((threadIdx.x & 31) == 0)
    red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = 0.0F;
#pragma unroll
  for (unsigned w = 0; w < kThreads / 32; ++w)
    v = fmaxf(v, red[w]);
  __syncthreads();
  return v;
}

__device__ __forceinline__ float BlockSum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  if ((threadIdx.x & 31) == 0)
    red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = 0.0F;
#pragma unroll
  for (unsigned w = 0; w < kThreads / 32; ++w)
    v += red[w];
  __syncthreads();
  return v;
}

/// Byte offset of element pair (e, e + 1) of row `row` in a fragment-major
/// int4 matrix of width k.
__device__ __forceinline__ std::size_t CodeOffset(int row, int e, int k) {
  return ((static_cast<std::size_t>(row / 16) * (k / 32) + (e / 32)) * 256) +
         ((row % 16) * 16) + ((e % 32) / 2);
}

__device__ __forceinline__ unsigned Code(float v, float inv) {
  const float q = fminf(fmaxf(__builtin_rintf(v * inv), -8.0F), 7.0F);
  return static_cast<unsigned>(static_cast<int>(q)) & 0xFU;
}

template<typename T>
__device__ __forceinline__ void LoadQuad(const T* p, float (&v)[4]) {
  if constexpr (sizeof(T) == 2) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __half2 h0 = __builtin_bit_cast(__half2, raw.x);
    const __half2 h1 = __builtin_bit_cast(__half2, raw.y);
    v[0] = __low2float(h0);
    v[1] = __high2float(h0);
    v[2] = __low2float(h1);
    v[3] = __high2float(h1);
  } else {
    const float4 f = *reinterpret_cast<const float4*>(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
  }
}

// One token row per block. Thread quad q = tid + 256 n holds elements
// 4q..4q+3, so a wave covers one aligned 128-element block (four 32-element
// blocks for b = 32): the butterflies run in registers (h = 1, 2) and across
// lanes (h = 4 .. b/2), in BlockHadamard's order. The cross-block transform
// goes through LDS (k floats). Padding rows (row >= rows) write zero codes
// and a zero scale.
template<typename T, bool kGroups>
__global__ __launch_bounds__(kThreads) void QuantizeRows(
    const T* __restrict__ x, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales, std::uint8_t* __restrict__ groups, int rows,
    int k_in, int k, Plan plan) {
  extern __shared__ float s[];
  __shared__ float red[kThreads / 32];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int quads = k / 4;
  const bool live = row < rows;
  float v[kMaxQuads][4];
#pragma unroll
  for (int n = 0; n < kMaxQuads; ++n) {
    const int q = tid + (n * static_cast<int>(kThreads));
    if (q >= quads)
      break;
    if (live && 4 * q < k_in) {
      LoadQuad(x + (static_cast<std::size_t>(row) * k_in) + (4 * q), v[n]);
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[n][e] = 0.0F;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
      v[n][e] *= Sign(((4 * q) + e) % plan.b);
    const float a0 = v[n][0] + v[n][1];
    const float a1 = v[n][0] - v[n][1];
    const float a2 = v[n][2] + v[n][3];
    const float a3 = v[n][2] - v[n][3];
    v[n][0] = a0 + a2;
    v[n][2] = a0 - a2;
    v[n][1] = a1 + a3;
    v[n][3] = a1 - a3;
    for (int d = 1; d < plan.b / 4; d <<= 1) {
      const bool upper = (lane & d) != 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float other = __shfl_xor(v[n][e], d);
        v[n][e] = upper ? other - v[n][e] : v[n][e] + other;
      }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
      v[n][e] *= plan.inner_norm;
  }
  if (plan.nb > 1) {
#pragma unroll
    for (int n = 0; n < kMaxQuads; ++n) {
      const int q = tid + (n * static_cast<int>(kThreads));
      if (q >= quads)
        break;
      *reinterpret_cast<float4*>(s + (4 * q)) =
          make_float4(v[n][0], v[n][1], v[n][2], v[n][3]);
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < kMaxQuads; ++n) {
      const int q = tid + (n * static_cast<int>(kThreads));
      if (q >= quads)
        break;
      const int r = (4 * q) / plan.b;
      const int j = (4 * q) % plan.b;
      const std::uint64_t signs = plan.row[r];
      float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      for (int i = 0; i < plan.nb; ++i) {
        const float sign = ((signs >> i) & 1U) != 0U ? -1.0F : 1.0F;
        const float4 t = *reinterpret_cast<const float4*>(s + (i * plan.b) + j);
        acc[0] = __fmaf_rn(sign, t.x, acc[0]);
        acc[1] = __fmaf_rn(sign, t.y, acc[1]);
        acc[2] = __fmaf_rn(sign, t.z, acc[2]);
        acc[3] = __fmaf_rn(sign, t.w, acc[3]);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[n][e] = acc[e] * plan.cross_norm;
    }
  }
  // A wave's quads of one n are one 128-wide slab (wave + 8n).
  float slab_max[kMaxQuads];
  float m = 0.0F;
#pragma unroll
  for (int n = 0; n < kMaxQuads; ++n) {
    slab_max[n] = 0.0F;
    const int q = tid + (n * static_cast<int>(kThreads));
    if (q >= quads)
      break;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      slab_max[n] = fmaxf(slab_max[n], fabsf(v[n][e]));
    if constexpr (kGroups) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        slab_max[n] = fmaxf(slab_max[n], __shfl_xor(slab_max[n], o));
    }
    m = fmaxf(m, slab_max[n]);
  }
  m = BlockMax(m, red);
  // Token scale; with groups the unit of the integer slab multiples.
  const float scale = kGroups ? m / (7.0F * 63.0F) : m / 7.0F;
#pragma unroll
  for (int n = 0; n < kMaxQuads; ++n) {
    const int q = tid + (n * static_cast<int>(kThreads));
    if (q >= quads)
      break;
    float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
    if constexpr (kGroups) {
      const int slab = (4 * q) / 128;
      const float multiple =
          scale > 0.0F
              ? fminf(fmaxf(ceilf(slab_max[n] / (7.0F * scale)), 1.0F), 63.0F)
              : 1.0F;
      inv = scale > 0.0F ? 1.0F / (scale * multiple) : 0.0F;
      if ((tid & 31) == 0)
        groups[((static_cast<std::size_t>(row / 16) * (k / 128) + slab) * 16) +
               (row % 16)] = static_cast<std::uint8_t>(multiple);
    }
    const unsigned packed = Code(v[n][0], inv) | (Code(v[n][1], inv) << 4) |
                            (Code(v[n][2], inv) << 8) |
                            (Code(v[n][3], inv) << 12);
    *reinterpret_cast<std::uint16_t*>(codes + CodeOffset(row, 4 * q, k)) =
        static_cast<std::uint16_t>(packed);
  }
  if (tid == 0)
    scales[row] = scale;
}

// One weight row per block: dequantize the Q8_0 row into LDS, rotate (the
// same transform as QuantizeRows, in LDS), choose the per-row scale by the
// clip search, write codes and the tile-permuted scale. kGroups: one scale
// per 128-wide slab (clip search per slab, scales rounded to integer
// multiples 1..63 of the row scale d = largest / 63), the multiples in
// `groups` at [row tile][slab][half][8]. Dynamic LDS: 2k floats. Rows past m
// stay zero (the buffer is cleared first).
template<bool kGroups>
__global__ __launch_bounds__(kThreads) void PackRows(
    const std::uint8_t* __restrict__ q8, int m, int k_in, int k, int streams,
    std::uint8_t* __restrict__ codes, float* __restrict__ scales,
    std::uint8_t* __restrict__ groups, Plan plan) {
  extern __shared__ float s[];
  float* t = s + k;
  __shared__ float red[kThreads / 32];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  // Packed row r holds source row (r % streams) * (m / streams) + r / streams.
  const int source = ((row % streams) * (m / streams)) + (row / streams);
  const std::uint8_t* rp =
      q8 + (static_cast<std::size_t>(source) * (k_in / 32) * 34);
  for (int i = tid; i < k; i += static_cast<int>(kThreads)) {
    float value = 0.0F;
    if (i < k_in) {
      const std::uint8_t* blk = rp + (static_cast<std::size_t>(i / 32) * 34);
      const auto dh = static_cast<std::uint16_t>(blk[0] | (blk[1] << 8));
      const float d = __half2float(__ushort_as_half(dh));
      value =
          d * static_cast<float>(static_cast<std::int8_t>(blk[2 + (i % 32)]));
    }
    s[i] = value * Sign(i % plan.b);
  }
  __syncthreads();
  const int half_block = plan.b / 2;
  for (int h = 1; h < plan.b; h <<= 1) {
    for (int p = tid; p < k / 2; p += static_cast<int>(kThreads)) {
      const int b = p / half_block;
      const int j = p - (b * half_block);
      const int i0 = (b * plan.b) + ((j / h) * (h * 2)) + (j % h);
      const float a = s[i0];
      const float c = s[i0 + h];
      s[i0] = a + c;
      s[i0 + h] = a - c;
    }
    __syncthreads();
  }
  for (int i = tid; i < k; i += static_cast<int>(kThreads))
    s[i] *= plan.inner_norm;
  __syncthreads();
  if (plan.nb > 1) {
    for (int e = tid; e < k; e += static_cast<int>(kThreads)) {
      const int r = e / plan.b;
      const int j = e % plan.b;
      float acc = 0.0F;
      for (int i = 0; i < plan.nb; ++i)
        acc = __fmaf_rn(((plan.row[r] >> i) & 1U) != 0U ? -1.0F : 1.0F,
                        s[(i * plan.b) + j], acc);
      t[e] = acc * plan.cross_norm;
    }
    __syncthreads();
    for (int i = tid; i < k; i += static_cast<int>(kThreads))
      s[i] = t[i];
    __syncthreads();
  }
  if constexpr (kGroups) {
    const int slabs = k / 128;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr float kClips[9] = {1.0F,  0.95F, 0.9F,  0.85F, 0.8F,
                                 0.75F, 0.7F,  0.65F, 0.6F};
    for (int sl = warp; sl < slabs; sl += static_cast<int>(kThreads / 32)) {
      float v[4];
      float slab_max = 0.0F;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = s[(sl * 128) + (lane * 4) + e];
        slab_max = fmaxf(slab_max, fabsf(v[e]));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        slab_max = fmaxf(slab_max, __shfl_xor(slab_max, o));
      float best_scale = 0.0F;
      float best_err = 3.0e38F;
      if (slab_max > 0.0F) {
        for (const float clip : kClips) {
          const float sc = slab_max * clip / 7.0F;
          const float inv = 1.0F / sc;
          float err = 0.0F;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float q =
                fminf(fmaxf(__builtin_rintf(v[e] * inv), -8.0F), 7.0F);
            const float diff = v[e] - (q * sc);
            err += diff * diff;
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1)
            err += __shfl_xor(err, o);
          if (err < best_err) {
            best_err = err;
            best_scale = sc;
          }
        }
      }
      if (lane == 0)
        t[sl] = best_scale;
    }
    __syncthreads();
    float scale_max = 0.0F;
    for (int sl = 0; sl < slabs; ++sl)
      scale_max = fmaxf(scale_max, t[sl]);
    const float d = scale_max / 63.0F;
    const int r = row % 16;
    for (int sl = warp; sl < slabs; sl += static_cast<int>(kThreads / 32)) {
      const float multiple =
          d > 0.0F ? fminf(fmaxf(__builtin_rintf(t[sl] / d), 1.0F), 63.0F)
                   : 1.0F;
      const float slab_scale = d * multiple;
      const float inv = slab_scale > 0.0F ? 1.0F / slab_scale : 0.0F;
      const int p = (sl * 32) + lane;
      const unsigned packed =
          Code(s[4 * p], inv) | (Code(s[(4 * p) + 1], inv) << 4) |
          (Code(s[(4 * p) + 2], inv) << 8) | (Code(s[(4 * p) + 3], inv) << 12);
      *reinterpret_cast<std::uint16_t*>(codes + CodeOffset(row, 4 * p, k)) =
          static_cast<std::uint16_t>(packed);
      if (lane == 0)
        groups[((static_cast<std::size_t>(row / 16) * slabs + sl) * 16) +
               ((r % 2) * 8) + (r / 2)] = static_cast<std::uint8_t>(multiple);
    }
    if (tid == 0)
      scales[((((row / 16) * 2) + (r % 2)) * 8) + (r / 2)] = d;
    return;
  }
  float amax = 0.0F;
  for (int i = tid; i < k; i += static_cast<int>(kThreads))
    amax = fmaxf(amax, fabsf(s[i]));
  amax = BlockMax(amax, red);
  constexpr float kClip[8] = {1.0F, 0.95F, 0.9F, 0.85F,
                              0.8F, 0.75F, 0.7F, 0.65F};
  float best_scale = amax / 7.0F;
  float best_err = 3.0e38F;
  if (amax > 0.0F) {
    for (float clip : kClip) {
      const float sc = amax * clip / 7.0F;
      const float inv = 1.0F / sc;
      float err = 0.0F;
      for (int i = tid; i < k; i += static_cast<int>(kThreads)) {
        const float q = fminf(fmaxf(__builtin_rintf(s[i] * inv), -8.0F), 7.0F);
        const float e = s[i] - (q * sc);
        err += e * e;
      }
      err = BlockSum(err, red);
      if (err < best_err) {
        best_err = err;
        best_scale = sc;
      }
    }
  }
  const float inv = best_scale > 0.0F ? 1.0F / best_scale : 0.0F;
  for (int p = tid; p < k / 4; p += static_cast<int>(kThreads)) {
    const unsigned packed =
        Code(s[4 * p], inv) | (Code(s[(4 * p) + 1], inv) << 4) |
        (Code(s[(4 * p) + 2], inv) << 8) | (Code(s[(4 * p) + 3], inv) << 12);
    *reinterpret_cast<std::uint16_t*>(codes + CodeOffset(row, 4 * p, k)) =
        static_cast<std::uint16_t>(packed);
  }
  if (tid == 0) {
    const int r = row % 16;
    scales[((((row / 16) * 2) + (r % 2)) * 8) + (r / 2)] = best_scale;
  }
}

// One offline-quantized row per block into row row0 + blockIdx.x of a
// grouped weight buffer (codes [rows][k] int8, sc [rows][k / 128], d [rows]).
__global__ __launch_bounds__(kThreads) void RepackS4(
    const std::int8_t* __restrict__ src, const std::uint8_t* __restrict__ sc,
    const float* __restrict__ d, int row0, int k,
    std::uint8_t* __restrict__ codes, float* __restrict__ scales,
    std::uint8_t* __restrict__ groups) {
  const int src_row = static_cast<int>(blockIdx.x);
  const int row = row0 + src_row;
  const int tid = static_cast<int>(threadIdx.x);
  const int slabs = k / 128;
  const int r = row % 16;
  for (int p = tid; p < k / 4; p += static_cast<int>(kThreads)) {
    const std::uint32_t quad = *reinterpret_cast<const std::uint32_t*>(
        src + (static_cast<std::size_t>(src_row) * k) + (4 * p));
    const unsigned packed = (quad & 0xFU) | ((quad >> 4) & 0xF0U) |
                            ((quad >> 8) & 0xF00U) | ((quad >> 12) & 0xF000U);
    *reinterpret_cast<std::uint16_t*>(codes + CodeOffset(row, 4 * p, k)) =
        static_cast<std::uint16_t>(packed);
  }
  for (int sl = tid; sl < slabs; sl += static_cast<int>(kThreads))
    groups[((static_cast<std::size_t>(row / 16) * slabs + sl) * 16) +
           ((r % 2) * 8) + (r / 2)] =
        sc[(static_cast<std::size_t>(src_row) * slabs) + sl];
  if (tid == 0)
    scales[((((row / 16) * 2) + (r % 2)) * 8) + (r / 2)] = d[src_row];
}

template<typename T>
bool QuantizeImpl(const T* x, std::uint32_t rows, std::uint32_t k_in,
                  Rotation rotation, void* buffer, hipStream_t stream,
                  bool groups) {
  if (x == nullptr || buffer == nullptr || rows == 0 || k_in % 4 != 0 ||
      !Supported(k_in, rotation))
    return false;
  const std::uint32_t k = Width(k_in);
  const Plan plan = MakePlan(k, rotation);
  const Matrix view = ActivationView(buffer, rows, k_in, true);
  const std::uint32_t padded = Padded(rows, kTokenPad);
  const auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(
        kernel, dim3(padded), dim3(kThreads),
        plan.nb > 1 ? k * sizeof(float) : 0, stream, x,
        static_cast<std::uint8_t*>(buffer), const_cast<float*>(view.scale),
        const_cast<std::uint8_t*>(view.group), static_cast<int>(rows),
        static_cast<int>(k_in), static_cast<int>(k), plan);
  };
  if (groups)
    launch(QuantizeRows<T, true>);
  else
    launch(QuantizeRows<T, false>);
  return true;
}

}  // namespace

std::uint32_t FullBlock(std::uint32_t k) {
  for (std::uint32_t b = 128; b >= 32; b >>= 1)
    if (k % b == 0 && k / b <= kMaxCross && HadamardOrder(k / b))
      return b;
  return 0;
}

bool Supported(std::uint32_t k, Rotation rotation) {
  return k % 32 == 0 && Width(k) <= kMaxK &&
         (rotation == Rotation::kBlock128 || FullBlock(Width(k)) != 0);
}

std::size_t WeightBytes(std::uint32_t m, std::uint32_t k, bool groups) {
  const std::size_t rows = Padded(m, kRowPad);
  return (rows * (Width(k) / 2)) + (rows * sizeof(float)) +
         (groups ? rows * (Width(k) / 128) : 0);
}

std::size_t ActivationBytes(std::uint32_t rows, std::uint32_t k) {
  k = Width(k);
  const std::size_t padded = Padded(rows, kTokenPad);
  // The GEMM's last K slab prefetches one (unused) slab past the last token
  // tile; the tail keeps it inside the allocation.
  constexpr std::size_t kSlabTail = 8 * 16 * 16;
  return (padded * (k / 2)) + (padded * sizeof(float)) + (padded * (k / 128)) +
         kSlabTail;
}

Matrix WeightView(const void* buffer, std::uint32_t m, std::uint32_t k,
                  bool groups) {
  k = Width(k);
  const auto* bytes = static_cast<const std::uint8_t*>(buffer);
  const std::size_t rows = Padded(m, kRowPad);
  return {buffer, reinterpret_cast<const float*>(bytes + (rows * (k / 2))), m,
          k,
          groups ? bytes + (rows * (k / 2)) + (rows * sizeof(float)) : nullptr};
}

Matrix ActivationView(const void* buffer, std::uint32_t rows, std::uint32_t k,
                      bool groups) {
  k = Width(k);
  const auto* bytes = static_cast<const std::uint8_t*>(buffer);
  const std::size_t padded = Padded(rows, kTokenPad);
  return {
      buffer, reinterpret_cast<const float*>(bytes + (padded * (k / 2))), rows,
      k,
      groups ? bytes + (padded * (k / 2)) + (padded * sizeof(float)) : nullptr};
}

bool PackQ8_0(const void* q8, std::uint32_t m, std::uint32_t k_in,
              Rotation rotation, void* buffer, hipStream_t stream,
              std::uint32_t streams, bool groups) {
  if (q8 == nullptr || buffer == nullptr || m == 0 || streams == 0 ||
      m % streams != 0 || !Supported(k_in, rotation))
    return false;
  const std::uint32_t k = Width(k_in);
  const Plan plan = MakePlan(k, rotation);
  const Matrix view = WeightView(buffer, m, k_in, groups);
  if (hipMemsetAsync(buffer, 0, WeightBytes(m, k_in, groups), stream) !=
      hipSuccess)
    return false;
  const auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(
        kernel, dim3(m), dim3(kThreads), 2 * k * sizeof(float), stream,
        static_cast<const std::uint8_t*>(q8), static_cast<int>(m),
        static_cast<int>(k_in), static_cast<int>(k), static_cast<int>(streams),
        static_cast<std::uint8_t*>(buffer), const_cast<float*>(view.scale),
        const_cast<std::uint8_t*>(view.group), plan);
  };
  if (groups)
    launch(PackRows<true>);
  else
    launch(PackRows<false>);
  return hipGetLastError() == hipSuccess;
}

bool PackS4(const std::int8_t* codes, const std::uint8_t* sc, const float* d,
            std::uint32_t rows, std::uint32_t k, std::uint32_t row0,
            std::uint32_t m_total, void* buffer, hipStream_t stream) {
  if (codes == nullptr || sc == nullptr || d == nullptr || buffer == nullptr ||
      rows == 0 || k % 128 != 0 || Width(k) != k || row0 + rows > m_total)
    return false;
  const Matrix view = WeightView(buffer, m_total, k, true);
  hipLaunchKernelGGL(RepackS4, dim3(rows), dim3(kThreads), 0, stream, codes, sc,
                     d, static_cast<int>(row0), static_cast<int>(k),
                     static_cast<std::uint8_t*>(buffer),
                     const_cast<float*>(view.scale),
                     const_cast<std::uint8_t*>(view.group));
  return hipGetLastError() == hipSuccess;
}

bool Quantize(const __half* x, std::uint32_t rows, std::uint32_t k,
              Rotation rotation, void* buffer, hipStream_t stream,
              bool groups) {
  return QuantizeImpl(x, rows, k, rotation, buffer, stream, groups);
}

bool Quantize(const float* x, std::uint32_t rows, std::uint32_t k,
              Rotation rotation, void* buffer, hipStream_t stream,
              bool groups) {
  return QuantizeImpl(x, rows, k, rotation, buffer, stream, groups);
}

}  // namespace gufo::models::qwen38_flash_next::rocm::dense_i4
