#include "src/models/qwen38_flash_next/kernels/rocm/routed_iu4.hpp"

#include <cmath>
#include <cstdio>
#include <set>

namespace gufo::models::qwen38_flash_next::rocm::routed_iu4 {
namespace {

constexpr int kThreads = 256;
constexpr int kPitch = 72;  // LDS bytes per row and 128-element slab (64 + 8)
constexpr int kMult = 255;  // activation group multiplier range
constexpr int kMaxChunkIt =
    10;  // quantizer: 8-element chunks per lane, k <= 2560
constexpr int kQ4KStep = 144 * 32;     // one super-block of a 32-row group
constexpr int kQ51Step = 24 * 2 * 32;  // two Q5_1 blocks of a 32-row group

using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
using f16x16 = __attribute__((__vector_size__(32))) _Float16;
using f32x8 = __attribute__((__vector_size__(32))) float;

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0F / (1.0F + __expf(-x));
}

/// Low or high nibbles of eight bytes (two words), compacted in byte order.
template<bool kHigh>
__device__ __forceinline__ std::uint32_t Nibbles(std::uint32_t w0,
                                                 std::uint32_t w1) {
  if constexpr (kHigh) {
    w0 >>= 4U;
    w1 >>= 4U;
  }
  w0 &= 0x0F0F0F0FU;
  w1 &= 0x0F0F0F0FU;
  w0 |= w0 >> 4U;
  w1 |= w1 >> 4U;
  return __builtin_amdgcn_perm(w1, w0, 0x06040200U);
}

/// Byte i of the 16-byte Q4_K header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h, int i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8 * (i & 3))) & 0xFFU;
}

/// Q4_K 6-bit scale and min of sub-block g (get_scale_min_k4).
__device__ __forceinline__ void ScaleMin(const uint4& h, int g, int& sc,
                                         int& mn) {
  if (g < 4) {
    sc = static_cast<int>(HeaderByte(h, 4 + g) & 0x3FU);
    mn = static_cast<int>(HeaderByte(h, 8 + g) & 0x3FU);
  } else {
    sc = static_cast<int>((HeaderByte(h, 8 + g) & 0x0FU) |
                          ((HeaderByte(h, g) >> 6U) << 4U));
    mn = static_cast<int>((HeaderByte(h, 8 + g) >> 4U) |
                          ((HeaderByte(h, 4 + g) >> 6U) << 4U));
  }
}

__device__ __forceinline__ void StoreRow16(std::uint8_t* p, std::uint32_t a,
                                           std::uint32_t b, std::uint32_t c,
                                           std::uint32_t d) {
  *reinterpret_cast<uint2*>(p) = make_uint2(a, b);
  *reinterpret_cast<uint2*>(p + 8) = make_uint2(c, d);
}

// ---------------------------------------------------------------------------
// Block Hadamard (128) of one lane's eight elements, sixteen lanes per block:
// the transform of BlockHadamard in spike.hip.cpp (sign vector on the local
// index, Sylvester-order butterflies, 1/sqrt(128)); h = 1, 2, 4 inside the
// lane, h = 8 .. 64 across lanes (xor 1 .. 8). Called by every lane.
__device__ __forceinline__ float HadSign(int i) {
  const unsigned h = static_cast<unsigned>(i) * 2654435761U;
  return ((h >> 29) & 1U) != 0U ? -1.0F : 1.0F;
}

__device__ __forceinline__ void Hadamard128(float* v, int lane) {
  const int base = (lane & 15) * 8;
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] *= HadSign(base + e);
#pragma unroll
  for (int h = 1; h < 8; h <<= 1)
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if ((e & h) == 0) {
        const float a = v[e], c = v[e + h];
        v[e] = a + c;
        v[e + h] = a - c;
      }
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    const bool upper = (lane & d) != 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float other = __shfl_xor(v[e], d);
      v[e] = upper ? other - v[e] : v[e] + other;
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] *= 0.08838834764831845F;  // 1/sqrt(128)
}

// Activation quantizer: one wave per row, lane chunks of eight elements,
// optionally rotated first (kRotate: 128-element blocks = 16 lanes).
template<int kGroup, bool kRotate>
__global__ __launch_bounds__(kThreads) void QuantizeKernel(
    const __half* __restrict__ x, int rows, int k, int xs_stride,
    std::uint8_t* __restrict__ x4, float* __restrict__ sx,
    std::uint8_t* __restrict__ xa, __half* __restrict__ xs,
    __half* __restrict__ xhat) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * (kThreads / 32) +
                  (static_cast<int>(threadIdx.x) >> 5);
  if (row >= rows)
    return;  // wave-uniform
  const int nch = k >> 3;
  const __half* xr = x + static_cast<std::size_t>(row) * k;
  float v[kMaxChunkIt][8];
  float m = 0.0F;
#pragma unroll
  for (int it = 0; it < kMaxChunkIt; ++it) {
    const int c = it * 32 + lane;
    if (c < nch) {
      const uint4 raw = *reinterpret_cast<const uint4*>(xr + c * 8);
      const std::uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const __half2 h = __builtin_bit_cast(__half2, w[p]);
        v[it][2 * p] = __low2float(h);
        v[it][2 * p + 1] = __high2float(h);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        v[it][e] = 0.0F;
    }
    if constexpr (kRotate) {
      if (it * 32 < nch)  // wave-uniform; k % 128 == 0 keeps blocks whole
        Hadamard128(v[it], lane);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e)
      m = fmaxf(m, fabsf(v[it][e]));
  }
  m = WaveMax(m);
  const float base = kGroup == 0 ? m / 7.0F : m / (7.0F * kMult);
  const int ngroups = k >> 5;
#pragma unroll
  for (int it = 0; it < kMaxChunkIt; ++it) {
    if (it * 32 >= nch)
      break;  // wave-uniform
    const int c = it * 32 + lane;
    const bool live = c < nch;
    float s = base;
    int a = 1;
    if constexpr (kGroup != 0) {
      float gm = 0.0F;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        gm = fmaxf(gm, fabsf(v[it][e]));
#pragma unroll
      for (int o = 1; o < kGroup / 8; o <<= 1)
        gm = fmaxf(gm, __shfl_xor(gm, o));
      a = m > 0.0F ? static_cast<int>(ceilf(gm * kMult / m)) : 1;
      a = a < 1 ? 1 : a > kMult ? kMult : a;
      s = base * static_cast<float>(a);
    }
    const float inv = s > 0.0F ? 1.0F / s : 0.0F;
    int sum = 0;
    std::uint32_t packed = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int q = static_cast<int>(
          fminf(fmaxf(__builtin_rintf(v[it][e] * inv), -8.0F), 7.0F));
      sum += q;
      packed |= static_cast<std::uint32_t>(q & 0xF) << (4 * e);
    }
    sum += __shfl_xor(sum, 1);
    sum += __shfl_xor(sum, 2);
    if (live) {
      *reinterpret_cast<std::uint32_t*>(
          x4 + static_cast<std::size_t>(row) * (k >> 1) + c * 4) = packed;
      if ((lane & 3) == 0) {
        const int g = c >> 2;
        xs[static_cast<std::size_t>(row) * xs_stride + g] =
            __float2half(s * static_cast<float>(sum));
        if constexpr (kGroup != 0)
          xa[static_cast<std::size_t>(row) * ngroups + g] =
              static_cast<std::uint8_t>(a);
      }
      if (xhat != nullptr) {
        std::uint32_t h[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const int q0 = static_cast<int>(packed << (28 - 8 * p)) >> 28;
          const int q1 = static_cast<int>(packed << (24 - 8 * p)) >> 28;
          h[p] = __builtin_bit_cast(
              std::uint32_t,
              __halves2half2(__float2half(static_cast<float>(q0) * s),
                             __float2half(static_cast<float>(q1) * s)));
        }
        *reinterpret_cast<uint4*>(xhat + static_cast<std::size_t>(row) * k +
                                  c * 8) = make_uint4(h[0], h[1], h[2], h[3]);
      }
    }
  }
  for (int g = ngroups + lane; g < xs_stride; g += 32)
    xs[static_cast<std::size_t>(row) * xs_stride + g] = __float2half(0.0F);
  if (lane == 0)
    sx[row] = base;
}

// ---------------------------------------------------------------------------
// Shared GEMM plumbing. 256 threads = 8 waves as 2 (tokens) x 4 (channels);
// block tile kBT = 32 kTI compact rows x 64 kTJ channels, wave tile
// 16 kTI x 16 kTJ (kTI x kTJ WMMA tiles), 128-element K slabs, double-
// buffered LDS with the next slab's global loads in flight during the
// matrix work. Shapes: (kTI, kTJ) = (2, 2) 64 x 128 and (4, 1) 128 x 64
// (the tall tile stages half the weights per WMMA).
//
// Accumulator layout (gfx11 wave32, iu4 and F16 alike): lane l, element e
// holds row 2e + l / 16 (a compact token row), column l % 16 (a channel).

struct Rows {
  int bucket_begin, bucket_rows, t_local;
};

/// The block's compact rows: a tile map entry holds `tile_rows` rows (64 or
/// 128); a block of `bt` < tile_rows rows takes part z of it, the parts in
/// consecutive x blocks so they read the same weights back to back (L2).
__device__ __forceinline__ Rows TileRows(const std::int32_t* tiles,
                                         const std::int32_t* pad_bounds,
                                         int tile_rows, int bt, int z) {
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  Rows r;
  r.bucket_begin = pad_bounds[expert];
  r.bucket_rows = pad_bounds[expert + 1] - r.bucket_begin;
  r.t_local = (tile >> 16) * tile_rows + bt * z;
  return r;
}

/// c + a * b with 24-bit signed operands as one v_mad_i32_i24 (the compiler
/// otherwise pairs two groups into mul + mul + add3).
__device__ __forceinline__ int MadI24(int a, int b, int c) {
  int r;
  asm("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}

/// c + (i16) lo/hi half of p * (i16) low half of b, one v_mad_i32_i16 (the
/// activation multiplier x weight scale products of two groups ride in one
/// register as u16 halves, all < 2^15).
template<bool kHi>
__device__ __forceinline__ int MadI16(std::uint32_t p, int b, int c) {
  int r;
  if constexpr (kHi)
    asm("v_mad_i32_i16 %0, %1, %2, %3 op_sel:[1,0,0,0]"
        : "=v"(r)
        : "v"(p), "v"(b), "v"(c));
  else
    asm("v_mad_i32_i16 %0, %1, %2, %3" : "=v"(r) : "v"(p), "v"(b), "v"(c));
  return r;
}

using u16x2 = __attribute__((ext_vector_type(2))) unsigned short;

/// Bytes (sel0, sel2) or (sel1, sel3) of w as two u16 halves.
template<bool kOdd>
__device__ __forceinline__ std::uint32_t BytePairs(std::uint32_t w) {
  return __builtin_amdgcn_perm(w, w, kOdd ? 0x0C030C01U : 0x0C020C00U);
}

// Paired Q4_K gate/up: block channels 0 .. 32 kTJ - 1 are gate rows r0..,
// the next 32 kTJ the matching up rows; waves 0-1 (per token half) compute
// gate, 2-3 up.
template<int kTI, int kTJ, int kGroup>
__global__ __launch_bounds__(kThreads) void GateUpKernel(
    const std::uint8_t* __restrict__ gate, const std::uint8_t* __restrict__ up,
    const std::uint8_t* __restrict__ x4, const float* __restrict__ sx,
    const std::uint8_t* __restrict__ xa, const __half* __restrict__ xs,
    int xs_stride, const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_in,
    const std::int32_t* __restrict__ rows_out, __half* __restrict__ out, int m,
    int k, int tile_rows) {
  constexpr int kBT = 32 * kTI;
  constexpr int kCh = 64 * kTJ;      // block channels (gate + up)
  constexpr int kHalfCh = 32 * kTJ;  // gate (or up) rows per block
  constexpr int kXBytes = kBT * kPitch;
  constexpr int kWBytes = kCh * kPitch;
  constexpr int kOffW = 2 * kXBytes;
  constexpr int kOffA = kOffW + 2 * kWBytes;
  constexpr int kOffSc = kOffA + 2 * kBT * 4;
  constexpr int kOffD = kOffSc + 2 * kCh * 4;
  constexpr int kOffWm = kOffD + 2 * kCh * 4;
  constexpr int kOffSx = kOffWm + 2 * kCh * 16 * 2;
  constexpr int kLds = kOffSx + kBT * 4;
  constexpr int kGs = kHalfCh + 4;  // epilogue gate row stride (floats)
  static_assert(kBT * kGs * 4 <= kLds);
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLds];
  auto* s_a = reinterpret_cast<std::uint32_t*>(lds + kOffA);
  auto* s_sc = reinterpret_cast<std::uint32_t*>(lds + kOffSc);
  auto* s_d = reinterpret_cast<float*>(lds + kOffD);
  auto* s_wm = reinterpret_cast<__half*>(lds + kOffWm);
  auto* s_sx = reinterpret_cast<float*>(lds + kOffSx);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5, lane = tid & 31;
  const int wave_t = wave >> 2, wave_n = wave & 3;
  const int half = lane >> 4, l16 = lane & 15;
  const int zs = tile_rows / kBT;
  const Rows tr = TileRows(tiles, pad_bounds, tile_rows, kBT,
                           static_cast<int>(blockIdx.x) % zs);
  if (tr.t_local >= tr.bucket_rows)
    return;  // block-uniform: the second half of a short 128-row tile
  const int expert = tiles[blockIdx.y] & 0xFFFF;
  const int r0 = static_cast<int>(blockIdx.x) / zs * kHalfCh;
  const int nslab = k >> 7;
  const std::size_t row_bytes = static_cast<std::size_t>(k >> 8) * 144;
  const std::size_t xrow = static_cast<std::size_t>(k) >> 1;

  // Activation staging: kBT rows x 4 16-byte parts per slab.
  constexpr int kXPer = kBT * 4 / kThreads;
  const std::uint8_t* xsrc[kXPer];
  int xdst[kXPer];
#pragma unroll
  for (int p = 0; p < kXPer; ++p) {
    const int c = tid + p * kThreads;
    const int row = c >> 2, part = c & 3;
    const int cr = tr.t_local + row;
    const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
    xsrc[p] = src >= 0 ? x4 + static_cast<std::size_t>(src) * xrow + part * 16
                       : nullptr;
    xdst[p] = row * kPitch + part * 16;
  }
  const std::uint8_t* asrc = nullptr;
  if constexpr (kGroup != 0) {
    if (tid < kBT) {
      const int cr = tr.t_local + tid;
      const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
      asrc = src >= 0 ? xa + static_cast<std::size_t>(src) * (k >> 5) : nullptr;
    }
  }
  // Weight staging: a slab of one row is the 64 nibble bytes of two Q4_K
  // nibble pairs, four 16-byte pieces. Job c = (piece p = c / kCh, row
  // c % kCh): 32 consecutive rows per wave read 512 contiguous bytes. Piece
  // p is half h = p & 1 of pair j = 2 (slab & 1) + (p >> 1), super-block
  // piece 1 + 2j + h. Jobs with p = 0 also stage the row's scales and d,
  // p = 1 its min terms.
  constexpr int kWPer = kCh * 4 / kThreads;
  const std::uint8_t* wbase[kWPer];
  int wrow[kWPer], wp[kWPer];
#pragma unroll
  for (int q = 0; q < kWPer; ++q) {
    const int c = tid + q * kThreads;
    wrow[q] = c % kCh;
    wp[q] = c / kCh;
    const int rr = r0 + wrow[q] % kHalfCh;
    wbase[q] = (wrow[q] < kHalfCh ? gate : up) +
               static_cast<std::size_t>(expert) * m * row_bytes +
               static_cast<std::size_t>(rr & ~31) * row_bytes + (rr & 31) * 16;
  }

  // Row scales of the block's tokens (LDS), and this lane's F16 offset
  // fragment row (l16) of each WMMA tile.
  if (tid < kBT) {
    const int cr = tr.t_local + tid;
    const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
    s_sx[tid] = src >= 0 ? sx[src] : 0.0F;
  }
  int xs_src[kTI];
  bool live_i[kTI];
#pragma unroll
  for (int i = 0; i < kTI; ++i) {
    const int t0 = tr.t_local + wave_t * 16 * kTI + 16 * i;
    live_i[i] = t0 < tr.bucket_rows;
    // Padding rows read row 0's sums: their outputs are never stored.
    const int t = t0 + l16;
    const int src = t < tr.bucket_rows ? rows_in[tr.bucket_begin + t] : -1;
    xs_src[i] = src >= 0 ? src : 0;
  }

  uint4 xr[kXPer];
  uint4 wq[kWPer], whdr[kWPer];
#pragma unroll
  for (int q = 0; q < kWPer; ++q)
    whdr[q] = make_uint4(0U, 0U, 0U, 0U);
  std::uint32_t ar = 0;

  const auto load_global = [&](int slab) {
#pragma unroll
    for (int q = 0; q < kWPer; ++q) {
      const std::uint8_t* step =
          wbase[q] + static_cast<std::size_t>(slab >> 1) * kQ4KStep;
      const int piece = 1 + 4 * (slab & 1) + wp[q];
      wq[q] = *reinterpret_cast<const uint4*>(step + piece * 512);
      if ((slab & 1) == 0 && wp[q] < 2)  // wave-uniform
        whdr[q] = *reinterpret_cast<const uint4*>(step);
    }
#pragma unroll
    for (int p = 0; p < kXPer; ++p)
      xr[p] = xsrc[p] != nullptr
                  ? *reinterpret_cast<const uint4*>(xsrc[p] + slab * 64)
                  : make_uint4(0U, 0U, 0U, 0U);
    if constexpr (kGroup != 0) {
      if (tid < kBT)
        ar = asrc != nullptr
                 ? *reinterpret_cast<const std::uint32_t*>(asrc + slab * 4)
                 : 0U;
    }
  };
  const auto commit = [&](int slab, int buf) {
    std::uint8_t* lx = lds + buf * kXBytes;
#pragma unroll
    for (int p = 0; p < kXPer; ++p)
      StoreRow16(lx + xdst[p], xr[p].x, xr[p].y, xr[p].z, xr[p].w);
    if constexpr (kGroup != 0) {
      if (tid < kBT)
        s_a[buf * kBT + tid] = ar;
    }
#pragma unroll
    for (int q = 0; q < kWPer; ++q) {
      // Low nibbles: elements 16 h.. of sub-block 2j (byte 32 (p >> 1) +
      // 8 h); high nibbles: the same of sub-block 2j + 1 (16 bytes later).
      std::uint8_t* wd = lds + kOffW + buf * kWBytes + wrow[q] * kPitch +
                         32 * (wp[q] >> 1) + 8 * (wp[q] & 1);
      *reinterpret_cast<uint2*>(wd) = make_uint2(
          Nibbles<false>(wq[q].x, wq[q].y), Nibbles<false>(wq[q].z, wq[q].w));
      *reinterpret_cast<uint2*>(wd + 16) = make_uint2(
          Nibbles<true>(wq[q].x, wq[q].y), Nibbles<true>(wq[q].z, wq[q].w));
      const __half2 dm = __builtin_bit_cast(__half2, whdr[q].x);
      if (wp[q] == 0) {
        std::uint32_t word = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          int sc, mn;
          ScaleMin(whdr[q], 4 * (slab & 1) + g, sc, mn);
          word |= static_cast<std::uint32_t>(sc) << (8 * g);
        }
        s_sc[buf * kCh + wrow[q]] = word;
        s_d[buf * kCh + wrow[q]] = __low2float(dm);
      } else if (wp[q] == 1 && (slab & 1) == 0) {
        const float dmin = __high2float(dm);
        std::uint32_t h[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          int sc0, mn0, sc1, mn1;
          ScaleMin(whdr[q], 2 * p, sc0, mn0);
          ScaleMin(whdr[q], 2 * p + 1, sc1, mn1);
          h[p] = __builtin_bit_cast(
              std::uint32_t,
              __halves2half2(__float2half(-dmin * static_cast<float>(mn0)),
                             __float2half(-dmin * static_cast<float>(mn1))));
        }
        auto* wm = reinterpret_cast<uint4*>(
            s_wm + ((slab >> 2) & 1) * kCh * 16 + wrow[q] * 16);
        const int sb = (slab >> 1) & 1;
        wm[sb] = make_uint4(h[0], h[1], h[2], h[3]);
        if (sb == 0)
          wm[1] = make_uint4(0U, 0U, 0U, 0U);
      }
    }
  };

  // Activation groups of 64/128 share one multiplier across kAgg weight
  // groups: their sums gather in `ts` (1 mad per group) and enter `t` once
  // per multiplier (1 more mad), instead of a multiply per group (g32).
  constexpr int kAgg = kGroup >= 64 ? kGroup / 32 : 1;
  i32x8 acc[kTI][kTJ];
  i32x8 t[kTI][kTJ];
  i32x8 ts[kAgg > 1 ? kTI : 1][kAgg > 1 ? kTJ : 1];
  f32x8 y[kTI][kTJ];
#pragma unroll
  for (int i = 0; i < kTI; ++i)
#pragma unroll
    for (int j = 0; j < kTJ; ++j) {
      acc[i][j] = i32x8{};
      t[i][j] = i32x8{};
      y[i][j] = f32x8{};
    }
  const int a_row = wave_t * 16 * kTI + l16;
  const int b_row = wave_n * 16 * kTJ + l16;

  const auto compute = [&](int slab, int cur) {
    const std::uint8_t* lx = lds + cur * kXBytes;
    const std::uint8_t* lw = lds + kOffW + cur * kWBytes;
    // The min-term WMMA closes a window of 16 groups (4 slabs): fetch its
    // activation sums now, they are needed after this slab's matrix work.
    const bool window_end = (slab & 3) == 3 || slab == nslab - 1;
    f16x16 xsf[kTI];
    if (window_end) {
#pragma unroll
      for (int i = 0; i < kTI; ++i)
        __builtin_memcpy(&xsf[i],
                         xs + static_cast<std::size_t>(xs_src[i]) * xs_stride +
                             (slab >> 2) * 16,
                         32);
    }
    std::uint32_t scw[kTJ];
#pragma unroll
    for (int j = 0; j < kTJ; ++j)
      scw[j] = s_sc[cur * kCh + b_row + 16 * j];
    std::uint32_t aw[kTI][8];
    if constexpr (kGroup != 0) {
#pragma unroll
      for (int i = 0; i < kTI; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          aw[i][e] = s_a[cur * kBT + wave_t * 16 * kTI + 16 * i + 2 * e + half];
    }
    // g32: groups in the order 0, 2, 1, 3 so the packed products of a byte
    // pair (multiplier x scale, two groups per v_pk_mul_lo_u16) are live
    // for two consecutive groups only.
    std::uint32_t prod[kGroup == 32 ? kTI : 1][8][kGroup == 32 ? kTJ : 1];
#pragma unroll
    for (int gi = 0; gi < 4; ++gi) {
      const int g = kGroup == 32 ? (gi >> 1) + 2 * (gi & 1) : gi;
      if constexpr (kGroup == 32) {
        if ((gi & 1) == 0) {  // g = 0 then g = 1: pair (g, g + 2)
#pragma unroll
          for (int j = 0; j < kTJ; ++j) {
            const u16x2 s =
                __builtin_bit_cast(u16x2, gi == 0 ? BytePairs<false>(scw[j])
                                                  : BytePairs<true>(scw[j]));
#pragma unroll
            for (int i = 0; i < kTI; ++i)
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                const u16x2 a = __builtin_bit_cast(
                    u16x2, gi == 0 ? BytePairs<false>(aw[i][e])
                                   : BytePairs<true>(aw[i][e]));
                prod[i][e][j] = __builtin_bit_cast(std::uint32_t, a* s);
              }
          }
        }
      }
#pragma unroll
      for (int st = 0; st < 2; ++st) {
        const int ks = 2 * g + st;
        // Compiler barrier: keep one k-step's LDS fragments live at a time.
        asm volatile("" ::: "memory");
        i32x2 a[kTI], b[kTJ];
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              lx + (a_row + 16 * i) * kPitch + ks * 8);
          __builtin_memcpy(&a[i], &v, 8);
        }
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              lw + (b_row + 16 * j) * kPitch + ks * 8);
          __builtin_memcpy(&b[j], &v, 8);
        }
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
#pragma unroll
          for (int j = 0; j < kTJ; ++j)
            acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a[i], false, b[j], st == 0 ? i32x8{} : acc[i][j], false);
        }
      }
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const int sc = static_cast<int>((scw[j] >> (8 * g)) & 0xFFU);
#pragma unroll
        for (int i = 0; i < kTI; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if constexpr (kGroup == 32) {
              // |acc| <= 32 * 8 * 15 fits the i16 operand.
              t[i][j][e] =
                  g >= 2
                      ? MadI16<true>(prod[i][e][j], acc[i][j][e], t[i][j][e])
                      : MadI16<false>(prod[i][e][j], acc[i][j][e], t[i][j][e]);
            } else if constexpr (kAgg > 1) {
              ts[i][j][e] = g % kAgg == 0
                                ? __mul24(sc, acc[i][j][e])
                                : MadI24(sc, acc[i][j][e], ts[i][j][e]);
            } else {
              t[i][j][e] = MadI24(sc, acc[i][j][e], t[i][j][e]);
            }
          }
      }
      if constexpr (kAgg > 1) {
        if (g % kAgg == kAgg - 1) {
#pragma unroll
          for (int i = 0; i < kTI; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const int a_ge = static_cast<int>((aw[i][e] >> (8 * g)) & 0xFFU);
#pragma unroll
              for (int j = 0; j < kTJ; ++j)
                t[i][j][e] = MadI24(a_ge, ts[i][j][e], t[i][j][e]);
            }
        }
      }
    }
    // Fold the integer sums at the end of each super-block (the Q4_K d
    // changes there): y += (sx d) t.
    if ((slab & 1) != 0) {
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const float d = s_d[cur * kCh + b_row + 16 * j];
#pragma unroll
        for (int i = 0; i < kTI; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float s = s_sx[wave_t * 16 * kTI + 16 * i + 2 * e + half];
            y[i][j][e] =
                fmaf(s * d, static_cast<float>(t[i][j][e]), y[i][j][e]);
            t[i][j][e] = 0;
          }
      }
    }
    // Min term: y -= sum_g (dmin m_g) * Xs_g, sixteen groups per F16 WMMA.
    if (window_end) {
      const __half* wm = s_wm + ((slab >> 2) & 1) * kCh * 16;
      f16x16 bm[kTJ];
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const auto* p =
            reinterpret_cast<const uint4*>(wm + (b_row + 16 * j) * 16);
        const uint4 lo = p[0], hi = p[1];
        __builtin_memcpy(&bm[j], &lo, 16);
        __builtin_memcpy(reinterpret_cast<char*>(&bm[j]) + 16, &hi, 16);
      }
#pragma unroll
      for (int i = 0; i < kTI; ++i) {
        const f16x16 am = xsf[i];
#pragma unroll
        for (int j = 0; j < kTJ; ++j)
          y[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(am, bm[j], y[i][j]);
      }
    }
  };

  load_global(0);
  commit(0, 0);
  __syncthreads();
  for (int slab = 0; slab < nslab; ++slab) {
    const int cur = slab & 1;
    if (slab + 1 < nslab)
      load_global(slab + 1);
    // A wave whose 32 rows are all bucket padding skips the matrix work
    // (branches inside the WMMA chain would copy the accumulators).
    if (live_i[0])
      compute(slab, cur);
    if (slab + 1 < nslab)
      commit(slab + 1, cur ^ 1);
    __syncthreads();
  }

  // Epilogue: gate waves publish their tile, up waves apply SwiGLU with the
  // F16 route's evaluation order and store F16.
  auto* gbuf = reinterpret_cast<float*>(lds);
  if (wave_n < 2) {
#pragma unroll
    for (int i = 0; i < kTI; ++i)
#pragma unroll
      for (int j = 0; j < kTJ; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int tok = wave_t * 16 * kTI + 16 * i + 2 * e + half;
          gbuf[tok * kGs + wave_n * 16 * kTJ + 16 * j + l16] = y[i][j][e];
        }
  }
  __syncthreads();
  if (wave_n >= 2) {
#pragma unroll
    for (int i = 0; i < kTI; ++i) {
      if (!live_i[i])
        continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int tok = wave_t * 16 * kTI + 16 * i + 2 * e + half;
        const int t_row = tr.t_local + tok;
        if (t_row >= tr.bucket_rows)
          continue;
        const std::int32_t dst = rows_out[tr.bucket_begin + t_row];
        if (dst < 0)
          continue;
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const int ch = (wave_n - 2) * 16 * kTJ + 16 * j + l16;
          const float g = gbuf[tok * kGs + ch];
          float product = y[i][j][e] * g;
          asm volatile("" : "+v"(product));
          float value = product * SigmoidF(g);
          asm volatile("" : "+v"(value));
          out[static_cast<std::size_t>(dst) * m + r0 + ch] =
              __float2half(value);
        }
      }
    }
  }
}

// Down projection on a packed Q5_1 container with 4-bit codes: one F16 d and
// m per 32-block; the integer sums are folded per group. Block tile kBT =
// 32 kTI rows x 64 kTJ output channels.
template<int kTI, int kTJ, int kGroup>
__global__ __launch_bounds__(kThreads) void DownKernel(
    const std::uint8_t* __restrict__ w, const std::uint8_t* __restrict__ x4,
    const float* __restrict__ sx, const std::uint8_t* __restrict__ xa,
    const __half* __restrict__ xs, int xs_stride,
    const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_in,
    const std::int32_t* __restrict__ rows_out, __half* __restrict__ out, int m,
    int k, int tile_rows) {
  constexpr int kBT = 32 * kTI;
  constexpr int kCh = 64 * kTJ;
  constexpr int kXBytes = kBT * kPitch;
  constexpr int kWBytes = kCh * kPitch;
  constexpr int kOffW = 2 * kXBytes;
  constexpr int kOffA = kOffW + 2 * kWBytes;
  constexpr int kOffD = kOffA + 2 * kBT * 4;
  constexpr int kOffWm = kOffD + 2 * kCh * 4 * 4;
  constexpr int kOffSx = kOffWm + 2 * kCh * 16 * 2;
  constexpr int kLds = kOffSx + kBT * 4;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLds];
  auto* s_a = reinterpret_cast<std::uint32_t*>(lds + kOffA);
  auto* s_d = reinterpret_cast<float*>(lds + kOffD);  // [2][kCh][4]
  auto* s_wm = reinterpret_cast<__half*>(lds + kOffWm);
  auto* s_sx = reinterpret_cast<float*>(lds + kOffSx);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5, lane = tid & 31;
  const int wave_t = wave >> 2, wave_n = wave & 3;
  const int half = lane >> 4, l16 = lane & 15;
  const int zs = tile_rows / kBT;
  const Rows tr = TileRows(tiles, pad_bounds, tile_rows, kBT,
                           static_cast<int>(blockIdx.x) % zs);
  if (tr.t_local >= tr.bucket_rows)
    return;  // block-uniform: the second half of a short 128-row tile
  const int expert = tiles[blockIdx.y] & 0xFFFF;
  const int r0 = static_cast<int>(blockIdx.x) / zs * kCh;
  const int nslab = k >> 7;
  const std::size_t row_bytes = static_cast<std::size_t>(k >> 5) * 24;
  const std::size_t xrow = static_cast<std::size_t>(k) >> 1;

  constexpr int kXPer = kBT * 4 / kThreads;
  const std::uint8_t* xsrc[kXPer];
  int xdst[kXPer];
#pragma unroll
  for (int p = 0; p < kXPer; ++p) {
    const int c = tid + p * kThreads;
    const int row = c >> 2, part = c & 3;
    const int cr = tr.t_local + row;
    const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
    xsrc[p] = src >= 0 ? x4 + static_cast<std::size_t>(src) * xrow + part * 16
                       : nullptr;
    xdst[p] = row * kPitch + part * 16;
  }
  const std::uint8_t* asrc = nullptr;
  if constexpr (kGroup != 0) {
    if (tid < kBT) {
      const int cr = tr.t_local + tid;
      const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
      asrc = src >= 0 ? xa + static_cast<std::size_t>(src) * (k >> 5) : nullptr;
    }
  }
  // Weight staging: job c = (block b = c / kCh of the slab, row c % kCh);
  // block b is piece b & 1 (qs0 / qs1) of step 2 slab + (b >> 1), its d | m
  // dword at byte 8 (b & 1) of the step's third piece.
  constexpr int kWPer = kCh * 4 / kThreads;
  const std::uint8_t* wbase[kWPer];
  int wrow[kWPer], wb[kWPer];
#pragma unroll
  for (int q = 0; q < kWPer; ++q) {
    const int c = tid + q * kThreads;
    wrow[q] = c % kCh;
    wb[q] = c / kCh;
    const int rr = r0 + wrow[q];
    wbase[q] = w + static_cast<std::size_t>(expert) * m * row_bytes +
               static_cast<std::size_t>(rr & ~31) * row_bytes + (rr & 31) * 16;
  }

  if (tid < kBT) {
    const int cr = tr.t_local + tid;
    const int src = cr < tr.bucket_rows ? rows_in[tr.bucket_begin + cr] : -1;
    s_sx[tid] = src >= 0 ? sx[src] : 0.0F;
  }
  int xs_src[kTI];
  bool live_i[kTI];
#pragma unroll
  for (int i = 0; i < kTI; ++i) {
    const int t0 = tr.t_local + wave_t * 16 * kTI + 16 * i;
    live_i[i] = t0 < tr.bucket_rows;
    const int t = t0 + l16;
    const int src = t < tr.bucket_rows ? rows_in[tr.bucket_begin + t] : -1;
    xs_src[i] = src >= 0 ? src : 0;
  }

  uint4 xr[kXPer];
  uint4 wq[kWPer];
  std::uint32_t wdm[kWPer];
  std::uint32_t ar = 0;

  const auto load_global = [&](int slab) {
#pragma unroll
    for (int q = 0; q < kWPer; ++q) {
      const std::uint8_t* step =
          wbase[q] +
          static_cast<std::size_t>(2 * slab + (wb[q] >> 1)) * kQ51Step;
      wq[q] = *reinterpret_cast<const uint4*>(step + 512 * (wb[q] & 1));
      wdm[q] = *reinterpret_cast<const std::uint32_t*>(step + 1024 +
                                                       8 * (wb[q] & 1));
    }
#pragma unroll
    for (int p = 0; p < kXPer; ++p)
      xr[p] = xsrc[p] != nullptr
                  ? *reinterpret_cast<const uint4*>(xsrc[p] + slab * 64)
                  : make_uint4(0U, 0U, 0U, 0U);
    if constexpr (kGroup != 0) {
      if (tid < kBT)
        ar = asrc != nullptr
                 ? *reinterpret_cast<const std::uint32_t*>(asrc + slab * 4)
                 : 0U;
    }
  };
  const auto commit = [&](int slab, int buf) {
    std::uint8_t* lx = lds + buf * kXBytes;
#pragma unroll
    for (int p = 0; p < kXPer; ++p)
      StoreRow16(lx + xdst[p], xr[p].x, xr[p].y, xr[p].z, xr[p].w);
    if constexpr (kGroup != 0) {
      if (tid < kBT)
        s_a[buf * kBT + tid] = ar;
    }
    const int q4 = slab & 3;
#pragma unroll
    for (int q = 0; q < kWPer; ++q) {
      // Elements 0-15 of the block are the low nibbles, 16-31 the high.
      StoreRow16(
          lds + kOffW + buf * kWBytes + wrow[q] * kPitch + 16 * wb[q],
          Nibbles<false>(wq[q].x, wq[q].y), Nibbles<false>(wq[q].z, wq[q].w),
          Nibbles<true>(wq[q].x, wq[q].y), Nibbles<true>(wq[q].z, wq[q].w));
      s_d[(buf * kCh + wrow[q]) * 4 + wb[q]] =
          __low2float(__builtin_bit_cast(__half2, wdm[q]));
      // The window's m: this job owns halves 4 q' + b of its row.
      auto* wm = reinterpret_cast<std::uint16_t*>(
          s_wm + ((slab >> 2) & 1) * kCh * 16 + wrow[q] * 16);
      wm[4 * q4 + wb[q]] = static_cast<std::uint16_t>(wdm[q] >> 16U);
      if (q4 == 0) {
#pragma unroll
        for (int qq = 1; qq < 4; ++qq)
          wm[4 * qq + wb[q]] = 0;
      }
    }
  };

  i32x8 acc[kTI][kTJ];
  f32x8 y[kTI][kTJ];
#pragma unroll
  for (int i = 0; i < kTI; ++i)
#pragma unroll
    for (int j = 0; j < kTJ; ++j) {
      acc[i][j] = i32x8{};
      y[i][j] = f32x8{};
    }
  const int a_row = wave_t * 16 * kTI + l16;
  const int b_row = wave_n * 16 * kTJ + l16;

  const auto compute = [&](int slab, int cur) {
    const std::uint8_t* lx = lds + cur * kXBytes;
    const std::uint8_t* lw = lds + kOffW + cur * kWBytes;
    const bool window_end = (slab & 3) == 3 || slab == nslab - 1;
    f16x16 xsf[kTI];
    if (window_end) {
#pragma unroll
      for (int i = 0; i < kTI; ++i)
        __builtin_memcpy(&xsf[i],
                         xs + static_cast<std::size_t>(xs_src[i]) * xs_stride +
                             (slab >> 2) * 16,
                         32);
    }
    float sa[kTI][8];
    std::uint32_t aw[kTI][8];
#pragma unroll
    for (int i = 0; i < kTI; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        sa[i][e] = s_sx[wave_t * 16 * kTI + 16 * i + 2 * e + half];
        if constexpr (kGroup != 0) {
          aw[i][e] = s_a[cur * kBT + wave_t * 16 * kTI + 16 * i + 2 * e + half];
          if constexpr (kGroup == 128)
            sa[i][e] *= static_cast<float>(aw[i][e] & 0xFFU);
        }
      }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
      for (int st = 0; st < 2; ++st) {
        const int ks = 2 * g + st;
        asm volatile("" ::: "memory");
        i32x2 a[kTI], b[kTJ];
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              lx + (a_row + 16 * i) * kPitch + ks * 8);
          __builtin_memcpy(&a[i], &v, 8);
        }
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              lw + (b_row + 16 * j) * kPitch + ks * 8);
          __builtin_memcpy(&b[j], &v, 8);
        }
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
#pragma unroll
          for (int j = 0; j < kTJ; ++j)
            acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a[i], false, b[j], st == 0 ? i32x8{} : acc[i][j], false);
        }
      }
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const float d = s_d[(cur * kCh + b_row + 16 * j) * 4 + g];
#pragma unroll
        for (int i = 0; i < kTI; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float s = sa[i][e];
            if constexpr (kGroup == 32 || kGroup == 64)
              s *= static_cast<float>((aw[i][e] >> (8 * g)) & 0xFFU);
            y[i][j][e] =
                fmaf(s * d, static_cast<float>(acc[i][j][e]), y[i][j][e]);
          }
      }
    }
    if (window_end) {
      const __half* wm = s_wm + ((slab >> 2) & 1) * kCh * 16;
      f16x16 bm[kTJ];
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const auto* p =
            reinterpret_cast<const uint4*>(wm + (b_row + 16 * j) * 16);
        const uint4 lo = p[0], hi = p[1];
        __builtin_memcpy(&bm[j], &lo, 16);
        __builtin_memcpy(reinterpret_cast<char*>(&bm[j]) + 16, &hi, 16);
      }
#pragma unroll
      for (int i = 0; i < kTI; ++i) {
        const f16x16 am = xsf[i];
#pragma unroll
        for (int j = 0; j < kTJ; ++j)
          y[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(am, bm[j], y[i][j]);
      }
    }
  };

  load_global(0);
  commit(0, 0);
  __syncthreads();
  for (int slab = 0; slab < nslab; ++slab) {
    const int cur = slab & 1;
    if (slab + 1 < nslab)
      load_global(slab + 1);
    if (live_i[0])
      compute(slab, cur);
    if (slab + 1 < nslab)
      commit(slab + 1, cur ^ 1);
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kTI; ++i) {
    if (!live_i[i])
      continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int t_row = tr.t_local + wave_t * 16 * kTI + 16 * i + 2 * e + half;
      if (t_row >= tr.bucket_rows)
        continue;
      const std::int32_t dst = rows_out[tr.bucket_begin + t_row];
      if (dst < 0)
        continue;
#pragma unroll
      for (int j = 0; j < kTJ; ++j)
        out[static_cast<std::size_t>(dst) * m + r0 + b_row + 16 * j] =
            __float2half(y[i][j][e]);
    }
  }
}

// ---------------------------------------------------------------------------
// Q5_1 -> 4-bit asymmetric requantization of one 32-block in registers.
__device__ void Requant32(const float* v, float& d_out, float& m_out,
                          int* codes) {
  float lo = v[0], hi = v[0];
#pragma unroll
  for (int i = 1; i < 32; ++i) {
    lo = fminf(lo, v[i]);
    hi = fmaxf(hi, v[i]);
  }
  float best = 3.0e38F, best_d = 0.0F, best_m = lo;
  const auto eval = [&](float d, float m) {
    const float dh = __half2float(__float2half(d));
    const float mh = __half2float(__float2half(m));
    const float inv = dh > 0.0F ? 1.0F / dh : 0.0F;
    float err = 0.0F;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float q =
          fminf(fmaxf(__builtin_rintf((v[i] - mh) * inv), 0.0F), 15.0F);
      const float r = v[i] - (mh + dh * q);
      err += r * r;
    }
    if (err < best) {
      best = err;
      best_d = dh;
      best_m = mh;
    }
  };
  const float ctr = 0.5F * (lo + hi), span = 0.5F * (hi - lo);
  constexpr float kShrink[6] = {1.0F, 0.97F, 0.94F, 0.91F, 0.88F, 0.85F};
#pragma unroll
  for (int c = 0; c < 6; ++c)
    eval(2.0F * span * kShrink[c] / 15.0F, ctr - span * kShrink[c]);
  // Least-squares refit of (d, m) for the best codes, twice.
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const float inv = best_d > 0.0F ? 1.0F / best_d : 0.0F;
    float sq = 0.0F, sqq = 0.0F, sv = 0.0F, sqv = 0.0F;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const float q =
          fminf(fmaxf(__builtin_rintf((v[i] - best_m) * inv), 0.0F), 15.0F);
      sq += q;
      sqq += q * q;
      sv += v[i];
      sqv += q * v[i];
    }
    const float den = 32.0F * sqq - sq * sq;
    if (den > 0.0F) {
      const float d = (32.0F * sqv - sq * sv) / den;
      const float m = (sv - d * sq) / 32.0F;
      if (d > 0.0F)
        eval(d, m);
    }
  }
  const float inv = best_d > 0.0F ? 1.0F / best_d : 0.0F;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    codes[i] = static_cast<int>(
        fminf(fmaxf(__builtin_rintf((v[i] - best_m) * inv), 0.0F), 15.0F));
  d_out = best_d;
  m_out = best_m;
}

__global__ __launch_bounds__(kThreads) void RequantQ51Kernel(
    std::uint8_t* __restrict__ w, std::size_t rows, int k) {
  const int steps = k >> 6;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * steps)
    return;
  const std::size_t r = idx / steps;
  const int s = static_cast<int>(idx % steps);
  const std::size_t row_bytes = static_cast<std::size_t>(k >> 5) * 24;
  std::uint8_t* step = w + (r & ~std::size_t{31}) * row_bytes + (r & 31) * 16 +
                       static_cast<std::size_t>(s) * kQ51Step;
  uint4 qs[2] = {*reinterpret_cast<const uint4*>(step),
                 *reinterpret_cast<const uint4*>(step + 512)};
  uint4 dmh = *reinterpret_cast<const uint4*>(step + 1024);
  std::uint32_t dm_new[2];
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
    const std::uint32_t dm = blk == 0 ? dmh.x : dmh.z;
    const std::uint32_t qh = blk == 0 ? dmh.y : dmh.w;
    const __half2 h = __builtin_bit_cast(__half2, dm);
    const float d = __low2float(h), m = __high2float(h);
    const std::uint32_t words[4] = {qs[blk].x, qs[blk].y, qs[blk].z, qs[blk].w};
    float v[32];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const std::uint32_t byte = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
      const std::uint32_t q0 = (byte & 0xFU) | (((qh >> i) & 1U) << 4U);
      const std::uint32_t q1 = (byte >> 4U) | (((qh >> (i + 16)) & 1U) << 4U);
      v[i] = d * static_cast<float>(q0) + m;
      v[i + 16] = d * static_cast<float>(q1) + m;
    }
    float nd, nm;
    int codes[32];
    Requant32(v, nd, nm, codes);
    std::uint32_t out[4] = {0U, 0U, 0U, 0U};
#pragma unroll
    for (int i = 0; i < 16; ++i)
      out[i >> 2] |= static_cast<std::uint32_t>(codes[i] | (codes[i + 16] << 4))
                     << (8 * (i & 3));
    qs[blk] = make_uint4(out[0], out[1], out[2], out[3]);
    dm_new[blk] = __builtin_bit_cast(
        std::uint32_t, __halves2half2(__float2half(nd), __float2half(nm)));
  }
  *reinterpret_cast<uint4*>(step) = qs[0];
  *reinterpret_cast<uint4*>(step + 512) = qs[1];
  *reinterpret_cast<uint4*>(step + 1024) =
      make_uint4(dm_new[0], 0U, dm_new[1], 0U);
}

std::size_t Align(std::size_t v) {
  return (v + 255) / 256 * 256;
}

}  // namespace

std::size_t XsStride(std::size_t k) {
  return (k / 32 + 15) / 16 * 16;
}

std::size_t QuantBytes(std::size_t rows, std::size_t k) {
  return Align(rows * (k / 2)) + Align(rows * sizeof(float)) +
         Align(rows * (k / 32)) + Align(rows * XsStride(k) * sizeof(__half));
}

QuantRows CarveQuant(void* base, std::size_t rows, std::size_t k) {
  auto* p = static_cast<std::uint8_t*>(base);
  QuantRows q;
  q.k = k;
  q.x4 = p;
  p += Align(rows * (k / 2));
  q.sx = reinterpret_cast<float*>(p);
  p += Align(rows * sizeof(float));
  q.a = p;
  p += Align(rows * (k / 32));
  q.xs = reinterpret_cast<__half*>(p);
  return q;
}

namespace {

template<int kGroup>
void LaunchQuantize(const __half* x, int rows, int k, int had,
                    const QuantRows& q, __half* xhat, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((rows + 7) / 8));
  const int st = static_cast<int>(XsStride(static_cast<std::size_t>(k)));
  if (had == 128)
    hipLaunchKernelGGL((QuantizeKernel<kGroup, true>), grid, dim3(kThreads), 0,
                       stream, x, rows, k, st, q.x4, q.sx, q.a, q.xs, xhat);
  else
    hipLaunchKernelGGL((QuantizeKernel<kGroup, false>), grid, dim3(kThreads), 0,
                       stream, x, rows, k, st, q.x4, q.sx, q.a, q.xs, xhat);
}

}  // namespace

bool Quantize(const __half* x, std::size_t rows, std::size_t k, int group,
              int had, const QuantRows& q, __half* xhat, hipStream_t stream) {
  if (rows == 0)
    return true;
  if (k == 0 || k % 128 != 0 || k > 8 * 32 * kMaxChunkIt || q.k != k ||
      (had != 0 && had != 128))
    return false;
  const int r = static_cast<int>(rows), kk = static_cast<int>(k);
  switch (group) {
    case 0:
      LaunchQuantize<0>(x, r, kk, had, q, xhat, stream);
      return true;
    case 32:
      LaunchQuantize<32>(x, r, kk, had, q, xhat, stream);
      return true;
    case 64:
      LaunchQuantize<64>(x, r, kk, had, q, xhat, stream);
      return true;
    case 128:
      LaunchQuantize<128>(x, r, kk, had, q, xhat, stream);
      return true;
    default:
      return false;
  }
}

namespace {

// Two block shapes: kTile64 = 64 rows x 128 channels (kTI 2, kTJ 2; a
// 128-row map entry runs as two consecutive 64-row blocks) and kTile128 =
// 128 rows x 64 channels (kTI 4, kTJ 1; 128-row maps only).
template<int kTI, int kTJ, int kGroup>
void LaunchGateUp(const void* gate, const void* up, const QuantRows& q,
                  const std::int32_t* tiles, std::uint32_t n_tiles,
                  std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                  const std::int32_t* rows_in, const std::int32_t* rows_out,
                  __half* out, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  hipLaunchKernelGGL(
      (GateUpKernel<kTI, kTJ, kGroup>),
      dim3(static_cast<unsigned>(m / (32 * kTJ) * (tile_rows / (32 * kTI))),
           n_tiles),
      dim3(kThreads), 0, stream, static_cast<const std::uint8_t*>(gate),
      static_cast<const std::uint8_t*>(up), q.x4, q.sx, q.a, q.xs,
      static_cast<int>(XsStride(k)), tiles, pad_bounds, rows_in, rows_out, out,
      static_cast<int>(m), static_cast<int>(k), static_cast<int>(tile_rows));
}

template<int kTI, int kTJ, int kGroup>
void LaunchDown(const void* w, const QuantRows& q, const std::int32_t* tiles,
                std::uint32_t n_tiles, std::uint32_t tile_rows,
                const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                const std::int32_t* rows_out, __half* out, std::size_t m,
                std::size_t k, hipStream_t stream) {
  hipLaunchKernelGGL(
      (DownKernel<kTI, kTJ, kGroup>),
      dim3(static_cast<unsigned>(m / (64 * kTJ) * (tile_rows / (32 * kTI))),
           n_tiles),
      dim3(kThreads), 0, stream, static_cast<const std::uint8_t*>(w), q.x4,
      q.sx, q.a, q.xs, static_cast<int>(XsStride(k)), tiles, pad_bounds,
      rows_in, rows_out, out, static_cast<int>(m), static_cast<int>(k),
      static_cast<int>(tile_rows));
}

/// Calls f.template operator()<kTI, kTJ, kGroup>() for a runtime shape and
/// group; false when unsupported.
template<class F>
bool Dispatch(int shape, int group, const F& f) {
  const auto by_group = [&]<int kTI, int kTJ>() {
    switch (group) {
      case 0:
        f.template operator()<kTI, kTJ, 0>();
        return true;
      case 32:
        f.template operator()<kTI, kTJ, 32>();
        return true;
      case 64:
        f.template operator()<kTI, kTJ, 64>();
        return true;
      case 128:
        f.template operator()<kTI, kTJ, 128>();
        return true;
      default:
        return false;
    }
  };
  switch (shape) {
    case kTile64:
      return by_group.template operator()<2, 2>();
    case kTile128:
      return by_group.template operator()<4, 1>();
    default:
      return false;
  }
}

bool ShapeFits(int shape, std::uint32_t tile_rows) {
  return (tile_rows == 64 || tile_rows == 128) &&
         (shape == kTile64 || (shape == kTile128 && tile_rows == 128));
}

}  // namespace

bool GateUp(const void* gate, const void* up, const QuantRows& q, int group,
            int shape, const std::int32_t* tiles, std::uint32_t n_tiles,
            std::uint32_t tile_rows, const std::int32_t* pad_bounds,
            const std::int32_t* rows_in, const std::int32_t* rows_out,
            __half* out, std::size_t m, std::size_t k, hipStream_t stream) {
  if (n_tiles == 0)
    return true;
  if (m == 0 || m % 64 != 0 || k == 0 || k % 256 != 0 || q.k != k ||
      !ShapeFits(shape, tile_rows))
    return false;
  return Dispatch(shape, group, [&]<int kTI, int kTJ, int kGroup>() {
    LaunchGateUp<kTI, kTJ, kGroup>(gate, up, q, tiles, n_tiles, tile_rows,
                                   pad_bounds, rows_in, rows_out, out, m, k,
                                   stream);
  });
}

bool Down(const void* w, const QuantRows& q, int group, int shape,
          const std::int32_t* tiles, std::uint32_t n_tiles,
          std::uint32_t tile_rows, const std::int32_t* pad_bounds,
          const std::int32_t* rows_in, const std::int32_t* rows_out,
          __half* out, std::size_t m, std::size_t k, hipStream_t stream) {
  if (n_tiles == 0)
    return true;
  if (m == 0 || m % 128 != 0 || k == 0 || k % 128 != 0 || q.k != k ||
      !ShapeFits(shape, tile_rows))
    return false;
  return Dispatch(shape, group, [&]<int kTI, int kTJ, int kGroup>() {
    LaunchDown<kTI, kTJ, kGroup>(w, q, tiles, n_tiles, tile_rows, pad_bounds,
                                 rows_in, rows_out, out, m, k, stream);
  });
}

bool RequantDownQ4(void* w, std::size_t rows, std::size_t k,
                   hipStream_t stream) {
  if (rows == 0 || rows % 32 != 0 || k == 0 || k % 64 != 0)
    return false;
  const std::size_t threads = rows * (k / 64);
  hipLaunchKernelGGL(
      RequantQ51Kernel,
      dim3(static_cast<unsigned>((threads + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, static_cast<std::uint8_t*>(w), rows,
      static_cast<int>(k));
  return true;
}

// ---------------------------------------------------------------------------
// Engine hook.

namespace {

/// Per-thread device scratch, grown on demand (prompt processing is never
/// graph-captured).
struct Scratch {
  void* p{nullptr};
  std::size_t bytes{0};
};

void* Grow(Scratch& s, std::size_t bytes) {
  if (bytes > s.bytes) {
    if (s.p != nullptr)
      (void)hipFree(s.p);
    s.p = nullptr;
    s.bytes = 0;
    if (hipMalloc(&s.p, bytes) != hipSuccess)
      return nullptr;
    s.bytes = bytes;
  }
  return s.p;
}

thread_local Scratch g_qx, g_qu, g_hat, g_ref, g_sums;

/// The 64/128-row map of the last gate/up call, for its down call.
struct Map {
  const std::int32_t* tiles{nullptr};
  const std::int32_t* pad_bounds{nullptr};
  std::uint32_t n_tiles{0}, tile_rows{0};
  bool valid{false};
};
thread_local Map g_map;
thread_local int g_checked[2] = {0, 0};

/// Matrices re-quantized by PrepareDown (Init, before any prefill).
std::set<const void*>& Down4() {
  static std::set<const void*> s;
  return s;
}

/// Upper bound on routed slots per token for the gate/up check buffer.
constexpr std::size_t kMaxUsed = 16;

/// sums[0] += sum (out - ref)^2, sums[1] += sum ref^2 over every row the
/// tile map writes (rows_out >= 0, below max_rows).
__global__ __launch_bounds__(kThreads) void CompareKernel(
    const __half* __restrict__ out, const __half* __restrict__ ref,
    const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_out, int tile_rows, int m,
    std::size_t max_rows, double* __restrict__ sums) {
  __shared__ float red[2][kThreads / 32];
  const std::int32_t tile = tiles[blockIdx.x];
  const int expert = tile & 0xFFFF;
  const int begin = pad_bounds[expert];
  const int rows = pad_bounds[expert + 1] - begin;
  const int t0 = (tile >> 16) * tile_rows;
  float num = 0.0F, den = 0.0F;
  for (int r = 0; r < tile_rows && t0 + r < rows; ++r) {
    const std::int32_t dst = rows_out[begin + t0 + r];
    if (dst < 0 || static_cast<std::size_t>(dst) >= max_rows)
      continue;
    for (int c = static_cast<int>(threadIdx.x); c < m; c += kThreads) {
      const std::size_t o = static_cast<std::size_t>(dst) * m + c;
      const float a = __half2float(out[o]), b = __half2float(ref[o]);
      num += (a - b) * (a - b);
      den += b * b;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    num += __shfl_xor(num, o);
    den += __shfl_xor(den, o);
  }
  if ((threadIdx.x & 31) == 0) {
    red[0][threadIdx.x >> 5] = num;
    red[1][threadIdx.x >> 5] = den;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double n = 0.0, d = 0.0;
    for (int w = 0; w < kThreads / 32; ++w) {
      n += red[0][w];
      d += red[1][w];
    }
    atomicAdd(&sums[0], n);
    atomicAdd(&sums[1], d);
  }
}

/// Logs the relative RMS of `out` against the emulation `ref` (synchronizes).
void Report(const char* what, int call, std::size_t x_rows, int group,
            const __half* out, const __half* ref, const std::int32_t* tiles,
            std::uint32_t n_tiles, std::uint32_t tile_rows,
            const std::int32_t* pad_bounds, const std::int32_t* rows_out,
            std::size_t m, std::size_t max_rows, hipStream_t stream) {
  auto* sums = static_cast<double*>(Grow(g_sums, 2 * sizeof(double)));
  if (sums == nullptr)
    return;
  (void)hipMemsetAsync(sums, 0, 2 * sizeof(double), stream);
  hipLaunchKernelGGL(CompareKernel, dim3(n_tiles), dim3(kThreads), 0, stream,
                     out, ref, tiles, pad_bounds, rows_out,
                     static_cast<int>(tile_rows), static_cast<int>(m), max_rows,
                     sums);
  double host[2] = {0.0, 0.0};
  if (hipMemcpyAsync(host, sums, sizeof(host), hipMemcpyDeviceToHost, stream) !=
          hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess)
    return;
  const double rel = host[1] > 0.0 ? std::sqrt(host[0] / host[1]) : 0.0;
  std::fprintf(stderr,
               "routed-iu4: check %s call %d rows %zu group %d: rel RMS vs "
               "emulation %.3e (%s)\n",
               what, call, x_rows, group, rel, rel <= 5e-3 ? "ok" : "FAIL");
}

bool GroupOk(int group) {
  return group == 0 || group == 32 || group == 64 || group == 128;
}

}  // namespace

bool PrepareDown(void* w, WeightType type, std::size_t rows, std::size_t cols,
                 hipStream_t stream) {
  if (type != WeightType::kQ5_1 || !RequantDownQ4(w, rows, cols, stream))
    return false;
  Down4().insert(w);
  return true;
}

bool HookGateUp(std::size_t x_rows, const void* gate, const void* up,
                WeightType type, const __half* x, const std::int32_t* tiles,
                std::uint32_t n_tiles, std::uint32_t tile_rows,
                const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                const std::int32_t* rows_out, __half* out, std::size_t m,
                std::size_t k, const HookOptions& options, hipStream_t stream) {
  // This call's map serves the down call of the same layer (even when the
  // gate/up itself stays on the F16 route).
  g_map = Map{tiles, pad_bounds, n_tiles, tile_rows,
              tile_rows == 64 || tile_rows == 128};
  if (type != WeightType::kQ4_K || !g_map.valid || !GroupOk(options.group) ||
      x_rows == 0 || k % 256 != 0 || k > 2560 || m % 64 != 0)
    return false;
  void* qbuf = Grow(g_qx, QuantBytes(x_rows, k));
  if (qbuf == nullptr)
    return false;
  const QuantRows q = CarveQuant(qbuf, x_rows, k);
  if (!Quantize(x, x_rows, k, options.group, options.had, q, nullptr, stream) ||
      !GateUp(gate, up, q, options.group,
              tile_rows == 128 ? options.shape : kTile64, tiles, n_tiles,
              tile_rows, pad_bounds, rows_in, rows_out, out, m, k, stream))
    return false;
  if (g_checked[0] < options.check_calls) {
    const int call = g_checked[0]++;
    auto* hat = static_cast<__half*>(Grow(g_hat, x_rows * k * sizeof(__half)));
    const std::size_t max_rows = x_rows * kMaxUsed;
    auto* ref =
        static_cast<__half*>(Grow(g_ref, max_rows * m * sizeof(__half)));
    if (hat != nullptr && ref != nullptr &&
        Quantize(x, x_rows, k, options.group, options.had, q, hat, stream) &&
        rocm::RoutedGatedF16Gemm(gate, up, type, hat, tiles, n_tiles, tile_rows,
                                 pad_bounds, rows_in, rows_out, ref, m, k,
                                 stream))
      Report("gate/up", call, x_rows, options.group, out, ref, tiles, n_tiles,
             tile_rows, pad_bounds, rows_out, m, max_rows, stream);
  }
  return true;
}

bool HookDown(std::size_t x_rows, const void* w, WeightType type,
              const __half* x, const std::int32_t* tiles, std::uint32_t n_tiles,
              std::uint32_t tile_rows, const std::int32_t* pad_bounds,
              const std::int32_t* rows_in, const std::int32_t* rows_out,
              const float* swiglu_gate, float* out, __half* out_half,
              std::size_t m, std::size_t k, const HookOptions& options,
              hipStream_t stream) {
  Map map{tiles, pad_bounds, n_tiles, tile_rows, true};
  if (tile_rows != 64) {
    if (!g_map.valid || g_map.pad_bounds != pad_bounds)
      return false;
    map = g_map;
  }
  g_map.valid = false;  // one down call per gate/up call
  if (type != WeightType::kQ5_1 || Down4().count(w) == 0 ||
      out_half == nullptr || out != nullptr || swiglu_gate != nullptr ||
      !GroupOk(options.group) || x_rows == 0 || k % 128 != 0 || k > 2560 ||
      m % 128 != 0)
    return false;
  void* qbuf = Grow(g_qu, QuantBytes(x_rows, k));
  if (qbuf == nullptr)
    return false;
  const QuantRows q = CarveQuant(qbuf, x_rows, k);
  if (!Quantize(x, x_rows, k, options.group, options.had, q, nullptr, stream) ||
      !Down(w, q, options.group, map.tile_rows == 128 ? options.shape : kTile64,
            map.tiles, map.n_tiles, map.tile_rows, map.pad_bounds, rows_in,
            rows_out, out_half, m, k, stream))
    return false;
  if (g_checked[1] < options.check_calls) {
    const int call = g_checked[1]++;
    auto* hat = static_cast<__half*>(Grow(g_hat, x_rows * k * sizeof(__half)));
    auto* ref = static_cast<__half*>(Grow(g_ref, x_rows * m * sizeof(__half)));
    // The emulation runs on the caller's own map (the F16 kernel's tiles).
    if (hat != nullptr && ref != nullptr &&
        Quantize(x, x_rows, k, options.group, options.had, q, hat, stream) &&
        rocm::RoutedF16Gemm(w, type, hat, tiles, n_tiles, tile_rows, pad_bounds,
                            rows_in, rows_out, nullptr, nullptr, ref, m, k,
                            stream))
      Report("down", call, x_rows, options.group, out_half, ref, tiles, n_tiles,
             tile_rows, pad_bounds, rows_out, m, x_rows, stream);
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::routed_iu4
