// Register-blocked routed-expert prefill GEMMs (see routed_prefill.hpp).
#include "src/models/qwen38_flash_next/kernels/rocm/routed_prefill.hpp"

#include <hip/hip_fp16.h>

namespace gufo::models::qwen38_flash_next::rocm::routed_prefill {
namespace {

constexpr int kThreads = 256;  // eight waves
constexpr int kWaves = kThreads / 32;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// The F16 route's helpers (kernels.hip.cpp), unchanged so the decoded
// halves and the SwiGLU match it bit for bit.
__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// ---------------------------------------------------------------------------
// iu4 gate/up.

using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;

/// Nibbles of codes c[base + 0..7] in WMMA K order [0,4,1,5,2,6,3,7]: the
/// order the Q4_K repack below produces from eight weight bytes.
__device__ __forceinline__ std::uint32_t PackNibbles(const int* c) {
  std::uint32_t w = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q)
    w |= ((static_cast<std::uint32_t>(c[q]) & 0xFU) << (8 * q)) |
         ((static_cast<std::uint32_t>(c[4 + q]) & 0xFU) << (8 * q + 4));
  return w;
}

/// One wave per token row; lane l takes 32-groups l, l + 32, ...; the
/// G / 32 lanes of an activation group share its maximum.
template<int G>
__launch_bounds__(kThreads) __global__
    void QuantizeA4Kernel(const __half* __restrict__ x, int rows, int k,
                          std::uint8_t* __restrict__ codes,
                          float* __restrict__ scale,
                          __half* __restrict__ sums) {
  constexpr int kLanes = G / 32;
  const int row = static_cast<int>(blockIdx.x) * kWaves +
                  static_cast<int>(threadIdx.x >> 5);
  if (row >= rows)
    return;  // wave-uniform
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int groups = k / 32;
  const __half* xr = x + static_cast<std::size_t>(row) * k;
  for (int g0 = 0; g0 < groups; g0 += 32) {
    const int g = g0 + lane;
    const bool active = g < groups;
    float v[32];
    float amax = 0.0F;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 raw = active ? reinterpret_cast<const uint4*>(xr + 32 * g)[c]
                               : make_uint4(0U, 0U, 0U, 0U);
      const std::uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const __half2 h2 = __builtin_bit_cast(__half2, words[i]);
        v[8 * c + 2 * i] = __low2float(h2);
        v[8 * c + 2 * i + 1] = __high2float(h2);
      }
    }
#pragma unroll
    for (int i = 0; i < 32; ++i)
      amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
    for (int msk = 1; msk < kLanes; msk <<= 1)
      amax = fmaxf(amax, __shfl_xor(amax, msk, 32));
    const float s = amax / 7.0F;
    const float inv = amax > 0.0F ? 7.0F / amax : 0.0F;
    int c[32];
    int csum = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      c[i] = min(7, max(-7, static_cast<int>(rintf(v[i] * inv))));
      csum += c[i];
    }
    if (active) {
      const uint4 packed = make_uint4(PackNibbles(c), PackNibbles(c + 8),
                                      PackNibbles(c + 16), PackNibbles(c + 24));
      *reinterpret_cast<uint4*>(
          codes + static_cast<std::size_t>(row) * (k / 2) + 16 * g) = packed;
      if (sums != nullptr)
        sums[static_cast<std::size_t>(row) * groups + g] =
            __float2half(s * static_cast<float>(csum));
      if (lane % kLanes == 0)
        scale[static_cast<std::size_t>(row) * (k / G) + g / kLanes] = s;
    }
  }
}

/// One block = eight waves = 128 gate/up output channels (wave w: gate rows
/// f0 + 16 w and the same up rows, the WMMA B operand, lane = row) x 64
/// tokens (four 16-token tiles, the A operand). K advances 128 per stage
/// (four 32-groups; a Q4_K super-block every two stages), single buffered
/// with the next stage prefetched into registers. kA4Out: the SwiGLU rows
/// leave as int4 down-projection activations (`hq`, one group of 128 per
/// block) instead of F16 `out`.
template<int G, bool kA4Out, bool kSym>
__launch_bounds__(kThreads) __global__
    void GateUpIu4Kernel(const std::uint8_t* __restrict__ gate,
                         const std::uint8_t* __restrict__ up,
                         const std::uint8_t* __restrict__ codes,
                         const float* __restrict__ scale,
                         const __half* __restrict__ sums,
                         const std::int32_t* __restrict__ tiles,
                         const std::int32_t* __restrict__ pad_bounds,
                         const std::int32_t* __restrict__ rows_in,
                         const std::int32_t* __restrict__ rows_out,
                         __half* __restrict__ out, int m, int k, A4Rows hq) {
  static_assert(G == 64 || G == 128 || G == 256);
  static_assert(!kSym || G != 256);
  constexpr int T = 4;
  constexpr int BN = 16 * T;
  constexpr int kPitch = 80;  // 4 groups x 16 bytes; 16-byte aligned, the
                              // 16 rows of a fragment on distinct banks
  constexpr int kNs = G >= 128 ? 1 : 2;  // token scales per stage
  constexpr int kOffA = 256 * kPitch;
  constexpr int kOffHdr = kOffA + BN * kPitch;
  constexpr int kOffSum = kOffHdr + 256 * 32;
  constexpr int kOffScl = kOffSum + BN * 16;
  constexpr int kLdsBytes = kOffScl + kNs * BN * 4;
  constexpr int kOutStride = 128 + 2;
  static_assert(16 * kOutStride * 2 <= kLdsBytes);
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  std::uint8_t* s_w = lds;
  std::uint8_t* s_a = lds + kOffA;
  auto* s_hdr = reinterpret_cast<std::uint32_t*>(lds + kOffHdr);
  std::uint8_t* s_sum = lds + kOffSum;
  auto* s_scl = reinterpret_cast<float*>(lds + kOffScl);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int half = lane >> 4;

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live = min(T, (bucket_rows - t_local + 15) / 16);
  const int stages = k / 128;
  const std::size_t row_bytes = static_cast<std::size_t>(k / 256) * 144;
  const int f0 = static_cast<int>(blockIdx.x) * 128;

  // Staging jobs: weight row tid (gate 0-127, up 128-255), token tid / 4
  // part tid % 4 (one 32-group of the stage), token tid < 64: sums, scales.
  const int ff = f0 + (tid & 127);
  const std::uint8_t* w_row = (tid < 128 ? gate : up) +
                              static_cast<std::size_t>(expert) * m * row_bytes +
                              static_cast<std::size_t>(ff & ~31) * row_bytes +
                              (ff & 31) * 16;
  const auto source = [&](int t) {
    const int c_row = t_local + t;
    return c_row < bucket_rows ? rows_in[bucket_begin + c_row] : -1;
  };
  const int a_tok = tid >> 2;
  const int a_part = tid & 3;
  const std::int32_t a_row = source(a_tok);
  const std::uint8_t* a_src =
      a_row >= 0
          ? codes + static_cast<std::size_t>(a_row) * (k / 2) + a_part * 16
          : nullptr;
  const std::int32_t t_row = tid < BN ? source(tid) : -1;
  const __half* sum_src =
      t_row >= 0 ? sums + static_cast<std::size_t>(t_row) * (k / 32) : nullptr;
  const float* scl_src =
      t_row >= 0 ? scale + static_cast<std::size_t>(t_row) * (k / G) : nullptr;
  // Token scales are read eight per lane: tile t, tokens 2 e + half.
  const int scl_slot = (tid >> 4) * 16 + (tid & 1) * 8 + ((tid & 15) >> 1);

  uint4 f_w[4];
  uint4 f_hdr = make_uint4(0U, 0U, 0U, 0U);
  uint4 f_a;
  uint4 f_sum = make_uint4(0U, 0U, 0U, 0U);
  float f_scl[kNs];

  // HH = s & 1 (the super-block half) is a template parameter so that the
  // per-group first/last/fold decisions are compile-time.
  const auto fetch = [&]<int HH>(int s) {
    constexpr int hh = HH;
    const std::uint8_t* step =
        w_row + static_cast<std::size_t>(s >> 1) * (32 * 144);
    if (hh == 0)
      f_hdr = *reinterpret_cast<const uint4*>(step);
#pragma unroll
    for (int p = 0; p < 4; ++p)
      f_w[p] = *reinterpret_cast<const uint4*>(step + (1 + 4 * hh + p) * 512);
    f_a = a_src != nullptr ? *reinterpret_cast<const uint4*>(a_src + s * 64)
                           : make_uint4(0U, 0U, 0U, 0U);
    if (tid < BN) {
      if (hh == 0 && !kSym)
        f_sum = sum_src != nullptr
                    ? *reinterpret_cast<const uint4*>(sum_src + 8 * (s >> 1))
                    : make_uint4(0U, 0U, 0U, 0U);
#pragma unroll
      for (int q = 0; q < kNs; ++q) {
        const int idx = G == 256 ? (s >> 1) : G == 128 ? s : 2 * s + q;
        f_scl[q] = scl_src != nullptr ? scl_src[idx] : 0.0F;
      }
    }
  };

  const auto commit = [&]<int HH>(int s) {
    std::uint8_t* row = s_w + tid * kPitch;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      // Piece p: nibble group 2 (s & 1) + p / 2, half p % 2. Its low
      // nibbles are stage group 2 (p / 2), its high nibbles the next one.
      constexpr std::uint32_t kM = 0x0F0F0F0FU;
      const uint4 v = f_w[p];
      const std::uint32_t lo0 = (v.x & kM) | ((v.y << 4) & ~kM);
      const std::uint32_t hi0 = ((v.x >> 4) & kM) | (v.y & ~kM);
      const std::uint32_t lo1 = (v.z & kM) | ((v.w << 4) & ~kM);
      const std::uint32_t hi1 = ((v.z >> 4) & kM) | (v.w & ~kM);
      const int jl = 2 * (p >> 1);
      const int hb = p & 1;
      constexpr std::uint32_t kSignFlip = kSym ? 0x88888888U : 0U;
      *reinterpret_cast<uint2*>(row + (jl * 2 + hb) * 8) =
          make_uint2(lo0 ^ kSignFlip, lo1 ^ kSignFlip);
      *reinterpret_cast<uint2*>(row + ((jl + 1) * 2 + hb) * 8) =
          make_uint2(hi0 ^ kSignFlip, hi1 ^ kSignFlip);
    }
    if constexpr (HH == 0 && kSym) {
      // Symmetric container (RequantQ4KSym): d * sc per weight group, the
      // only scales the iu4 fold needs; no mins.
      const uint4 h = f_hdr;
      const float dd = __low2float(__builtin_bit_cast(__half2, h.x));
      auto* hd = reinterpret_cast<float*>(s_hdr + tid * 8);
#pragma unroll
      for (int g = 0; g < 256 / G; ++g) {
        const int j = g * (G / 32);
        const std::uint32_t scj = j < 4 ? HeaderByte(h, 4 + j) & 0x3FU
                                        : (HeaderByte(h, 8 + j) & 0x0FU) |
                                              ((HeaderByte(h, j) >> 6U) << 4U);
        hd[g] = dd * static_cast<float>(scj);
      }
    }
    if constexpr (HH == 0 && !kSym) {
      const uint4 h = f_hdr;
      std::uint32_t sc[8];
      std::uint32_t mn[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < 4) {
          sc[j] = HeaderByte(h, 4 + j) & 0x3FU;
          mn[j] = HeaderByte(h, 8 + j) & 0x3FU;
        } else {
          sc[j] =
              (HeaderByte(h, 8 + j) & 0x0FU) | ((HeaderByte(h, j) >> 6U) << 4U);
          mn[j] = (HeaderByte(h, 8 + j) >> 4U) |
                  ((HeaderByte(h, 4 + j) >> 6U) << 4U);
        }
      }
      const __half2 dm = __builtin_bit_cast(__half2, h.x);
      const float dmin = __high2float(dm);
      std::uint32_t* hd = s_hdr + tid * 8;
      hd[0] = sc[0] | (sc[1] << 8) | (sc[2] << 16) | (sc[3] << 24);
      hd[1] = sc[4] | (sc[5] << 8) | (sc[6] << 16) | (sc[7] << 24);
      hd[2] = __float_as_uint(__low2float(dm));
      hd[3] = 0U;
#pragma unroll
      for (int p = 0; p < 4; ++p)
        hd[4 + p] = __builtin_bit_cast(
            std::uint32_t,
            __floats2half2_rn(-dmin * static_cast<float>(mn[2 * p]),
                              -dmin * static_cast<float>(mn[2 * p + 1])));
    }
    *reinterpret_cast<uint4*>(s_a + a_tok * kPitch + a_part * 16) = f_a;
    if (tid < BN) {
      if constexpr (HH == 0 && !kSym)
        *reinterpret_cast<uint4*>(s_sum + tid * 16) = f_sum;
#pragma unroll
      for (int q = 0; q < kNs; ++q)
        s_scl[q * BN + scl_slot] = f_scl[q];
    }
  };

  i32x8 P[2][T];
  v8f F[2][T];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int t = 0; t < T; ++t) {
      P[u][t] = i32x8{};
      F[u][t] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  const int r_row[2] = {16 * wave + l16, 128 + 16 * wave + l16};

  const auto compute = [&]<int HH>() {
    constexpr int hh = HH;
    if constexpr (kSym) {
      // One integer fold per weight/activation group of G elements: the
      // int32 dot products of its 32-groups accumulate in P, then
      // F += P * (d sc) * token scale. Signed weights (codes - 8), no min.
      constexpr int GP = G / 32;
      constexpr int NF = 4 / GP;
      float ws[NF][2];
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int u = 0; u < 2; ++u)
          ws[f][u] =
              reinterpret_cast<const float*>(s_hdr)[r_row[u] * 8 + hh * NF + f];
#pragma unroll
      for (int f = 0; f < NF; ++f) {
#pragma unroll
        for (int jg = 0; jg < GP; ++jg) {
          const int jl = f * GP + jg;
          uint4 bw[2];
#pragma unroll
          for (int u = 0; u < 2; ++u)
            bw[u] = *reinterpret_cast<const uint4*>(s_w + r_row[u] * kPitch +
                                                    jl * 16);
          uint4 av[T];
#pragma unroll
          for (int t = 0; t < T; ++t) {
            asm volatile("" ::: "memory");
            av[t] = t < live ? *reinterpret_cast<const uint4*>(
                                   s_a + (16 * t + l16) * kPitch + jl * 16)
                             : make_uint4(0U, 0U, 0U, 0U);
          }
#pragma unroll
          for (int part = 0; part < 2; ++part)
#pragma unroll
            for (int t = 0; t < T; ++t) {
              if (t >= live)
                continue;
              const i32x2 a = part == 0 ? i32x2{static_cast<int>(av[t].x),
                                                static_cast<int>(av[t].y)}
                                        : i32x2{static_cast<int>(av[t].z),
                                                static_cast<int>(av[t].w)};
#pragma unroll
              for (int u = 0; u < 2; ++u) {
                const i32x2 b = part == 0 ? i32x2{static_cast<int>(bw[u].x),
                                                  static_cast<int>(bw[u].y)}
                                          : i32x2{static_cast<int>(bw[u].z),
                                                  static_cast<int>(bw[u].w)};
                i32x8 c = (jg == 0 && part == 0) ? i32x8{} : P[u][t];
                P[u][t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    true, a, true, b, c, false);
              }
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (t >= live)
            continue;
          const float* sp = s_scl + (G == 64 ? f : 0) * BN + 16 * t + 8 * half;
          const float4 s0 = *reinterpret_cast<const float4*>(sp);
          const float4 s1 = *reinterpret_cast<const float4*>(sp + 4);
          const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
          for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              F[u][t][e] = fmaf(static_cast<float>(P[u][t][e]),
                                ws[f][u] * sv[e], F[u][t][e]);
        }
      }
      return;
    }
    std::uint32_t scw[2];
#pragma unroll
    for (int u = 0; u < 2; ++u)
      scw[u] = s_hdr[r_row[u] * 8 + hh];
#pragma unroll
    for (int jl = 0; jl < 4; ++jl) {
      const int j = 4 * hh + jl;  // 32-group in the super-block
      const bool first = G == 256   ? j == 0
                         : G == 128 ? (j & 3) == 0
                                    : (j & 1) == 0;
      const bool last = G == 256   ? j == 7
                        : G == 128 ? (j & 3) == 3
                                   : (j & 1) == 1;
      // jl is unrolled and hh constant: both are compile-time.
      uint4 bw[2];
      int sc[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        bw[u] =
            *reinterpret_cast<const uint4*>(s_w + r_row[u] * kPitch + jl * 16);
        sc[u] = static_cast<int>((scw[u] >> (8 * jl)) & 0xFFU);
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
        asm volatile("" ::: "memory");
        if (t >= live)
          continue;
        const uint4 av = *reinterpret_cast<const uint4*>(
            s_a + (16 * t + l16) * kPitch + jl * 16);
        const i32x2 a0 = {static_cast<int>(av.x), static_cast<int>(av.y)};
        const i32x2 a1 = {static_cast<int>(av.z), static_cast<int>(av.w)};
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const i32x2 b0 = {static_cast<int>(bw[u].x),
                            static_cast<int>(bw[u].y)};
          const i32x2 b1 = {static_cast<int>(bw[u].z),
                            static_cast<int>(bw[u].w)};
          i32x8 d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, a0, false, b0, i32x8{}, false);
          d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a1, false, b1, d,
                                                         false);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if (first)
              P[u][t][e] = __mul24(d[e], sc[u]);
            else
              P[u][t][e] = __mul24(d[e], sc[u]) + P[u][t][e];
          }
        }
      }
      if (last) {
        // F += (d * token scale) * P, the activation group done.
        const int q = G == 64 ? (jl >> 1) : 0;
        float dw[2];
#pragma unroll
        for (int u = 0; u < 2; ++u)
          dw[u] = __uint_as_float(s_hdr[r_row[u] * 8 + 2]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (t >= live)
            continue;
          const float* sp = s_scl + q * BN + 16 * t + 8 * half;
          const float4 s0 = *reinterpret_cast<const float4*>(sp);
          const float4 s1 = *reinterpret_cast<const float4*>(sp + 4);
          const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
          for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              F[u][t][e] = fmaf(static_cast<float>(P[u][t][e]), dw[u] * sv[e],
                                F[u][t][e]);
        }
      }
    }
    if constexpr (hh == 1) {
      // Min term of the super-block: F += S . (-dmin m) over its 8 groups.
      v16h bm[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const uint4 mins =
            *reinterpret_cast<const uint4*>(s_hdr + r_row[u] * 8 + 4);
        const uint4 parts[2] = {mins, make_uint4(0U, 0U, 0U, 0U)};
        __builtin_memcpy(&bm[u], parts, 32);
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t >= live)
          continue;
        const uint4 sv =
            *reinterpret_cast<const uint4*>(s_sum + (16 * t + l16) * 16);
        const uint4 parts[2] = {sv, make_uint4(0U, 0U, 0U, 0U)};
        v16h am;
        __builtin_memcpy(&am, parts, 32);
#pragma unroll
        for (int u = 0; u < 2; ++u)
          F[u][t] = Wmma(am, bm[u], F[u][t]);
      }
    }
  };

  // stages is even (k % 256 == 0): one super-block per iteration.
  fetch.template operator()<0>(0);
  for (int s = 0; s < stages; s += 2) {
    __syncthreads();
    commit.template operator()<0>(s);
    __syncthreads();
    fetch.template operator()<1>(s + 1);
    compute.template operator()<0>();
    __syncthreads();
    commit.template operator()<1>(s + 1);
    __syncthreads();
    if (s + 2 < stages)
      fetch.template operator()<0>(s + 2);
    compute.template operator()<1>();
  }

  // Epilogue: SwiGLU in registers (gate u = 0, up u = 1 share the element
  // mapping), staged [token][channel] in LDS, whole 16-byte stores.
  if constexpr (kA4Out) {
    // int4 rows for the down projection: the block's 128 channels are one
    // activation group per token (scale = max / 7), codes in WMMA nibble
    // order, dequantized sums per 32 channels for the down's min term.
    auto* s_max = reinterpret_cast<float*>(lds);         // [16 tok][8 waves]
    auto* s_amax = reinterpret_cast<float*>(lds + 512);  // slot order
    auto* s_code = reinterpret_cast<std::int8_t*>(lds + 1024);  // [16][128]
    const int groups_out = static_cast<int>(hq.k / 32);
    const int scales_out = static_cast<int>(hq.k / 128);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t >= live)
        break;  // block-uniform
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float g = F[0][t][e];
        float product = F[1][t][e] * g;
        asm volatile("" : "+v"(product));
        float value = product * SigmoidF(g);
        asm volatile("" : "+v"(value));
        // Round through F16 like the F16 route's SwiGLU rows.
        v[e] = __half2float(__float2half(value));
      }
      __syncthreads();
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float a = fabsf(v[e]);
#pragma unroll
        for (int msk = 1; msk < 16; msk <<= 1)
          a = fmaxf(a, __shfl_xor(a, msk, 32));
        if (l16 == 0)
          s_max[(2 * e + half) * 8 + wave] = a;
      }
      __syncthreads();
      if (tid < 16) {
        float a = 0.0F;
#pragma unroll
        for (int w8 = 0; w8 < 8; ++w8)
          a = fmaxf(a, s_max[tid * 8 + w8]);
        s_amax[(tid & 1) * 8 + (tid >> 1)] = a;  // token 2 e + half
      }
      __syncthreads();
      {
        const float4 a0 = *reinterpret_cast<const float4*>(s_amax + 8 * half);
        const float4 a1 =
            *reinterpret_cast<const float4*>(s_amax + 8 * half + 4);
        const float am[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float inv = am[e] > 0.0F ? 7.0F / am[e] : 0.0F;
          const int c = min(7, max(-7, static_cast<int>(rintf(v[e] * inv))));
          s_code[(2 * e + half) * 128 + 16 * wave + l16] =
              static_cast<std::int8_t>(c);
        }
      }
      __syncthreads();
      if (tid < 128) {
        // Thread: token tid / 8, 16-channel slice tid % 8 (32-group sl / 2,
        // half sl % 2).
        const int tr = tid >> 3;
        const int sl = tid & 7;
        const uint4 b =
            *reinterpret_cast<const uint4*>(s_code + tr * 128 + 16 * sl);
        constexpr std::uint32_t kM = 0x0F0F0F0FU;
        const std::uint32_t dw0 = (b.x & kM) | ((b.y << 4) & ~kM);
        const std::uint32_t dw1 = (b.z & kM) | ((b.w << 4) & ~kM);
        int sum = __builtin_amdgcn_sudot4(true, static_cast<int>(b.x), true,
                                          0x01010101, 0, false);
        sum = __builtin_amdgcn_sudot4(true, static_cast<int>(b.y), true,
                                      0x01010101, sum, false);
        sum = __builtin_amdgcn_sudot4(true, static_cast<int>(b.z), true,
                                      0x01010101, sum, false);
        sum = __builtin_amdgcn_sudot4(true, static_cast<int>(b.w), true,
                                      0x01010101, sum, false);
        sum += __shfl_xor(sum, 1, 32);
        const int tt = t_local + 16 * t + tr;
        if (tt < bucket_rows) {
          const std::int32_t dst = rows_out[bucket_begin + tt];
          if (dst >= 0) {
            const float sc = s_amax[(tr & 1) * 8 + (tr >> 1)] / 7.0F;
            const int g32 = 4 * static_cast<int>(blockIdx.x) + (sl >> 1);
            *reinterpret_cast<uint2*>(
                hq.codes + static_cast<std::size_t>(dst) * (hq.k / 2) +
                g32 * 16 + (sl & 1) * 8) = make_uint2(dw0, dw1);
            if (!kSym && (sl & 1) == 0)
              hq.sums[static_cast<std::size_t>(dst) * groups_out + g32] =
                  __float2half(sc * static_cast<float>(sum));
            if (sl == 0)
              hq.scale[static_cast<std::size_t>(dst) * scales_out +
                       blockIdx.x] = sc;
          }
        }
      }
    }
    return;
  }
  auto* stage = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (t >= live)
      break;  // block-uniform
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float g = F[0][t][e];
      float product = F[1][t][e] * g;
      asm volatile("" : "+v"(product));
      float value = product * SigmoidF(g);
      asm volatile("" : "+v"(value));
      stage[(2 * e + half) * kOutStride + 16 * wave + l16] =
          __float2half(value);
    }
    __syncthreads();
    const int tr = tid >> 4;
    const int c8 = (tid & 15) * 8;
    const int tt = t_local + 16 * t + tr;
    if (tt < bucket_rows) {
      const std::int32_t dst = rows_out[bucket_begin + tt];
      if (dst >= 0) {
        const __half* src = stage + tr * kOutStride + c8;
        uint4 v;
        v.x = *reinterpret_cast<const std::uint32_t*>(src);
        v.y = *reinterpret_cast<const std::uint32_t*>(src + 2);
        v.z = *reinterpret_cast<const std::uint32_t*>(src + 4);
        v.w = *reinterpret_cast<const std::uint32_t*>(src + 6);
        *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(dst) * m + f0 +
                                  c8) = v;
      }
    }
  }
}

/// RequantDown4: one thread per (row, 128-element super-block = two packed
/// Q5_1 steps). Each 32-block gets asymmetric 4-bit codes; its scale is
/// rounded to sc * 2^e with one power of two e per super-block and the
/// largest sc in [32, 63], so the Q5_1 d (= sc * 2^e, exact in F16) still
/// serves every F16/int8 route while the iu4 down recovers the integer sc
/// and e from the four d values. m is the least-squares offset for those
/// codes; qh = 0.
__global__ __launch_bounds__(kThreads) void RequantDown4Kernel(
    std::uint8_t* __restrict__ w, std::size_t rows, int k) {
  const int supers = k / 128;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * supers)
    return;
  const std::size_t row = idx / supers;
  const int sb = static_cast<int>(idx % supers);
  const std::size_t row_bytes = static_cast<std::size_t>(k / 64) * 48;
  std::uint8_t* base =
      w + (row & ~std::size_t{31}) * row_bytes + (row & 31) * 16;
  float v[4][32];
  float scale[4], lo[4];
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    std::uint8_t* step = base + static_cast<std::size_t>(2 * sb + st) * 1536;
    const uint4 dmh = *reinterpret_cast<const uint4*>(step + 1024);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint4 qs = *reinterpret_cast<const uint4*>(step + 512 * h);
      const __half2 dm = __builtin_bit_cast(__half2, h ? dmh.z : dmh.x);
      const std::uint32_t qh = h ? dmh.w : dmh.y;
      const float d = __low2float(dm), mn = __high2float(dm);
      const std::uint32_t words[4] = {qs.x, qs.y, qs.z, qs.w};
      const int b = 2 * st + h;
      float vmin = 1e30F, vmax = -1e30F;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t byte = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
        const int q0 =
            static_cast<int>((byte & 0xFU) | (((qh >> i) & 1U) << 4));
        const int q1 =
            static_cast<int>((byte >> 4) | (((qh >> (16 + i)) & 1U) << 4));
        v[b][i] = d * static_cast<float>(q0) + mn;
        v[b][16 + i] = d * static_cast<float>(q1) + mn;
        vmin = fminf(vmin, fminf(v[b][i], v[b][16 + i]));
        vmax = fmaxf(vmax, fmaxf(v[b][i], v[b][16 + i]));
      }
      scale[b] = (vmax - vmin) / 15.0F;
      lo[b] = vmin;
    }
  }
  const float smax =
      fmaxf(fmaxf(scale[0], scale[1]), fmaxf(scale[2], scale[3]));
  // e: smax / 2^e in (31.5, 63], so the largest rounded sc is 32..63.
  const int e = smax > 0.0F ? static_cast<int>(ceilf(log2f(smax / 63.0F))) : 0;
  const float pe = ldexpf(1.0F, e);
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    std::uint8_t* step = base + static_cast<std::size_t>(2 * sb + st) * 1536;
    std::uint32_t dm_out[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int b = 2 * st + h;
      const float sc = fminf(63.0F, rintf(scale[b] / pe));
      const float d = sc * pe;
      int q[32];
      float acc = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        q[i] = d > 0.0F
                   ? min(15,
                         max(0, static_cast<int>(rintf((v[b][i] - lo[b]) / d))))
                   : 0;
        acc += v[b][i] - d * static_cast<float>(q[i]);
      }
      const float mn = acc / 32.0F;
      std::uint32_t words[4] = {0U, 0U, 0U, 0U};
#pragma unroll
      for (int i = 0; i < 16; ++i)
        words[i >> 2] |= (static_cast<std::uint32_t>(q[i]) |
                          (static_cast<std::uint32_t>(q[16 + i]) << 4))
                         << (8 * (i & 3));
      *reinterpret_cast<uint4*>(step + 512 * h) =
          make_uint4(words[0], words[1], words[2], words[3]);
      dm_out[h] = __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
    }
    *reinterpret_cast<uint4*>(step + 1024) =
        make_uint4(dm_out[0], 0U, dm_out[1], 0U);
  }
}

/// Integer scale and power of two of a RequantDown4 super-block's four d.
__device__ __forceinline__ float SuperScale(const float d[4], int sc[4]) {
  const float dmax = fmaxf(fmaxf(d[0], d[1]), fmaxf(d[2], d[3]));
  int ex = 0;
  (void)frexpf(dmax, &ex);  // dmax = f 2^ex, f in [0.5, 1): sc max in 32..63
  const float inv = dmax > 0.0F ? ldexpf(1.0F, 6 - ex) : 0.0F;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    sc[j] = static_cast<int>(d[j] * inv);
  return dmax > 0.0F ? ldexpf(1.0F, ex - 6) : 0.0F;
}

/// Down projection on the iu4 WMMA over packed Q5_1 experts whose codes are
/// 4-bit (RequantDown4, qh = 0): w = d_b q + m_b per 32-block b. Eight
/// waves, wave w rows r0 + 32 w (two 16-row tiles, B operand) x 64 slot rows
/// (A operand: the int4 SwiGLU rows of GateUpIu4, groups of 128). Per block
/// the integer dot products enter an F32 partial with d_b, per activation
/// group (one stage of 128) the partial is scaled by the row's group scale;
/// the m_b terms are one F16 WMMA pass over all blocks at the end.
constexpr int kDownMaxGroups = 24;

__launch_bounds__(kThreads) __global__
    void DownIu4Kernel(const std::uint8_t* __restrict__ w,
                       const std::uint8_t* __restrict__ codes,
                       const float* __restrict__ scale,
                       const __half* __restrict__ sums,
                       const std::int32_t* __restrict__ tiles,
                       const std::int32_t* __restrict__ pad_bounds,
                       const std::int32_t* __restrict__ rows_in,
                       const std::int32_t* __restrict__ rows_out,
                       __half* __restrict__ out, int m, int k) {
  constexpr int T = 4;
  constexpr int BN = 16 * T;
  constexpr int kPitch = 80;                   // 4 groups x 16 bytes (see
                                               // GateUpIu4Kernel)
  constexpr int kMPitch = kDownMaxGroups + 8;  // halves per row (m_b, sums)
  constexpr int kOffA = 256 * kPitch;
  constexpr int kOffD = kOffA + BN * kPitch;          // [256][4] F32 d_b
  constexpr int kOffM = kOffD + 256 * 16;             // [256][kMPitch] F16 m_b
  constexpr int kOffSum = kOffM + 256 * kMPitch * 2;  // [64][kMPitch] F16
  constexpr int kOffScl = kOffSum + BN * kMPitch * 2;
  constexpr int kLdsBytes = kOffScl + BN * 4;
  constexpr int kOutStride = 256 + 2;
  static_assert(16 * kOutStride * 4 <= kLdsBytes);
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  std::uint8_t* s_w = lds;
  std::uint8_t* s_a = lds + kOffA;
  auto* s_d = reinterpret_cast<float*>(lds + kOffD);
  auto* s_m = reinterpret_cast<__half*>(lds + kOffM);
  auto* s_sum = reinterpret_cast<__half*>(lds + kOffSum);
  auto* s_scl = reinterpret_cast<float*>(lds + kOffScl);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int half = lane >> 4;

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live = min(T, (bucket_rows - t_local + 15) / 16);
  const int stages = k / 128;
  const int groups = k / 32;
  const std::size_t row_bytes = static_cast<std::size_t>(k / 64) * 48;
  const int r0 = static_cast<int>(blockIdx.x) * 256;

  // Blocks past k contribute nothing to the final min-term WMMA.
  for (int i = tid; i < 256 * kMPitch; i += kThreads)
    s_m[i] = __float2half(0.0F);
  for (int i = tid; i < BN * kMPitch; i += kThreads)
    s_sum[i] = __float2half(0.0F);

  const int rr = r0 + tid;  // staged weight row
  const std::uint8_t* w_row =
      w + static_cast<std::size_t>(expert) * m * row_bytes +
      static_cast<std::size_t>(rr & ~31) * row_bytes + (rr & 31) * 16;
  const auto source = [&](int t) {
    const int c_row = t_local + t;
    return c_row < bucket_rows ? rows_in[bucket_begin + c_row] : -1;
  };
  const int a_tok = tid >> 2;
  const int a_part = tid & 3;
  const std::int32_t a_row = source(a_tok);
  const std::uint8_t* a_src =
      a_row >= 0
          ? codes + static_cast<std::size_t>(a_row) * (k / 2) + a_part * 16
          : nullptr;
  const std::int32_t t_row = tid < BN ? source(tid) : -1;
  const __half* sum_src =
      t_row >= 0 ? sums + static_cast<std::size_t>(t_row) * groups : nullptr;
  const float* scl_src =
      t_row >= 0 ? scale + static_cast<std::size_t>(t_row) * (k / 128)
                 : nullptr;
  const int scl_slot = (tid >> 4) * 16 + (tid & 1) * 8 + ((tid & 15) >> 1);

  uint4 f_qs[4];
  uint4 f_dmh[2];
  uint4 f_a;
  uint2 f_sum = make_uint2(0U, 0U);
  float f_scl = 0.0F;
  const auto fetch = [&](int s) {
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const std::uint8_t* step =
          w_row + static_cast<std::size_t>(2 * s + st) * 1536;
      f_qs[2 * st] = *reinterpret_cast<const uint4*>(step);
      f_qs[2 * st + 1] = *reinterpret_cast<const uint4*>(step + 512);
      f_dmh[st] = *reinterpret_cast<const uint4*>(step + 1024);
    }
    f_a = a_src != nullptr ? *reinterpret_cast<const uint4*>(a_src + s * 64)
                           : make_uint4(0U, 0U, 0U, 0U);
    if (tid < BN) {
      f_sum = sum_src != nullptr
                  ? *reinterpret_cast<const uint2*>(sum_src + 4 * s)
                  : make_uint2(0U, 0U);
      f_scl = scl_src != nullptr ? scl_src[s] : 0.0F;
    }
  };
  const auto commit = [&](int s) {
    std::uint8_t* row = s_w + tid * kPitch;
    constexpr std::uint32_t kM = 0x0F0F0F0FU;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      // Q5_1 nibbles: byte i holds element i (low) and 16 + i (high).
      const uint4 v = f_qs[b];
      const std::uint32_t s0d0 = (v.x & kM) | ((v.y << 4) & ~kM);
      const std::uint32_t s0d1 = (v.z & kM) | ((v.w << 4) & ~kM);
      const std::uint32_t s1d0 = ((v.x >> 4) & kM) | (v.y & ~kM);
      const std::uint32_t s1d1 = ((v.z >> 4) & kM) | (v.w & ~kM);
      *reinterpret_cast<uint4*>(row + 16 * b) =
          make_uint4(s0d0, s0d1, s1d0, s1d1);
      const uint4 dmh = f_dmh[b >> 1];
      const __half2 dm = __builtin_bit_cast(__half2, (b & 1) ? dmh.z : dmh.x);
      s_m[tid * kMPitch + 4 * s + b] = __high2half(dm);
    }
    {
      // The stage is one super-block: integer sc per block and 2^e.
      const float d[4] = {__low2float(__builtin_bit_cast(__half2, f_dmh[0].x)),
                          __low2float(__builtin_bit_cast(__half2, f_dmh[0].z)),
                          __low2float(__builtin_bit_cast(__half2, f_dmh[1].x)),
                          __low2float(__builtin_bit_cast(__half2, f_dmh[1].z))};
      int sc[4];
      const float pe = SuperScale(d, sc);
      s_d[tid * 4] = pe;
      s_d[tid * 4 + 1] = __uint_as_float(static_cast<std::uint32_t>(
          sc[0] | (sc[1] << 8) | (sc[2] << 16) | (sc[3] << 24)));
    }
    *reinterpret_cast<uint4*>(s_a + a_tok * kPitch + a_part * 16) = f_a;
    if (tid < BN) {
      *reinterpret_cast<uint2*>(s_sum + tid * kMPitch + 4 * s) = f_sum;
      s_scl[scl_slot] = f_scl;
    }
  };

  i32x8 P[2][T];
  v8f F[2][T];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int t = 0; t < T; ++t)
      F[u][t] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const int r_row[2] = {32 * wave + l16, 32 * wave + 16 + l16};

  const auto compute = [&]() {
#pragma unroll
    for (int jl = 0; jl < 4; ++jl) {
      uint4 bw[2];
      int scv[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        bw[u] =
            *reinterpret_cast<const uint4*>(s_w + r_row[u] * kPitch + jl * 16);
        scv[u] = static_cast<int>(
            (__float_as_uint(s_d[r_row[u] * 4 + 1]) >> (8 * jl)) & 0xFFU);
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
        asm volatile("" ::: "memory");
        if (t >= live)
          continue;
        const uint4 av = *reinterpret_cast<const uint4*>(
            s_a + (16 * t + l16) * kPitch + jl * 16);
        const i32x2 a0 = {static_cast<int>(av.x), static_cast<int>(av.y)};
        const i32x2 a1 = {static_cast<int>(av.z), static_cast<int>(av.w)};
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const i32x2 b0 = {static_cast<int>(bw[u].x),
                            static_cast<int>(bw[u].y)};
          const i32x2 b1 = {static_cast<int>(bw[u].z),
                            static_cast<int>(bw[u].w)};
          i32x8 d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, a0, false, b0, i32x8{}, false);
          d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a1, false, b1, d,
                                                         false);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if (jl == 0)
              P[u][t][e] = __mul24(d[e], scv[u]);
            else
              P[u][t][e] = __mul24(d[e], scv[u]) + P[u][t][e];
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t >= live)
        continue;
      const float* sp = s_scl + 16 * t + 8 * half;
      const float4 s0 = *reinterpret_cast<const float4*>(sp);
      const float4 s1 = *reinterpret_cast<const float4*>(sp + 4);
      const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const float pe = s_d[r_row[u] * 4];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          F[u][t][e] =
              fmaf(static_cast<float>(P[u][t][e]), pe * sv[e], F[u][t][e]);
      }
    }
  };

  fetch(0);
  for (int s = 0; s < stages; ++s) {
    __syncthreads();
    commit(s);
    __syncthreads();
    if (s + 1 < stages)
      fetch(s + 1);
    compute();
  }
  // m_b terms: F += S . m over all blocks (two K16 F16 WMMAs).
  __syncthreads();
#pragma unroll
  for (int kh = 0; kh < 2; ++kh) {
    v16h bm[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const uint4* p =
          reinterpret_cast<const uint4*>(s_m + r_row[u] * kMPitch + 16 * kh);
      const uint4 parts[2] = {p[0], p[1]};
      __builtin_memcpy(&bm[u], parts, 32);
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t >= live)
        continue;
      const uint4* p = reinterpret_cast<const uint4*>(
          s_sum + (16 * t + l16) * kMPitch + 16 * kh);
      const uint4 parts[2] = {p[0], p[1]};
      v16h am;
      __builtin_memcpy(&am, parts, 32);
#pragma unroll
      for (int u = 0; u < 2; ++u)
        F[u][t] = Wmma(am, bm[u], F[u][t]);
    }
  }

  // Epilogue: [slot][row] through LDS, F16 pairs.
  auto* stage = reinterpret_cast<float*>(lds);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (t >= live)
      break;  // block-uniform
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        stage[(2 * e + half) * kOutStride + 32 * wave + 16 * u + l16] =
            F[u][t][e];
    __syncthreads();
#pragma unroll
    for (int round = 0; round < 16 * 256 / (2 * kThreads); ++round) {
      const int flat = (round * kThreads + tid) * 2;
      const int tr = flat / 256;
      const int row = flat % 256;
      const int tt = t_local + 16 * t + tr;
      if (tt < bucket_rows) {
        const std::int32_t dst = rows_out[bucket_begin + tt];
        if (dst >= 0) {
          const float2 v =
              *reinterpret_cast<const float2*>(stage + tr * kOutStride + row);
          *reinterpret_cast<__half2*>(out + static_cast<std::size_t>(dst) * m +
                                      r0 + row) = __floats2half2_rn(v.x, v.y);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Symmetric int4 experts (one scale per 64/128 K, no minimum).

/// Best symmetric step for `n` values: amax / c, c in [6, 8], minimizing the
/// squared error of codes clamp(rint(v / step), -8, 7).
template<int N>
__device__ float SymStep(const float* v) {
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < N; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  if (amax == 0.0F)
    return 0.0F;
  float best = amax / 7.0F;
  float best_err = 3.0e38F;
#pragma unroll 1
  for (int ci = 0; ci < 9; ++ci) {
    const float step = amax / (6.0F + 0.25F * static_cast<float>(ci));
    const float inv = 1.0F / step;
    float err = 0.0F;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const float q = fminf(7.0F, fmaxf(-8.0F, rintf(v[i] * inv)));
      const float r = v[i] - step * q;
      err += r * r;
    }
    if (err < best_err) {
      best_err = err;
      best = step;
    }
  }
  return best;
}

__device__ __forceinline__ int SymCode(float v, float step) {
  return step > 0.0F
             ? static_cast<int>(fminf(7.0F, fmaxf(-8.0F, rintf(v / step))))
             : 0;
}

/// One thread per (row, super-block of 256): Q4_K -> the symmetric Q4_K
/// container. d = max group step / 63 (F16), every WG group an integer
/// 1..63 times d, dmin = 8 d and mins = scales, so d sc (q - 8) is also what
/// every Q4_K reader (decode, F16 prefill) computes.
template<int WG>
__global__ __launch_bounds__(kThreads) void RequantQ4KSymKernel(
    std::uint8_t* __restrict__ w, std::size_t rows, int k) {
  constexpr int NG = 256 / WG;
  const int supers = k / 256;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * supers)
    return;
  const std::size_t row = idx / supers;
  const int sb = static_cast<int>(idx % supers);
  const std::size_t row_bytes = static_cast<std::size_t>(supers) * 144;
  std::uint8_t* step = w + (row & ~std::size_t{31}) * row_bytes +
                       (row & 31) * 16 +
                       static_cast<std::size_t>(sb) * (32 * 144);
  const uint4 h = *reinterpret_cast<const uint4*>(step);
  std::uint32_t words[8][4];
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint4 piece = *reinterpret_cast<const uint4*>(step + (1 + p) * 512);
    words[p][0] = piece.x;
    words[p][1] = piece.y;
    words[p][2] = piece.z;
    words[p][3] = piece.w;
  }
  const __half2 dm = __builtin_bit_cast(__half2, h.x);
  const float d = __low2float(dm);
  const float dmin = __high2float(dm);
  float v[256];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    std::uint32_t sc, mn;
    if (j < 4) {
      sc = HeaderByte(h, 4 + j) & 0x3FU;
      mn = HeaderByte(h, 8 + j) & 0x3FU;
    } else {
      sc = (HeaderByte(h, 8 + j) & 0x0FU) | ((HeaderByte(h, j) >> 6U) << 4U);
      mn = (HeaderByte(h, 8 + j) >> 4U) | ((HeaderByte(h, 4 + j) >> 6U) << 4U);
    }
    const float a = d * static_cast<float>(sc);
    const float b = dmin * static_cast<float>(mn);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int p = 2 * (j >> 1) + (i >> 4);
      const std::uint32_t byte =
          (words[p][(i & 15) >> 2] >> (8 * (i & 3))) & 0xFFU;
      const std::uint32_t q = (j & 1) ? byte >> 4 : byte & 0xFU;
      v[32 * j + i] = a * static_cast<float>(q) - b;
    }
  }
  float steps[NG];
  float smax = 0.0F;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    steps[g] = SymStep<WG>(v + g * WG);
    smax = fmaxf(smax, steps[g]);
  }
  const __half dh = __float2half(smax / 63.0F);
  const float df = __half2float(dh);
  std::uint32_t sc[8];
  std::uint32_t scg[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g)
    scg[g] = df > 0.0F ? static_cast<std::uint32_t>(
                             fminf(63.0F, fmaxf(1.0F, rintf(steps[g] / df))))
                       : 1U;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    sc[j] = scg[j / (WG / 32)];
  std::uint32_t out_words[8][4];
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const int g = p >> 1;
    const int half = p & 1;
    const float lo_step = df * static_cast<float>(sc[2 * g]);
    const float hi_step = df * static_cast<float>(sc[2 * g + 1]);
#pragma unroll
    for (int dw = 0; dw < 4; ++dw) {
      std::uint32_t word = 0;
#pragma unroll
      for (int bb = 0; bb < 4; ++bb) {
        const int i = 16 * half + 4 * dw + bb;
        const std::uint32_t qlo =
            static_cast<std::uint32_t>(SymCode(v[64 * g + i], lo_step) + 8);
        const std::uint32_t qhi = static_cast<std::uint32_t>(
            SymCode(v[64 * g + 32 + i], hi_step) + 8);
        word |= (qlo | (qhi << 4)) << (8 * bb);
      }
      out_words[p][dw] = word;
    }
  }
  std::uint32_t scales[12];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    scales[j] = sc[j] | ((sc[j + 4] >> 4) << 6);
    scales[j + 4] = sc[j] | ((sc[j + 4] >> 4) << 6);
    scales[j + 8] = (sc[j + 4] & 0xFU) | ((sc[j + 4] & 0xFU) << 4);
  }
  const __half2 dm_out = __halves2half2(dh, __float2half(8.0F * df));
  std::uint32_t hw[4];
  hw[0] = __builtin_bit_cast(std::uint32_t, dm_out);
#pragma unroll
  for (int q = 0; q < 3; ++q)
    hw[1 + q] = scales[4 * q] | (scales[4 * q + 1] << 8) |
                (scales[4 * q + 2] << 16) | (scales[4 * q + 3] << 24);
  *reinterpret_cast<uint4*>(step) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
#pragma unroll
  for (int p = 0; p < 8; ++p)
    *reinterpret_cast<uint4*>(step + (1 + p) * 512) = make_uint4(
        out_words[p][0], out_words[p][1], out_words[p][2], out_words[p][3]);
}

/// One thread per (row, 128 K = two packed Q5_1 steps): codes of
/// d (q - 8) with one F16 d per 128 in all four 32-blocks, m = -8 d, qh = 0.
__global__ __launch_bounds__(kThreads) void RequantDownSymKernel(
    std::uint8_t* __restrict__ w, std::size_t rows, int k) {
  const int supers = k / 128;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * supers)
    return;
  const std::size_t row = idx / supers;
  const int sb = static_cast<int>(idx % supers);
  const std::size_t row_bytes = static_cast<std::size_t>(k / 64) * 48;
  std::uint8_t* base =
      w + (row & ~std::size_t{31}) * row_bytes + (row & 31) * 16;
  float v[128];
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    std::uint8_t* step = base + static_cast<std::size_t>(2 * sb + st) * 1536;
    const uint4 dmh = *reinterpret_cast<const uint4*>(step + 1024);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint4 qs = *reinterpret_cast<const uint4*>(step + 512 * h);
      const __half2 dm = __builtin_bit_cast(__half2, h ? dmh.z : dmh.x);
      const std::uint32_t qh = h ? dmh.w : dmh.y;
      const float d = __low2float(dm), mn = __high2float(dm);
      const std::uint32_t words[4] = {qs.x, qs.y, qs.z, qs.w};
      const int b = 2 * st + h;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t byte = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
        const int q0 =
            static_cast<int>((byte & 0xFU) | (((qh >> i) & 1U) << 4));
        const int q1 =
            static_cast<int>((byte >> 4) | (((qh >> (16 + i)) & 1U) << 4));
        v[32 * b + i] = d * static_cast<float>(q0) + mn;
        v[32 * b + 16 + i] = d * static_cast<float>(q1) + mn;
      }
    }
  }
  const __half dh = __float2half(SymStep<128>(v));
  const float df = __half2float(dh);
  const __half2 dm_out = __halves2half2(dh, __float2half(-8.0F * df));
  const std::uint32_t dmw = __builtin_bit_cast(std::uint32_t, dm_out);
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    std::uint8_t* step = base + static_cast<std::size_t>(2 * sb + st) * 1536;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int b = 2 * st + h;
      std::uint32_t words[4] = {0U, 0U, 0U, 0U};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t q0 =
            static_cast<std::uint32_t>(SymCode(v[32 * b + i], df) + 8);
        const std::uint32_t q1 =
            static_cast<std::uint32_t>(SymCode(v[32 * b + 16 + i], df) + 8);
        words[i >> 2] |= (q0 | (q1 << 4)) << (8 * (i & 3));
      }
      *reinterpret_cast<uint4*>(step + 512 * h) =
          make_uint4(words[0], words[1], words[2], words[3]);
    }
    *reinterpret_cast<uint4*>(step + 1024) = make_uint4(dmw, 0U, dmw, 0U);
  }
}

/// Down projection on the iu4 WMMA over symmetric Q5_1 experts
/// (RequantDownSym: one d per 128 K, w = d (q - 8)). Eight waves, wave w rows
/// r0 + 32 w (two 16-row tiles, B operand) x 64 slot rows (A operand: the
/// int4 SwiGLU rows of GateUpIu4, groups of 128). One stage = one 128 K
/// group: eight WMMAs per tile into an int32 accumulator, then one F32 fold
/// F += P * d * token scale.
__launch_bounds__(kThreads) __global__
    void DownSymKernel(const std::uint8_t* __restrict__ w,
                       const std::uint8_t* __restrict__ codes,
                       const float* __restrict__ scale,
                       const std::int32_t* __restrict__ tiles,
                       const std::int32_t* __restrict__ pad_bounds,
                       const std::int32_t* __restrict__ rows_in,
                       const std::int32_t* __restrict__ rows_out,
                       __half* __restrict__ out, int m, int k) {
  constexpr int T = 4;
  constexpr int BN = 16 * T;
  constexpr int kPitch = 80;
  constexpr int kOffA = 256 * kPitch;
  constexpr int kOffD = kOffA + BN * kPitch;  // [256] F32 d
  constexpr int kOffScl = kOffD + 256 * 4;    // [64] F32 token scales
  constexpr int kLdsBytes = kOffScl + BN * 4;
  constexpr int kOutStride = 256 + 8;  // halves per token row, 16-byte aligned
  static_assert(16 * kOutStride * 2 <= kLdsBytes);
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  std::uint8_t* s_w = lds;
  std::uint8_t* s_a = lds + kOffA;
  auto* s_d = reinterpret_cast<float*>(lds + kOffD);
  auto* s_scl = reinterpret_cast<float*>(lds + kOffScl);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int l16 = lane & 15;
  const int half = lane >> 4;

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live = min(T, (bucket_rows - t_local + 15) / 16);
  const int stages = k / 128;
  const std::size_t row_bytes = static_cast<std::size_t>(k / 64) * 48;
  const int r0 = static_cast<int>(blockIdx.x) * 256;

  const int rr = r0 + tid;  // staged weight row
  const std::uint8_t* w_row =
      w + static_cast<std::size_t>(expert) * m * row_bytes +
      static_cast<std::size_t>(rr & ~31) * row_bytes + (rr & 31) * 16;
  const auto source = [&](int t) {
    const int c_row = t_local + t;
    return c_row < bucket_rows ? rows_in[bucket_begin + c_row] : -1;
  };
  const int a_tok = tid >> 2;
  const int a_part = tid & 3;
  const std::int32_t a_row = source(a_tok);
  const std::uint8_t* a_src =
      a_row >= 0
          ? codes + static_cast<std::size_t>(a_row) * (k / 2) + a_part * 16
          : nullptr;
  const std::int32_t t_row = tid < BN ? source(tid) : -1;
  const float* scl_src =
      t_row >= 0 ? scale + static_cast<std::size_t>(t_row) * (k / 128)
                 : nullptr;
  const int scl_slot = (tid >> 4) * 16 + (tid & 1) * 8 + ((tid & 15) >> 1);

  uint4 f_qs[4];
  std::uint32_t f_dm = 0U;
  uint4 f_a;
  float f_scl = 0.0F;
  const auto fetch = [&](int s) {
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const std::uint8_t* step =
          w_row + static_cast<std::size_t>(2 * s + st) * 1536;
      f_qs[2 * st] = *reinterpret_cast<const uint4*>(step);
      f_qs[2 * st + 1] = *reinterpret_cast<const uint4*>(step + 512);
      if (st == 0)
        f_dm = reinterpret_cast<const std::uint32_t*>(step + 1024)[0];
    }
    f_a = a_src != nullptr ? *reinterpret_cast<const uint4*>(a_src + s * 64)
                           : make_uint4(0U, 0U, 0U, 0U);
    if (tid < BN)
      f_scl = scl_src != nullptr ? scl_src[s] : 0.0F;
  };
  const auto commit = [&]() {
    std::uint8_t* row = s_w + tid * kPitch;
    constexpr std::uint32_t kM = 0x0F0F0F0FU;
    constexpr std::uint32_t kFlip = 0x88888888U;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      // Q5_1 nibbles: byte i holds element i (low) and 16 + i (high).
      const uint4 v = f_qs[b];
      const std::uint32_t s0d0 = (v.x & kM) | ((v.y << 4) & ~kM);
      const std::uint32_t s0d1 = (v.z & kM) | ((v.w << 4) & ~kM);
      const std::uint32_t s1d0 = ((v.x >> 4) & kM) | (v.y & ~kM);
      const std::uint32_t s1d1 = ((v.z >> 4) & kM) | (v.w & ~kM);
      *reinterpret_cast<uint4*>(row + 16 * b) =
          make_uint4(s0d0 ^ kFlip, s0d1 ^ kFlip, s1d0 ^ kFlip, s1d1 ^ kFlip);
    }
    s_d[tid] = __low2float(__builtin_bit_cast(__half2, f_dm));
    *reinterpret_cast<uint4*>(s_a + a_tok * kPitch + a_part * 16) = f_a;
    if (tid < BN)
      s_scl[scl_slot] = f_scl;
  };

  i32x8 P[2][T];
  v8f F[2][T];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int t = 0; t < T; ++t)
      F[u][t] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const int r_row[2] = {32 * wave + l16, 32 * wave + 16 + l16};

  const auto compute = [&]() {
    float ws[2];
#pragma unroll
    for (int u = 0; u < 2; ++u)
      ws[u] = s_d[r_row[u]];
#pragma unroll
    for (int jl = 0; jl < 4; ++jl) {
      uint4 bw[2];
#pragma unroll
      for (int u = 0; u < 2; ++u)
        bw[u] =
            *reinterpret_cast<const uint4*>(s_w + r_row[u] * kPitch + jl * 16);
      uint4 av[T];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        asm volatile("" ::: "memory");
        av[t] = t < live ? *reinterpret_cast<const uint4*>(
                               s_a + (16 * t + l16) * kPitch + jl * 16)
                         : make_uint4(0U, 0U, 0U, 0U);
      }
#pragma unroll
      for (int part = 0; part < 2; ++part)
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (t >= live)
            continue;
          const i32x2 a =
              part == 0
                  ? i32x2{static_cast<int>(av[t].x), static_cast<int>(av[t].y)}
                  : i32x2{static_cast<int>(av[t].z), static_cast<int>(av[t].w)};
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const i32x2 b = part == 0 ? i32x2{static_cast<int>(bw[u].x),
                                              static_cast<int>(bw[u].y)}
                                      : i32x2{static_cast<int>(bw[u].z),
                                              static_cast<int>(bw[u].w)};
            i32x8 c = (jl == 0 && part == 0) ? i32x8{} : P[u][t];
            P[u][t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true,
                                                                 b, c, false);
          }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t >= live)
        continue;
      const float* sp = s_scl + 16 * t + 8 * half;
      const float4 s0 = *reinterpret_cast<const float4*>(sp);
      const float4 s1 = *reinterpret_cast<const float4*>(sp + 4);
      const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          F[u][t][e] =
              fmaf(static_cast<float>(P[u][t][e]), ws[u] * sv[e], F[u][t][e]);
    }
  };

  fetch(0);
  for (int s = 0; s < stages; ++s) {
    __syncthreads();
    commit();
    __syncthreads();
    if (s + 1 < stages)
      fetch(s + 1);
    compute();
  }

  // Epilogue: [slot][row] F16 through LDS, whole 16-byte stores.
  auto* stage = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (t >= live)
      break;  // block-uniform
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        stage[(2 * e + half) * kOutStride + 32 * wave + 16 * u + l16] =
            __float2half(F[u][t][e]);
    __syncthreads();
#pragma unroll
    for (int round = 0; round < 2; ++round) {
      const int flat = round * kThreads + tid;
      const int tr = flat >> 5;
      const int c8 = (flat & 31) * 8;
      const int tt = t_local + 16 * t + tr;
      if (tt < bucket_rows) {
        const std::int32_t dst = rows_out[bucket_begin + tt];
        if (dst >= 0)
          *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(dst) * m +
                                    r0 + c8) =
              *reinterpret_cast<const uint4*>(stage + tr * kOutStride + c8);
      }
    }
  }
}

}  // namespace

std::size_t A4Bytes(std::size_t rows, std::size_t k, int group) {
  const auto align = [](std::size_t n) { return (n + 255) / 256 * 256; };
  if (group <= 0)
    return 0;
  return align(rows * (k / 2)) + align(rows * (k / group) * 4) +
         align(rows * (k / 32) * 2);
}

A4Rows CarveA4(void* base, std::size_t rows, std::size_t k, int group) {
  const auto align = [](std::size_t n) { return (n + 255) / 256 * 256; };
  auto* p = static_cast<std::uint8_t*>(base);
  A4Rows q;
  q.group = group;
  q.k = k;
  q.codes = p;
  p += align(rows * (k / 2));
  q.scale = reinterpret_cast<float*>(p);
  p += align(rows * (k / group) * 4);
  q.sums = reinterpret_cast<__half*>(p);
  return q;
}

bool QuantizeA4(const __half* x, std::size_t rows, const A4Rows& q,
                hipStream_t stream) {
  if (rows == 0)
    return true;
  if (q.k == 0 || q.k % 256 != 0)
    return false;
  const dim3 grid(static_cast<unsigned>((rows + kWaves - 1) / kWaves));
  const int r = static_cast<int>(rows);
  const int k = static_cast<int>(q.k);
  switch (q.group) {
    case 64:
      hipLaunchKernelGGL(QuantizeA4Kernel<64>, grid, dim3(kThreads), 0, stream,
                         x, r, k, q.codes, q.scale, q.sums);
      return true;
    case 128:
      hipLaunchKernelGGL(QuantizeA4Kernel<128>, grid, dim3(kThreads), 0, stream,
                         x, r, k, q.codes, q.scale, q.sums);
      return true;
    case 256:
      hipLaunchKernelGGL(QuantizeA4Kernel<256>, grid, dim3(kThreads), 0, stream,
                         x, r, k, q.codes, q.scale, q.sums);
      return true;
    default:
      return false;
  }
}

bool GateUpIu4(const void* gate, const void* up, WeightType type,
               const A4Rows& q, const std::int32_t* tiles,
               std::uint32_t n_tiles, std::uint32_t tile_rows,
               const std::int32_t* pad_bounds, const std::int32_t* rows_in,
               const std::int32_t* rows_out, __half* out, const A4Rows* h4,
               std::size_t m, bool sym, hipStream_t stream) {
  if (type != WeightType::kQ4_K || tile_rows != 64 || n_tiles == 0 || m == 0 ||
      m % 128 != 0 || q.k == 0 || q.k % 256 != 0 || (sym && q.group == 256) ||
      (h4 != nullptr && (h4->group != 128 || h4->k != m)) ||
      ((h4 == nullptr) == (out == nullptr)))
    return false;
  const dim3 grid(static_cast<unsigned>(m / 128), n_tiles);
  const auto* g = static_cast<const std::uint8_t*>(gate);
  const auto* u = static_cast<const std::uint8_t*>(up);
  const int mi = static_cast<int>(m);
  const int k = static_cast<int>(q.k);
  const A4Rows hq = h4 != nullptr ? *h4 : A4Rows{};
  const auto launch = [&]<int G, bool S>() {
    if (h4 != nullptr)
      hipLaunchKernelGGL((GateUpIu4Kernel<G, true, S>), grid, dim3(kThreads), 0,
                         stream, g, u, q.codes, q.scale, q.sums, tiles,
                         pad_bounds, rows_in, rows_out, out, mi, k, hq);
    else
      hipLaunchKernelGGL((GateUpIu4Kernel<G, false, S>), grid, dim3(kThreads),
                         0, stream, g, u, q.codes, q.scale, q.sums, tiles,
                         pad_bounds, rows_in, rows_out, out, mi, k, hq);
    return true;
  };
  switch (q.group) {
    case 64:
      return sym ? launch.template operator()<64, true>()
                 : launch.template operator()<64, false>();
    case 128:
      return sym ? launch.template operator()<128, true>()
                 : launch.template operator()<128, false>();
    case 256:
      return launch.template operator()<256, false>();
    default:
      return false;
  }
}

bool RequantDown4(void* w, WeightType type, std::size_t rows, std::size_t k,
                  hipStream_t stream) {
  if (type != WeightType::kQ5_1 || rows == 0 || rows % 32 != 0 || k == 0 ||
      k % 128 != 0)
    return false;
  const std::size_t threads = rows * (k / 128);
  hipLaunchKernelGGL(
      RequantDown4Kernel,
      dim3(static_cast<unsigned>((threads + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, static_cast<std::uint8_t*>(w), rows,
      static_cast<int>(k));
  return true;
}

bool DownIu4(const void* w, WeightType type, const A4Rows& h,
             const std::int32_t* tiles, std::uint32_t n_tiles,
             std::uint32_t tile_rows, const std::int32_t* pad_bounds,
             const std::int32_t* rows_in, const std::int32_t* rows_out,
             __half* out, std::size_t m, hipStream_t stream) {
  if (type != WeightType::kQ5_1 || tile_rows != 64 || n_tiles == 0 || m == 0 ||
      m % 256 != 0 || h.group != 128 || h.k == 0 || h.k % 128 != 0 ||
      h.k > 32 * kDownMaxGroups)
    return false;
  const dim3 grid(static_cast<unsigned>(m / 256), n_tiles);
  hipLaunchKernelGGL(DownIu4Kernel, grid, dim3(kThreads), 0, stream,
                     static_cast<const std::uint8_t*>(w), h.codes, h.scale,
                     h.sums, tiles, pad_bounds, rows_in, rows_out, out,
                     static_cast<int>(m), static_cast<int>(h.k));
  return true;
}

bool RequantQ4KSym(void* w, WeightType type, std::size_t rows, std::size_t k,
                   int group, hipStream_t stream) {
  if (type != WeightType::kQ4_K || rows == 0 || rows % 32 != 0 || k == 0 ||
      k % 256 != 0 || (group != 64 && group != 128))
    return false;
  const std::size_t threads = rows * (k / 256);
  const dim3 grid(static_cast<unsigned>((threads + kThreads - 1) / kThreads));
  auto* p = static_cast<std::uint8_t*>(w);
  if (group == 64)
    hipLaunchKernelGGL(RequantQ4KSymKernel<64>, grid, dim3(kThreads), 0, stream,
                       p, rows, static_cast<int>(k));
  else
    hipLaunchKernelGGL(RequantQ4KSymKernel<128>, grid, dim3(kThreads), 0,
                       stream, p, rows, static_cast<int>(k));
  return true;
}

bool RequantDownSym(void* w, WeightType type, std::size_t rows, std::size_t k,
                    hipStream_t stream) {
  if (type != WeightType::kQ5_1 || rows == 0 || rows % 32 != 0 || k == 0 ||
      k % 128 != 0)
    return false;
  const std::size_t threads = rows * (k / 128);
  hipLaunchKernelGGL(
      RequantDownSymKernel,
      dim3(static_cast<unsigned>((threads + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, static_cast<std::uint8_t*>(w), rows,
      static_cast<int>(k));
  return true;
}

bool DownSym(const void* w, WeightType type, const A4Rows& h,
             const std::int32_t* tiles, std::uint32_t n_tiles,
             std::uint32_t tile_rows, const std::int32_t* pad_bounds,
             const std::int32_t* rows_in, const std::int32_t* rows_out,
             __half* out, std::size_t m, hipStream_t stream) {
  if (type != WeightType::kQ5_1 || tile_rows != 64 || n_tiles == 0 || m == 0 ||
      m % 256 != 0 || h.group != 128 || h.k == 0 || h.k % 128 != 0)
    return false;
  const dim3 grid(static_cast<unsigned>(m / 256), n_tiles);
  hipLaunchKernelGGL(DownSymKernel, grid, dim3(kThreads), 0, stream,
                     static_cast<const std::uint8_t*>(w), h.codes, h.scale,
                     tiles, pad_bounds, rows_in, rows_out, out,
                     static_cast<int>(m), static_cast<int>(h.k));
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::routed_prefill
