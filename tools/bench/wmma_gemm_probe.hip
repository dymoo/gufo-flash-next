// Dense WMMA GEMM probe for gfx1151: the same tiling, staging and epilogue
// structure for fp16, int8 (iu8) and int4 (iu4) operands, at Flash-Next's
// dense prefill shapes. It answers "what can a well-structured kernel of each
// type reach on this chip", independent of Gufo's Q8_0->F16 production kernel
// (compare against `qwen38_flash_next_batch_dense_bench <shard> prefill`).
//
// Build: nix develop -c tools/bench/build.sh wmma_gemm_probe
//        (or: hipcc -O3 --offload-arch=gfx1151 -std=c++20 wmma_gemm_probe.hip)
// Run:   /tmp/wmma_gemm_probe [check|shapes|sweep]
//
// Problem: Y[T][N] = sum over K of X[T][K] * W[N][K], with
//   X  activations, one scale per token (SX), A operand of the WMMA
//   W  weights, one scale per output channel (DW) and one per K group (SC),
//      B operand; the lane's accumulator column is one output channel, so a
//      group scale is one scalar per lane
//   fp16: no scales (reference ideal), fp32 accumulation across all of K
//   iu8/iu4: int32 WMMA accumulation per K group, folded into the running sum
//      with an integer scale (v_mad_i32_i24, 1 VALU per element per group,
//      Q4_K-style 6-bit group scales) or a float scale (cvt + fma, 2 VALU)
//
// Tiling: 256 threads (8 waves), 128 tokens x 128 channels per block, wave
// tile 64 tokens x 32 channels (4 x 2 WMMA tiles), 64-byte rows per K slab
// (32 fp16 / 64 int8 / 128 int4 elements), double-buffered LDS with global
// loads in flight during the matrix work, one barrier per slab.
//
// Device memory: at most the largest shape's weights x 4 rotating copies,
// bounded by --max-mib (default 1536).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x4 = __attribute__((__vector_size__(16))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
using f16x16 = __attribute__((__vector_size__(32))) _Float16;
using f32x8 = __attribute__((__vector_size__(32))) float;

enum class Kind { kF16, kIu8, kIu4 };

template<Kind K>
struct Tr;
template<>
struct Tr<Kind::kF16> {
  static constexpr int kSlabElems = 32;   // 64 B per row
  static constexpr int kPitch = 80;       // LDS row bytes (conflict-free b128)
  static constexpr int kFragBytes = 32;   // per lane per K=16 step
  static constexpr double kElemBytes = 2.0;
  using Frag = f16x16;
  using Acc = f32x8;
  __device__ static Frag Load(const std::uint8_t* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 16);
    Frag f;
    __builtin_memcpy(&f, &lo, 16);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16, &hi, 16);
    return f;
  }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  }
};
template<>
struct Tr<Kind::kIu8> {
  static constexpr int kSlabElems = 64;
  static constexpr int kPitch = 80;
  static constexpr int kFragBytes = 16;
  static constexpr double kElemBytes = 1.0;
  using Frag = i32x4;
  using Acc = i32x8;
  __device__ static Frag Load(const std::uint8_t* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    Frag f;
    __builtin_memcpy(&f, &v, 16);
    return f;
  }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
  }
};
template<>
struct Tr<Kind::kIu4> {
  static constexpr int kSlabElems = 128;
  static constexpr int kPitch = 72;       // 8-byte fragment reads, conflict-free
  static constexpr int kFragBytes = 8;
  static constexpr double kElemBytes = 0.5;
  using Frag = i32x2;
  using Acc = i32x8;
  __device__ static Frag Load(const std::uint8_t* p) {
    const uint2 v = *reinterpret_cast<const uint2*>(p);
    Frag f;
    __builtin_memcpy(&f, &v, 8);
    return f;
  }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, true);
  }
};

constexpr int kBM = 128, kBN = 128, kTI = 4, kTJ = 2, kThreads = 256;

// Scale modes for the K-group epilogue.
//   kNone : accumulate the whole K extent in the matrix accumulator
//   kInt  : weight group scale is a small integer (u8 in SC); int32 mad
//   kFloat: weight group scale is a float (SC holds floats); cvt + fma
enum class ScaleMode { kNone, kInt, kFloat };

__device__ __forceinline__ int Mad24(int a, int b, int c) {
  return __mul24(a, b) + c;
}

// SC layout: [slab][channel] 32-bit words. kInt: byte g = scale of group g in
// the slab. kFloat: [slab][channel][groups per slab] floats.
template<Kind KD, ScaleMode SM, int G, int DBG = 0>
__global__ __launch_bounds__(kThreads) void Gemm(
    const std::uint8_t* __restrict__ X, const std::uint8_t* __restrict__ W,
    const void* __restrict__ SC, const float* __restrict__ SX,
    const float* __restrict__ DW, float* __restrict__ Y, int T, int N, int Kd) {
  using TR = Tr<KD>;
  constexpr int kElems = TR::kSlabElems;
  constexpr int kKSteps = kElems / 16;
  constexpr int kPitch = TR::kPitch;
  constexpr int kGroup = SM == ScaleMode::kNone ? kElems : G;
  static_assert(kGroup <= kElems && kElems % kGroup == 0, "group within a slab");
  constexpr int kGps = kElems / kGroup;          // groups per slab
  constexpr int kStepsPerGroup = kKSteps / kGps;  // K=16 WMMA steps per group
  constexpr double kBytesPerElem = TR::kElemBytes;

  __shared__ __attribute__((aligned(16))) std::uint8_t s_ab[2][2][kBM * kPitch];
  __shared__ std::uint32_t s_sc[2][kBN * (SM == ScaleMode::kFloat ? kGps : 1)];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int wave_t = wave >> 2;  // 0..1 along tokens
  const int wave_n = wave & 3;   // 0..3 along channels
  const int tok0 = blockIdx.x * kBM;
  const int ch0 = blockIdx.y * kBN;
  const std::size_t row_bytes = static_cast<std::size_t>(Kd * kBytesPerElem);
  const int nslab = Kd / kElems;

  // Global->register staging: 512 16-byte chunks per operand per slab.
  const std::uint8_t* xsrc[2];
  const std::uint8_t* wsrc[2];
  int sdst[2];
  for (int p = 0; p < 2; ++p) {
    const int c = tid + p * kThreads;
    const int row = c >> 2, part = c & 3;
    xsrc[p] = X + static_cast<std::size_t>(tok0 + row) * row_bytes + part * 16;
    wsrc[p] = W + static_cast<std::size_t>(ch0 + row) * row_bytes + part * 16;
    sdst[p] = row * kPitch + part * 16;
  }
  uint4 xr[2], wr[2];
  std::uint32_t scr[kGps];
  const auto load_global = [&](int slab) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      xr[p] = *reinterpret_cast<const uint4*>(xsrc[p] + slab * 64);
      wr[p] = *reinterpret_cast<const uint4*>(wsrc[p] + slab * 64);
    }
    if constexpr (SM == ScaleMode::kInt) {
      if (tid < kBN)
        scr[0] = reinterpret_cast<const std::uint32_t*>(SC)
                     [static_cast<std::size_t>(slab) * N + ch0 + tid];
    } else if constexpr (SM == ScaleMode::kFloat) {
      if (tid < kBN) {
#pragma unroll
        for (int g = 0; g < kGps; ++g)
          scr[g] = reinterpret_cast<const std::uint32_t*>(SC)
                       [(static_cast<std::size_t>(slab) * N + ch0 + tid) * kGps + g];
      }
    }
  };
  const auto store_lds = [&](int buf) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      std::uint8_t* a = &s_ab[buf][0][sdst[p]];
      std::uint8_t* b = &s_ab[buf][1][sdst[p]];
      if constexpr (kPitch % 16 == 0) {
        *reinterpret_cast<uint4*>(a) = xr[p];
        *reinterpret_cast<uint4*>(b) = wr[p];
      } else {
        *reinterpret_cast<uint2*>(a) = make_uint2(xr[p].x, xr[p].y);
        *reinterpret_cast<uint2*>(a + 8) = make_uint2(xr[p].z, xr[p].w);
        *reinterpret_cast<uint2*>(b) = make_uint2(wr[p].x, wr[p].y);
        *reinterpret_cast<uint2*>(b + 8) = make_uint2(wr[p].z, wr[p].w);
      }
    }
    if constexpr (SM == ScaleMode::kInt) {
      if (tid < kBN)
        s_sc[buf][tid] = scr[0];
    } else if constexpr (SM == ScaleMode::kFloat) {
      if (tid < kBN) {
#pragma unroll
        for (int g = 0; g < kGps; ++g)
          s_sc[buf][tid * kGps + g] = scr[g];
      }
    }
  };

  typename TR::Acc acc[kTI][kTJ];
  int ti[kTI][kTJ][8];
  float tf[kTI][kTJ][8];
#pragma unroll
  for (int i = 0; i < kTI; ++i) {
#pragma unroll
    for (int j = 0; j < kTJ; ++j) {
      acc[i][j] = typename TR::Acc{};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        ti[i][j][e] = 0;
        tf[i][j][e] = 0.0F;
      }
    }
  }

  const int a_row = wave_t * 64 + (lane & 15);
  const int b_row = wave_n * 32 + (lane & 15);
  const int b_ch[kTJ] = {wave_n * 32 + (lane & 15), wave_n * 32 + 16 + (lane & 15)};

  load_global(0);
  store_lds(0);
  __syncthreads();
  for (int slab = 0; slab < nslab; ++slab) {
    const int cur = slab & 1;
    if (slab + 1 < nslab)
      load_global((DBG & 4) ? 0 : slab + 1);
    const std::uint8_t* la = s_ab[cur][0];
    const std::uint8_t* lb = s_ab[cur][1];
    std::uint32_t scw[kTJ] = {0, 0};
    if constexpr (SM == ScaleMode::kInt) {
#pragma unroll
      for (int j = 0; j < kTJ; ++j)
        scw[j] = s_sc[cur][b_ch[j]];
    }
#pragma unroll
    for (int g = 0; g < kGps; ++g) {
#pragma unroll
      for (int s = 0; s < kStepsPerGroup; ++s) {
        const int ks = g * kStepsPerGroup + s;
        typename TR::Frag a[kTI], b[kTJ];
#pragma unroll
        for (int i = 0; i < kTI; ++i)
          a[i] = TR::Load(la + (a_row + i * 16) * kPitch + ks * TR::kFragBytes);
#pragma unroll
        for (int j = 0; j < kTJ; ++j)
          b[j] = TR::Load(lb + (b_row + j * 16) * kPitch + ks * TR::kFragBytes);
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
#pragma unroll
          for (int j = 0; j < kTJ; ++j) {
            if constexpr (SM == ScaleMode::kNone)
              acc[i][j] = TR::Mma(a[i], b[j], acc[i][j]);
            else
              acc[i][j] = TR::Mma(a[i], b[j], s == 0 ? typename TR::Acc{} : acc[i][j]);
          }
        }
      }
      if constexpr (SM == ScaleMode::kInt) {
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const int sc = static_cast<int>((scw[j] >> (8 * g)) & 0xFFU);
#pragma unroll
          for (int i = 0; i < kTI; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
              ti[i][j][e] = Mad24(sc, acc[i][j][e], ti[i][j][e]);
          }
        }
      } else if constexpr (SM == ScaleMode::kFloat) {
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const float sc = __builtin_bit_cast(float, s_sc[cur][b_ch[j] * kGps + g]);
#pragma unroll
          for (int i = 0; i < kTI; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
              tf[i][j][e] = __builtin_fmaf(static_cast<float>(acc[i][j][e]), sc, tf[i][j][e]);
          }
        }
      }
    }
    if (slab + 1 < nslab)
      store_lds(cur ^ 1);
    __syncthreads();
  }

  // acc[i][j][e] <-> D[token 16*i + 2e + (lane>>4)][channel 16*j + (lane&15)]
  const int half = lane >> 4;
#pragma unroll
  for (int j = 0; j < kTJ; ++j) {
    const int ch = ch0 + b_ch[j];
    const float dw = KD == Kind::kF16 ? 1.0F : DW[ch];
#pragma unroll
    for (int i = 0; i < kTI; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int tok = tok0 + wave_t * 64 + i * 16 + 2 * e + half;
        float v;
        if constexpr (KD == Kind::kF16)
          v = acc[i][j][e];
        else if constexpr (SM == ScaleMode::kInt)
          v = static_cast<float>(ti[i][j][e]) * (SX[tok] * dw);
        else if constexpr (SM == ScaleMode::kFloat)
          v = tf[i][j][e] * SX[tok];
        else
          v = static_cast<float>(acc[i][j][e]) * (SX[tok] * dw);
        if constexpr (DBG & 1) {
          if (v == 12345.678F)
            Y[static_cast<std::size_t>(tok) * N + ch] = v;  // keep the math alive, no traffic
        } else if constexpr (DBG & 2) {
          // tile-major: each block writes one contiguous 128x128 tile
          const std::size_t tile = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
          Y[tile * (kBM * kBN) + static_cast<std::size_t>(tok - tok0) * kBN + (ch - ch0)] = v;
        } else {
          Y[static_cast<std::size_t>(tok) * N + ch] = v;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- host side

struct Shape {
  const char* name;
  int n, k;  // output channels, reduction
};

// Flash-Next dense prefill projections (hidden 2560, 6144 SSM inner, ...).
// hc_down (320 outputs) and hc_up (K=320) are not 128-multiples; the probe
// skips them (together about 7% of a d0 prefill).
static const Shape kShapes[] = {
    {"ssm_in", 16384, 2560}, {"attn_qkv", 13312, 2560}, {"ssm_out", 2560, 6144},
    {"attn_out", 2560, 6144}, {"shexp_gate_up", 1280, 2560}, {"shexp_down", 2560, 640},
};

struct Buf {
  void* p = nullptr;
  std::size_t bytes = 0;
};

static std::uint32_t g_rng = 12345;
static std::uint32_t Rng() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

static std::uint16_t F2H(float f) {  // round-to-nearest-even, normal range
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000U;
  int exp = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
  std::uint32_t man = u & 0x7FFFFFU;
  if (exp <= 0)
    return static_cast<std::uint16_t>(sign);
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  std::uint32_t h = (static_cast<std::uint32_t>(exp) << 10) | (man >> 13);
  const std::uint32_t rem = man & 0x1FFFU;
  if (rem > 0x1000U || (rem == 0x1000U && (h & 1U)))
    ++h;
  return static_cast<std::uint16_t>(sign | h);
}
static float H2F(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000U) << 16;
  const int exp = (h >> 10) & 0x1F;
  const std::uint32_t man = h & 0x3FFU;
  std::uint32_t u;
  if (exp == 0)
    u = sign;
  else
    u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Operand generation: values, packed bytes, and a float view for the CPU check.
struct Operand {
  std::vector<std::uint8_t> bytes;
  std::vector<float> val;  // [rows][k] as floats
};
static Operand MakeOperand(Kind kd, int rows, int k, bool keep_val) {
  Operand o;
  const std::size_t n = static_cast<std::size_t>(rows) * k;
  if (keep_val)
    o.val.resize(n);
  if (kd == Kind::kF16) {
    o.bytes.resize(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
      const float f = static_cast<float>(static_cast<int>(Rng() & 0x3FFU) - 512) / 512.0F;
      const std::uint16_t h = F2H(f);
      if (keep_val)
        o.val[i] = H2F(h);
      std::memcpy(&o.bytes[i * 2], &h, 2);
    }
  } else if (kd == Kind::kIu8) {
    o.bytes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const int v = static_cast<int>(Rng() & 0xFFU) - 128;
      if (keep_val)
        o.val[i] = static_cast<float>(v);
      o.bytes[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(v));
    }
  } else {
    o.bytes.assign(n / 2, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const int v = static_cast<int>(Rng() & 0xFU) - 8;
      if (keep_val)
        o.val[i] = static_cast<float>(v);
      const std::uint8_t nib = static_cast<std::uint8_t>(v) & 0xFU;
      o.bytes[i / 2] |= static_cast<std::uint8_t>(i & 1U ? nib << 4 : nib);
    }
  }
  return o;
}

struct Problem {
  Kind kd;
  ScaleMode sm;
  int group;  // elements per scale group (0 = none)
  int T, N, K;
};

static const char* KindName(Kind k) {
  return k == Kind::kF16 ? "f16" : k == Kind::kIu8 ? "iu8" : "iu4";
}

static int g_dbg = 0;  // custom-mode experiments: 1 no stores, 2 tile-major stores, 4 no loads

template<Kind KD, ScaleMode SM, int G>
static void Launch(const Problem& p, const std::uint8_t* x, const std::uint8_t* w,
                   const void* sc, const float* sx, const float* dw, float* y,
                   hipStream_t s) {
  dim3 grid(p.T / kBM, p.N / kBN);
  if constexpr (KD == Kind::kIu4 && SM == ScaleMode::kInt) {
    switch (g_dbg) {
      case 1: Gemm<KD, SM, G, 1><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K); return;
      case 2: Gemm<KD, SM, G, 2><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K); return;
      case 4: Gemm<KD, SM, G, 4><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K); return;
      case 5: Gemm<KD, SM, G, 5><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K); return;
      case 6: Gemm<KD, SM, G, 6><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K); return;
      default: break;
    }
  }
  Gemm<KD, SM, G><<<grid, kThreads, 0, s>>>(x, w, sc, sx, dw, y, p.T, p.N, p.K);
}

static bool Dispatch(const Problem& p, const std::uint8_t* x, const std::uint8_t* w,
                     const void* sc, const float* sx, const float* dw, float* y,
                     hipStream_t s) {
  if (p.kd == Kind::kF16 && p.sm == ScaleMode::kNone)
    return Launch<Kind::kF16, ScaleMode::kNone, 0>(p, x, w, sc, sx, dw, y, s), true;
  if (p.kd == Kind::kIu8) {
    if (p.sm == ScaleMode::kNone)
      return Launch<Kind::kIu8, ScaleMode::kNone, 0>(p, x, w, sc, sx, dw, y, s), true;
    if (p.sm == ScaleMode::kInt && p.group == 32)
      return Launch<Kind::kIu8, ScaleMode::kInt, 32>(p, x, w, sc, sx, dw, y, s), true;
    if (p.sm == ScaleMode::kFloat && p.group == 32)
      return Launch<Kind::kIu8, ScaleMode::kFloat, 32>(p, x, w, sc, sx, dw, y, s), true;
  }
  if (p.kd == Kind::kIu4) {
    if (p.sm == ScaleMode::kNone)
      return Launch<Kind::kIu4, ScaleMode::kNone, 0>(p, x, w, sc, sx, dw, y, s), true;
    if (p.sm == ScaleMode::kInt) {
      if (p.group == 32)
        return Launch<Kind::kIu4, ScaleMode::kInt, 32>(p, x, w, sc, sx, dw, y, s), true;
      if (p.group == 64)
        return Launch<Kind::kIu4, ScaleMode::kInt, 64>(p, x, w, sc, sx, dw, y, s), true;
      if (p.group == 128)
        return Launch<Kind::kIu4, ScaleMode::kInt, 128>(p, x, w, sc, sx, dw, y, s), true;
    }
    if (p.sm == ScaleMode::kFloat && p.group == 32)
      return Launch<Kind::kIu4, ScaleMode::kFloat, 32>(p, x, w, sc, sx, dw, y, s), true;
  }
  return false;
}

static int Elems(Kind kd) { return kd == Kind::kF16 ? 32 : kd == Kind::kIu8 ? 64 : 128; }

// Scale arrays. Integer scales are 1..63; float scales are their product with
// a per-row factor folded into DW (so the CPU check is the same).
struct Scales {
  std::vector<std::uint8_t> sc_i;  // [N][K/G] integer scales (CPU view)
  std::vector<std::uint32_t> dev_i;  // [slab][N] packed bytes
  std::vector<float> dev_f;          // [slab][N][gps]
  std::vector<float> dw, sx;
};
static Scales MakeScales(const Problem& p) {
  Scales s;
  s.dw.resize(p.N);
  s.sx.resize(p.T);
  for (auto& v : s.dw)
    v = 0.001F * static_cast<float>(1 + (Rng() & 15U));
  for (auto& v : s.sx)
    v = 0.01F * static_cast<float>(1 + (Rng() & 7U));
  if (p.sm == ScaleMode::kNone)
    return s;
  const int ng = p.K / p.group;
  const int elems = Elems(p.kd);
  const int gps = elems / p.group;
  const int nslab = p.K / elems;
  s.sc_i.resize(static_cast<std::size_t>(p.N) * ng);
  for (auto& v : s.sc_i)
    v = static_cast<std::uint8_t>(1 + (Rng() % 63U));
  if (p.sm == ScaleMode::kInt) {
    s.dev_i.assign(static_cast<std::size_t>(nslab) * p.N, 0);
    for (int n = 0; n < p.N; ++n)
      for (int g = 0; g < ng; ++g)
        s.dev_i[static_cast<std::size_t>(g / gps) * p.N + n] |=
            static_cast<std::uint32_t>(s.sc_i[static_cast<std::size_t>(n) * ng + g])
            << (8 * (g % gps));
  } else {
    s.dev_f.resize(static_cast<std::size_t>(nslab) * p.N * gps);
    for (int n = 0; n < p.N; ++n)
      for (int g = 0; g < ng; ++g)
        s.dev_f[(static_cast<std::size_t>(g / gps) * p.N + n) * gps + (g % gps)] =
            static_cast<float>(s.sc_i[static_cast<std::size_t>(n) * ng + g]) * s.dw[n];
  }
  return s;
}

// CPU reference on sampled outputs, in double.
static double CheckSamples(const Problem& p, const Operand& xo, const Operand& wo,
                           const Scales& sc, const std::vector<float>& y) {
  double max_rel = 0, rms = 0;
  const int samples = 256;
  std::vector<double> ref(samples);
  std::vector<int> ts(samples), ns(samples);
  for (int i = 0; i < samples; ++i) {
    ts[i] = static_cast<int>((i * 2654435761U) % static_cast<unsigned>(p.T));
    ns[i] = static_cast<int>((i * 40503U + 7) % static_cast<unsigned>(p.N));
    const float* xr = &xo.val[static_cast<std::size_t>(ts[i]) * p.K];
    const float* wr = &wo.val[static_cast<std::size_t>(ns[i]) * p.K];
    double acc = 0;
    if (p.sm == ScaleMode::kNone) {
      for (int k = 0; k < p.K; ++k)
        acc += static_cast<double>(xr[k]) * wr[k];
      if (p.kd != Kind::kF16)
        acc *= static_cast<double>(sc.sx[ts[i]]) * sc.dw[ns[i]];
    } else {
      const int ng = p.K / p.group;
      for (int g = 0; g < ng; ++g) {
        double part = 0;
        for (int k = g * p.group; k < (g + 1) * p.group; ++k)
          part += static_cast<double>(xr[k]) * wr[k];
        acc += part * sc.sc_i[static_cast<std::size_t>(ns[i]) * ng + g];
      }
      acc *= static_cast<double>(sc.sx[ts[i]]) * sc.dw[ns[i]];
    }
    ref[i] = acc;
    rms += acc * acc;
  }
  rms = std::sqrt(rms / samples);
  for (int i = 0; i < samples; ++i) {
    const double got = y[static_cast<std::size_t>(ts[i]) * p.N + ns[i]];
    max_rel = std::max(max_rel, std::abs(got - ref[i]) / (rms + 1e-30));
  }
  return max_rel;
}

struct Result {
  double us, tops;
  double max_err;  // relative to rms; <0 when not checked
};

static std::shared_ptr<const Operand> GetOperand(Kind kd, int rows, int k, bool keep_val) {
  struct Entry {
    Kind kd;
    int rows, k;
    std::shared_ptr<const Operand> op;
  };
  static std::vector<Entry> cache;
  if (keep_val)  // checks need the float view and are never cached
    return std::make_shared<const Operand>(MakeOperand(kd, rows, k, true));
  for (const auto& e : cache)
    if (e.kd == kd && e.rows == rows && e.k == k)
      return e.op;
  if (cache.size() >= 3)
    cache.erase(cache.begin());
  cache.push_back({kd, rows, k, std::make_shared<const Operand>(MakeOperand(kd, rows, k, false))});
  return cache.back().op;
}

// One arena for the whole process: fresh hipMalloc buffers per run changed
// placement (and, on this UMA part, TLB behaviour) enough to move a kernel by
// 10x, so every run reuses the same device buffers.
struct Arena {
  void* x = nullptr;
  void* w[4] = {};
  void* sc[4] = {};
  float* sx = nullptr;
  float* dw = nullptr;
  float* y = nullptr;
  bool ready = false;
  void Init() {
    if (ready)
      return;
    ready = true;
    HIP_OK(hipMalloc(&x, std::size_t{4096} * 10240 * 2));
    for (int c = 0; c < 4; ++c) {
      HIP_OK(hipMalloc(&w[c], std::size_t{16384} * 2560 * 2));
      HIP_OK(hipMalloc(&sc[c], std::size_t{16384} * 2560 / 32 * 4 + 4096));
    }
    HIP_OK(hipMalloc(&sx, 4096 * 4));
    HIP_OK(hipMalloc(&dw, 16384 * 4));
    HIP_OK(hipMalloc(&y, std::size_t{4096} * 16384 * 4));
  }
};
static Arena g_arena;

static Result RunProblem(const Problem& p, int copies, bool check, hipStream_t s) {
  g_arena.Init();
  const auto xo_ptr = GetOperand(p.kd, p.T, p.K, check);
  const auto wo_ptr = GetOperand(p.kd, p.N, p.K, check);
  const Operand& xo = *xo_ptr;
  const Operand& wo = *wo_ptr;
  const Scales sc = MakeScales(p);
  auto* dx = static_cast<std::uint8_t*>(g_arena.x);
  float* dsx = g_arena.sx;
  float* ddw = g_arena.dw;
  float* dy = g_arena.y;
  std::uint8_t* dw_bytes[4];
  void* dsc[4] = {};
  HIP_OK(hipMemcpy(dx, xo.bytes.data(), xo.bytes.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dsx, sc.sx.data(), sc.sx.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(ddw, sc.dw.data(), sc.dw.size() * 4, hipMemcpyHostToDevice));
  copies = std::min(copies, 4);
  for (int c = 0; c < copies; ++c) {
    dw_bytes[c] = static_cast<std::uint8_t*>(g_arena.w[c]);
    HIP_OK(hipMemcpy(dw_bytes[c], wo.bytes.data(), wo.bytes.size(), hipMemcpyHostToDevice));
    dsc[c] = g_arena.sc[c];
    if (p.sm == ScaleMode::kInt)
      HIP_OK(hipMemcpy(dsc[c], sc.dev_i.data(), sc.dev_i.size() * 4, hipMemcpyHostToDevice));
    else if (p.sm == ScaleMode::kFloat)
      HIP_OK(hipMemcpy(dsc[c], sc.dev_f.data(), sc.dev_f.size() * 4, hipMemcpyHostToDevice));
  }
  Result r{0, 0, -1};
  if (!Dispatch(p, dx, dw_bytes[0], dsc[0], dsx, ddw, dy, s)) {
    r.us = -1;
    return r;
  }
  HIP_OK(hipStreamSynchronize(s));
  if (check) {
    std::vector<float> y(static_cast<std::size_t>(p.T) * p.N);
    HIP_OK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
    r.max_err = CheckSamples(p, xo, wo, sc, y);
  }
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  std::vector<double> us;
  for (int i = 0; i < 4; ++i)  // warm-up: clocks and TLBs
    Dispatch(p, dx, dw_bytes[i % copies], dsc[i % copies], dsx, ddw, dy, s);
  HIP_OK(hipStreamSynchronize(s));
  for (int rep = 0; rep < 7; ++rep) {
    HIP_OK(hipEventRecord(e0, s));
    for (int i = 0; i < 8; ++i)
      Dispatch(p, dx, dw_bytes[i % copies], dsc[i % copies], dsx, ddw, dy, s);
    HIP_OK(hipEventRecord(e1, s));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    us.push_back(1000.0 * ms / 8);
  }
  std::sort(us.begin(), us.end());
  r.us = us[3];
  r.tops = 2.0 * p.T * p.N * p.K / r.us / 1e6;
  HIP_OK(hipEventDestroy(e0));
  HIP_OK(hipEventDestroy(e1));
  return r;
}

static double ReadSys(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f)
    return -1;
  double v = -1;
  if (std::fscanf(f, "%lf", &v) != 1)
    v = -1;
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "shapes";
  double max_mib = 1536;
  for (int i = 2; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--max-mib")
      max_mib = std::atof(argv[i + 1]);
  hipStream_t s;
  HIP_OK(hipStreamCreate(&s));
  std::string hw;
  for (int i = 0; i < 16 && hw.empty(); ++i) {
    char p[96];
    std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/name", i);
    if (FILE* f = std::fopen(p, "r")) {
      char n[32] = {};
      if (std::fgets(n, sizeof n, f) && std::strncmp(n, "amdgpu", 6) == 0) {
        std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/", i);
        hw = p;
      }
      std::fclose(f);
    }
  }
  const auto t_start = std::chrono::steady_clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  };

  struct Variant {
    Kind kd;
    ScaleMode sm;
    int group;
    const char* label;
  };
  const Variant variants[] = {
      {Kind::kF16, ScaleMode::kNone, 0, "f16 ideal"},
      {Kind::kIu8, ScaleMode::kNone, 0, "iu8 no-group-scale"},
      {Kind::kIu8, ScaleMode::kFloat, 32, "iu8 g32 float"},
      {Kind::kIu8, ScaleMode::kInt, 32, "iu8 g32 int"},
      {Kind::kIu4, ScaleMode::kNone, 0, "iu4 no-group-scale"},
      {Kind::kIu4, ScaleMode::kFloat, 32, "iu4 g32 float"},
      {Kind::kIu4, ScaleMode::kInt, 32, "iu4 g32 int"},
      {Kind::kIu4, ScaleMode::kInt, 64, "iu4 g64 int"},
      {Kind::kIu4, ScaleMode::kInt, 128, "iu4 g128 int"},
  };

  if (mode == "check") {
    std::printf("== correctness (T=256 N=256 K=1024; max |err|/rms over 256 samples) ==\n");
    int bad = 0;
    for (const auto& v : variants) {
      const Problem p{v.kd, v.sm, v.group, 256, 256, 1024};
      const auto r = RunProblem(p, 1, true, s);
      const bool ok = r.max_err >= 0 && r.max_err < (v.kd == Kind::kF16 ? 2e-3 : 2e-5);
      bad += ok ? 0 : 1;
      std::printf("  %-20s max_err/rms %.2e  %s\n", v.label, r.max_err, ok ? "ok" : "FAIL");
    }
    return bad == 0 ? 0 : 1;
  }

  if (mode == "custom") {  // custom VARIANT_INDEX DBG T N K [T N K ...]
    const int vi = std::atoi(argv[2]);
    g_dbg = std::atoi(argv[3]);
    std::printf("shape,N,K,T,variant,us,TOPS\n");
    for (int a = 4; a + 2 < argc; a += 3) {
      const int t = std::atoi(argv[a]), n = std::atoi(argv[a + 1]), k = std::atoi(argv[a + 2]);
      const auto& v = variants[vi];
      const Problem p{v.kd, v.sm, v.group, t, n, k};
      if (k % Elems(v.kd) || n % kBN || t % kBM) {
        std::printf("# skipped %d %d %d (not a multiple of the tile)\n", t, n, k);
        continue;
      }
      const double wbytes = static_cast<double>(n) * k * (v.kd == Kind::kF16 ? 2.0 : v.kd == Kind::kIu8 ? 1.0 : 0.5);
      const int copies = std::max(1, std::min(4, static_cast<int>(max_mib * 1048576.0 / (wbytes * 1.25 + t * n * 4.0))));
      const auto r = RunProblem(p, copies, false, s);
      std::printf("custom,%d,%d,%d,%s,%.1f,%.2f\n", n, k, t, v.label, r.us, r.tops);
    }
    return 0;
  }
  const std::vector<int> ts = mode == "sweep" ? std::vector<int>{256, 512, 1024, 2048, 4096}
                                              : std::vector<int>{2048};
  std::printf("shape,N,K,T,variant,us,TOPS\n");
  for (const auto& sh : kShapes) {
    for (const int t : ts) {
      for (const auto& v : variants) {
        if (elapsed() > 50.0) {
          std::printf("# deadline reached\n");
          return 0;
        }
        const Problem p{v.kd, v.sm, v.group, t, sh.n, sh.k};
        if (sh.k % Elems(v.kd) != 0 || sh.n % kBN != 0 || t % kBM != 0)
          continue;
        const double wbytes = static_cast<double>(sh.n) * sh.k * Tr<Kind::kIu8>::kElemBytes *
                              (v.kd == Kind::kF16 ? 2.0 : v.kd == Kind::kIu8 ? 1.0 : 0.5);
        const int copies = std::max(1, std::min(4, static_cast<int>(max_mib * 1048576.0 /
                                                                    (wbytes * 1.25 + t * sh.n * 4.0))));
        const auto r = RunProblem(p, copies, false, s);
        std::printf("%s,%d,%d,%d,%s,%.1f,%.2f\n", sh.name, sh.n, sh.k, t, v.label, r.us, r.tops);
        std::fflush(stdout);
      }
    }
  }
  if (!hw.empty())
    std::printf("# sclk %.0f MHz, %.0f W, %.0f C at exit\n", ReadSys(hw + "freq1_input") / 1e6,
                ReadSys(hw + "power1_average") / 1e6, ReadSys(hw + "temp1_input") / 1e3);
  return 0;
}
