// Routed expert FFNs for decode and verification rows (see moe.hpp).
//
// A plan launch buckets the (row, slot) assignments by expert and quantizes
// the token rows. The gate/up launch streams each routed expert's gate and
// up rows once for all of its token rows and writes the SwiGLU result as Q8;
// the down launch does the same for the down rows. A work-group stages its
// token rows in LDS and keeps the next weight step in flight while it
// computes; one lane owns one weight row. Integer 32-element dot products are
// exact and every FP32 operation after them is explicitly rounded in a fixed
// order, so a (row, expert) result is independent of batch width, position
// and grouping.
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <type_traits>

#include "src/models/qwen38_flash_next/kernels/rocm/iq.hpp"

// Benchmark hooks around each expert work item (empty in production).
#ifndef MOE_TRACE_BEGIN
#define MOE_TRACE_BEGIN(kind)
#define MOE_TRACE_END(kind)
#endif

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr int kMaxExperts = 512;
constexpr int kXBlocks = kHidden / 32;  // Q8 blocks per token row
constexpr int kHBlocks = kFf / 32;      // Q8 blocks per SwiGLU row
constexpr int kSuper = kHidden / 256;   // K-quant super-blocks per row
/// Token rows per work item: an expert with more rows gets more items.
constexpr int kGroup = 16;
/// Output rows per gate/up work-group (two gate waves, two up waves) and per
/// down work-group (four waves).
constexpr int kGateRows = 64;
constexpr int kDownRows = 128;

using V4 = __attribute__((ext_vector_type(4))) unsigned;

/// Q8 activation block: F16 scale, the exact sum of its 32 codes (for the
/// weight offsets: rounding errors of the codes then cancel against the
/// offset as in the dequantized dot product), 32 codes.
struct Q8Block {
  __half d;
  short sum;
  int qs[8];
};
static_assert(sizeof(Q8Block) == 36);

/// An activation block's codes in LDS.
struct Act {
  int4 q[2];
};

std::size_t Align(std::size_t n) {
  return (n + 255) / 256 * 256;
}

int MaxGroups(std::uint32_t rows, std::uint32_t used, std::uint32_t experts) {
  const int slots = static_cast<int>(rows * used);
  return std::min<int>(slots, static_cast<int>(experts)) +
         (slots + kGroup - 1) / kGroup;
}

struct Layout {
  Q8Block* xq;
  Q8Block* hq;
  int* entries;
  int4* groups;
  int* n_groups;
};

/// `h_blocks`: Q8 blocks per SwiGLU row, the instantiated FF / 32.
Layout Carve(void* scratch, std::uint32_t rows, std::uint32_t used,
             int h_blocks = kHBlocks) {
  auto* p = static_cast<std::uint8_t*>(scratch);
  Layout l{};
  l.xq = reinterpret_cast<Q8Block*>(p);
  p += Align(std::size_t{rows} * kXBlocks * sizeof(Q8Block));
  l.hq = reinterpret_cast<Q8Block*>(p);
  p += Align(std::size_t{rows} * used * h_blocks * sizeof(Q8Block));
  l.entries = reinterpret_cast<int*>(p);
  p += Align(std::size_t{rows} * used * sizeof(int));
  l.groups = reinterpret_cast<int4*>(p);
  p += Align(std::size_t(MaxGroups(rows, used, kMaxExperts)) * sizeof(int4));
  l.n_groups = reinterpret_cast<int*>(p);
  return l;
}

__device__ __forceinline__ int Dot8(const int* w, int4 x0, int4 x1) {
  int s = __builtin_amdgcn_sudot4(true, w[0], true, x0.x, 0, false);
  s = __builtin_amdgcn_sudot4(true, w[1], true, x0.y, s, false);
  s = __builtin_amdgcn_sudot4(true, w[2], true, x0.z, s, false);
  s = __builtin_amdgcn_sudot4(true, w[3], true, x0.w, s, false);
  s = __builtin_amdgcn_sudot4(true, w[4], true, x1.x, s, false);
  s = __builtin_amdgcn_sudot4(true, w[5], true, x1.y, s, false);
  s = __builtin_amdgcn_sudot4(true, w[6], true, x1.z, s, false);
  return __builtin_amdgcn_sudot4(true, w[7], true, x1.w, s, false);
}

__device__ __forceinline__ float HalfLo(unsigned v) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(v)));
}
__device__ __forceinline__ float HalfHi(unsigned v) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(v >> 16)));
}
__device__ __forceinline__ unsigned Byte(unsigned v, int i) {
  return (v >> (8 * i)) & 0xFFu;
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int m = 16; m; m >>= 1)
    v = fmaxf(v, __shfl_xor(v, m, 32));
  return v;
}

/// Scale and exact offset operand (scale * code sum) of an activation block.
__device__ __forceinline__ float2 Operands(const Q8Block& b) {
  const float d = __half2float(b.d);
  return make_float2(d, __fmul_rn(d, float(b.sum)));
}

/// Copies a global activation block into LDS (codes, operands).
__device__ __forceinline__ void StoreAct(const Q8Block& b, Act* q, float2* o) {
  q->q[0] = make_int4(b.qs[0], b.qs[1], b.qs[2], b.qs[3]);
  q->q[1] = make_int4(b.qs[4], b.qs[5], b.qs[6], b.qs[7]);
  *o = Operands(b);
}

/// One wave writes a 32-value Q8 block, one value per lane.
__device__ __forceinline__ void QuantizeLane(float v, Q8Block* out, int lane) {
  const float amax = WaveMax(fabsf(v));
  const float d = __fdiv_rn(amax, 127.0f);
  const int q = amax == 0.0f ? 0 : static_cast<int>(roundf(__fdiv_rn(v, d)));
  int sum = q;
#pragma unroll
  for (int m = 16; m; m >>= 1)
    sum += __shfl_xor(sum, m, 32);
  reinterpret_cast<std::int8_t*>(out->qs)[lane] = static_cast<std::int8_t>(q);
  if (lane == 0) {
    out->d = __float2half_rn(d);
    out->sum = static_cast<short>(sum);
  }
}

/// Spike (GUFO_SPIKE_ROUTED_HAD): rotates Q8 rows in place by spike.hip.cpp's
/// BlockHadamard(128) (sign vector on the 128-block index, Sylvester-order
/// butterflies, 1/sqrt(128)) and requantizes them like QuantizeLane. One
/// wave per 128 values (four Q8 blocks), four consecutive values per lane.
__global__ __launch_bounds__(256) void RotateQ8Kernel(Q8Block* __restrict__ q,
                                                      std::size_t groups) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
  if (g >= groups)
    return;  // wave-uniform
  const int lane = static_cast<int>(threadIdx.x) & 31;
  Q8Block* blk = q + g * 4 + (lane >> 3);
  const int word = blk->qs[lane & 7];
  const float d = __half2float(blk->d);
  float v[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const unsigned h = static_cast<unsigned>(4 * lane + e) * 2654435761U;
    v[e] = d * static_cast<float>(static_cast<std::int8_t>(word >> (8 * e))) *
           (((h >> 29) & 1U) != 0U ? -1.0f : 1.0f);
  }
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[2] = a0 - a2;
  v[1] = a1 + a3;
  v[3] = a1 - a3;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool upper = (lane & m) != 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float other = __shfl_xor(v[e], m, 32);
      v[e] = upper ? other - v[e] : v[e] + other;
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    v[e] *= 0.08838834764831845f;  // 1/sqrt(128)
    amax = fmaxf(amax, fabsf(v[e]));
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1)
    amax = fmaxf(amax, __shfl_xor(amax, m, 32));
  const float nd = __fdiv_rn(amax, 127.0f);
  int sum = 0;
  unsigned packed = 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int c =
        amax == 0.0f ? 0 : static_cast<int>(roundf(__fdiv_rn(v[e], nd)));
    sum += c;
    packed |= (static_cast<unsigned>(c) & 0xFFu) << (8 * e);
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1)
    sum += __shfl_xor(sum, m, 32);
  blk->qs[lane & 7] = static_cast<int>(packed);
  if ((lane & 7) == 0) {
    blk->d = __float2half_rn(nd);
    blk->sum = static_cast<short>(sum);
  }
}

void RotateQ8(Q8Block* q, std::size_t blocks, hipStream_t stream) {
  const std::size_t groups = blocks / 4;
  RotateQ8Kernel<<<static_cast<unsigned>((groups + 7) / 8), 256, 0, stream>>>(
      q, groups);
}

/// Block 0 buckets the assignments by expert: `entries` holds
/// (row << 4 | slot) sorted by expert and `groups` (expert, first entry,
/// rows) items of at most kGroup rows. Blocks 1..rows quantize one row.
template<int H>
__global__ __launch_bounds__(256) void PlanKernel(
    const float* __restrict__ x, const std::int32_t* __restrict__ ids,
    Q8Block* __restrict__ xq, int* __restrict__ entries,
    int4* __restrict__ groups, int* __restrict__ n_groups,
    float* __restrict__ out, int rows, int used, int n_experts) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  if (blockIdx.x != 0) {
    const int row = blockIdx.x - 1;
    for (int b = wave; b < (H / 32); b += 8)
      QuantizeLane(x[std::size_t(row) * H + b * 32 + lane],
                   xq + std::size_t(row) * (H / 32) + b, lane);
    return;
  }
  __shared__ int count[kMaxExperts];
  __shared__ int cursor[kMaxExperts];
  __shared__ int wave_total[8][2];
  for (int e = tid; e < kMaxExperts; e += 256)
    count[e] = 0;
  __syncthreads();
  const int n = rows * used;
  for (int i = tid; i < n; i += 256) {
    const int e = ids[i];
    if (e >= 0 && e < n_experts) {
      atomicAdd(&count[e], 1);
    } else {
      // Unrouted slots produce zero rows (rare; weights may still be read).
      for (int k = 0; k < H; ++k)
        out[std::size_t(i) * H + k] = 0.0f;
    }
  }
  __syncthreads();
  const int c0 = count[2 * tid];
  const int c1 = count[2 * tid + 1];
  const int ng = (c0 + kGroup - 1) / kGroup + (c1 + kGroup - 1) / kGroup;
  const int ne = c0 + c1;
  int sg = ng;
  int se = ne;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int a = __shfl_up(sg, d, 32);
    const int b = __shfl_up(se, d, 32);
    if (lane >= d) {
      sg += a;
      se += b;
    }
  }
  if (lane == 31) {
    wave_total[wave][0] = sg;
    wave_total[wave][1] = se;
  }
  __syncthreads();
  int pg = sg - ng;
  int pe = se - ne;
  for (int w = 0; w < wave; ++w) {
    pg += wave_total[w][0];
    pe += wave_total[w][1];
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int e = 2 * tid + k;
    const int c = k ? c1 : c0;
    cursor[e] = pe;
    for (int b = 0; b < c; b += kGroup)
      groups[pg++] = make_int4(e, pe + b, min(kGroup, c - b), 0);
    pe += c;
  }
  if (tid == 255)
    *n_groups = pg;
  __syncthreads();
  for (int i = tid; i < n; i += 256) {
    const int e = ids[i];
    if (e >= 0 && e < n_experts)
      entries[atomicAdd(&cursor[e], 1)] = (i / used) << 4 | (i % used);
  }
}

/// Q4_K/Q5_K scales of sub-blocks j, j + 1 (j even): A = d * scale,
/// B = dmin * min (both exact).
__device__ __forceinline__ void KScales(V4 h, int j, float* A, float* B) {
  const float d = HalfLo(h.x);
  const float dmin = HalfHi(h.x);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int i = j + k;
    unsigned sc, m;
    if (i < 4) {
      sc = Byte(h.y, i) & 63u;
      m = Byte(h.z, i) & 63u;
    } else {
      sc = (Byte(h.w, i - 4) & 15u) | ((Byte(h.y, i - 4) >> 6) << 4);
      m = (Byte(h.w, i - 4) >> 4) | ((Byte(h.z, i - 4) >> 6) << 4);
    }
    A[k] = __fmul_rn(d, float(sc));
    B[k] = __fmul_rn(dmin, float(m));
  }
}

enum GateFormat { kGateQ4K, kGateQ5K, kGateQ80, kGateQ40 };

/// Codes and scales of blocks 2a, 2a + 1 of one 256-value step: weight
/// value = A * code - B.
template<int F, int P>
__device__ __forceinline__ void GateBlocks(const V4 (&w)[P], int a, int* lo,
                                           int* hi, float* A, float* B) {
  if constexpr (F == kGateQ80) {
    // Eight 34-byte blocks: F16 scale, 32 codes (2-byte aligned).
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int blk = 2 * a + k;
      const int b = (34 * blk) / 4;
      const auto word = [&](int i) { return w[i / 4][i % 4]; };
      int* q = k ? hi : lo;
      A[k] = blk % 2 ? HalfHi(word(b)) : HalfLo(word(b));
      B[k] = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        q[i] = static_cast<int>(blk % 2 ? word(b + 1 + i)
                                        : __builtin_amdgcn_alignbyte(
                                              word(b + 1 + i), word(b + i), 2));
    }
  } else {
    constexpr int kQs = F == kGateQ5K ? 3 : 1;  // first piece of the codes
    if constexpr (F == kGateQ40) {
      // Q4_0 steps (PackRoutedExperts): the eight block scales, then the
      // Q4_K nibble layout; weight value = d * code - 8 d.
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const unsigned word = w[0][(2 * a + k) / 2];
        A[k] = (2 * a + k) % 2 ? HalfHi(word) : HalfLo(word);
        B[k] = __fmul_rn(A[k], 8.0f);
      }
    } else {
      KScales(w[0], 2 * a, A, B);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned q = w[kQs + 2 * a + i / 4][i % 4];
      lo[i] = static_cast<int>(q & 0x0F0F0F0Fu);
      hi[i] = static_cast<int>((q >> 4) & 0x0F0F0F0Fu);
      if constexpr (F == kGateQ5K) {
        const unsigned h = w[1 + i / 4][i % 4];
        lo[i] |= static_cast<int>(((h >> (2 * a)) & 0x01010101u) << 4);
        hi[i] |= static_cast<int>(((h >> (2 * a + 1)) & 0x01010101u) << 4);
      }
    }
  }
}

/// Gate (waves 0-1) or up (waves 2-3) rows of one expert for up to kGroup
/// token rows, one lane per row; the up waves then form silu(gate) * up and
/// write it as Q8 (one 32-row block per wave).
template<int F, int H, int FF>
__global__ __launch_bounds__(128) void GateUpKernel(
    const std::uint8_t* __restrict__ gate, const std::uint8_t* __restrict__ up,
    const Q8Block* __restrict__ xq, const int* __restrict__ entries,
    const int4* __restrict__ groups, const int* __restrict__ n_groups,
    Q8Block* __restrict__ hq, int used) {
  // 16-byte pieces per 256-value step.
  constexpr int kPieces = F == kGateQ80 ? 17 : F == kGateQ5K ? 11 : 9;
  constexpr int kRowBytes = kPieces * 16 * (H / 256);
  __shared__ Act act[2][kGroup][8];
  __shared__ float2 opr[2][kGroup][8];
  __shared__ float gate_out[kGateRows][kGroup + 1];
  if (static_cast<int>(blockIdx.y) >= *n_groups)
    return;
  MOE_TRACE_BEGIN(0);
  const int4 g = groups[blockIdx.y];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const bool is_up = wave >= 2;
  const int local = (wave & 1) * 32 + lane;
  const int row = blockIdx.x * kGateRows + local;
  // K-quant experts are packed (moe.hpp): piece i of this lane's row sits
  // 32 pieces after piece i - 1. The MTP block's Q8_0 experts are GGUF.
  constexpr int kStride = F == kGateQ80 ? 1 : 32;
  const std::size_t flat = std::size_t(g.x) * FF + row;
  const auto* w = reinterpret_cast<const V4*>(
      (is_up ? up : gate) +
      (F == kGateQ80 ? flat * kRowBytes
                     : (flat - lane) * kRowBytes + std::size_t(lane) * 16));
  // Thread tid < 8 * rows stages block (tid & 7) of token row tid >> 3.
  const int st = tid >> 3;
  const int sb = tid & 7;
  const bool stager = st < g.z;
  const Q8Block* src =
      xq + (stager ? std::size_t(entries[g.y + st] >> 4) * (H / 32) + sb : 0);
  V4 cur[kPieces];
#pragma unroll
  for (int i = 0; i < kPieces; ++i)
    cur[i] = w[i * kStride];
  if (stager)
    StoreAct(src[0], &act[0][st][sb], &opr[0][st][sb]);
  __syncthreads();
  float acc[kGroup] = {};
  for (int s = 0; s < (H / 256); ++s) {
    const int buf = s & 1;
    V4 next[kPieces];
    Q8Block xr;
    if (s + 1 < (H / 256)) {
#pragma unroll
      for (int i = 0; i < kPieces; ++i)
        next[i] = w[((s + 1) * kPieces + i) * kStride];
      if (stager)
        xr = src[(s + 1) * 8];
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      float A[2], B[2];
      int lo[8], hi[8];
      GateBlocks<F>(cur, a, lo, hi, A, B);
#pragma unroll
      for (int t = 0; t < kGroup; ++t) {
        if (t >= g.z)
          break;
        const Act x0 = act[buf][t][2 * a];
        const Act x1 = act[buf][t][2 * a + 1];
        const float2 o0 = opr[buf][t][2 * a];
        const float2 o1 = opr[buf][t][2 * a + 1];
        const int d0 = Dot8(lo, x0.q[0], x0.q[1]);
        const int d1 = Dot8(hi, x1.q[0], x1.q[1]);
        acc[t] = __fmaf_rn(A[0], __fmul_rn(o0.x, float(d0)), acc[t]);
        if constexpr (F != kGateQ80)
          acc[t] = __fmaf_rn(-B[0], o0.y, acc[t]);
        acc[t] = __fmaf_rn(A[1], __fmul_rn(o1.x, float(d1)), acc[t]);
        if constexpr (F != kGateQ80)
          acc[t] = __fmaf_rn(-B[1], o1.y, acc[t]);
      }
    }
    if (s + 1 < (H / 256)) {
      if (stager)
        StoreAct(xr, &act[buf ^ 1][st][sb], &opr[buf ^ 1][st][sb]);
#pragma unroll
      for (int i = 0; i < kPieces; ++i)
        cur[i] = next[i];
    }
    __syncthreads();
  }
  if (!is_up) {
#pragma unroll
    for (int t = 0; t < kGroup; ++t)
      gate_out[local][t] = acc[t];
  }
  __syncthreads();
  if (is_up) {
#pragma unroll
    for (int t = 0; t < kGroup; ++t) {
      if (t >= g.z)
        break;
      const float gv = isfinite(gate_out[local][t]) ? gate_out[local][t] : 0.0f;
      const float uv = isfinite(acc[t]) ? acc[t] : 0.0f;
      const float e = __expf(-gv);
      const float h = __fmul_rn(
          __fmul_rn(gv, __builtin_amdgcn_rcpf(__fadd_rn(1.0f, e))), uv);
      const int entry = entries[g.y + t];
      QuantizeLane(
          h,
          hq + (std::size_t(entry >> 4) * used + (entry & 15)) * (FF / 32) +
              blockIdx.x * 2 + (wave & 1),
          lane);
    }
  }
  MOE_TRACE_END(0);
}

/// Down rows of one expert for up to kGroup (row, slot) inputs, one lane per
/// row; Q5_1 (24-byte blocks) or Q8_0 (34-byte blocks), two blocks per step,
/// packed (moe.hpp) unless GGUF Q8_0 (the MTP block).
template<bool Q8, bool Packed, int H, int FF>
__global__ __launch_bounds__(128) void DownKernel(
    const std::uint8_t* __restrict__ down, const Q8Block* __restrict__ hq,
    const int* __restrict__ entries, const int4* __restrict__ groups,
    const int* __restrict__ n_groups, float* __restrict__ out, int used) {
  constexpr int kBlockBytes = Q8 ? 34 : 24;
  constexpr int kRowBytes = kBlockBytes * (FF / 32);
  constexpr int kWords = Q8 ? 17 : 12;  // dwords per two blocks
  constexpr int kSteps = (FF / 32) / 2;
  // Steps in flight beyond the current, at most the row's steps.
  constexpr int kAhead = std::min(Q8 ? 5 : 9, kSteps);
  __shared__ Act act[kGroup][(FF / 32)];
  __shared__ float2 opr[kGroup][(FF / 32)];
  if (static_cast<int>(blockIdx.y) >= *n_groups)
    return;
  MOE_TRACE_BEGIN(1);
  const int4 g = groups[blockIdx.y];
  const int tid = threadIdx.x;
  const int row = blockIdx.x * kDownRows + tid;
  static_assert(Packed || Q8);
  const int lane = tid & 31;
  const std::size_t flat = std::size_t(g.x) * H + row;
  const std::uint8_t* w =
      down + (Packed ? (flat - lane) * kRowBytes + std::size_t(lane) * 16
                     : flat * kRowBytes);
  const auto load = [&](int c, unsigned* wd) {
    if constexpr (Packed) {
      // Q5_1 step: qs0 | qs1 | d0 m0 qh0 d1 m1 qh1, rebuilt in GGUF order;
      // Q8_0 step: four code pieces, then the lanes' d0 | d1 dwords.
      const auto* p = reinterpret_cast<const V4*>(w + c * 32 * kWords * 4);
      V4 v[Q8 ? 4 : 3];
#pragma unroll
      for (int i = 0; i < (Q8 ? 4 : 3); ++i)
        v[i] = p[32 * i];
      if constexpr (Q8) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
          wd[i] = v[i / 4][i % 4];
        wd[16] = *reinterpret_cast<const unsigned*>(
            w - lane * 16 + c * 32 * 68 + 2048 + lane * 4);
      } else {
        const unsigned order[12] = {v[2].x, v[2].y, v[0].x, v[0].y,
                                    v[0].z, v[0].w, v[2].z, v[2].w,
                                    v[1].x, v[1].y, v[1].z, v[1].w};
#pragma unroll
        for (int i = 0; i < 12; ++i)
          wd[i] = order[i];
      }
    } else if constexpr (Q8) {
      const auto* p = reinterpret_cast<const unsigned*>(w + c * 68);
#pragma unroll
      for (int i = 0; i < 17; ++i)
        wd[i] = p[i];
    } else {
      const auto* p = reinterpret_cast<const V4*>(w + c * 48);
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const V4 v = p[i];
        wd[4 * i] = v.x;
        wd[4 * i + 1] = v.y;
        wd[4 * i + 2] = v.z;
        wd[4 * i + 3] = v.w;
      }
    }
  };
  unsigned wd[kSteps][kWords];
#pragma unroll
  for (int c = 0; c < kAhead; ++c)
    load(c, wd[c]);
  for (int i = tid; i < g.z * (FF / 32); i += 128) {
    const int t = i / (FF / 32);
    const int b = i % (FF / 32);
    const int entry = entries[g.y + t];
    StoreAct(
        hq[(std::size_t(entry >> 4) * used + (entry & 15)) * (FF / 32) + b],
        &act[t][b], &opr[t][b]);
  }
  __syncthreads();
  float acc[kGroup] = {};
#pragma unroll
  for (int c = 0; c < kSteps; ++c) {
    if (c + kAhead < kSteps)
      load(c + kAhead, wd[c + kAhead]);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      int q[8];
      float scale, bias = 0.0f;
      if constexpr (Q8 && Packed) {
        scale = k ? HalfHi(wd[c][16]) : HalfLo(wd[c][16]);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          q[i] = static_cast<int>(wd[c][8 * k + i]);
      } else if constexpr (Q8) {
        // Block k starts at byte 34k: scale, then 32 codes.
        if (k == 0) {
          scale = HalfLo(wd[c][0]);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            q[i] = static_cast<int>(
                __builtin_amdgcn_alignbyte(wd[c][i + 1], wd[c][i], 2));
        } else {
          scale = HalfHi(wd[c][8]);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            q[i] = static_cast<int>(wd[c][9 + i]);
        }
      } else {
        const unsigned* b = wd[c] + 6 * k;  // dm, qh, qs[4]
        scale = HalfLo(b[0]);
        bias = HalfHi(b[0]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned lo =
              (((b[1] >> (4 * i)) & 15u) * 0x00204081u) & 0x01010101u;
          const unsigned hi =
              (((b[1] >> (16 + 4 * i)) & 15u) * 0x00204081u) & 0x01010101u;
          q[i] = static_cast<int>((b[2 + i] & 0x0F0F0F0Fu) | (lo << 4));
          q[i + 4] =
              static_cast<int>(((b[2 + i] >> 4) & 0x0F0F0F0Fu) | (hi << 4));
        }
      }
#pragma unroll
      for (int t = 0; t < kGroup; ++t) {
        if (t >= g.z)
          break;
        const Act x = act[t][2 * c + k];
        const float2 o = opr[t][2 * c + k];
        const int dot = Dot8(q, x.q[0], x.q[1]);
        acc[t] = __fmaf_rn(scale, __fmul_rn(o.x, float(dot)), acc[t]);
        if constexpr (!Q8)
          acc[t] = __fmaf_rn(bias, o.y, acc[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < kGroup; ++t) {
    if (t >= g.z)
      break;
    const int entry = entries[g.y + t];
    out[(std::size_t(entry >> 4) * used + (entry & 15)) * H + row] =
        isfinite(acc[t]) ? acc[t] : 0.0f;
  }
  MOE_TRACE_END(1);
}

/// Canonical step byte -> GGUF byte within the step (see moe.hpp).
__device__ __forceinline__ int RawStepByte(int fmt, int b) {
  if (fmt == 1)  // Q5_1: qs0 | qs1 | d0 m0 qh0 d1 m1 qh1
    return b < 16 ? 8 + b : b < 32 ? 16 + b : 24 * ((b - 32) / 8) + (b % 8);
  if (fmt == 2)  // Q8_0: qs0 | qs1 | d0 d1
    return b < 32 ? 2 + b : b < 64 ? 4 + b : 34 * ((b - 64) / 2) + (b % 2);
  return b;
}

/// One work-group permutes one 32-row group through LDS.
__global__ __launch_bounds__(256) void PackKernel(std::uint8_t* data,
                                                  int row_bytes, int step,
                                                  int pieces, int fmt) {
  extern __shared__ std::uint32_t tile_words[];
  const auto* tile_bytes = reinterpret_cast<const std::uint8_t*>(tile_words);
  auto* tile = reinterpret_cast<std::uint32_t*>(data + std::size_t(blockIdx.x) *
                                                           32 * row_bytes);
  const int words = 8 * row_bytes;  // 32 rows / 4 bytes
  for (int i = threadIdx.x; i < words; i += 256)
    tile_words[i] = tile[i];
  __syncthreads();
  const int tail = step - 16 * pieces;
  for (int i = threadIdx.x; i < words; i += 256) {
    const int o = 4 * i;
    const int c = o / (32 * step);
    const int rem = o % (32 * step);
    int lane, b;
    if (rem < 512 * pieces) {
      lane = rem % 512 / 16;
      b = rem / 512 * 16 + rem % 16;
    } else {
      lane = (rem - 512 * pieces) / tail;
      b = 16 * pieces + (rem - 512 * pieces) % tail;
    }
    const int base = lane * row_bytes + c * step;
    std::uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v |= std::uint32_t(tile_bytes[base + RawStepByte(fmt, b + j)]) << (8 * j);
    tile[i] = v;
  }
}

/// Rewrites each 256-value step of Q4_0 rows (eight 18-byte blocks: F16
/// d, then element j in the low and element j + 16 in the high nibble of
/// byte j) in place as the eight scales followed by the Q4_K nibble layout
/// (byte j of 32-byte group g: element j of block 2g low, of block 2g + 1
/// high), so the packed Q4_K routes read it with only the header changed.
__global__ void Q4_0StepKernel(std::uint8_t* data, std::size_t steps) {
  const std::size_t step = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (step >= steps)
    return;
  std::uint8_t* p = data + step * 144;
  std::uint8_t raw[144];
  for (int i = 0; i < 144; ++i)
    raw[i] = p[i];
  for (int b = 0; b < 8; ++b) {
    p[2 * b] = raw[18 * b];
    p[2 * b + 1] = raw[18 * b + 1];
  }
  const auto code = [&](int b, int e) {
    const std::uint8_t q = raw[18 * b + 2 + (e % 16)];
    return e < 16 ? q & 15 : q >> 4;
  };
  for (int g = 0; g < 4; ++g)
    for (int j = 0; j < 32; ++j)
      p[16 + 32 * g + j] =
          static_cast<std::uint8_t>(code(2 * g, j) | code(2 * g + 1, j) << 4);
}

#include "src/models/qwen38_flash_next/kernels/rocm/moe_iq.inc"

}  // namespace

bool RoutedPacked(WeightType gate, WeightType down) {
  const bool iq_gate = gate == WeightType::kIQ3_S ||
                       gate == WeightType::kIQ3_XXS ||
                       gate == WeightType::kIQ4_XS;
  return ((gate == WeightType::kQ4_K || gate == WeightType::kQ5_K ||
           gate == WeightType::kQ4_0) &&
          (down == WeightType::kQ5_1 || down == WeightType::kQ8_0)) ||
         (iq_gate &&
          (down == WeightType::kIQ4_NL || down == WeightType::kQ8_0));
}

bool PackRoutedExperts(void* data, WeightType type, std::size_t rows,
                       std::size_t cols, hipStream_t stream) {
  int step = 0, pieces = 0, fmt = 0, per_step = 0;
  switch (type) {
    case WeightType::kQ4_0: {
      if (cols % 256 != 0)
        return false;
      const std::size_t steps = rows * cols / 256;
      Q4_0StepKernel<<<(steps + 255) / 256, 256, 0, stream>>>(
          static_cast<std::uint8_t*>(data), steps);
      step = 144, pieces = 9, per_step = 256;
      break;
    }
    case WeightType::kQ4_K:
      step = 144, pieces = 9, per_step = 256;
      break;
    case WeightType::kQ5_K:
      step = 176, pieces = 11, per_step = 256;
      break;
    case WeightType::kQ5_1:
      step = 48, pieces = 3, fmt = 1, per_step = 64;
      break;
    case WeightType::kQ8_0:
      step = 68, pieces = 4, fmt = 2, per_step = 64;
      break;
    default: {
      // IQ formats (iq.hpp).
      const iq::Step st = iq::StepOf(type);
      const std::size_t row_bytes = cols / std::max(st.values, 1) * st.bytes;
      if (st.bytes == 0 || rows == 0 || rows % 32 != 0 ||
          cols % st.values != 0 || 32 * row_bytes > 64 * 1024)
        return false;
      PackIqKernel<<<rows / 32, 256, 32 * row_bytes, stream>>>(
          static_cast<std::uint8_t*>(data), int(row_bytes), type);
      return hipGetLastError() == hipSuccess;
    }
  }
  const std::size_t row_bytes = cols / per_step * step;
  if (rows == 0 || rows % 32 != 0 || cols % per_step != 0 ||
      32 * row_bytes > 64 * 1024)
    return false;
  PackKernel<<<rows / 32, 256, 32 * row_bytes, stream>>>(
      static_cast<std::uint8_t*>(data), int(row_bytes), step, pieces, fmt);
  return hipGetLastError() == hipSuccess;
}

std::size_t RoutedExpertsScratchBytes(std::uint32_t rows, std::uint32_t used) {
  return Align(std::size_t{rows} * kXBlocks * sizeof(Q8Block)) +
         Align(std::size_t{rows} * used * kHBlocks * sizeof(Q8Block)) +
         Align(std::size_t{rows} * used * sizeof(int)) +
         Align(std::size_t(MaxGroups(rows, used, kMaxExperts)) * sizeof(int4)) +
         256;
}

namespace {

template<int H, int FF>
bool LaunchRoutedExperts(const void* gate, const void* up, WeightType gate_type,
                         const void* down, WeightType down_type,
                         std::uint32_t n_experts, const float* x,
                         const std::int32_t* ids, float* out,
                         std::uint32_t rows, std::uint32_t used, void* scratch,
                         hipStream_t stream, const hipEvent_t* phases,
                         int hadamard) {
  const bool q8 = down_type == WeightType::kQ8_0;
  const bool packed = RoutedPacked(gate_type, down_type);
  const bool iq_down = down_type == WeightType::kIQ4_NL;
  const Layout l = Carve(scratch, rows, used, FF / 32);
  const int groups = MaxGroups(rows, used, n_experts);
  const auto* g = static_cast<const std::uint8_t*>(gate);
  const auto* u = static_cast<const std::uint8_t*>(up);
  const auto* d = static_cast<const std::uint8_t*>(down);
  PlanKernel<H><<<rows + 1, 256, 0, stream>>>(x, ids, l.xq, l.entries, l.groups,
                                              l.n_groups, out, int(rows),
                                              int(used), int(n_experts));
  if (hadamard == 128)
    RotateQ8(l.xq, std::size_t{rows} * (H / 32), stream);
  if (phases)
    (void)hipEventRecord(phases[0], stream);
  const dim3 gate_grid(FF / kGateRows, groups);
  const auto launch = [&](auto kernel) {
    kernel<<<gate_grid, 128, 0, stream>>>(g, u, l.xq, l.entries, l.groups,
                                          l.n_groups, l.hq, int(used));
  };
  // The IQ kernels (moe_iq.inc) keep the Flash-Next geometry.
  if constexpr (H == kHidden && FF == kFf) {
    if (gate_type == WeightType::kIQ4_XS)
      launch(GateUpIqKernel<WeightType::kIQ4_XS>);
    else if (gate_type == WeightType::kIQ3_S)
      launch(GateUpIqKernel<WeightType::kIQ3_S>);
    else if (gate_type == WeightType::kIQ3_XXS)
      launch(GateUpIqKernel<WeightType::kIQ3_XXS>);
  }
  if (gate_type == WeightType::kQ4_0)
    launch(GateUpKernel<kGateQ40, H, FF>);
  else if (gate_type == WeightType::kQ5_K)
    launch(GateUpKernel<kGateQ5K, H, FF>);
  else if (gate_type == WeightType::kQ8_0)
    launch(GateUpKernel<kGateQ80, H, FF>);
  else if (gate_type == WeightType::kQ4_K)
    launch(GateUpKernel<kGateQ4K, H, FF>);
  if (phases)
    (void)hipEventRecord(phases[1], stream);
  if (hadamard == 128)
    RotateQ8(l.hq, std::size_t{rows} * used * (FF / 32), stream);
  const dim3 down_grid(H / kDownRows, groups);
  const auto down_launch = [&](auto kernel) {
    kernel<<<down_grid, 128, 0, stream>>>(d, l.hq, l.entries, l.groups,
                                          l.n_groups, out, int(used));
  };
  if constexpr (H == kHidden && FF == kFf) {
    if (iq_down)
      down_launch(DownIq4NlKernel);
  }
  if (!iq_down && !q8)
    down_launch(DownKernel<false, true, H, FF>);
  else if (!iq_down && packed)
    down_launch(DownKernel<true, true, H, FF>);
  else if (!iq_down)
    down_launch(DownKernel<true, false, H, FF>);
  return hipGetLastError() == hipSuccess;
}

}  // namespace

bool RoutedExperts(const void* gate, const void* up, WeightType gate_type,
                   const void* down, WeightType down_type,
                   std::uint32_t n_experts, const float* x,
                   const std::int32_t* ids, float* out, std::uint32_t rows,
                   std::uint32_t used, void* scratch, hipStream_t stream,
                   const hipEvent_t* phases, std::uint32_t hidden,
                   std::uint32_t ff, int hadamard) {
  const bool q8 = down_type == WeightType::kQ8_0;
  const bool packed = RoutedPacked(gate_type, down_type);
  if (rows == 0 || used == 0 || used > 16 || n_experts > kMaxExperts ||
      !(packed || (gate_type == WeightType::kQ8_0 && q8)) ||
      (hadamard != 0 && (hadamard != 128 || hidden != kHidden || ff != kFf)))
    return false;
  if (hidden == kHidden && ff == kFf)
    return LaunchRoutedExperts<kHidden, kFf>(
        gate, up, gate_type, down, down_type, n_experts, x, ids, out, rows,
        used, scratch, stream, phases, hadamard);
  // qwen35moe (MiniCPM-V 4.7): K-quant or Q8_0 experts only.
  const bool iq =
      down_type == WeightType::kIQ4_NL || gate_type == WeightType::kIQ4_XS ||
      gate_type == WeightType::kIQ3_S || gate_type == WeightType::kIQ3_XXS;
  if (hidden == 2048 && ff == 512 && !iq)
    return LaunchRoutedExperts<2048, 512>(gate, up, gate_type, down, down_type,
                                          n_experts, x, ids, out, rows, used,
                                          scratch, stream, phases, 0);
  // lfm2moe (LFM2.5-24B-A2B): 64 experts of 1536, the same formats.
  if (hidden == 2048 && ff == 1536 && !iq)
    return LaunchRoutedExperts<2048, 1536>(gate, up, gate_type, down, down_type,
                                           n_experts, x, ids, out, rows, used,
                                           scratch, stream, phases, 0);
  return false;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
