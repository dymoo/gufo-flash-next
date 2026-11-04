#include "src/models/qwen38_flash_next/kernels/rocm/spike.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/routed_iu4.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/spike_knobs.hpp"

namespace gufo::models::qwen38_flash_next::rocm::spike {
namespace {

const Knobs& Opt() {
  static const Knobs o =
      ParseKnobs([](const char* name) { return std::getenv(name); });
  return o;
}

// GUFO_SPIKE_* names in the environment that this binary does not read
// (a stale stages file must not silently run a different recipe).
std::vector<std::string> UnknownEnvKnobs() {
  std::vector<std::string> names;
  for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
    const char* eq = std::strchr(*e, '=');
    names.emplace_back(*e, eq != nullptr ? static_cast<std::size_t>(eq - *e)
                                         : std::strlen(*e));
  }
  return UnknownKnobs(names);
}

struct PackedWeights {
  std::uint8_t* w4{nullptr};   // [n][k/2] signed nibbles
  std::uint32_t* sc{nullptr};  // [k/128][n] slab words of 6-bit group scales
  float* d{nullptr};           // [n] row scale
  std::uint32_t n{0}, k{0}, group{0};
};

struct Entry {
  const void* shadow{nullptr};
  bool rotated{false};
  unsigned role{0};
  int layer{0};
  std::uint32_t cols{0};
  PackedWeights packed;
};

std::unordered_map<const void*, Entry>& Registry() {
  static std::unordered_map<const void*, Entry> r;
  return r;
}

/// GUFO_SPIKE_ROUTED_HAD: every routed expert matrix of the main layers is
/// stored rotated (W' = W R^T), so all of its calls need rotated activations,
/// whatever the targets, KEEP, MIN_ROWS or phase.
std::unordered_map<const void*, int>& Rotated() {
  static std::unordered_map<const void*, int> r;
  return r;
}

int RotationOf(const void* w) {
  const auto it = Rotated().find(w);
  return it == Rotated().end() ? 0 : it->second;
}

constexpr int kThreads = 256;
constexpr int kMaxGroups =
    10240 / 32;  // widest row (hc) over the narrowest group

__device__ __forceinline__ float Sign(int i) {
  const unsigned h = static_cast<unsigned>(i) * 2654435761U;
  return ((h >> 29) & 1U) != 0U ? -1.0F : 1.0F;
}

// Entry (r, c) of a Hadamard matrix of order n: Sylvester for powers of two,
// Paley I (n - 1 a prime congruent to 3 mod 4) otherwise. FullRotationBlock
// only picks supported orders.
__device__ __forceinline__ float HadamardEntry(int n, int r, int c) {
  if ((n & (n - 1)) == 0)
    return (__popc(static_cast<unsigned>(r & c)) & 1) != 0 ? -1.0F : 1.0F;
  if (r == 0)
    return 1.0F;
  if (c == 0)
    return -1.0F;
  if (r == c)
    return 1.0F;
  const int q = n - 1;
  const int a = ((c - r) % q + q) % q;
  // Legendre symbol by Euler's criterion: a^((q-1)/2) mod q.
  int p = 1;
  for (int e = 0; e < (q - 1) / 2; ++e)
    p = (p * a) % q;
  return p == 1 ? 1.0F : -1.0F;
}

__host__ __device__ constexpr bool HadamardOrder(int n) {
  if (n >= 1 && (n & (n - 1)) == 0)
    return true;
  const int q = n - 1;
  if (q < 3 || q % 4 != 3)
    return false;
  for (int d = 2; d * d <= q; ++d)
    if (q % d == 0)
      return false;
  return true;
}

/// GUFO_SPIKE_HAD=1 (kFullRotation): the inner block size b of the
/// Kronecker rotation (H_{k/b} x H_b D) / sqrt(k), the largest of 128/64/32
/// whose block count k / b has a Hadamard order of at most 48; 0 if none.
/// 2560 = 20 x 128, 6144 = 48 x 128, 640 = 20 x 32.
constexpr int kFullRotation = 1;
constexpr int kMaxCrossBlocks = 48;
__host__ __device__ constexpr int FullRotationBlock(int k) {
  for (int b = 128; b >= 32; b >>= 1)
    if (k % b == 0 && k / b <= kMaxCrossBlocks && HadamardOrder(k / b))
      return b;
  return 0;
}

// In-place block-diagonal Walsh-Hadamard transform of `k` floats in shared
// memory (blocks of `had`), with the fixed sign vector applied first and the
// orthonormal 1/sqrt(had) scale last. had == kFullRotation rotates the whole
// row: the block transform with b = FullRotationBlock(k), then H_{k/b}
// across the blocks (element j of every block), normalized by
// 1/sqrt(k / b). Called by every thread of the block.
__device__ void BlockHadamard(float* s, int k, int had) {
  const int nb = had == kFullRotation ? k / FullRotationBlock(k) : 1;
  if (had == kFullRotation)
    had = FullRotationBlock(k);
  for (int i = threadIdx.x; i < k; i += kThreads)
    s[i] *= Sign(i % had);
  __syncthreads();
  const int half_block = had >> 1;
  for (int h = 1; h < had; h <<= 1) {
    for (int p = threadIdx.x; p < (k >> 1); p += kThreads) {
      const int b = p / half_block;
      const int j = p - b * half_block;
      const int i0 = b * had + (j / h) * (h << 1) + (j % h);
      const float a = s[i0];
      const float c = s[i0 + h];
      s[i0] = a + c;
      s[i0 + h] = a - c;
    }
    __syncthreads();
  }
  const float norm = rsqrtf(static_cast<float>(had));
  for (int i = threadIdx.x; i < k; i += kThreads)
    s[i] *= norm;
  __syncthreads();
  if (nb == 1)
    return;
  // Each thread owns one in-block position j across the nb blocks.
  const float cross = rsqrtf(static_cast<float>(nb));
  for (int j = threadIdx.x; j < had; j += kThreads) {
    float v[kMaxCrossBlocks];
    for (int i = 0; i < nb; ++i)
      v[i] = s[i * had + j];
    for (int r = 0; r < nb; ++r) {
      float acc = 0.0F;
      for (int i = 0; i < nb; ++i)
        acc = __fmaf_rn(HadamardEntry(nb, r, i), v[i], acc);
      s[r * had + j] = acc * cross;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  return v;
}

// Block-wide max / sum (every thread gets the result). `red` holds one value
// per wave; the trailing barrier lets the caller reuse it at once.
__device__ __forceinline__ float BlockMax(float v, float* red) {
  v = WaveMax(v);
  if ((threadIdx.x & 31) == 0)
    red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = 0.0F;
  for (int w = 0; w < kThreads / 32; ++w)
    v = fmaxf(v, red[w]);
  __syncthreads();
  return v;
}

__device__ __forceinline__ float BlockSum(float v, float* red) {
  v = WaveSum(v);
  if ((threadIdx.x & 31) == 0)
    red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = 0.0F;
  for (int w = 0; w < kThreads / 32; ++w)
    v += red[w];
  __syncthreads();
  return v;
}

template<bool kBlock>
__device__ __forceinline__ float SpanMax(float v, float* red) {
  if constexpr (kBlock)
    return BlockMax(v, red);
  else
    return WaveMax(v);
}

template<bool kBlock>
__device__ __forceinline__ float SpanSum(float v, float* red) {
  if constexpr (kBlock)
    return BlockSum(v, red);
  else
    return WaveSum(v);
}

// Marks the `count` largest |s[i]| (i < n) not yet marked in `fl` (lowest
// index on ties), one block-wide argmax per pick. Called by every thread.
__device__ void MarkLargest(const float* s, std::uint8_t* fl, int n, int count,
                            float* red, int* red_i) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int r = 0; r < count; ++r) {
    float bv = -1.0F;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n; i += kThreads) {
      const float a = fabsf(s[i]);
      if (fl[i] == 0 && a > bv) {
        bv = a;
        bi = i;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o);
      const int oi = __shfl_xor(bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      red[wave] = bv;
      red_i[wave] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float v = red[0];
      int idx = red_i[0];
      for (int w = 1; w < kThreads / 32; ++w)
        if (red[w] > v || (red[w] == v && red_i[w] < idx)) {
          v = red[w];
          idx = red_i[w];
        }
      if (v >= 0.0F && idx < n)
        fl[idx] = 1;
    }
    __syncthreads();
  }
}

// Launch parameters of one activation emulation pass (see ActRecipe).
struct ActLaunch {
  int grid{kActNone};
  int group{0};  // 0 = one scale per row
  float clip{1.0F};
  int had{0};
  int clip_search{0};
  int outliers{0};                    // per-row K (row mode)
  const std::uint8_t* keep{nullptr};  // per-call channel set (call mode)
  std::uint32_t jitter{0};            // JitterKey of the tensor (0 = off)
};

// Quantize-dequantize one span (a group, or the whole row when kBlock):
// absmax over the unmarked elements, scale from the clip (or the MSE-best of
// the clip search), marked elements pass through unchanged. With a jitter
// key, the chosen integer-grid scale of group `gidx` (index within the row)
// gets its GUFO_SPIKE_JITTER perturbation.
template<bool kBlock>
__device__ void QuantSpan(const float* v, __half* out, const std::uint8_t* fl,
                          int n, const ActLaunch& p, float gs, float* red,
                          std::uint32_t gidx) {
  const int start = kBlock ? static_cast<int>(threadIdx.x)
                           : static_cast<int>(threadIdx.x & 31);
  const int step = kBlock ? kThreads : 32;
  float amax = 0.0F;
  for (int i = start; i < n; i += step)
    if (fl == nullptr || fl[i] == 0)
      amax = fmaxf(amax, fabsf(v[i]));
  amax = SpanMax<kBlock>(amax, red);
  float sc = ActScale(p.clip * amax, p.grid, gs);
  if (p.clip_search != 0 && amax > 0.0F) {
    float best = 3.0e38F;
    for (int c = 0; c < kAClipCount; ++c) {
      const float t = ActScale(AClip(c) * amax, p.grid, gs);
      const float inv = t > 0.0F ? 1.0F / t : 0.0F;
      float err = 0.0F;
      for (int i = start; i < n; i += step) {
        if (fl != nullptr && fl[i] != 0)
          continue;
        const float e = v[i] - ActQuant(v[i], inv, t, p.grid);
        err += e * e;
      }
      err = SpanSum<kBlock>(err, red);
      if (err < best) {
        best = err;
        sc = t;
      }
    }
  }
  if (p.jitter != 0 && p.grid != kActFp4)
    sc += sc * JitterEps(p.jitter, gidx);
  const float inv = sc > 0.0F ? 1.0F / sc : 0.0F;
  for (int i = start; i < n; i += step)
    out[i] = __float2half_rn(
        fl != nullptr && fl[i] != 0 ? v[i] : ActQuant(v[i], inv, sc, p.grid));
}

// Row-wise activation emulation: optional rotation, optional F16-retained
// outlier channels, then quantize and dequantize on the recipe's grid (none =
// rotate only) with one scale per row or per `group` elements. F16 in, F16
// out, same layout; x == y is allowed. Dynamic LDS: k floats, plus k bytes
// when outliers are retained. Integer grids without outliers or clip search
// are bit-identical to the original per-row/per-group emulation.
__global__ __launch_bounds__(kThreads) void FqRows(const __half* x, __half* y,
                                                   int k, ActLaunch p) {
  extern __shared__ float s[];
  __shared__ float red[kThreads / 32];
  __shared__ int red_i[kThreads / 32];
  const std::size_t row = blockIdx.x;
  const __half* xr = x + row * k;
  __half* yr = y + row * k;
  for (int i = threadIdx.x; i < k; i += kThreads)
    s[i] = __half2float(xr[i]);
  __syncthreads();
  if (p.had > 0)
    BlockHadamard(s, k, p.had);
  std::uint8_t* fl = nullptr;
  if (p.grid != kActNone && (p.outliers > 0 || p.keep != nullptr)) {
    fl = reinterpret_cast<std::uint8_t*>(s + k);
    for (int i = threadIdx.x; i < k; i += kThreads)
      fl[i] = p.keep != nullptr ? p.keep[i] : 0;
    __syncthreads();
    if (p.keep == nullptr)
      MarkLargest(s, fl, k, p.outliers, red, red_i);
  }
  if (p.grid == kActNone) {
    for (int i = threadIdx.x; i < k; i += kThreads)
      yr[i] = __float2half_rn(s[i]);
    return;
  }
  float gs = 1.0F;
  if (p.grid == kActFp4) {
    float m = 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads)
      if (fl == nullptr || fl[i] == 0)
        m = fmaxf(m, fabsf(s[i]));
    gs = Fp4RowScale(BlockMax(m, red));
    // FP4 jitters the per-row FP32 scale (the block scales stay UE4M3).
    if (p.jitter != 0)
      gs += gs * JitterEps(p.jitter, 0xFFFFFFFFU);
  }
  if (p.group == 0) {
    QuantSpan<true>(s, yr, fl, k, p, gs, red, 0);
    return;
  }
  const int wave = threadIdx.x >> 5;
  for (int g = wave; g < k / p.group; g += kThreads / 32)
    QuantSpan<false>(s + g * p.group, yr + g * p.group,
                     fl != nullptr ? fl + g * p.group : nullptr, p.group, p, gs,
                     red, static_cast<std::uint32_t>(g));
}

// Call-mode outliers, step 1: colmax[c] = max over rows of |x[r][c]| as the
// float's bit pattern (monotonic for non-negative floats). grid.x covers the
// columns, grid.y splits the rows; colmax must be zeroed first.
__global__ __launch_bounds__(kThreads) void ColAbsMax(
    const __half* __restrict__ x, std::size_t rows, int k,
    unsigned* __restrict__ colmax) {
  const int col = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (col >= k)
    return;
  const std::size_t per = (rows + gridDim.y - 1) / gridDim.y;
  const std::size_t r0 = blockIdx.y * per;
  const std::size_t r1 = std::min(rows, r0 + per);
  float m = 0.0F;
  for (std::size_t r = r0; r < r1; ++r)
    m = fmaxf(m, fabsf(__half2float(x[r * k + col])));
  if (r0 < r1)
    atomicMax(&colmax[col], __float_as_uint(m));
}

// Call-mode outliers, step 2 (one block): keep[c] = 1 for the `count`
// columns with the largest colmax. Dynamic LDS: k floats + k bytes.
__global__ __launch_bounds__(kThreads) void SelectColumns(
    const unsigned* __restrict__ colmax, int k, int count,
    std::uint8_t* __restrict__ keep) {
  extern __shared__ float s[];
  __shared__ float red[kThreads / 32];
  __shared__ int red_i[kThreads / 32];
  std::uint8_t* fl = reinterpret_cast<std::uint8_t*>(s + k);
  for (int i = threadIdx.x; i < k; i += kThreads) {
    s[i] = __uint_as_float(colmax[i]);
    fl[i] = 0;
  }
  __syncthreads();
  MarkLargest(s, fl, k, count, red, red_i);
  for (int i = threadIdx.x; i < k; i += kThreads)
    keep[i] = fl[i];
}

// Re-quantizes one Q8_0 weight row to a 4-bit grid. Optional block-Hadamard
// rotation of the row first (the activations get the same rotation). Per
// group of `group` elements the scale comes from a small clipping search that
// minimizes the squared error; with `scale6` the group scales are then
// restricted to the integer kernel's format, one float per row times a 6-bit
// integer (1..63) per group.
//
//   kPacked == false: writes the result back in place into the Q8_0 container
//     (fp16 scale per 32-block, int8 codes in [-8, 7]) so the existing
//     Q8_0 -> F16 kernels read exactly the 4-bit values (emulation).
//   kPacked == true: writes the integer kernel's format instead -- signed
//     nibbles w4 [n][k/2], 6-bit scales as 32-bit slab words sc [k/128][n]
//     (byte g = group g within the 128-element slab) and the row scale d [n].
//     The effective weights are the same values the emulation produces.
template<bool kPacked>
__global__ __launch_bounds__(kThreads) void RequantRow(
    std::uint8_t* __restrict__ w, int n, int k, int group, int had, int scale6,
    std::uint8_t* __restrict__ w4, std::uint32_t* __restrict__ sc,
    float* __restrict__ d_row, int wbits, std::uint32_t jitter) {
  const float qpos = WeightQmax(wbits);
  const float qneg = -qpos - 1.0F;
  extern __shared__ float s[];
  __shared__ float gscale[kMaxGroups];
  __shared__ float red[kThreads / 32];
  const int nb = k / 32;
  const int row = blockIdx.x;
  std::uint8_t* rp = w + static_cast<std::size_t>(row) * nb * 34;
  for (int b = threadIdx.x; b < nb; b += kThreads) {
    const std::uint8_t* blk = rp + static_cast<std::size_t>(b) * 34;
    const unsigned short dh =
        static_cast<unsigned short>(blk[0] | (blk[1] << 8));
    const float d = __half2float(__ushort_as_half(dh));
    for (int j = 0; j < 32; ++j)
      s[b * 32 + j] =
          d * static_cast<float>(static_cast<std::int8_t>(blk[2 + j]));
  }
  __syncthreads();
  if (had > 0)
    BlockHadamard(s, k, had);
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int ngroups = k / group;
  constexpr float kClip[8] = {1.0F, 0.95F, 0.9F, 0.85F,
                              0.8F, 0.75F, 0.7F, 0.65F};
  // Pass 1: the best scale per group from a clipping search.
  for (int g = wave; g < ngroups; g += kThreads / 32) {
    const float* gp = s + static_cast<std::size_t>(g) * group;
    float amax = 0.0F;
    for (int i = lane; i < group; i += 32)
      amax = fmaxf(amax, fabsf(gp[i]));
    amax = WaveMax(amax);
    float best_scale = amax / qpos;
    float best_err = 3.0e38F;
    if (amax > 0.0F) {
      for (int c = 0; c < 8; ++c) {
        const float sc_try = amax * kClip[c] / qpos;
        const float inv = 1.0F / sc_try;
        float err = 0.0F;
        for (int i = lane; i < group; i += 32) {
          const float q =
              fminf(fmaxf(__builtin_rintf(gp[i] * inv), qneg), qpos);
          const float e = gp[i] - q * sc_try;
          err += e * e;
        }
        err = WaveSum(err);
        if (err < best_err) {
          best_err = err;
          best_scale = sc_try;
        }
      }
    }
    if (lane == 0)
      gscale[g] = best_scale;
  }
  __syncthreads();
  float row_d = 0.0F;
  if (scale6 != 0 || kPacked) {
    float m = 0.0F;
    for (int g = threadIdx.x; g < ngroups; g += kThreads)
      m = fmaxf(m, gscale[g]);
    m = WaveMax(m);
    if (lane == 0)
      red[wave] = m;
    __syncthreads();
    m = 0.0F;
    for (int i = 0; i < kThreads / 32; ++i)
      m = fmaxf(m, red[i]);
    row_d = m / 63.0F;
    for (int g = threadIdx.x; g < ngroups; g += kThreads) {
      const float ideal = row_d > 0.0F ? gscale[g] / row_d : 0.0F;
      const float sc_int = fminf(fmaxf(__builtin_rintf(ideal), 1.0F), 63.0F);
      gscale[g] = row_d * sc_int;
      if constexpr (kPacked) {
        // slab word: byte (g % groups_per_slab) of word (g / groups_per_slab)
        const int gps = 128 / group;
        const int slab = g / gps;
        const int pos = g % gps;
        // bytes of one word come from different groups; combine with atomics
        atomicOr(&sc[static_cast<std::size_t>(slab) * n + row],
                 static_cast<std::uint32_t>(sc_int) << (8 * pos));
      }
    }
    if constexpr (kPacked) {
      if (threadIdx.x == 0)
        d_row[row] = row_d;
    }
    __syncthreads();
  }
  // Pass 2: codes. GUFO_SPIKE_JITTER perturbs the scale the codes are
  // rounded against (group index row * ngroups + g); the stored F16 scale
  // stays, so the jitter flips boundary codes even below F16 resolution.
  for (int g = wave; g < ngroups; g += kThreads / 32) {
    const float* gp = s + static_cast<std::size_t>(g) * group;
    float scale_value =
        kPacked ? gscale[g] : __half2float(__float2half_rn(gscale[g]));
    if (jitter != 0)
      scale_value +=
          scale_value *
          JitterEps(jitter, static_cast<std::uint32_t>(row) *
                                    static_cast<std::uint32_t>(ngroups) +
                                static_cast<std::uint32_t>(g));
    const float inv = scale_value > 0.0F ? 1.0F / scale_value : 0.0F;
    if constexpr (!kPacked) {
      const unsigned short sbits = __half_as_ushort(__float2half_rn(gscale[g]));
      for (int bb = 0; bb < group / 32; ++bb) {
        std::uint8_t* blk =
            rp + static_cast<std::size_t>(g * (group / 32) + bb) * 34;
        if (lane == 0) {
          blk[0] = static_cast<std::uint8_t>(sbits & 0xFFU);
          blk[1] = static_cast<std::uint8_t>(sbits >> 8);
        }
        const float q =
            fminf(fmaxf(__builtin_rintf(gp[bb * 32 + lane] * inv), qneg), qpos);
        blk[2 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      }
    } else {
      // two codes per byte: lane handles element pairs within the group
      std::uint8_t* out = w4 + static_cast<std::size_t>(row) * (k / 2) +
                          static_cast<std::size_t>(g) * (group / 2);
      for (int i = lane; i < group / 2; i += 32) {
        const float q0 =
            fminf(fmaxf(__builtin_rintf(gp[2 * i] * inv), qneg), qpos);
        const float q1 =
            fminf(fmaxf(__builtin_rintf(gp[2 * i + 1] * inv), qneg), qpos);
        out[i] = static_cast<std::uint8_t>((static_cast<int>(q0) & 0xF) |
                                           ((static_cast<int>(q1) & 0xF) << 4));
      }
    }
  }
}

// Row-wise int4 activation quantizer for the integer GEMM: optional
// rotation, then symmetric per-row quantization (scale = absmax / 7) into
// signed nibbles x4 [row][k/2] (low nibble = even element) and sx [row].
// Matches FqRows(bits = 4, group = 0) bit for bit up to the final dequantize.
__global__ __launch_bounds__(kThreads) void QuantRowsI4(
    const __half* __restrict__ x, std::uint8_t* __restrict__ x4,
    float* __restrict__ sx, int k, int had, float clip) {
  extern __shared__ float s[];
  __shared__ float red[kThreads / 32];
  const std::size_t row = blockIdx.x;
  const __half* xr = x + row * k;
  for (int i = threadIdx.x; i < k; i += kThreads)
    s[i] = __half2float(xr[i]);
  __syncthreads();
  if (had > 0)
    BlockHadamard(s, k, had);
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  float m = 0.0F;
  for (int i = threadIdx.x; i < k; i += kThreads)
    m = fmaxf(m, fabsf(s[i]));
  m = WaveMax(m);
  if (lane == 0)
    red[wave] = m;
  __syncthreads();
  m = 0.0F;
  for (int w = 0; w < kThreads / 32; ++w)
    m = fmaxf(m, red[w]);
  const float sc = clip * m / 7.0F;
  const float inv = sc > 0.0F ? 1.0F / sc : 0.0F;
  std::uint8_t* out = x4 + row * (k / 2);
  for (int i = threadIdx.x; i < k / 2; i += kThreads) {
    const float q0 = fminf(fmaxf(__builtin_rintf(s[2 * i] * inv), -8.0F), 7.0F);
    const float q1 =
        fminf(fmaxf(__builtin_rintf(s[2 * i + 1] * inv), -8.0F), 7.0F);
    out[i] = static_cast<std::uint8_t>((static_cast<int>(q0) & 0xF) |
                                       ((static_cast<int>(q1) & 0xF) << 4));
  }
  if (threadIdx.x == 0)
    sx[row] = sc;
}

// Fast path of QuantRowsI4 for rows of at most kFastMaxBlocks * 8 * 128 =
// 6144 elements whose Hadamard block is 128 (or no rotation): one wave handles
// 128-element blocks with four consecutive elements per lane, the butterflies
// run in registers (h = 1, 2) and with __shfl_xor (h = 4..64), the row maximum
// needs one barrier, and nibbles are written two bytes per lane. Same
// arithmetic as QuantRowsI4 (identical sign vector, normalization, rounding).
constexpr int kFastMaxBlocks = 6;  // 128-element blocks per wave (8 waves)

template<bool kRotate>
__global__ __launch_bounds__(kThreads) void QuantRowsI4Fast(
    const __half* __restrict__ x, std::uint8_t* __restrict__ x4,
    float* __restrict__ sx, int k, float clip) {
  __shared__ float red[kThreads / 32];
  const std::size_t row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int nblocks = k / 128;
  const __half* xr = x + row * k;
  float v[kFastMaxBlocks][4];
  float m = 0.0F;
#pragma unroll
  for (int b = 0; b < kFastMaxBlocks; ++b) {
    const int blk = wave + b * (kThreads / 32);
    if (blk < nblocks) {
      const uint2 raw =
          *reinterpret_cast<const uint2*>(xr + blk * 128 + lane * 4);
      const __half2 h0 = __builtin_bit_cast(__half2, raw.x);
      const __half2 h1 = __builtin_bit_cast(__half2, raw.y);
      v[b][0] = __low2float(h0);
      v[b][1] = __high2float(h0);
      v[b][2] = __low2float(h1);
      v[b][3] = __high2float(h1);
      if constexpr (kRotate) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[b][e] *= Sign(lane * 4 + e);
        // h = 1 and h = 2 inside the lane
        float a0 = v[b][0] + v[b][1], a1 = v[b][0] - v[b][1];
        float a2 = v[b][2] + v[b][3], a3 = v[b][2] - v[b][3];
        v[b][0] = a0 + a2;
        v[b][2] = a0 - a2;
        v[b][1] = a1 + a3;
        v[b][3] = a1 - a3;
        // h = 4, 8, 16, 32, 64 across lanes (lane xor h/4)
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const bool upper = (lane & d) != 0;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float other = __shfl_xor(v[b][e], d);
            v[b][e] = upper ? other - v[b][e] : v[b][e] + other;
          }
        }
        const float norm = 0.08838834764831845F;  // 1/sqrt(128)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[b][e] *= norm;
      }
#pragma unroll
      for (int e = 0; e < 4; ++e)
        m = fmaxf(m, fabsf(v[b][e]));
    }
  }
  m = WaveMax(m);
  if (lane == 0)
    red[wave] = m;
  __syncthreads();
  m = 0.0F;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w)
    m = fmaxf(m, red[w]);
  const float sc = clip * m / 7.0F;
  const float inv = sc > 0.0F ? 1.0F / sc : 0.0F;
  std::uint8_t* out = x4 + row * (k / 2);
#pragma unroll
  for (int b = 0; b < kFastMaxBlocks; ++b) {
    const int blk = wave + b * (kThreads / 32);
    if (blk < nblocks) {
      int q[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        q[e] = static_cast<int>(
            fminf(fmaxf(__builtin_rintf(v[b][e] * inv), -8.0F), 7.0F));
      const unsigned short packed = static_cast<unsigned short>(
          (q[0] & 0xF) | ((q[1] & 0xF) << 4) | ((q[2] & 0xF) << 8) |
          ((q[3] & 0xF) << 12));
      *reinterpret_cast<unsigned short*>(out + blk * 64 + lane * 2) = packed;
    }
  }
  if (threadIdx.x == 0)
    sx[row] = sc;
}

using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;

constexpr int kBM = 128, kBN = 128, kTI = 4, kTJ = 2;
constexpr int kPitch = 72;  // LDS row bytes: 64 + 8, conflict-free 8-byte reads

__device__ __forceinline__ int Mad24(int a, int b, int c) {
  return __mul24(a, b) + c;
}

// Y[t][ch] = sx[t] * d[ch] * sum_g sc[ch][g] * sum_{k in g} x4[t][k] *
// w4[ch][k] with signed int4 x signed int4 -> int32 WMMA accumulation per scale
// group, 128 tokens x 128 channels per block (4x2 WMMA tiles per wave),
// 128-element K slabs, double-buffered LDS. Tokens past `T` are computed on
// padded rows and not stored. N and K must be multiples of 128. The kernel body
// is the one benchmarked in tools/bench/wmma_gemm_probe.hip.
template<int G>
__global__ __launch_bounds__(kThreads) void GemmI4(
    const std::uint8_t* __restrict__ X, const std::uint8_t* __restrict__ W,
    const std::uint32_t* __restrict__ SC, const float* __restrict__ SX,
    const float* __restrict__ DW, float* __restrict__ Y, int T, int N, int K) {
  constexpr int kElems = 128;
  constexpr int kKSteps = kElems / 16;
  constexpr int kGps = kElems / G;
  constexpr int kStepsPerGroup = kKSteps / kGps;
  __shared__ __attribute__((aligned(16))) std::uint8_t s_ab[2][2][kBM * kPitch];
  __shared__ std::uint32_t s_sc[2][kBN];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int wave_t = wave >> 2;
  const int wave_n = wave & 3;
  const int tok0 = blockIdx.x * kBM;
  const int ch0 = blockIdx.y * kBN;
  const std::size_t row_bytes = static_cast<std::size_t>(K) / 2;
  const int nslab = K / kElems;
  const std::uint8_t* xsrc[2];
  const std::uint8_t* wsrc[2];
  int sdst[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int c = tid + p * kThreads;
    const int row = c >> 2, part = c & 3;
    xsrc[p] = X + static_cast<std::size_t>(tok0 + row) * row_bytes + part * 16;
    wsrc[p] = W + static_cast<std::size_t>(ch0 + row) * row_bytes + part * 16;
    sdst[p] = row * kPitch + part * 16;
  }
  uint4 xr[2], wr[2];
  std::uint32_t scr = 0;
  const auto load_global = [&](int slab) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      xr[p] = *reinterpret_cast<const uint4*>(xsrc[p] + slab * 64);
      wr[p] = *reinterpret_cast<const uint4*>(wsrc[p] + slab * 64);
    }
    if (tid < kBN)
      scr = SC[static_cast<std::size_t>(slab) * N + ch0 + tid];
  };
  const auto store_lds = [&](int buf) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      std::uint8_t* a = &s_ab[buf][0][sdst[p]];
      std::uint8_t* b = &s_ab[buf][1][sdst[p]];
      *reinterpret_cast<uint2*>(a) = make_uint2(xr[p].x, xr[p].y);
      *reinterpret_cast<uint2*>(a + 8) = make_uint2(xr[p].z, xr[p].w);
      *reinterpret_cast<uint2*>(b) = make_uint2(wr[p].x, wr[p].y);
      *reinterpret_cast<uint2*>(b + 8) = make_uint2(wr[p].z, wr[p].w);
    }
    if (tid < kBN)
      s_sc[buf][tid] = scr;
  };
  i32x8 acc[kTI][kTJ];
  int ti[kTI][kTJ][8];
#pragma unroll
  for (int i = 0; i < kTI; ++i)
#pragma unroll
    for (int j = 0; j < kTJ; ++j) {
      acc[i][j] = i32x8{};
#pragma unroll
      for (int e = 0; e < 8; ++e)
        ti[i][j][e] = 0;
    }
  const int a_row = wave_t * 64 + (lane & 15);
  const int b_row = wave_n * 32 + (lane & 15);
  const int b_ch[kTJ] = {wave_n * 32 + (lane & 15),
                         wave_n * 32 + 16 + (lane & 15)};
  load_global(0);
  store_lds(0);
  __syncthreads();
  for (int slab = 0; slab < nslab; ++slab) {
    const int cur = slab & 1;
    if (slab + 1 < nslab)
      load_global(slab + 1);
    const std::uint8_t* la = s_ab[cur][0];
    const std::uint8_t* lb = s_ab[cur][1];
    std::uint32_t scw[kTJ];
#pragma unroll
    for (int j = 0; j < kTJ; ++j)
      scw[j] = s_sc[cur][b_ch[j]];
#pragma unroll
    for (int g = 0; g < kGps; ++g) {
#pragma unroll
      for (int st = 0; st < kStepsPerGroup; ++st) {
        const int ks = g * kStepsPerGroup + st;
        i32x2 a[kTI], b[kTJ];
#pragma unroll
        for (int i = 0; i < kTI; ++i) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              la + (a_row + i * 16) * kPitch + ks * 8);
          __builtin_memcpy(&a[i], &v, 8);
        }
#pragma unroll
        for (int j = 0; j < kTJ; ++j) {
          const uint2 v = *reinterpret_cast<const uint2*>(
              lb + (b_row + j * 16) * kPitch + ks * 8);
          __builtin_memcpy(&b[j], &v, 8);
        }
#pragma unroll
        for (int i = 0; i < kTI; ++i)
#pragma unroll
          for (int j = 0; j < kTJ; ++j)
            acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a[i], true, b[j], st == 0 ? i32x8{} : acc[i][j], true);
      }
#pragma unroll
      for (int j = 0; j < kTJ; ++j) {
        const int scale = static_cast<int>((scw[j] >> (8 * g)) & 0xFFU);
#pragma unroll
        for (int i = 0; i < kTI; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e)
            ti[i][j][e] = Mad24(scale, acc[i][j][e], ti[i][j][e]);
      }
    }
    if (slab + 1 < nslab)
      store_lds(cur ^ 1);
    __syncthreads();
  }
  const int half = lane >> 4;
#pragma unroll
  for (int j = 0; j < kTJ; ++j) {
    const int ch = ch0 + b_ch[j];
    const float dw = DW[ch];
#pragma unroll
    for (int i = 0; i < kTI; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int tok = tok0 + wave_t * 64 + i * 16 + 2 * e + half;
        if (tok < T)
          Y[static_cast<std::size_t>(tok) * N + ch] =
              static_cast<float>(ti[i][j][e]) * (SX[tok] * dw);
      }
  }
}

// One device scratch per thread, grown on demand, for the emulated activation
// rows. Prompt processing is never graph-captured, so allocating here is safe.
thread_local __half* g_scratch = nullptr;
thread_local std::size_t g_scratch_bytes = 0;

__half* Scratch(std::size_t bytes) {
  if (bytes > g_scratch_bytes) {
    if (g_scratch != nullptr)
      (void)hipFree(g_scratch);
    g_scratch = nullptr;
    g_scratch_bytes = 0;
    if (hipMalloc(reinterpret_cast<void**>(&g_scratch), bytes) != hipSuccess)
      return nullptr;
    g_scratch_bytes = bytes;
  }
  return g_scratch;
}

// A second per-thread scratch for the integer path (nibbles + scales).
thread_local std::uint8_t* g_bytes = nullptr;
thread_local std::size_t g_bytes_cap = 0;

std::uint8_t* ScratchBytes(std::size_t bytes) {
  if (bytes > g_bytes_cap) {
    if (g_bytes != nullptr)
      (void)hipFree(g_bytes);
    g_bytes = nullptr;
    g_bytes_cap = 0;
    if (hipMalloc(reinterpret_cast<void**>(&g_bytes), bytes) != hipSuccess)
      return nullptr;
    g_bytes_cap = bytes;
  }
  return g_bytes;
}

struct Plan {
  const void* w;
  const __half* x;
  bool ok{true};  // false: a required rotation could not run (fail the call)
};

// Integer W4A4 prefill GEMM for a packed tensor: quantizes the F16 rows to
// int4 (with the registered rotation), then runs the iu4 WMMA kernel.
bool RunReal(const PackedWeights& pk, bool rotated, const __half* x, float* out,
             std::size_t rows, hipStream_t stream) {
  const Knobs& o = Opt();
  const std::size_t rows_pad = (rows + kBM - 1) / kBM * kBM;
  const std::size_t x4_bytes = rows_pad * (pk.k / 2);
  std::uint8_t* buf = ScratchBytes(x4_bytes + rows_pad * sizeof(float));
  if (buf == nullptr)
    return false;
  float* sx = reinterpret_cast<float*>(buf + x4_bytes);
  const bool fast = pk.k % 128 == 0 && pk.k <= kFastMaxBlocks * 8 * 128 &&
                    (!rotated || o.had == 128);
  if (fast && rotated) {
    hipLaunchKernelGGL(QuantRowsI4Fast<true>, dim3(static_cast<unsigned>(rows)),
                       dim3(kThreads), 0, stream, x, buf, sx,
                       static_cast<int>(pk.k), o.dense.clip);
  } else if (fast) {
    hipLaunchKernelGGL(QuantRowsI4Fast<false>,
                       dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                       stream, x, buf, sx, static_cast<int>(pk.k),
                       o.dense.clip);
  } else {
    hipLaunchKernelGGL(QuantRowsI4, dim3(static_cast<unsigned>(rows)),
                       dim3(kThreads), pk.k * sizeof(float), stream, x, buf, sx,
                       static_cast<int>(pk.k), rotated ? o.had : 0,
                       o.dense.clip);
  }
  const dim3 grid(static_cast<unsigned>(rows_pad / kBM), pk.n / kBN);
  const int t = static_cast<int>(rows), n = static_cast<int>(pk.n),
            k = static_cast<int>(pk.k);
  switch (pk.group) {
    case 32:
      hipLaunchKernelGGL(GemmI4<32>, grid, dim3(kThreads), 0, stream, buf,
                         pk.w4, pk.sc, sx, pk.d, out, t, n, k);
      break;
    case 64:
      hipLaunchKernelGGL(GemmI4<64>, grid, dim3(kThreads), 0, stream, buf,
                         pk.w4, pk.sc, sx, pk.d, out, t, n, k);
      break;
    case 128:
      hipLaunchKernelGGL(GemmI4<128>, grid, dim3(kThreads), 0, stream, buf,
                         pk.w4, pk.sc, sx, pk.d, out, t, n, k);
      break;
    default:
      return false;
  }
  return true;
}

// A third per-thread scratch for call-mode outliers: k column maxima (u32)
// followed by k keep flags.
thread_local std::uint8_t* g_cols = nullptr;
thread_local std::size_t g_cols_cap = 0;

std::uint8_t* ScratchCols(std::size_t bytes) {
  if (bytes > g_cols_cap) {
    if (g_cols != nullptr)
      (void)hipFree(g_cols);
    g_cols = nullptr;
    g_cols_cap = 0;
    if (hipMalloc(reinterpret_cast<void**>(&g_cols), bytes) != hipSuccess)
      return nullptr;
    g_cols_cap = bytes;
  }
  return g_cols;
}

// The activation recipe of a registered tensor: the routed recipe for roles
// 32/64, the dense one otherwise; GUFO_SPIKE_A8_LAYERS turns a quantized
// grid into int8 (same group and clip) for the listed layers.
ActRecipe RecipeFor(const Entry& e) {
  const Knobs& o = Opt();
  ActRecipe r =
      (e.role & (kRoutedGateUp | kRoutedDown)) != 0 ? o.routed : o.dense;
  if (r.grid != kActNone && o.a8_layers.count(e.layer) != 0)
    r.grid = kActInt8;
  return r;
}

void LaunchFq(const __half* x, __half* y, std::size_t rows, std::size_t k,
              const ActLaunch& a, hipStream_t stream) {
  const bool flags =
      a.grid != kActNone && (a.outliers > 0 || a.keep != nullptr);
  hipLaunchKernelGGL(FqRows, dim3(static_cast<unsigned>(rows)), dim3(kThreads),
                     k * sizeof(float) + (flags ? k : 0), stream, x, y,
                     static_cast<int>(k), a);
}

// Emulates `a` on rows x k into `buf`. Call-mode outliers first rotate into
// `buf` (when rotating), pick the call's column set from the (rotated) rows,
// then quantize in place. False when a scratch allocation fails.
bool EmulateRows(const __half* x, __half* buf, std::size_t rows, std::size_t k,
                 ActLaunch a, hipStream_t stream) {
  const Knobs& o = Opt();
  if (a.grid == kActNone || o.outliers == 0 ||
      o.outlier_mode != OutlierMode::kCall) {
    a.outliers = a.grid == kActNone ? 0 : o.outliers;
    LaunchFq(x, buf, rows, k, a, stream);
    return true;
  }
  std::uint8_t* cols = ScratchCols(k * sizeof(unsigned) + k);
  if (cols == nullptr)
    return false;
  auto* colmax = reinterpret_cast<unsigned*>(cols);
  std::uint8_t* keep = cols + k * sizeof(unsigned);
  const __half* src = x;
  if (a.had > 0) {
    ActLaunch rot;
    rot.had = a.had;
    LaunchFq(x, buf, rows, k, rot, stream);
    src = buf;
    a.had = 0;
  }
  (void)hipMemsetAsync(colmax, 0, k * sizeof(unsigned), stream);
  const unsigned splits = static_cast<unsigned>(
      std::min<std::size_t>(64, std::max<std::size_t>(1, rows / 32)));
  hipLaunchKernelGGL(
      ColAbsMax,
      dim3(static_cast<unsigned>((k + kThreads - 1) / kThreads), splits),
      dim3(kThreads), 0, stream, src, rows, static_cast<int>(k), colmax);
  hipLaunchKernelGGL(SelectColumns, dim3(1), dim3(kThreads),
                     k * sizeof(float) + k, stream, colmax, static_cast<int>(k),
                     std::min(o.outliers, static_cast<int>(k)), keep);
  a.outliers = 0;
  a.keep = keep;
  LaunchFq(src, buf, rows, k, a, stream);
  return true;
}

// True when a registered tensor has spike work: a shadow or packed weight, a
// rotation, or a quantizing activation recipe.
bool Active(const Entry& e) {
  return e.shadow != nullptr || e.packed.w4 != nullptr ||
         (e.rotated && Opt().had > 0) || RecipeFor(e).grid != kActNone;
}

// Per-role dispatch counters (host side), per kind:
//   kFull   the recipe (prefill call at or above MIN_ROWS),
//   kSmall  MIN_ROWS_MODE=act below MIN_ROWS or outside prefill: W4 shadow
//           (and rotation) with unquantized activations, F16 GEMM route,
//   kGemv   MIN_ROWS_MODE=act decode GEMV (<= 8 rows): W4 shadow, rotation.
// The first dispatch of every (role, K, kind) is logged so a run shows which
// targeted families really reached the spike; Init registers an exit-time
// summary.
enum DispatchKind : int { kFull = 0, kSmall = 1, kGemv = 2 };
constexpr const char* kRoleNames[7] = {
    "ssm_in", "ssm_out",        "attn_qkv",   "attn_out",
    "shared", "routed_gate_up", "routed_down"};
constexpr const char* kKindText[3] = {"prefill", "w4-only", "w4-only gemv"};
std::size_t g_dispatch_calls[3][7] = {};
std::size_t g_dispatch_rows[3][7] = {};
std::set<std::tuple<unsigned, std::size_t, int>> g_dispatch_seen;

// The shared role has two widths (gate/up K = 2560, down K = 640), so each
// (role, K) shows up separately.
void NoteDispatch(unsigned role, std::size_t rows, std::size_t k,
                  DispatchKind kind = kFull) {
  for (int b = 0; b < 7; ++b) {
    if (role != (1U << b))
      continue;
    ++g_dispatch_calls[kind][b];
    g_dispatch_rows[kind][b] += rows;
    if (g_dispatch_seen.emplace(role, k, kind).second)
      std::fprintf(stderr, "spike: first %s dispatch of %s (k=%zu, %zu rows)\n",
                   kKindText[kind], kRoleNames[b], k, rows);
  }
}

void PrintDispatchSummary() {
  for (int kind = 0; kind < 3; ++kind) {
    std::string s;
    for (int b = 0; b < 7; ++b)
      s += std::string(" ") + kRoleNames[b] + "=" +
           std::to_string(g_dispatch_calls[kind][b]) + "/" +
           std::to_string(g_dispatch_rows[kind][b]);
    std::fprintf(stderr, "spike: %s dispatches (calls/rows):%s\n",
                 kKindText[kind], s.c_str());
  }
}

bool Capturing(hipStream_t stream) {
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  return hipStreamIsCapturing(stream, &st) == hipSuccess &&
         st != hipStreamCaptureStatusNone;
}

// Looks `w` up; emulates the activation grid for its rows when registered.
// Calls with fewer than GUFO_SPIKE_MIN_ROWS rows (and calls outside prefill)
// keep the original weights and activations (MIN_ROWS_MODE=all, the path a
// deployment would take below the threshold), or with MIN_ROWS_MODE=act the
// W4 shadow and its rotation with unquantized activations (weights 4-bit at
// all times, A4 only on big prefill calls). A required rotation that cannot
// run fails the call (ok = false) and never falls through to unrotated rows.
Plan Prepare(bool prefill, const void* w, const __half* x, std::size_t rows,
             std::size_t k, hipStream_t stream) {
  const Knobs& o = Opt();
  const int rot = RotationOf(w);
  const auto rotate_only = [&]() -> Plan {
    if (rows == 0 || k == 0)
      return {w, x};
    if (Capturing(stream)) {
      std::fprintf(stderr, "spike: routed rotation inside graph capture\n");
      return {w, x, false};
    }
    __half* buf = Scratch(rows * k * sizeof(__half));
    ActLaunch a;
    a.grid = kActNone;
    a.had = rot;
    if (buf == nullptr || !EmulateRows(x, buf, rows, k, a, stream)) {
      std::fprintf(stderr, "spike: routed rotation failed\n");
      return {w, x, false};
    }
    return {w, buf};
  };
  const bool small = !prefill || rows < o.min_rows;
  if (!o.enabled || (small && !o.min_rows_act))
    return rot != 0 ? rotate_only() : Plan{w, x};
  const auto it = Registry().find(w);
  if (it == Registry().end())
    return rot != 0 ? rotate_only() : Plan{w, x};
  const Entry& e = it->second;
  const bool rotate = (e.rotated && o.had > 0) || rot != 0;
  if (small) {
    // MIN_ROWS_MODE=act: W4 shadow and rotation only; outside prefill a
    // captured graph cannot take the scratch path, so it keeps the original
    // (consistent) weights and rows.
    if (e.shadow == nullptr && !rotate)
      return {w, x};
    if (Capturing(stream))
      return rot != 0 ? rotate_only() : Plan{w, x};
    Plan p{e.shadow != nullptr ? e.shadow : w, x};
    if (rotate && rows > 0 && k > 0) {
      __half* buf = Scratch(rows * k * sizeof(__half));
      ActLaunch a;
      a.had = rot != 0 ? rot : o.had;
      if (buf == nullptr || !EmulateRows(x, buf, rows, k, a, stream)) {
        std::fprintf(stderr, "spike: rotation failed\n");
        return {w, x, false};
      }
      p.x = buf;
    }
    NoteDispatch(e.role, rows, k, kSmall);
    return p;
  }
  if (Active(e))
    NoteDispatch(e.role, rows, k, kFull);
  Plan p{e.shadow != nullptr ? e.shadow : w, x};
  ActRecipe r = RecipeFor(e);
  if (r.group != 0 && k % static_cast<std::size_t>(r.group) != 0) {
    static bool warned = false;
    if (!warned)
      std::fprintf(stderr,
                   "spike: width %zu not a multiple of group %d; per-row "
                   "scales for such calls\n",
                   k, r.group);
    warned = true;
    r.group = 0;
  }
  if ((r.grid != kActNone || rotate) && rows > 0 && k > 0) {
    __half* buf = Scratch(rows * k * sizeof(__half));
    ActLaunch a;
    a.grid = r.grid;
    a.group = r.group;
    a.clip = r.clip;
    a.had = rot != 0 ? rot : rotate ? o.had : 0;
    a.clip_search = o.clip_search ? 1 : 0;
    a.jitter = JitterKey(o.jitter, e.role, e.layer);
    if (buf != nullptr && EmulateRows(x, buf, rows, k, a, stream)) {
      p.x = buf;
    } else if (rotate) {
      std::fprintf(stderr, "spike: rotation failed\n");
      p.ok = false;
    }
  }
  return p;
}

// MIN_ROWS_MODE=act decode GEMV: one row-rotation buffer per executor stream,
// allocated at Init (decode may be graph-captured; no allocation later).
constexpr std::uint32_t kGemvRows = 8;  // the executor's kVecBatch
std::unordered_map<hipStream_t, std::pair<float*, std::size_t>>& GemvBuffers() {
  static std::unordered_map<hipStream_t, std::pair<float*, std::size_t>> b;
  return b;
}

// Row rotation of F32 rows (block Hadamard, the sign vector and
// normalization of FqRows). Dynamic LDS: k floats.
__global__ __launch_bounds__(kThreads) void RotateRowsF32(
    const float* __restrict__ x, float* __restrict__ y, int k, int had) {
  extern __shared__ float s[];
  const std::size_t row = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += kThreads)
    s[i] = x[row * k + i];
  __syncthreads();
  BlockHadamard(s, k, had);
  for (int i = threadIdx.x; i < k; i += kThreads)
    y[row * k + i] = s[i];
}

// A fourth per-thread scratch: F32 activations narrowed to F16 for callers
// whose own route has no F16 staging of this width (DenseF32Gemm).
thread_local __half* g_narrow = nullptr;
thread_local std::size_t g_narrow_bytes = 0;

__half* ScratchNarrow(std::size_t bytes) {
  if (bytes > g_narrow_bytes) {
    if (g_narrow != nullptr)
      (void)hipFree(g_narrow);
    g_narrow = nullptr;
    g_narrow_bytes = 0;
    if (hipMalloc(reinterpret_cast<void**>(&g_narrow), bytes) != hipSuccess)
      return nullptr;
    g_narrow_bytes = bytes;
  }
  return g_narrow;
}

}  // namespace

bool Enabled() noexcept {
  return Opt().enabled;
}

const void* ShadowOf(const void* w) {
  const auto it = Registry().find(w);
  return it == Registry().end() ? nullptr : it->second.shadow;
}

namespace {
std::size_t g_shadow_bytes = 0;
std::size_t g_counts[8] = {};
}  // namespace

bool RegisterTensor(const DeviceTensor& t, unsigned role, int layer, bool dense,
                    hipStream_t stream) {
  const Knobs& o = Opt();
  if (t.empty() || (o.targets & role) == 0 || o.keep.count(layer) != 0)
    return false;
  Entry e;
  e.role = role;
  e.layer = layer;
  e.cols = t.cols;
  if (dense && o.wgroup > 0) {
    // GUFO_SPIKE_WGROUP=1: one scale per row (group = the row width).
    const int wgroup = o.wgroup == 1 ? static_cast<int>(t.cols) : o.wgroup;
    const bool fits =
        t.type == core::GgmlType::kQ8_0 && t.cols % 32 == 0 &&
        t.cols % wgroup == 0 &&
        (o.had == 0 || (o.had == kFullRotation ? FullRotationBlock(t.cols) != 0
                                               : t.cols % o.had == 0));
    if (!fits) {
      std::fprintf(stderr,
                   "spike: layer %d role %u skipped (not Q8_0 or width)\n",
                   layer, role);
      return false;
    }
    if (o.real) {
      const bool kernel_ok =
          t.cols % 128 == 0 && t.rows % 128 == 0 &&
          (o.wgroup == 32 || o.wgroup == 64 || o.wgroup == 128);
      if (!kernel_ok) {
        std::fprintf(
            stderr,
            "spike: layer %d role %u not packable for the integer kernel\n",
            layer, role);
        return false;
      }
      PackedWeights pk;
      pk.n = t.rows;
      pk.k = t.cols;
      pk.group = static_cast<std::uint32_t>(o.wgroup);
      const std::size_t w4_bytes =
          static_cast<std::size_t>(t.rows) * (t.cols / 2);
      const std::size_t sc_bytes =
          static_cast<std::size_t>(t.cols / 128) * t.rows * 4;
      if (hipMalloc(reinterpret_cast<void**>(&pk.w4), w4_bytes) != hipSuccess ||
          hipMalloc(reinterpret_cast<void**>(&pk.sc), sc_bytes) != hipSuccess ||
          hipMalloc(reinterpret_cast<void**>(&pk.d), t.rows * sizeof(float)) !=
              hipSuccess) {
        std::fprintf(stderr, "spike: packed allocation failed\n");
        return false;
      }
      (void)hipMemsetAsync(pk.sc, 0, sc_bytes, stream);
      hipLaunchKernelGGL(RequantRow<true>, dim3(t.rows), dim3(kThreads),
                         t.cols * sizeof(float), stream,
                         static_cast<std::uint8_t*>(t.data),
                         static_cast<int>(t.rows), static_cast<int>(t.cols),
                         o.wgroup, o.had, 1, pk.w4, pk.sc, pk.d, 4, 0U);
      e.packed = pk;
      e.rotated = o.had > 0;
      g_shadow_bytes += w4_bytes + sc_bytes + t.rows * sizeof(float);
    }
    if (!o.real || o.check) {
      const std::size_t bytes =
          static_cast<std::size_t>(t.rows) * (t.cols / 32) * 34;
      void* shadow = nullptr;
      if (hipMalloc(&shadow, bytes) != hipSuccess) {
        std::fprintf(stderr, "spike: shadow allocation failed (%zu bytes)\n",
                     bytes);
        return false;
      }
      (void)hipMemcpyAsync(shadow, t.data, bytes, hipMemcpyDeviceToDevice,
                           stream);
      hipLaunchKernelGGL(RequantRow<false>, dim3(t.rows), dim3(kThreads),
                         t.cols * sizeof(float), stream,
                         static_cast<std::uint8_t*>(shadow),
                         static_cast<int>(t.rows), static_cast<int>(t.cols),
                         wgroup, o.had, o.scale6, nullptr, nullptr, nullptr,
                         o.wbits, JitterKey(o.jitter, role, layer));
      e.shadow = shadow;
      e.rotated = o.had > 0;
      g_shadow_bytes += bytes;
    }
  }
  Registry()[t.data] = e;
  for (int b = 0; b < 7; ++b)
    if (role == (1U << b))
      ++g_counts[b];
  return true;
}

void Init(const DeviceModel& model, hipStream_t stream) {
  static bool done = false;
  if (done)
    return;
  done = true;
  const Knobs& o = Opt();
  for (const std::string& w : o.warnings)
    std::fprintf(stderr, "spike: warning: %s\n", w.c_str());
  for (const std::string& n : UnknownEnvKnobs())
    std::fprintf(stderr, "spike: warning: %s is not a knob of this binary\n",
                 n.c_str());
  if (!o.enabled)
    return;
  std::atexit(PrintDispatchSummary);
  const auto& layers = model.layers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const DeviceLayer& l = layers[i];
    const int li = static_cast<int>(i);
    if (l.linear()) {
      RegisterTensor(l.ssm_in, kSsmIn, li, true, stream);
      RegisterTensor(l.ssm_out, kSsmOut, li, true, stream);
    } else {
      RegisterTensor(l.attn_qkv, kAttnQkv, li, true, stream);
      RegisterTensor(l.attn_out, kAttnOut, li, true, stream);
    }
    RegisterTensor(l.shexp_gate, kShared, li, true, stream);
    RegisterTensor(l.shexp_up, kShared, li, true, stream);
    RegisterTensor(l.shexp_down, kShared, li, true, stream);
    RegisterTensor(l.ffn_gate_exps, kRoutedGateUp, li, false, stream);
    RegisterTensor(l.ffn_up_exps, kRoutedGateUp, li, false, stream);
    RegisterTensor(l.ffn_down_exps, kRoutedDown, li, false, stream);
  }
  // GUFO_SPIKE_ROUTED_DOWN4: registered Q5_1 down experts get 4-bit codes in
  // place (every route then reads the same 4-bit weights).
  std::size_t down4 = 0;
  if (o.routed_down4) {
    for (const DeviceLayer& l : layers) {
      const DeviceTensor& d = l.ffn_down_exps;
      if (!d.empty() && Registry().count(d.data) != 0 &&
          routed_iu4::PrepareDown(d.data, static_cast<WeightType>(d.type),
                                  std::size_t{d.rows} * d.experts, d.cols,
                                  stream))
        ++down4;
    }
  }
  // GUFO_SPIKE_ROUTED_HAD: the GGUF stores every main-layer routed expert
  // matrix rotated; all their calls rotate the activations (Prepare, the
  // iu4 hook, and RoutedExperts via RoutedRotation for decode rows).
  if (o.routed_had != 0) {
    for (const DeviceLayer& l : layers)
      for (const DeviceTensor* t :
           {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps})
        if (!t->empty())
          Rotated()[t->data] = o.routed_had;
  }
  // MIN_ROWS_MODE=act: decode GEMVs over rotated W4 shadows rotate their rows
  // into this stream's buffer (allocated now; decode may be graph-captured).
  if (o.min_rows_act && o.had > 0) {
    std::size_t cols = 0;
    for (const auto& [ptr, e] : Registry())
      if (e.shadow != nullptr && e.rotated)
        cols = std::max<std::size_t>(cols, e.cols);
    float* buf = nullptr;
    if (cols > 0 && hipMalloc(reinterpret_cast<void**>(&buf),
                              kGemvRows * cols * sizeof(float)) == hipSuccess)
      GemvBuffers()[stream] = {buf, kGemvRows * cols};
    else if (cols > 0)
      std::fprintf(stderr, "spike: GEMV rotation buffer allocation failed\n");
  }
  (void)hipStreamSynchronize(stream);
  std::fprintf(stderr,
               "spike: routed_real=%d routed_down4=%d (%zu down matrices "
               "re-quantized) routed_check=%d routed_had=%d (%zu matrices) "
               "routed_tall=%d\n",
               o.routed_real ? 1 : 0, o.routed_down4 ? 1 : 0, down4,
               o.routed_check, o.routed_had, Rotated().size(),
               o.routed_tall ? 1 : 0);
  std::string keep, a8;
  for (int l : o.keep)
    keep += (keep.empty() ? "" : ",") + std::to_string(l);
  for (int l : o.a8_layers)
    a8 += (a8.empty() ? "" : ",") + std::to_string(l);
  std::fprintf(
      stderr,
      "spike: dense=%s routed=%s had=%d wgroup=%d wbits=%d scale6=%d "
      "targets=%u min_rows=%zu/%s aclip_search=%d outliers=%d/%s keep=[%s] "
      "a8_layers=[%s] real=%d jitter=%u; registered ssm_in=%zu ssm_out=%zu "
      "attn_qkv=%zu attn_out=%zu shared=%zu routed_gate_up=%zu "
      "routed_down=%zu; shadow weights %.2f GiB\n",
      RecipeText(o.dense).c_str(), RecipeText(o.routed).c_str(), o.had,
      o.wgroup, o.wbits, o.scale6, o.targets, o.min_rows,
      o.min_rows_act ? "act" : "all", o.clip_search ? 1 : 0, o.outliers,
      o.outlier_mode == OutlierMode::kCall ? "call" : "row", keep.c_str(),
      a8.c_str(), o.real ? 1 : 0, o.jitter, g_counts[0], g_counts[1],
      g_counts[2], g_counts[3], g_counts[4], g_counts[5], g_counts[6],
      static_cast<double>(g_shadow_bytes) / (1ULL << 30));
}

bool Handles(const void* w) {
  if (!Opt().real)
    return false;
  const auto it = Registry().find(w);
  return it != Registry().end() && it->second.packed.w4 != nullptr;
}

bool DenseF16Gemm(bool prefill, const void* w, const __half* x, float* out,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  if (prefill && Opt().real && batch >= Opt().min_rows) {
    const auto it = Registry().find(w);
    if (it != Registry().end() && it->second.packed.w4 != nullptr &&
        it->second.packed.n == m && it->second.packed.k == k &&
        RunReal(it->second.packed, it->second.rotated, x, out, batch, stream)) {
      NoteDispatch(it->second.role, batch, k);
      return true;
    }
  }
  const Plan p = Prepare(prefill, w, x, batch, k, stream);
  return p.ok && rocm::DenseF16Gemm(p.w, p.x, out, batch, m, k, stream);
}

bool Covers(bool prefill, const void* w, std::size_t rows, hipStream_t stream) {
  const Knobs& o = Opt();
  if (!o.enabled || rows == 0)
    return false;
  const auto it = Registry().find(w);
  if (it == Registry().end())
    return false;
  const Entry& e = it->second;
  if (prefill && rows >= o.min_rows)
    return Active(e);
  // Small or non-prefill calls: only MIN_ROWS_MODE=act has work (the W4
  // shadow and its rotation), and only outside graph capture (scratch).
  return o.min_rows_act && (e.shadow != nullptr || e.rotated) &&
         !Capturing(stream);
}

const void* GemvWeight(const void* w) {
  const Knobs& o = Opt();
  if (!o.enabled || !o.min_rows_act)
    return w;
  const auto it = Registry().find(w);
  return it != Registry().end() && it->second.shadow != nullptr
             ? it->second.shadow
             : w;
}

bool GemvOperands(const void* w, const float* x, std::uint32_t rows,
                  std::uint32_t k, const void** w_out, const float** x_out,
                  hipStream_t stream) {
  *w_out = w;
  *x_out = x;
  const Knobs& o = Opt();
  if (!o.enabled || !o.min_rows_act)
    return true;
  const auto it = Registry().find(w);
  if (it == Registry().end() || it->second.shadow == nullptr)
    return true;
  const Entry& e = it->second;
  if (e.rotated && o.had > 0) {
    const auto b = GemvBuffers().find(stream);
    if (b == GemvBuffers().end() || std::size_t{rows} * k > b->second.second) {
      std::fprintf(stderr,
                   "spike: no GEMV rotation buffer for %u x %u on this "
                   "stream\n",
                   rows, k);
      return false;  // never pair the rotated shadow with unrotated rows
    }
    hipLaunchKernelGGL(RotateRowsF32, dim3(rows), dim3(kThreads),
                       k * sizeof(float), stream, x, b->second.first,
                       static_cast<int>(k), o.had);
    *x_out = b->second.first;
  }
  *w_out = e.shadow;
  NoteDispatch(e.role, rows, k, kGemv);
  return true;
}

bool DenseF32Gemm(bool prefill, const void* w, const float* x, float* out,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  __half* xh = ScratchNarrow(batch * k * sizeof(__half));
  if (xh == nullptr)
    return false;
  rocm::NarrowActivations(x, xh, false, batch * k, stream);
  return DenseF16Gemm(prefill, w, xh, out, batch, m, k, stream);
}

bool DenseF16GemmEmulated(const void* w, const __half* x, float* out,
                          std::size_t batch, std::size_t m, std::size_t k,
                          hipStream_t stream) {
  const auto it = Registry().find(w);
  if (it == Registry().end() || it->second.shadow == nullptr)
    return false;
  const Knobs& o = Opt();
  const Entry& e = it->second;
  __half* buf = Scratch(batch * k * sizeof(__half));
  if (buf == nullptr)
    return false;
  ActLaunch a;  // the integer kernel's arithmetic: per-row int4 by default
  a.grid = o.dense.grid == kActNone ? kActInt4 : o.dense.grid;
  a.group = o.dense.group;
  a.clip = o.dense.clip;
  a.had = e.rotated && o.had > 0 ? o.had : 0;
  LaunchFq(x, buf, batch, k, a, stream);
  return rocm::DenseF16Gemm(e.shadow, buf, out, batch, m, k, stream);
}

bool DenseF16SsmGemm(bool prefill, const void* w, const __half* x,
                     const float* conv_w, const float* history, float* qkvz,
                     float* convolved, std::uint32_t n_tokens, std::uint32_t m,
                     std::uint32_t k, std::uint32_t channels,
                     std::uint32_t kernel, hipStream_t stream) {
  if (prefill && Handles(w))
    return false;  // the integer path has no fused conv; the executor unfuses
  const Plan p = Prepare(prefill, w, x, n_tokens, k, stream);
  return p.ok &&
         rocm::DenseF16SsmGemm(p.w, p.x, conv_w, history, qkvz, convolved,
                               n_tokens, m, k, channels, kernel, stream);
}

bool AttentionF16Gemm(bool prefill, const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, const KvView& kv, std::uint32_t n_tokens,
                      const std::uint32_t* position, float theta, float eps,
                      hipStream_t stream,
                      const qwen::vision::DeviceRope* rope) {
  if (prefill && Handles(weights))
    return false;  // unfused projection; the executor checks Handles first
  const Plan p = Prepare(prefill, weights, input, n_tokens, 2560, stream);
  return p.ok &&
         rocm::AttentionF16Gemm(p.w, p.x, q_gamma, k_gamma, query, gate, kv,
                                n_tokens, position, theta, eps, stream, rope);
}

namespace {

/// The routed iu4 hook applies to prefill calls of registered routed
/// tensors with an int4 recipe the integer kernels implement.
bool RoutedReal(bool prefill, const void* w, unsigned role, std::size_t rows) {
  const Knobs& o = Opt();
  if (!prefill || !o.routed_real || rows < o.min_rows ||
      o.routed.grid != kActInt4 || o.routed.clip != 1.0F)
    return false;
  const auto it = Registry().find(w);
  return it != Registry().end() && it->second.role == role;
}

routed_iu4::HookOptions RoutedOptions(const void* w) {
  routed_iu4::HookOptions h;
  h.group = Opt().routed.group;
  h.check_calls = Opt().routed_check;
  h.had = RotationOf(w);
  h.shape = Opt().routed_tall ? routed_iu4::kTile128 : routed_iu4::kTile64;
  return h;
}

}  // namespace

int RoutedRotation(const void* w) noexcept {
  return RotationOf(w);
}

bool RoutedGatedF16Gemm(bool prefill, std::size_t x_rows, const void* gate,
                        const void* up, WeightType type, const __half* x,
                        const std::int32_t* tiles, std::uint32_t n_tiles,
                        std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  if (RoutedReal(prefill, gate, kRoutedGateUp, x_rows) &&
      routed_iu4::HookGateUp(x_rows, gate, up, type, x, tiles, n_tiles,
                             tile_rows, pad_bounds, rows_in, rows_out, out, m,
                             k, RoutedOptions(gate), stream)) {
    NoteDispatch(kRoutedGateUp, x_rows, k);
    return true;
  }
  const Plan p = Prepare(prefill, gate, x, x_rows, k, stream);
  if (!p.ok)
    return false;  // rotated experts must never see unrotated rows
  return rocm::RoutedGatedF16Gemm(gate, up, type, p.x, tiles, n_tiles,
                                  tile_rows, pad_bounds, rows_in, rows_out, out,
                                  m, k, stream);
}

bool RoutedF16Gemm(bool prefill, std::size_t x_rows, const void* w,
                   WeightType type, const __half* x, const std::int32_t* tiles,
                   std::uint32_t n_tiles, std::uint32_t tile_rows,
                   const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                   const std::int32_t* rows_out, const float* swiglu_gate,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream) {
  if (RoutedReal(prefill, w, kRoutedDown, x_rows) &&
      routed_iu4::HookDown(x_rows, w, type, x, tiles, n_tiles, tile_rows,
                           pad_bounds, rows_in, rows_out, swiglu_gate, out,
                           out_half, m, k, RoutedOptions(w), stream)) {
    NoteDispatch(kRoutedDown, x_rows, k);
    return true;
  }
  const Plan p = Prepare(prefill, w, x, x_rows, k, stream);
  if (!p.ok)
    return false;  // rotated experts must never see unrotated rows
  return rocm::RoutedF16Gemm(w, type, p.x, tiles, n_tiles, tile_rows,
                             pad_bounds, rows_in, rows_out, swiglu_gate, out,
                             out_half, m, k, stream);
}

}  // namespace gufo::models::qwen38_flash_next::rocm::spike
