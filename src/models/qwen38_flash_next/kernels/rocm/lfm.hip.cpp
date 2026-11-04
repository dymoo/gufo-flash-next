#include "src/models/qwen38_flash_next/kernels/rocm/lfm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kD = kLfmHeadDim;
constexpr std::uint32_t kGroup = kLfmHeads / kLfmKvHeads;  // 4 q heads
constexpr std::uint32_t kKvRow = kLfmKvHeads * kD;         // 512
constexpr std::uint32_t kTile = 64;                        // keys per tile
constexpr std::uint32_t kKPad = kD + 2;  // halves: rows 33 words apart
constexpr std::uint32_t kSplits = kMaxAttentionSplits;
constexpr std::uint32_t kRecord = kD + 2;  // max, sum, acc[64]
constexpr std::uint32_t kExperts = 64;

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

// --------------------------------------------- TurboQuant, 64-wide heads --
// A head row lives in one wave: lane L holds elements L and L + 32. Rows are
// rotated by R = H S / 8 (fixed +-1 signs S from kTqSigns, 64-point
// Walsh-Hadamard H, orthonormal and symmetric, so R^T = S H / 8), then
// stored as Lloyd-Max codes against the same integer centroids as the
// 256-wide format plus one FP16 least-squares scale (KvCodeBytes x 64/256):
//   tq4: 32 bytes, byte L = code(L) | code(L + 32) << 4.
//   tq3: 24 bytes, a 16-byte plane of low bit pairs (byte L / 2, nibble
//        L % 2 = low2(L) | low2(L + 32) << 2) then two u32 words of high bits
//        (bit L of word c is the high bit of element L + 32c).
//   tq8: 64 bytes, byte e the signed code of element e; the scale is the
//        row's absolute maximum / 127 (F16), codes rounded against it.

__device__ __forceinline__ void Hadamard64(float& a, float& b) {
  const std::uint32_t lane = threadIdx.x & 31U;
#pragma unroll
  for (std::uint32_t h = 1; h < 32; h <<= 1) {
    const bool upper = (lane & h) != 0;
    const float oa = __shfl_xor(a, h), ob = __shfl_xor(b, h);
    a = upper ? oa - a : a + oa;
    b = upper ? ob - b : b + ob;
  }
  const float sum = a + b, diff = a - b;
  a = sum * 0.125F;
  b = diff * 0.125F;
}

__device__ __forceinline__ void Signs64(float& a, float& b) {
  const std::uint32_t lane = threadIdx.x & 31U;
  if ((kTqSigns[0] >> lane) & 1U)
    a = -a;
  if ((kTqSigns[1] >> lane) & 1U)
    b = -b;
}

__device__ __forceinline__ void Rotate64(float& a, float& b) {
  Signs64(a, b);
  Hadamard64(a, b);
}

__device__ __forceinline__ void Unrotate64(float& a, float& b) {
  Hadamard64(a, b);
  Signs64(a, b);
}

template<KvCacheFormat kFmt>
constexpr std::uint32_t RowBytes64() {
  return KvCodeBytes(kFmt) * kD / 256;
}

/// Nearest signed centroid of a unit-scaled value: code and value.
template<bool k4>
__device__ __forceinline__ float Nearest(float u, std::uint32_t* code) {
  constexpr std::uint32_t half = k4 ? 8U : 4U;
  const float a = fabsf(u);
  std::uint32_t m = 0;
  float centroid = k4 ? kTq4Centroids[0] : kTq3Centroids[0];
#pragma unroll
  for (std::uint32_t j = 1; j < half; ++j) {
    const float lo = k4 ? kTq4Centroids[j - 1] : kTq3Centroids[j - 1];
    const float hi = k4 ? kTq4Centroids[j] : kTq3Centroids[j];
    const bool above = a > 0.5F * (lo + hi);
    m += above ? 1U : 0U;
    centroid = above ? hi : centroid;
  }
  *code = u < 0.0F ? half - 1 - m : half + m;
  return u < 0.0F ? -centroid : centroid;
}

/// Quantizes one rotated row (whole wave; lane holds elements L, L + 32).
template<KvCacheFormat kFmt>
__device__ __forceinline__ void Store64(float a, float b, std::uint8_t* codes,
                                        __half* scale) {
  constexpr bool k4 = kFmt == KvCacheFormat::kTq4;
  const std::uint32_t lane = threadIdx.x & 31U;
  if constexpr (kFmt == KvCacheFormat::kTq8) {
    const __half s =
        __float2half_rn(WaveMax(fmaxf(fabsf(a), fabsf(b))) / 127.0F);
    const float sf = __half2float(s);
    const float inv = sf > 0.0F ? 1.0F / sf : 0.0F;
    const auto code = [inv](float v) {
      return static_cast<std::uint8_t>(static_cast<std::int8_t>(
          fminf(fmaxf(rintf(v * inv), -127.0F), 127.0F)));
    };
    codes[lane] = code(a);
    codes[lane + 32] = code(b);
    if (lane == 0)
      *scale = s;
    return;
  }
  const float ss = WaveSum(a * a + b * b);
  const float inv = ss > 0.0F ? (k4 ? kTq4Unit : kTq3Unit) *
                                    rsqrtf(ss / static_cast<float>(kD))
                              : 0.0F;
  std::uint32_t ca = 0, cb = 0;
  const float va = Nearest<k4>(a * inv, &ca), vb = Nearest<k4>(b * inv, &cb);
  const float dot = WaveSum(a * va + b * vb), cc = WaveSum(va * va + vb * vb);
  if (lane == 0)
    *scale = __float2half_rn(cc > 0.0F ? dot / cc : 0.0F);
  if constexpr (k4) {
    codes[lane] = static_cast<std::uint8_t>(ca | cb << 4);
  } else {
    const std::uint32_t nib = (ca & 3U) | (cb & 3U) << 2;
    const std::uint32_t next = __shfl_down(nib, 1);
    if ((lane & 1U) == 0)
      codes[lane >> 1] = static_cast<std::uint8_t>(nib | next << 4);
    const std::uint32_t h0 = static_cast<std::uint32_t>(__ballot(ca >> 2));
    const std::uint32_t h1 = static_cast<std::uint32_t>(__ballot(cb >> 2));
    if (lane == 0) {
      reinterpret_cast<std::uint32_t*>(codes + 16)[0] = h0;
      reinterpret_cast<std::uint32_t*>(codes + 16)[1] = h1;
    }
  }
}

/// Element e (0..63) of a stored row, in rotated space, times its scale.
template<KvCacheFormat kFmt>
__device__ __forceinline__ float Load64(const std::uint8_t* codes, float scale,
                                        std::uint32_t e) {
  if constexpr (kFmt == KvCacheFormat::kTq8)
    return static_cast<float>(static_cast<std::int8_t>(codes[e])) * scale;
  constexpr bool k4 = kFmt == KvCacheFormat::kTq4;
  constexpr std::uint32_t half = k4 ? 8U : 4U;
  const std::uint32_t lane = e & 31U, c = e >> 5;
  std::uint32_t code;
  if constexpr (k4) {
    code = (codes[lane] >> (4 * c)) & 15U;
  } else {
    const std::uint32_t nib = (codes[lane >> 1] >> (4 * (lane & 1U))) & 15U;
    const std::uint32_t high =
        (reinterpret_cast<const std::uint32_t*>(codes + 16)[c] >> lane) & 1U;
    code = ((nib >> (2 * c)) & 3U) | high << 2;
  }
  const float v = code < half ? -(k4 ? kTq4Centroids[half - 1 - code]
                                     : kTq3Centroids[half - 1 - code])
                              : (k4 ? kTq4Centroids[code - half]
                                    : kTq3Centroids[code - half]);
  return v * scale;
}

// ---------------------------------------------------------- short conv --

/// One thread per (row, channel): y[t] = C[t] * (w0 bx[t-2] + w1 bx[t-1] +
/// w2 bx[t]) with bx[r] = B[r] x[r], and the carried state for r < 0
/// (state row 0 is bx[-2], row 1 bx[-1]). grid.z is the item.
__global__ void ShortConvKernel(const float* __restrict__ bcx,
                                std::uint32_t stride,
                                const float* __restrict__ w, const float* state,
                                float* __restrict__ y, std::uint32_t rows,
                                std::uint32_t channels,
                                const LfmConvItem* items) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  std::uint32_t row0 = 0;
  const float* st = state;
  if (items != nullptr) {
    const LfmConvItem& item = items[blockIdx.z];
    row0 = item.row;
    rows = item.rows;
    st = item.state;
  }
  if (i >= channels || t >= rows)
    return;
  const auto bx = [&](std::int32_t r) {
    if (r < 0)
      return st[static_cast<std::size_t>(2 + r) * channels + i];
    const float* row = bcx + static_cast<std::size_t>(row0 + r) * stride;
    return row[i] * row[2 * channels + i];
  };
  const auto ti = static_cast<std::int32_t>(t);
  const float* row = bcx + static_cast<std::size_t>(row0 + t) * stride;
  y[static_cast<std::size_t>(row0 + t) * channels + i] =
      row[channels + i] * (w[3 * i] * bx(ti - 2) + w[3 * i + 1] * bx(ti - 1) +
                           w[3 * i + 2] * bx(ti));
}

/// After ShortConvKernel: the state becomes the last two bx rows.
__global__ void ShortConvStateKernel(const float* __restrict__ bcx,
                                     std::uint32_t stride, float* state,
                                     std::uint32_t rows, std::uint32_t channels,
                                     const LfmConvItem* items) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t row0 = 0;
  float* st = state;
  if (items != nullptr) {
    const LfmConvItem& item = items[blockIdx.y];
    row0 = item.row;
    rows = item.rows;
    st = item.state;
  }
  if (i >= channels || rows == 0)
    return;
  const auto bx = [&](std::uint32_t r) {
    const float* row = bcx + static_cast<std::size_t>(row0 + r) * stride;
    return row[i] * row[2 * channels + i];
  };
  const float s0 = rows >= 2 ? bx(rows - 2) : st[channels + i];
  st[i] = s0;
  st[channels + i] = bx(rows - 1);
}

// ------------------------------------------------------ attention prep --

/// One work-group per token: eight waves over the 48 q/k/v heads (lane l
/// owns dims l and l + 32, the NEOX rotation pair and the TurboQuant lane
/// layout). TurboQuant caches store rotated K/V codes and the queries are
/// rotated alike, so attention runs in the rotated basis.
template<KvCacheFormat kFmt>
__global__ void PrepareKernel(const float* __restrict__ qkv,
                              const float* __restrict__ q_gamma,
                              const float* __restrict__ k_gamma, float* q,
                              KvView kv, const std::uint32_t* start_pos,
                              float theta, float eps,
                              const AttentionBatchItem* items) {
  constexpr bool kTq = kFmt != KvCacheFormat::kF16;
  std::uint32_t t = blockIdx.x, pos;
  if (items != nullptr) {
    const AttentionBatchItem& item = items[blockIdx.y];
    if (blockIdx.x >= item.rows)
      return;
    t = item.row + blockIdx.x;
    pos = *item.position + blockIdx.x;
    kv = item.kv;
  } else {
    pos = *start_pos + t;
  }
  const float* row = qkv + static_cast<std::size_t>(t) * kLfmQkvRow;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5;
  float sn = 0.0F, cs = 0.0F;
  sincosf(
      static_cast<float>(pos) * powf(theta, -2.0F * static_cast<float>(lane) /
                                                static_cast<float>(kD)),
      &sn, &cs);
  for (std::uint32_t h = wave; h < kLfmHeads + 2 * kLfmKvHeads; h += 8) {
    const bool is_q = h < kLfmHeads;
    const bool is_v = h >= kLfmHeads + kLfmKvHeads;
    const float* src = row + h * kD;  // q heads, then k heads, then v heads
    float a = src[lane], b = src[lane + 32];
    if (!is_v) {
      const float inv =
          rsqrtf(WaveSum(a * a + b * b) / static_cast<float>(kD) + eps);
      const float* g = is_q ? q_gamma : k_gamma;
      a *= inv * g[lane];
      b *= inv * g[lane + 32];
      const float ra = a * cs - b * sn;
      b = a * sn + b * cs;
      a = ra;
    }
    if (kTq)
      Rotate64(a, b);
    if (is_q) {
      float* dst = q + (static_cast<std::size_t>(t) * kLfmHeads + h) * kD;
      dst[lane] = a;
      dst[lane + 32] = b;
      continue;
    }
    const std::uint32_t g = h - kLfmHeads - (is_v ? kLfmKvHeads : 0U);
    const std::size_t slot = static_cast<std::size_t>(pos) * kLfmKvHeads + g;
    if constexpr (kTq) {
      Store64<kFmt>(a, b,
                    static_cast<std::uint8_t*>(is_v ? kv.v : kv.k) +
                        slot * RowBytes64<kFmt>(),
                    (is_v ? kv.v_scale : kv.k_scale) + slot);
    } else {
      __half* dst = static_cast<__half*>(is_v ? kv.v : kv.k) + slot * kD;
      dst[lane] = __float2half(a);
      dst[lane + 32] = __float2half(b);
    }
  }
}

/// Rotates attention output rows ([rows][32][64]) back from the TurboQuant
/// basis: one wave per head row.
__global__ void UnrotateKernel(float* x, std::uint32_t head_rows) {
  const std::uint32_t r = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
  if (r >= head_rows)
    return;
  const std::uint32_t lane = threadIdx.x & 31U;
  float* p = x + static_cast<std::size_t>(r) * kD;
  float a = p[lane], b = p[lane + 32];
  Unrotate64(a, b);
  p[lane] = a;
  p[lane + 32] = b;
}

// ----------------------------------------------------------- attention --

/// Flash attention for one KV head and QR consecutive query rows: K/V tiles
/// of 64 keys are staged in LDS once for the 4 * QR query vectors. Thread
/// tid owns output dim tid % 64 of query vectors tid / 64 + 4c. Split z of
/// `splits` covers a contiguous, tile-aligned share of the visible keys;
/// with more than one split the unnormalized result goes to a record.
template<std::uint32_t QR, KvCacheFormat kFmt = KvCacheFormat::kF16>
__global__ __launch_bounds__(kThreads) void AttentionKernel(
    const float* __restrict__ q, KvView kv, float* __restrict__ out,
    float* __restrict__ partials, const std::uint32_t* start_pos,
    std::uint32_t n_rows, const AttentionBatchItem* items) {
  constexpr std::uint32_t kVec = QR * kGroup;           // query vectors
  constexpr std::uint32_t kOut = kVec * kD / kThreads;  // outputs per thread
  constexpr std::uint32_t kScores = kVec * kTile / kThreads;
  static_assert(kOut >= 1 && kScores >= 1);
  __shared__ float qs[kVec][kD];
  __shared__ __align__(16) __half ks[kTile][kKPad];
  __shared__ __align__(16) __half vs[kTile][kD];
  __shared__ float ps[kVec][kTile + 1];
  __shared__ float m_row[kVec], l_row[kVec], scale_row[kVec];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t g = blockIdx.x;  // KV head
  const std::uint32_t t0 = blockIdx.y * QR;
  std::uint32_t split = blockIdx.z, splits = gridDim.z, base_row = 0, pos0;
  std::size_t record_base = 0;
  bool batched = false;
  if (items != nullptr) {
    const AttentionBatchItem& item = items[blockIdx.z / kSplits];
    split = blockIdx.z % kSplits;
    splits = item.splits;
    if (split >= splits || t0 >= item.rows)
      return;
    n_rows = item.rows;
    base_row = item.row;
    pos0 = *item.position;
    kv = item.kv;
    record_base = item.partial;
    batched = true;
  } else {
    if (t0 >= n_rows)
      return;
    pos0 = *start_pos;
  }
  const std::uint32_t live = min(QR, n_rows - t0);
  const std::uint32_t n_keys = pos0 + t0 + live;  // keys the last row sees
  const std::uint32_t per =
      ((n_keys + splits - 1) / splits + kTile - 1) / kTile * kTile;
  const std::uint32_t k_begin = min(split * per, n_keys);
  const std::uint32_t k_end = min(n_keys, k_begin + per);

  // 1/sqrt(64) folded into q.
  for (std::uint32_t idx = tid; idx < kVec * kD; idx += kThreads) {
    const std::uint32_t v = idx / kD, d = idx % kD;
    const std::uint32_t r = v / kGroup, h = v % kGroup;
    qs[v][d] =
        r < live ? q[(static_cast<std::size_t>(base_row + t0 + r) * kLfmHeads +
                      g * kGroup + h) *
                         kD +
                     d] *
                       0.125F
                 : 0.0F;
  }
  if (tid < kVec) {
    m_row[tid] = -INFINITY;
    l_row[tid] = 0.0F;
  }
  float acc[kOut];
#pragma unroll
  for (std::uint32_t c = 0; c < kOut; ++c)
    acc[c] = 0.0F;

  for (std::uint32_t kb = k_begin; kb < k_end; kb += kTile) {
    __syncthreads();
    // Stage the tile: 64 keys x 64 dims of K and of V, eight halves a load
    // (TurboQuant rows decode to the same rotated-basis halves).
    for (std::uint32_t idx = tid; idx < kTile * kD / 8; idx += kThreads) {
      const std::uint32_t key = idx / (kD / 8), c8 = (idx % (kD / 8)) * 8;
      const std::uint32_t p = kb + key;
      uint4 kv4{0, 0, 0, 0}, vv4{0, 0, 0, 0};
      if (p < k_end) {
        if constexpr (kFmt == KvCacheFormat::kF16) {
          const std::size_t off =
              static_cast<std::size_t>(p) * kKvRow + g * kD + c8;
          kv4 = *reinterpret_cast<const uint4*>(
              static_cast<const __half*>(kv.k) + off);
          vv4 = *reinterpret_cast<const uint4*>(
              static_cast<const __half*>(kv.v) + off);
        } else {
          const std::size_t slot =
              static_cast<std::size_t>(p) * kLfmKvHeads + g;
          const auto* kc = static_cast<const std::uint8_t*>(kv.k) +
                           slot * RowBytes64<kFmt>();
          const auto* vc = static_cast<const std::uint8_t*>(kv.v) +
                           slot * RowBytes64<kFmt>();
          const float k_scale = __half2float(kv.k_scale[slot]);
          const float v_scale = __half2float(kv.v_scale[slot]);
          __half kh[8], vh[8];
#pragma unroll
          for (std::uint32_t j = 0; j < 8; ++j) {
            kh[j] = __float2half(Load64<kFmt>(kc, k_scale, c8 + j));
            vh[j] = __float2half(Load64<kFmt>(vc, v_scale, c8 + j));
          }
          kv4 = *reinterpret_cast<const uint4*>(kh);
          vv4 = *reinterpret_cast<const uint4*>(vh);
        }
      }
      auto* kd = reinterpret_cast<std::uint32_t*>(&ks[key][c8]);
      kd[0] = kv4.x;
      kd[1] = kv4.y;
      kd[2] = kv4.z;
      kd[3] = kv4.w;
      *reinterpret_cast<uint4*>(&vs[key][c8]) = vv4;
    }
    __syncthreads();
    // Scores: a wave shares its query vector and spans 32 keys.
#pragma unroll
    for (std::uint32_t c = 0; c < kScores; ++c) {
      const std::uint32_t idx = tid + c * kThreads;
      const std::uint32_t v = idx / kTile, key = idx % kTile;
      float s = 0.0F;
#pragma unroll 8
      for (std::uint32_t d = 0; d < kD; d += 2) {
        const float2 kk =
            __half22float2(*reinterpret_cast<const __half2*>(&ks[key][d]));
        s += qs[v][d] * kk.x + qs[v][d + 1] * kk.y;
      }
      const std::uint32_t r = v / kGroup, p = kb + key;
      ps[v][key] =
          (r < live && p < k_end && p <= pos0 + t0 + r) ? s : -INFINITY;
    }
    __syncthreads();
    // Online softmax: four lanes per query vector, 16 keys each.
    if (tid < kVec * 4) {
      const std::uint32_t v = tid >> 2, sub = tid & 3U;
      float m = -INFINITY;
      for (std::uint32_t k = sub; k < kTile; k += 4)
        m = fmaxf(m, ps[v][k]);
      m = fmaxf(m, __shfl_xor(m, 1));
      m = fmaxf(m, __shfl_xor(m, 2));
      const float m_old = m_row[v];
      const float m_new = fmaxf(m_old, m);
      float sum = 0.0F;
      for (std::uint32_t k = sub; k < kTile; k += 4) {
        const float p = m_new == -INFINITY ? 0.0F : __expf(ps[v][k] - m_new);
        ps[v][k] = p;
        sum += p;
      }
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (sub == 0) {
        const float sc = m_old == -INFINITY ? 0.0F : __expf(m_old - m_new);
        l_row[v] = l_row[v] * sc + sum;
        m_row[v] = m_new;
        scale_row[v] = sc;
      }
    }
    __syncthreads();
    // P V: lanes span dims, so ps reads broadcast and vs reads stay dense.
    const std::uint32_t d = tid % kD;
#pragma unroll
    for (std::uint32_t c = 0; c < kOut; ++c) {
      const std::uint32_t v = tid / kD + c * (kThreads / kD);
      float a = 0.0F;
#pragma unroll 8
      for (std::uint32_t k = 0; k < kTile; ++k)
        a += ps[v][k] * __half2float(vs[k][d]);
      acc[c] = acc[c] * scale_row[v] + a;
    }
  }
  __syncthreads();
  const std::uint32_t d = tid % kD;
#pragma unroll
  for (std::uint32_t c = 0; c < kOut; ++c) {
    const std::uint32_t v = tid / kD + c * (kThreads / kD);
    const std::uint32_t r = v / kGroup, h = v % kGroup;
    if (r >= live)
      continue;
    const std::uint32_t head = g * kGroup + h;
    if (splits == 1) {
      const float l = l_row[v];
      out[(static_cast<std::size_t>(base_row + t0 + r) * kLfmHeads + head) *
              kD +
          d] = l > 0.0F ? acc[c] / l : 0.0F;
      continue;
    }
    const std::size_t record =
        batched
            ? record_base +
                  (static_cast<std::size_t>(t0 + r) * kLfmHeads + head) *
                      splits +
                  split
            : (static_cast<std::size_t>(t0 + r) * kLfmHeads + head) * splits +
                  split;
    float* p = partials + record * kRecord;
    if (d == 0) {
      p[0] = m_row[v];
      p[1] = l_row[v];
    }
    p[2 + d] = acc[c];
  }
}

// F16 WMMA fragments (wave32). A holds row L % 16 and 16 contiguous k, B
// column L % 16 and 16 contiguous k; C element i is row 2i + L / 16, column
// L % 16.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaQueries = 32;     // queries per work-group
constexpr std::uint32_t kWmmaStride = kD + 8;  // halves; 16-byte rows
constexpr std::uint32_t kWmmaVtStride = kTile + 8;

/// Prompt attention on the matrix cores: one work-group per (KV head, 32
/// queries). Wave w owns query head 4g + w / 2 for 16 queries starting at
/// t0 + 16 (w % 2): one 16-row WMMA block, so scores, the online softmax
/// and the output accumulator stay in its registers. K and transposed V
/// tiles of 64 keys are staged once for all eight waves.
template<KvCacheFormat kFmt>
__global__ __launch_bounds__(kThreads) void WmmaAttentionKernel(
    const float* __restrict__ q, KvView kv, float* __restrict__ out,
    const std::uint32_t* start_pos, std::uint32_t n_rows) {
  __shared__ __align__(16) __half ks[kTile][kWmmaStride];
  __shared__ __align__(16) __half vt[kD][kWmmaVtStride];
  __shared__ __align__(16) __half ps[kThreads / 32][16][kWmmaVtStride];

  const std::uint32_t tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5;
  const std::uint32_t sub = lane & 15U, half = lane >> 4;
  const std::uint32_t g = blockIdx.x;
  const std::uint32_t t0 = blockIdx.y * kWmmaQueries;
  const std::uint32_t pos0 = *start_pos;
  const std::uint32_t head = g * kGroup + wave / 2;
  const std::uint32_t first = t0 + (wave % 2) * 16;  // query of row 0
  const std::uint32_t n_keys = pos0 + min(t0 + kWmmaQueries, n_rows);

  // 1/sqrt(64) folded into q.
  v16h qf[kD / 16];
  {
    const bool live = first + sub < n_rows;
    const float* row =
        q +
        (static_cast<std::size_t>(live ? first + sub : 0) * kLfmHeads + head) *
            kD;
#pragma unroll
    for (std::uint32_t ks16 = 0; ks16 < kD / 16; ++ks16)
#pragma unroll
      for (std::uint32_t j = 0; j < 16; ++j)
        qf[ks16][j] =
            static_cast<_Float16>(live ? row[ks16 * 16 + j] * 0.125F : 0.0F);
  }
  v8f o[kD / 16] = {};
  float m[8], l[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0F;
  }

  for (std::uint32_t kb = 0; kb < n_keys; kb += kTile) {
    __syncthreads();
    // K: (key, eight dims) per load. V: one key per lane, 16 dims, so the
    // transposed writes of a wave fill consecutive keys of one dim row.
    // Keys past n_keys stage as zero (never NaN under a zero probability).
    for (std::uint32_t idx = tid; idx < kTile * kD / 8; idx += kThreads) {
      const std::uint32_t key = idx / (kD / 8), c8 = (idx % (kD / 8)) * 8;
      const std::uint32_t p = kb + key;
      uint4 k4{0, 0, 0, 0};
      if (p < n_keys) {
        if constexpr (kFmt == KvCacheFormat::kF16) {
          k4 = *reinterpret_cast<const uint4*>(
              static_cast<const __half*>(kv.k) +
              static_cast<std::size_t>(p) * kKvRow + g * kD + c8);
        } else {
          const std::size_t slot =
              static_cast<std::size_t>(p) * kLfmKvHeads + g;
          const auto* kc = static_cast<const std::uint8_t*>(kv.k) +
                           slot * RowBytes64<kFmt>();
          const float scale = __half2float(kv.k_scale[slot]);
          __half kh[8];
#pragma unroll
          for (std::uint32_t j = 0; j < 8; ++j)
            kh[j] = __float2half(Load64<kFmt>(kc, scale, c8 + j));
          k4 = *reinterpret_cast<const uint4*>(kh);
        }
      }
      *reinterpret_cast<uint4*>(&ks[key][c8]) = k4;
    }
    {
      const std::uint32_t key = tid % kTile, d0 = (tid / kTile) * 16;
      const std::uint32_t p = kb + key;
      __half vh[16];
      if (p < n_keys) {
        if constexpr (kFmt == KvCacheFormat::kF16) {
          const auto* src = static_cast<const __half*>(kv.v) +
                            static_cast<std::size_t>(p) * kKvRow + g * kD + d0;
          *reinterpret_cast<uint4*>(vh) = *reinterpret_cast<const uint4*>(src);
          *reinterpret_cast<uint4*>(vh + 8) =
              *reinterpret_cast<const uint4*>(src + 8);
        } else {
          const std::size_t slot =
              static_cast<std::size_t>(p) * kLfmKvHeads + g;
          const auto* vc = static_cast<const std::uint8_t*>(kv.v) +
                           slot * RowBytes64<kFmt>();
          const float scale = __half2float(kv.v_scale[slot]);
#pragma unroll
          for (std::uint32_t j = 0; j < 16; ++j)
            vh[j] = __float2half(Load64<kFmt>(vc, scale, d0 + j));
        }
      } else {
#pragma unroll
        for (std::uint32_t j = 0; j < 16; ++j)
          vh[j] = __float2half(0.0F);
      }
#pragma unroll
      for (std::uint32_t j = 0; j < 16; ++j)
        vt[d0 + j][key] = vh[j];
    }
    __syncthreads();
    // No live row, or every row of this wave is causal-masked.
    if (first >= n_rows || kb > pos0 + first + 15)
      continue;

    // S = Q K^T for the 64 keys: four 16-key tiles.
    v8f s[kTile / 16];
#pragma unroll
    for (std::uint32_t kt = 0; kt < kTile / 16; ++kt) {
      s[kt] = v8f{};
#pragma unroll
      for (std::uint32_t ks16 = 0; ks16 < kD / 16; ++ks16)
        s[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            qf[ks16], LoadFrag(&ks[kt * 16 + sub][ks16 * 16]), s[kt]);
    }
    // Online softmax: row 2i + half's 64 scores sit in element i of the 16
    // lanes of its half (four tiles each).
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t r = 2 * i + half;
      const std::uint32_t visible = pos0 + first + r;  // last visible key
      const bool live = first + r < n_rows;
      float mx = -INFINITY;
#pragma unroll
      for (std::uint32_t kt = 0; kt < kTile / 16; ++kt) {
        const std::uint32_t p = kb + kt * 16 + sub;
        if (!live || p > visible)
          s[kt][i] = -INFINITY;
        mx = fmaxf(mx, s[kt][i]);
      }
#pragma unroll
      for (std::uint32_t o2 = 1; o2 < 16; o2 <<= 1)
        mx = fmaxf(mx, __shfl_xor(mx, o2));
      const float m_new = fmaxf(m[i], mx);
      const float sc = m[i] == -INFINITY ? 0.0F : __expf(m[i] - m_new);
      float sum = 0.0F;
#pragma unroll
      for (std::uint32_t kt = 0; kt < kTile / 16; ++kt) {
        const float p = s[kt][i] == -INFINITY ? 0.0F : __expf(s[kt][i] - m_new);
        sum += p;
        ps[wave][r][kt * 16 + sub] = __float2half(p);
      }
#pragma unroll
      for (std::uint32_t o2 = 1; o2 < 16; o2 <<= 1)
        sum += __shfl_xor(sum, o2);
      m[i] = m_new;
      l[i] = l[i] * sc + sum;
#pragma unroll
      for (std::uint32_t dt = 0; dt < kD / 16; ++dt)
        o[dt][i] *= sc;
    }
    // The wave reads back its own probability rows in the A layout.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    // O += P V.
#pragma unroll
    for (std::uint32_t kt = 0; kt < kTile / 16; ++kt) {
      const v16h pf = LoadFrag(&ps[wave][sub][kt * 16]);
#pragma unroll
      for (std::uint32_t dt = 0; dt < kD / 16; ++dt)
        o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            pf, LoadFrag(&vt[dt * 16 + sub][kt * 16]), o[dt]);
    }
  }

#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t t = first + 2 * i + half;
    if (t >= n_rows)
      continue;
    float* dst = out + (static_cast<std::size_t>(t) * kLfmHeads + head) * kD;
    const float inv = l[i] > 0.0F ? 1.0F / l[i] : 0.0F;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kD / 16; ++dt)
      dst[dt * 16 + sub] = o[dt][i] * inv;
  }
}

/// Combines a row's split records: grid (heads, rows[, items]).
__global__ void MergeKernel(const float* partials, float* out,
                            std::uint32_t splits,
                            const AttentionBatchItem* items) {
  const std::uint32_t h = blockIdx.x, d = threadIdx.x;
  std::uint32_t t = blockIdx.y, local = blockIdx.y;
  std::size_t base = 0;
  if (items != nullptr) {
    const AttentionBatchItem& item = items[blockIdx.z];
    if (blockIdx.y >= item.rows || item.splits == 1)
      return;
    splits = item.splits;
    t = item.row + blockIdx.y;
    base = item.partial;
  }
  const float* rec =
      partials +
      (base + (static_cast<std::size_t>(local) * kLfmHeads + h) * splits) *
          kRecord;
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s)
    m = fmaxf(m, rec[s * kRecord]);
  float l = 0.0F, a = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float ms = rec[s * kRecord];
    const float sc = ms == -INFINITY ? 0.0F : __expf(ms - m);
    l += rec[s * kRecord + 1] * sc;
    a += rec[s * kRecord + 2 + d] * sc;
  }
  out[(static_cast<std::size_t>(t) * kLfmHeads + h) * kD + d] =
      l > 0.0F ? a / l : 0.0F;
}

// -------------------------------------------------------------- router --

constexpr std::uint32_t kRouterTokens = 4;  // tokens per work-group

/// Router logits and top-k in one pass. A work-group stages four tokens'
/// inputs in LDS and scores them against the 64 F32 expert rows (wave w:
/// experts 8w..8w+7; the 16 float4 weight loads of a lane are unrolled so
/// they are in flight together, each reused for the four tokens), then
/// wave t selects token t's experts: sigmoid scores, top-k on score + bias
/// (lowest index on ties), weights normalized to sum 1.
__global__ __launch_bounds__(kThreads) void RouterKernel(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, std::int32_t* __restrict__ ids,
    float* __restrict__ weights, std::uint32_t n_tokens, std::uint32_t k) {
  constexpr std::uint32_t kQuads = kLfmHidden / 4;
  __shared__ float4 xs[kRouterTokens][kQuads];
  __shared__ float logits[kRouterTokens][kExperts];
  const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5;
  const std::uint32_t t0 = blockIdx.x * kRouterTokens;
  const std::uint32_t live = min(kRouterTokens, n_tokens - t0);
  for (std::uint32_t i = threadIdx.x; i < kRouterTokens * kQuads;
       i += kThreads) {
    const std::uint32_t t = i / kQuads;
    xs[t][i % kQuads] =
        t < live
            ? reinterpret_cast<const float4*>(x)[std::size_t{t0} * kQuads + i]
            : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  }
  __syncthreads();
  for (std::uint32_t e = wave * 8; e < wave * 8 + 8; ++e) {
    const auto* wr =
        reinterpret_cast<const float4*>(w + std::size_t{e} * kLfmHidden);
    float acc[kRouterTokens] = {};
#pragma unroll
    for (std::uint32_t jj = 0; jj < kQuads / 32; ++jj) {
      const std::uint32_t j = lane + 32 * jj;
      const float4 wv = wr[j];
#pragma unroll
      for (std::uint32_t t = 0; t < kRouterTokens; ++t) {
        const float4 xv = xs[t][j];
        acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
      }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < kRouterTokens; ++t) {
      const float sum = WaveSum(acc[t]);
      if (lane == 0)
        logits[t][e] = sum;
    }
  }
  __syncthreads();
  if (wave >= live)
    return;
  const std::uint32_t t = t0 + wave;
  float score[2], sel[2];
#pragma unroll
  for (std::uint32_t j = 0; j < 2; ++j) {
    const std::uint32_t e = lane + 32 * j;
    score[j] = 1.0F / (1.0F + __expf(-logits[wave][e]));
    sel[j] = score[j] + bias[e];
  }
  float chosen_sum = 0.0F;
  float chosen[16];
  std::int32_t chosen_id[16];
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float best = sel[0] >= sel[1] ? sel[0] : sel[1];
    std::uint32_t index = sel[0] >= sel[1] ? lane : lane + 32;
    float best_score = sel[0] >= sel[1] ? score[0] : score[1];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor(best, o);
      const std::uint32_t oi = __shfl_xor(index, o);
      const float os = __shfl_xor(best_score, o);
      if (ob > best || (ob == best && oi < index)) {
        best = ob;
        index = oi;
        best_score = os;
      }
    }
    chosen[slot] = best_score;
    chosen_id[slot] = static_cast<std::int32_t>(index);
    chosen_sum += best_score;
    if (index == lane)
      sel[0] = -INFINITY;
    if (index == lane + 32)
      sel[1] = -INFINITY;
  }
  if (lane == 0) {
    const float denom = fmaxf(chosen_sum, 6.103515625e-5F);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = chosen_id[slot];
      weights[t * k + slot] = chosen[slot] / denom;
    }
  }
}

}  // namespace

void LfmShortConv(const float* bcx, std::uint32_t stride, const float* w,
                  float* state, float* y, std::uint32_t n_tokens,
                  std::uint32_t channels, hipStream_t stream) {
  if (n_tokens == 0)
    return;
  const std::uint32_t blocks = (channels + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(ShortConvKernel, dim3(blocks, n_tokens), dim3(kThreads), 0,
                     stream, bcx, stride, w, state, y, n_tokens, channels,
                     nullptr);
  hipLaunchKernelGGL(ShortConvStateKernel, dim3(blocks), dim3(kThreads), 0,
                     stream, bcx, stride, state, n_tokens, channels, nullptr);
}

void LfmShortConvBatch(const float* bcx, std::uint32_t stride, const float* w,
                       const LfmConvItem* items, std::uint32_t count,
                       std::uint32_t max_rows, float* y, std::uint32_t channels,
                       hipStream_t stream) {
  if (count == 0 || max_rows == 0)
    return;
  const std::uint32_t blocks = (channels + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(ShortConvKernel, dim3(blocks, max_rows, count),
                     dim3(kThreads), 0, stream, bcx, stride, w, nullptr, y, 0U,
                     channels, items);
  hipLaunchKernelGGL(ShortConvStateKernel, dim3(blocks, count), dim3(kThreads),
                     0, stream, bcx, stride, nullptr, 0U, channels, items);
}

namespace {

template<typename Launch>
void ByFormat(KvCacheFormat format, Launch&& launch) {
  switch (format) {
    case KvCacheFormat::kF16:
      launch(std::integral_constant<KvCacheFormat, KvCacheFormat::kF16>{});
      break;
    case KvCacheFormat::kTq4:
      launch(std::integral_constant<KvCacheFormat, KvCacheFormat::kTq4>{});
      break;
    case KvCacheFormat::kTq3:
      launch(std::integral_constant<KvCacheFormat, KvCacheFormat::kTq3>{});
      break;
    case KvCacheFormat::kTq8:
      launch(std::integral_constant<KvCacheFormat, KvCacheFormat::kTq8>{});
      break;
  }
}

void Unrotate(KvCacheFormat format, float* out, std::uint32_t rows,
              hipStream_t stream) {
  const std::uint32_t head_rows = rows * kLfmHeads;
  if (format != KvCacheFormat::kF16 && head_rows != 0)
    hipLaunchKernelGGL(UnrotateKernel,
                       dim3((head_rows + kThreads / 32 - 1) / (kThreads / 32)),
                       dim3(kThreads), 0, stream, out, head_rows);
}

}  // namespace

void LfmPrepareAttention(const float* qkv, const float* q_gamma,
                         const float* k_gamma, float* q, const KvView& kv,
                         std::uint32_t n_tokens, const std::uint32_t* start_pos,
                         float theta, float eps, hipStream_t stream) {
  if (n_tokens == 0)
    return;
  ByFormat(kv.format, [&](auto f) {
    hipLaunchKernelGGL(PrepareKernel<decltype(f)::value>, dim3(n_tokens),
                       dim3(kThreads), 0, stream, qkv, q_gamma, k_gamma, q, kv,
                       start_pos, theta, eps, nullptr);
  });
}

void LfmPrepareAttentionBatch(const float* qkv, const float* q_gamma,
                              const float* k_gamma, float* q,
                              KvCacheFormat format,
                              const AttentionBatchItem* items,
                              std::uint32_t count, std::uint32_t max_rows,
                              float theta, float eps, hipStream_t stream) {
  if (count == 0 || max_rows == 0)
    return;
  ByFormat(format, [&](auto f) {
    hipLaunchKernelGGL(PrepareKernel<decltype(f)::value>, dim3(max_rows, count),
                       dim3(kThreads), 0, stream, qkv, q_gamma, k_gamma, q,
                       KvView{}, nullptr, theta, eps, items);
  });
}

void LfmAttention(const float* q, const KvView& kv, float* out, float* partials,
                  std::uint32_t n_tokens, const std::uint32_t* start_pos,
                  hipStream_t stream) {
  if (n_tokens == 0)
    return;
  ByFormat(kv.format, [&](auto f) {
    constexpr KvCacheFormat kFmt = decltype(f)::value;
    if (n_tokens <= 8 && partials != nullptr) {
      hipLaunchKernelGGL((AttentionKernel<1, kFmt>),
                         dim3(kLfmKvHeads, n_tokens, kSplits), dim3(kThreads),
                         0, stream, q, kv, out, partials, start_pos, n_tokens,
                         nullptr);
      hipLaunchKernelGGL(MergeKernel, dim3(kLfmHeads, n_tokens), dim3(kD), 0,
                         stream, partials, out, kSplits, nullptr);
    } else {
      hipLaunchKernelGGL(
          WmmaAttentionKernel<kFmt>,
          dim3(kLfmKvHeads, (n_tokens + kWmmaQueries - 1) / kWmmaQueries),
          dim3(kThreads), 0, stream, q, kv, out, start_pos, n_tokens);
    }
  });
  Unrotate(kv.format, out, n_tokens, stream);
}

void LfmAttentionBatch(const float* q, KvCacheFormat format,
                       const AttentionBatchItem* items, std::uint32_t count,
                       std::uint32_t max_rows, std::uint32_t rows, float* out,
                       float* partials, hipStream_t stream) {
  if (count == 0 || max_rows == 0)
    return;
  ByFormat(format, [&](auto f) {
    hipLaunchKernelGGL((AttentionKernel<1, decltype(f)::value>),
                       dim3(kLfmKvHeads, max_rows, count * kSplits),
                       dim3(kThreads), 0, stream, q, KvView{}, out, partials,
                       nullptr, 0U, items);
  });
  hipLaunchKernelGGL(MergeKernel, dim3(kLfmHeads, max_rows, count), dim3(kD), 0,
                     stream, partials, out, 1U, items);
  Unrotate(format, out, rows, stream);
}

void LfmRouter(const float* x, const float* w, const float* bias,
               std::int32_t* ids, float* weights, std::uint32_t n_tokens,
               std::uint32_t k, hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(
      RouterKernel, dim3((n_tokens + kRouterTokens - 1) / kRouterTokens),
      dim3(kThreads), 0, stream, x, w, bias, ids, weights, n_tokens, k);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
