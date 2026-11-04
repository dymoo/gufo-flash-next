// MiMo-V2.6-Flash elementwise, routing and attention kernels for gfx1151
// (wave32). Geometry is fixed at compile time; see config.hpp.
#include "src/models/mimo_v26_flash/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>

#include <cfloat>
#include <cmath>

namespace gufo::models::mimo_v26_flash::rocm {
namespace {

constexpr int kHidden = 4096;
constexpr int kHeads = 64;
constexpr int kDqk = 192;
constexpr int kDv = 128;
constexpr int kRope = 64;
constexpr int kWindow = 128;
constexpr int kExperts = 256;
constexpr int kUsed = 8;
constexpr int kQDim = kHeads * kDqk;
/// Halves of one layer inside a page or ring: K then V for every KV head.
/// Full layers: 4 heads x 256 tokens; sliding layers: 8 heads x 128 slots.
constexpr int kLayerHalves = 4 * 256 * (kDqk + kDv);
static_assert(kLayerHalves == 8 * kWindow * (kDqk + kDv));
constexpr int kFullLayers = 9;
constexpr int kPageHalves = kFullLayers * kLayerHalves;

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o, 32);
  return v;
}

// ---------------------------------------------------------------- embedding

struct BlockQ8_0 {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

__global__ void EmbedQ8_0Kernel(const BlockQ8_0* __restrict__ table,
                                const std::int32_t* __restrict__ tokens,
                                float* __restrict__ out) {
  const int row = blockIdx.x;
  const BlockQ8_0* b =
      table + static_cast<std::size_t>(tokens[row]) * (kHidden / 32);
  const int i = threadIdx.x * 16;  // 256 threads x 16 values
  const BlockQ8_0& blk = b[i / 32];
  const float d = __half2float(blk.d);
  float* o = out + static_cast<std::size_t>(row) * kHidden + i;
#pragma unroll
  for (int k = 0; k < 16; ++k)
    o[k] = d * blk.qs[i % 32 + k];
}

// ------------------------------------------------------------------ rmsnorm

__global__ __launch_bounds__(256) void RmsNormKernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, float eps) {
  const float4* row = reinterpret_cast<const float4*>(
      x + static_cast<std::size_t>(blockIdx.x) * kHidden);
  float4 v[4];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] = row[threadIdx.x + k * 256];
    ss += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
  }
  __shared__ float partial[8];
  ss = WaveSum(ss);
  if ((threadIdx.x & 31) == 0)
    partial[threadIdx.x >> 5] = ss;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    total += partial[k];
  const float scale = rsqrtf(total / kHidden + eps);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4* o = reinterpret_cast<float4*>(
      out + static_cast<std::size_t>(blockIdx.x) * kHidden);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 g = w4[threadIdx.x + k * 256];
    o[threadIdx.x + k * 256] =
        make_float4(v[k].x * scale * g.x, v[k].y * scale * g.y,
                    v[k].z * scale * g.z, v[k].w * scale * g.w);
  }
}

// --------------------------------------------------------------- qkv + rope

/// grid (n, 64 + kv_heads), block 64. Heads [0, 64) are queries; then keys
/// (and each key block also moves the matching value head).
__global__ __launch_bounds__(64) void QkvRopeKernel(
    const float* __restrict__ qkv, const std::int32_t* __restrict__ positions,
    float* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v,
    int kv_heads, float theta_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int width = kQDim + kv_heads * (kDqk + kDv);
  const float* src = qkv + static_cast<std::size_t>(row) * width;
  const float pos = static_cast<float>(positions[row]);
  const bool is_q = head < kHeads;
  const int kh = head - kHeads;
  const float* in = is_q ? src + head * kDqk : src + kQDim + kh * kDqk;
  for (int d = threadIdx.x; d < kDqk; d += 64) {
    float value;
    if (d < kRope) {
      const int i = d % (kRope / 2);
      const float theta = pos * powf(theta_scale, static_cast<float>(i));
      const float c = cosf(theta);
      const float s = sinf(theta);
      const float x0 = in[i];
      const float x1 = in[i + kRope / 2];
      value = d < kRope / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    } else {
      value = in[d];
    }
    if (is_q) {
      q[(static_cast<std::size_t>(row) * kHeads + head) * kDqk + d] = value;
    } else {
      k[(static_cast<std::size_t>(row) * kv_heads + kh) * kDqk + d] =
          __float2half(value);
    }
  }
  if (!is_q) {
    const float* vin = src + kQDim + kv_heads * kDqk + kh * kDv;
    for (int d = threadIdx.x; d < kDv; d += 64)
      v[(static_cast<std::size_t>(row) * kv_heads + kh) * kDv + d] =
          __float2half(vin[d]);
  }
}

// ---------------------------------------------------------------- kv cache

/// Address of key/value head `h` at position `pos` in a session's cache.
struct CacheAddr {
  const __half* k;
  const __half* v;
};

__device__ __forceinline__ std::size_t FullOffset(const std::int32_t* pages,
                                                  int full_index, int h,
                                                  int pos) {
  const std::size_t page = static_cast<std::size_t>(pages[pos / kPageTokens]);
  return page * kPageHalves +
         static_cast<std::size_t>(full_index) * kLayerHalves;
}

__device__ __forceinline__ CacheAddr CacheAt(const SessionKv& s,
                                             const __half* pool, int full_index,
                                             int swa_index, int h, int pos) {
  if (swa_index >= 0) {
    const __half* base = reinterpret_cast<const __half*>(s.ring) +
                         static_cast<std::size_t>(swa_index) * kLayerHalves;
    const int slot = h * kWindow + pos % kWindow;
    return {base + slot * kDqk, base + 8 * kWindow * kDqk + slot * kDv};
  }
  const __half* base = pool + FullOffset(s.pages, full_index, h, pos);
  const int slot = h * kPageTokens + pos % kPageTokens;
  return {base + slot * kDqk, base + 4 * kPageTokens * kDqk + slot * kDv};
}

/// grid (max_count, items), block 128: one row of one item per block.
__global__ __launch_bounds__(128) void KvCommitKernel(
    const __half* __restrict__ k, const __half* __restrict__ v,
    const AttnItem* __restrict__ items, const SessionKv* __restrict__ sessions,
    __half* __restrict__ pool, int full_index, int swa_index, int kv_heads) {
  const AttnItem item = items[blockIdx.y];
  const int r = blockIdx.x;
  if (r >= item.count)
    return;
  // Later rows of a long chunk own a ring slot; only the last window lands.
  if (swa_index >= 0 && r < item.count - kWindow)
    return;
  const int row = item.row + r;
  const int pos = item.start + r;
  const SessionKv s = sessions[item.session];
  for (int h = 0; h < kv_heads; ++h) {
    const CacheAddr a = CacheAt(s, pool, full_index, swa_index, h, pos);
    __half* dk = const_cast<__half*>(a.k);
    __half* dv = const_cast<__half*>(a.v);
    const __half* sk =
        k + (static_cast<std::size_t>(row) * kv_heads + h) * kDqk;
    const __half* sv = v + (static_cast<std::size_t>(row) * kv_heads + h) * kDv;
    for (int d = threadIdx.x; d < kDqk; d += 128)
      dk[d] = sk[d];
    dv[threadIdx.x] = sv[threadIdx.x];
  }
}

// ---------------------------------------------------------------- attention

/// Floats per attention partial: acc[128], m, l, padded to 16 bytes.
constexpr int kPartial = kDv + 4;
constexpr int kKeyTile = 32;
constexpr int kKStride = kDqk + 2;  // odd word stride: conflict-free rows

/// One workgroup: `TQ` consecutive query rows of one item x the `G` query
/// heads sharing KV head `g`, over one split of the key range. 8 waves; each
/// wave owns G*TQ/8 query vectors and each lane one key of every 32-key tile.
/// Writes an unnormalised partial (m, l, acc[128]) per (row, head, split).
template<int G, int TQ>
__global__ __launch_bounds__(256) void AttentionKernel(
    const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, const AttnItem* __restrict__ items,
    const SessionKv* __restrict__ sessions, const __half* __restrict__ pool,
    int full_index, int swa_index, float* __restrict__ partial, int splits) {
  constexpr int kVectors = G * TQ;
  constexpr int kPerWave = kVectors / 8;
  constexpr int kKvHeads = kHeads / G;
  // q in f16 (as llama.cpp's flash attention) so a workgroup holds 64 query
  // vectors beside the key tile within 64 KiB of LDS.
  __shared__ __half qs[kVectors][kDqk];
  __shared__ __half ks[kKeyTile][kKStride];
  __shared__ __half vs[kKeyTile][kDv];

  const int item_index = blockIdx.y / kKvHeads;
  const int g = blockIdx.y % kKvHeads;
  const AttnItem item = items[item_index];
  const int r0 = blockIdx.x * TQ;
  if (r0 >= item.count)
    return;
  const int rows = min(TQ, item.count - r0);
  const int split = blockIdx.z;
  const SessionKv session = sessions[item.session];
  const bool swa = swa_index >= 0;

  // Keys any query of this tile may see: [lo, hi].
  const int p_first = item.start + r0;
  const int p_last = p_first + rows - 1;
  const int lo = swa ? max(0, p_first - (kWindow - 1)) : 0;
  const int hi = p_last;
  const int span = hi - lo + 1;
  const int per_split = (span + splits - 1) / splits;
  const int key_begin = lo + split * per_split;
  const int key_end = min(hi + 1, key_begin + per_split);

  for (int i = threadIdx.x; i < kVectors * kDqk; i += 256) {
    const int vec = i / kDqk;
    const int d = i % kDqk;
    const int rr = vec / G;
    const int h = g * G + vec % G;
    qs[vec][d] = __float2half(
        rr < rows
            ? q[(static_cast<std::size_t>(item.row + r0 + rr) * kHeads + h) *
                    kDqk +
                d]
            : 0.0f);
  }

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const float scale = 0.07216878364870323f;  // 1/sqrt(192)
  float m[kPerWave], l[kPerWave], acc[kPerWave][4];
#pragma unroll
  for (int e = 0; e < kPerWave; ++e) {
    m[e] = -FLT_MAX;
    l[e] = 0.0f;
    acc[e][0] = acc[e][1] = acc[e][2] = acc[e][3] = 0.0f;
  }

  for (int t0 = key_begin; t0 < key_end; t0 += kKeyTile) {
    __syncthreads();
    // Stage the tile: 32 keys x (192 + 128) halves as 32-bit words.
    for (int i = threadIdx.x; i < kKeyTile * (kDqk + kDv) / 2; i += 256) {
      const int key = i / ((kDqk + kDv) / 2);
      const int w = i % ((kDqk + kDv) / 2);
      const int j = t0 + key;
      __half2 val = __float2half2_rn(0.0f);
      if (j < key_end) {
        const __half* src_k;
        const __half* src_v;
        if (j < item.start) {
          const CacheAddr a =
              CacheAt(session, pool, full_index, swa_index, g, j);
          src_k = a.k;
          src_v = a.v;
        } else {
          const std::size_t row = item.row + (j - item.start);
          src_k = k + (row * kKvHeads + g) * kDqk;
          src_v = v + (row * kKvHeads + g) * kDv;
        }
        val = w < kDqk / 2
                  ? reinterpret_cast<const __half2*>(src_k)[w]
                  : reinterpret_cast<const __half2*>(src_v)[w - kDqk / 2];
      }
      if (w < kDqk / 2)
        reinterpret_cast<__half2*>(&ks[key][0])[w] = val;
      else
        reinterpret_cast<__half2*>(&vs[key][0])[w - kDqk / 2] = val;
    }
    __syncthreads();

    const int j = t0 + lane;
#pragma unroll
    for (int e = 0; e < kPerWave; ++e) {
      const int vec = wave * kPerWave + e;
      const int rr = vec / G;
      const int p = p_first + rr;
      const bool visible =
          rr < rows && j < key_end && j <= p && (!swa || j > p - kWindow);
      float s = -FLT_MAX;
      if (visible) {
        float dot = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kDqk; d += 2) {
          const float2 kk =
              __half22float2(*reinterpret_cast<const __half2*>(&ks[lane][d]));
          const float2 qq =
              __half22float2(*reinterpret_cast<const __half2*>(&qs[vec][d]));
          dot += qq.x * kk.x + qq.y * kk.y;
        }
        s = dot * scale;
      }
      const float tile_max = WaveMax(s);
      if (tile_max == -FLT_MAX)
        continue;
      const float m_new = fmaxf(m[e], tile_max);
      const float correction = __expf(m[e] - m_new);
      const float pj = visible ? __expf(s - m_new) : 0.0f;
      l[e] = l[e] * correction + WaveSum(pj);
#pragma unroll
      for (int c = 0; c < 4; ++c)
        acc[e][c] *= correction;
#pragma unroll 8
      for (int key = 0; key < kKeyTile; ++key) {
        const float pk = __shfl(pj, key, 32);
        const float2 v01 = __half22float2(
            *reinterpret_cast<const __half2*>(&vs[key][lane * 4]));
        const float2 v23 = __half22float2(
            *reinterpret_cast<const __half2*>(&vs[key][lane * 4 + 2]));
        acc[e][0] += pk * v01.x;
        acc[e][1] += pk * v01.y;
        acc[e][2] += pk * v23.x;
        acc[e][3] += pk * v23.y;
      }
      m[e] = m_new;
    }
  }

#pragma unroll
  for (int e = 0; e < kPerWave; ++e) {
    const int vec = wave * kPerWave + e;
    const int rr = vec / G;
    if (rr >= rows)
      continue;
    const int h = g * G + vec % G;
    const std::size_t row = item.row + r0 + rr;
    float* dst = partial + ((row * kHeads + h) * splits + split) * kPartial;
    reinterpret_cast<float4*>(dst)[lane] =
        make_float4(acc[e][0], acc[e][1], acc[e][2], acc[e][3]);
    if (lane == 0) {
      dst[kDv] = m[e];
      dst[kDv + 1] = l[e];
    }
  }
}

/// grid (n, 64), block 128: merges the splits and the sink of one head.
__global__ __launch_bounds__(128) void AttentionCombineKernel(
    const float* __restrict__ partial, const float* __restrict__ sinks,
    float* __restrict__ out, int splits) {
  const std::size_t row = blockIdx.x;
  const int h = blockIdx.y;
  const float* base = partial + (row * kHeads + h) * splits * kPartial;
  float big = sinks != nullptr ? sinks[h] : -FLT_MAX;
  for (int s = 0; s < splits; ++s)
    big = fmaxf(big, base[s * kPartial + kDv]);
  float denom = sinks != nullptr ? __expf(sinks[h] - big) : 0.0f;
  float value = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float* p = base + s * kPartial;
    if (p[kDv + 1] == 0.0f)
      continue;
    const float w = __expf(p[kDv] - big);
    denom += p[kDv + 1] * w;
    value += p[threadIdx.x] * w;
  }
  out[(row * kHeads + h) * kDv + threadIdx.x] = value / denom;
}

// ------------------------------------------------------------------ routing

/// logits[row][e] = x[row] . W[e] for decode-width steps (F32 router, 4 MiB
/// per layer): grid (256 / 8, n), one wave per expert.
__global__ __launch_bounds__(256) void RouterLogitsKernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ logits) {
  const int row = blockIdx.y;
  const int e = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const float4* xr = reinterpret_cast<const float4*>(
      x + static_cast<std::size_t>(row) * kHidden);
  const float4* wr = reinterpret_cast<const float4*>(
      w + static_cast<std::size_t>(e) * kHidden);
  float sum = 0.0f;
#pragma unroll 8
  for (int i = lane; i < kHidden / 4; i += 32) {
    const float4 a = xr[i];
    const float4 b = wr[i];
    sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  sum = WaveSum(sum);
  if (lane == 0)
    logits[row * kExperts + e] = sum;
}

__global__ __launch_bounds__(256) void RouterTopKKernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    std::int32_t* __restrict__ ids, float* __restrict__ weights) {
  const int row = blockIdx.x;
  const int e = threadIdx.x;
  const float score = 1.0f / (1.0f + __expf(-logits[row * kExperts + e]));
  float key = score + bias[e];
  __shared__ float best_key[8];
  __shared__ int best_id[8];
  __shared__ float chosen_score[kUsed];
  __shared__ int chosen_id[kUsed];
  for (int round = 0; round < kUsed; ++round) {
    float bk = key;
    int bi = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ok = __shfl_xor(bk, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ok > bk || (ok == bk && oi < bi)) {
        bk = ok;
        bi = oi;
      }
    }
    if ((e & 31) == 0) {
      best_key[e >> 5] = bk;
      best_id[e >> 5] = bi;
    }
    __syncthreads();
    if (e == 0) {
      float k0 = best_key[0];
      int i0 = best_id[0];
      for (int w = 1; w < 8; ++w)
        if (best_key[w] > k0 || (best_key[w] == k0 && best_id[w] < i0)) {
          k0 = best_key[w];
          i0 = best_id[w];
        }
      chosen_id[round] = i0;
    }
    __syncthreads();
    if (e == chosen_id[round]) {
      chosen_score[round] = score;
      key = -FLT_MAX;
    }
    __syncthreads();
  }
  if (e < kUsed) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kUsed; ++i)
      sum += chosen_score[i];
    sum = fmaxf(sum, 6.103515625e-5f);
    ids[row * kUsed + e] = chosen_id[e];
    weights[row * kUsed + e] = chosen_score[e] / sum;
  }
}

// -------------------------------------------------------------- elementwise

__global__ void SwiGluKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out, std::int64_t count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i];
  out[i] = g / (1.0f + __expf(-g)) * up[i];
}

__global__ void AddScaledKernel(float* __restrict__ hidden,
                                const float* __restrict__ x, float scale,
                                std::int64_t count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += scale * x[i];
}

__global__ __launch_bounds__(256) void MoeCombineKernel(
    float* __restrict__ hidden, const float* __restrict__ y,
    const float* __restrict__ weights) {
  const std::size_t row = blockIdx.x;
  const float* w = weights + row * kUsed;
  for (int d = threadIdx.x; d < kHidden; d += 256) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kUsed; ++k)
      sum += w[k] * y[(row * kUsed + k) * kHidden + d];
    hidden[row * kHidden + d] += sum;
  }
}

__global__ __launch_bounds__(256) void GatherRowsKernel(
    const float* __restrict__ src, const std::int32_t* __restrict__ rows,
    float* __restrict__ dst) {
  const float4* s = reinterpret_cast<const float4*>(
      src + static_cast<std::size_t>(rows[blockIdx.x]) * kHidden);
  float4* d = reinterpret_cast<float4*>(
      dst + static_cast<std::size_t>(blockIdx.x) * kHidden);
  for (int i = threadIdx.x; i < kHidden / 4; i += 256)
    d[i] = s[i];
}

unsigned Blocks(std::int64_t count, int per) {
  return static_cast<unsigned>((count + per - 1) / per);
}

}  // namespace

void EmbedQ8_0(const void* table, const std::int32_t* tokens, float* out, int n,
               hipStream_t stream) {
  EmbedQ8_0Kernel<<<n, 256, 0, stream>>>(static_cast<const BlockQ8_0*>(table),
                                         tokens, out);
}

void RmsNorm(const float* x, const float* weight, float* out, int n, float eps,
             hipStream_t stream) {
  RmsNormKernel<<<n, 256, 0, stream>>>(x, weight, out, eps);
}

void QkvRope(const float* qkv, const std::int32_t* positions, float* q,
             std::uint16_t* k, std::uint16_t* v, int n, int kv_heads,
             float rope_base, hipStream_t stream) {
  const float theta_scale = powf(rope_base, -2.0f / kRope);
  QkvRopeKernel<<<dim3(n, kHeads + kv_heads), 64, 0, stream>>>(
      qkv, positions, q, reinterpret_cast<__half*>(k),
      reinterpret_cast<__half*>(v), kv_heads, theta_scale);
}

void KvCommit(const std::uint16_t* k, const std::uint16_t* v,
              const AttnItem* items, int item_count, int max_count,
              const SessionKv* sessions, std::uint16_t* page_pool,
              int full_index, int swa_index, int kv_heads, hipStream_t stream) {
  KvCommitKernel<<<dim3(max_count, item_count), 128, 0, stream>>>(
      reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(v),
      items, sessions, reinterpret_cast<__half*>(page_pool), full_index,
      swa_index, kv_heads);
}

std::size_t AttentionScratchFloats(int n, int splits) {
  return static_cast<std::size_t>(n) * kHeads * splits * kPartial;
}

void Attention(const float* q, const std::uint16_t* k, const std::uint16_t* v,
               const AttnItem* items, int item_count, int max_count,
               int max_keys, const SessionKv* sessions,
               const std::uint16_t* page_pool, int full_index, int swa_index,
               const float* sinks, float* out, float* scratch, int splits,
               int n, hipStream_t stream) {
  (void)max_keys;
  const auto* kh = reinterpret_cast<const __half*>(k);
  const auto* vh = reinterpret_cast<const __half*>(v);
  const auto* pool = reinterpret_cast<const __half*>(page_pool);
  const bool swa = swa_index >= 0;
  const bool decode = max_count == 1;
  // Decode: 16 query vectors per group; prefill: 64 sharing each key tile.
  if (swa) {
    if (decode) {
      AttentionKernel<8, 2>
          <<<dim3(1, item_count * 8, splits), 256, 0, stream>>>(
              q, kh, vh, items, sessions, pool, full_index, swa_index, scratch,
              splits);
    } else {
      AttentionKernel<8, 8><<<dim3((max_count + 7) / 8, item_count * 8, splits),
                              256, 0, stream>>>(q, kh, vh, items, sessions,
                                                pool, full_index, swa_index,
                                                scratch, splits);
    }
  } else if (decode) {
    AttentionKernel<16, 1><<<dim3(1, item_count * 4, splits), 256, 0, stream>>>(
        q, kh, vh, items, sessions, pool, full_index, swa_index, scratch,
        splits);
  } else {
    AttentionKernel<16, 4>
        <<<dim3((max_count + 3) / 4, item_count * 4, splits), 256, 0, stream>>>(
            q, kh, vh, items, sessions, pool, full_index, swa_index, scratch,
            splits);
  }
  AttentionCombineKernel<<<dim3(n, kHeads), 128, 0, stream>>>(
      scratch, swa ? sinks : nullptr, out, splits);
}

void RouterLogits(const float* x, const float* weight, float* logits, int n,
                  hipStream_t stream) {
  RouterLogitsKernel<<<dim3(kExperts / 8, n), 256, 0, stream>>>(x, weight,
                                                                logits);
}

void RouterTopK(const float* logits, const float* bias, std::int32_t* ids,
                float* weights, int n, hipStream_t stream) {
  RouterTopKKernel<<<n, 256, 0, stream>>>(logits, bias, ids, weights);
}

void SwiGlu(const float* gate, const float* up, float* out, std::int64_t count,
            hipStream_t stream) {
  SwiGluKernel<<<Blocks(count, 256), 256, 0, stream>>>(gate, up, out, count);
}

void AddScaled(float* hidden, const float* x, float scale, std::int64_t count,
               hipStream_t stream) {
  AddScaledKernel<<<Blocks(count, 256), 256, 0, stream>>>(hidden, x, scale,
                                                          count);
}

void MoeCombine(float* hidden, const float* y, const float* weights, int n,
                hipStream_t stream) {
  MoeCombineKernel<<<n, 256, 0, stream>>>(hidden, y, weights);
}

void GatherRows(const float* src, const std::int32_t* rows, float* dst, int n,
                hipStream_t stream) {
  GatherRowsKernel<<<n, 256, 0, stream>>>(src, rows, dst);
}

}  // namespace gufo::models::mimo_v26_flash::rocm
