// Kev packed-pass kernels. The attention kernel is the Qwen27B WMMA prefill
// kernel (src/models/qwen/hip/kernels/attention_wmma.hip, 32 queries x 2 heads
// against 16 keys, V prefetched a tile ahead) with three changes: 16 query
// heads, keys addressed through two slot ranges (prefix, then the sequence's
// own keys), and one tile descriptor per block instead of one sequence per
// launch. The delta rule follows the Flash-Next barrier-free token-serial
// kernel with per-sequence initial and final state slots.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/kev/hip/kernels.hpp"

#include <hip/hip_fp16.h>

#include <cmath>

namespace gufo::models::kev::hip {
namespace {

constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kWmmaHeads = 2;
constexpr std::uint32_t kQueryRows = kAttentionTileQueries;
constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

__device__ __forceinline__ float Softplus(float x) {
  return x > 20.0F ? x : log1pf(expf(x));
}

}  // namespace

// ---------------------------------------------------------------------------
// Q/K norm + RoPE + KV write
// ---------------------------------------------------------------------------

__launch_bounds__(256) __global__ void KevQkNormRopeKvKernel(
    float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, const std::uint32_t* __restrict__ pos,
    const std::uint32_t* __restrict__ kv_slots, __half* __restrict__ pool,
    std::size_t pool_slots, std::uint32_t layer, std::uint32_t rotary_dim,
    float rope_theta, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t block = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t half_rot = rotary_dim / 2;
  const float position = static_cast<float>(pos[b]);
  __shared__ float s_norm[kHeadDim];
  __shared__ float s_sum[256];

  const bool is_query = block < kQueryHeads;
  const std::uint32_t head = is_query ? block : block - kQueryHeads;
  const float* x = is_query
                       ? q + (b * kAttnWidth) + (head * kHeadDim)
                       : k + (b * kKvWidth) + (head * kHeadDim);
  const float value = x[tid];
  s_sum[tid] = value * value;
  __syncthreads();
  for (std::uint32_t stride = 128; stride > 0; stride /= 2) {
    if (tid < stride)
      s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(s_sum[0] / static_cast<float>(kHeadDim) + eps);
  const float* w = is_query ? q_weight : k_weight;
  s_norm[tid] = (value * inv_rms) * w[tid];
  __syncthreads();

  float out = s_norm[tid];
  if (tid < rotary_dim) {
    const std::uint32_t p = tid % half_rot;
    const float exponent =
        static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
    const float angle = position * (1.0F / powf(rope_theta, exponent));
    const float c = cosf(angle);
    const float s = sinf(angle);
    out = tid < half_rot ? (s_norm[p] * c) - (s_norm[p + half_rot] * s)
                         : (s_norm[p] * s) + (s_norm[p + half_rot] * c);
  }
  if (is_query) {
    q[(b * kAttnWidth) + (head * kHeadDim) + tid] = out;
    return;
  }
  __half* slot = pool + ((static_cast<std::size_t>(layer) * pool_slots +
                          kv_slots[b]) *
                         kSlotHalves) +
                 (head * kHeadDim);
  slot[tid] = __float2half_rn(out);
  slot[kKvWidth + tid] = __float2half_rn(v[(b * kKvWidth) + (head * kHeadDim) +
                                           tid]);
}

void LaunchQkNormRopeKv(float* q, const float* k, const float* v,
                        const float* q_norm, const float* k_norm,
                        const std::uint32_t* positions,
                        const std::uint32_t* kv_slots, void* kv_pool,
                        std::size_t pool_slots, std::uint32_t layer,
                        std::size_t tokens, std::uint32_t rotary_dim,
                        float rope_theta, float eps, hipStream_t stream) {
  const dim3 grid(kQueryHeads + kKvHeads, static_cast<unsigned>(tokens));
  KevQkNormRopeKvKernel<<<grid, 256, 0, stream>>>(
      q, k, v, q_norm, k_norm, positions, kv_slots,
      static_cast<__half*>(kv_pool), pool_slots, layer, rotary_dim, rope_theta,
      eps);
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

__launch_bounds__(256, 2) __global__ void KevAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ pool, std::size_t pool_slots,
    std::uint32_t layer, const AttentionTile* __restrict__ tiles,
    float* __restrict__ out_context) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");

  const AttentionTile tile = tiles[blockIdx.x];
  const std::uint32_t start_pos = tile.prefix_len + tile.own_first;
  const std::uint32_t batch_size = tile.n;
  const std::uint32_t context_end = start_pos + batch_size;
  const std::uint32_t max_visible = context_end;

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves = kKeys * kWmmaKStride > kHeadDim * kVtStride
                                             ? kKeys * kWmmaKStride
                                             : kHeadDim * kVtStride;
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  const float* q_base = q + static_cast<std::size_t>(tile.q_packed) * kAttnWidth;
  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query = row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row = q_base +
                         (static_cast<std::size_t>(local_query) * kAttnWidth) +
                         (static_cast<std::size_t>(row_block_head(s_rb)) *
                          kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
#pragma unroll
      for (std::uint32_t vi = 0; vi < 4; ++vi) {
        const float4 f =
            live ? reinterpret_cast<const float4*>(q_row + d0)[vi]
                 : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(vi * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(vi * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(vi * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(vi * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  v8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  // Key index -> pool slot: the prefix range, then the own range.
  const __half* layer_pool =
      pool + static_cast<std::size_t>(layer) * pool_slots * kSlotHalves +
      static_cast<std::size_t>(kv_head) * kHeadDim;
  const auto key_row = [&](std::uint32_t key) {
    const std::uint32_t slot = key < tile.prefix_len
                                   ? tile.prefix_slot + key
                                   : tile.own_slot + (key - tile.prefix_len);
    return layer_pool + static_cast<std::size_t>(slot) * kSlotHalves;
  };

  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    const std::uint32_t key = key_start + v_key;
    const bool live = key < context_end;
    const __half* src = live ? key_row(key) + kKvWidth + v_slice : nullptr;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j)
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
  };
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] = key < context_end
                   ? *reinterpret_cast<const uint4*>(key_row(key) + d8)
                   : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs] = {};
  uint4 v_pre[kVRegs] = {};
  load_k(0, k_cur);
  load_v(0, v_cur);

  for (std::uint32_t key_start = 0; key_start < max_visible;
       key_start += kKeys) {
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(row * kWmmaKStride) + d8]) = k_cur[n];
    }
    __syncthreads();

    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      load_v(next_start, v_pre);
    }

    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
    }
    __syncthreads();

    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query = row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key = key_start + col;
        const bool valid = local_query < batch_size &&
                           key <= absolute_query && key < context_end;
        const std::uint32_t t = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][t][row][col % 16] +
                           s_lds[1][t][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_sum += __shfl_xor(part_sum, off);
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t)
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          o_acc[rb][t][i] *= scale[i];
    }

#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + t * 8;
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb)
        v_frag[kb] =
            LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const std::uint32_t first_key = key_start + kb * 16;
          // Sum a query's final partial tile over visible keys only, as the
          // Qwen27B kernel does, so masked probabilities never meet values.
          if (first_key + 15 > start_pos) {
            v8f tail = o_acc[rb][t];
            const int relative_key =
                static_cast<int>(first_key) -
                static_cast<int>(start_pos + row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >=
                        relative_key + static_cast<int>(key) -
                            static_cast<int>(2 * i)) {
                  tail[i] = fmaf(
                      __half2float(p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                      value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i))
                next[i] = tail[i];
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n)
      k_cur[n] = k_pre[n];
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n)
      v_cur[n] = v_pre[n];
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = row_offset + row;
        if (local_query >= batch_size)
          continue;
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(tile.q_packed + local_query) *
             kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        value *= 1.0F / (1.0F + __expf(-gate[offset]));
        out_context[offset] = value;
      }
    }
  }
}

void LaunchAttention(const float* q, const float* gate, const void* kv_pool,
                     std::size_t pool_slots, std::uint32_t layer,
                     const AttentionTile* tiles, std::uint32_t tile_count,
                     float* out, hipStream_t stream) {
  if (tile_count == 0)
    return;
  const dim3 grid(tile_count, kQueryHeads / kWmmaHeads);
  KevAttentionKernel<<<grid, 256, 0, stream>>>(
      q, gate, static_cast<const __half*>(kv_pool), pool_slots, layer, tiles,
      out);
}

// ---------------------------------------------------------------------------
// Conv
// ---------------------------------------------------------------------------

__device__ __forceinline__ const float* ConvWindow(const RecurrentPools& pools,
                                                   std::int32_t slot,
                                                   std::uint32_t ssm_layer,
                                                   std::uint32_t channel) {
  if (slot == kNoSlot)
    return nullptr;
  return pools.conv +
         ((static_cast<std::size_t>(slot) * pools.ssm_layers + ssm_layer) *
              pools.channels +
          channel) *
             kConvTaps;
}

__global__ void KevConvKernel(const float* __restrict__ qkv,
                              const float* __restrict__ weights,
                              const std::uint32_t* __restrict__ token_sequence,
                              const Sequence* __restrict__ sequences,
                              const std::uint8_t* __restrict__ phases,
                              std::uint8_t phase, RecurrentPools pools,
                              std::uint32_t ssm_layer,
                              float* __restrict__ out) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t t = blockIdx.y;
  const std::uint32_t s = token_sequence[t];
  if (c >= pools.channels || phases[s] != phase)
    return;
  const Sequence seq = sequences[s];
  const std::uint32_t i = static_cast<std::uint32_t>(t) - seq.offset;
  const float* window = ConvWindow(pools, seq.init_slot, ssm_layer, c);
  const std::size_t n = pools.channels;
  const auto history = [&](std::uint32_t back) {
    if (i >= back)
      return qkv[(t - back) * n + c];
    return window != nullptr ? window[kConvTaps - (back - i)] : 0.0F;
  };
  const float* w = weights + static_cast<std::size_t>(c) * kConvTaps;
  const float dot = (history(3) * w[0]) + (history(2) * w[1]) +
                    (history(1) * w[2]) + (qkv[t * n + c] * w[3]);
  out[t * n + c] = dot * (1.0F / (1.0F + expf(-dot)));
}

void LaunchConv(const float* qkv, const float* weights,
                const std::uint32_t* token_sequence, const Sequence* sequences,
                const std::uint8_t* sequence_phase, std::uint8_t phase,
                RecurrentPools pools, std::uint32_t ssm_layer,
                std::size_t tokens, float* out, hipStream_t stream) {
  if (tokens == 0)
    return;
  const dim3 grid((pools.channels + 255) / 256, static_cast<unsigned>(tokens));
  KevConvKernel<<<grid, 256, 0, stream>>>(qkv, weights, token_sequence,
                                          sequences, sequence_phase, phase,
                                          pools, ssm_layer, out);
}

__global__ void KevConvStoreKernel(const float* __restrict__ qkv,
                                   const Sequence* __restrict__ sequences,
                                   const std::uint32_t* __restrict__ list,
                                   RecurrentPools pools,
                                   std::uint32_t ssm_layer) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const Sequence seq = sequences[list[blockIdx.y]];
  if (c >= pools.channels || seq.final_slot == kNoSlot)
    return;
  const float* window = ConvWindow(pools, seq.init_slot, ssm_layer, c);
  float tail[kConvTaps];
#pragma unroll
  for (std::uint32_t j = 0; j < kConvTaps; ++j) {
    const int index = static_cast<int>(seq.length) -
                      static_cast<int>(kConvTaps) + static_cast<int>(j);
    tail[j] = index >= 0 ? qkv[(static_cast<std::size_t>(seq.offset) + index) *
                                   pools.channels +
                               c]
              : window != nullptr ? window[kConvTaps + index]
                                  : 0.0F;
  }
  float* final_window = pools.conv + ((static_cast<std::size_t>(seq.final_slot) *
                                           pools.ssm_layers +
                                       ssm_layer) *
                                          pools.channels +
                                      c) *
                                         kConvTaps;
#pragma unroll
  for (std::uint32_t j = 0; j < kConvTaps; ++j)
    final_window[j] = tail[j];
}

void LaunchConvStore(const float* qkv, const Sequence* sequences,
                     const std::uint32_t* list, std::uint32_t count,
                     RecurrentPools pools, std::uint32_t ssm_layer,
                     hipStream_t stream) {
  if (count == 0)
    return;
  const dim3 grid((pools.channels + 255) / 256, count);
  KevConvStoreKernel<<<grid, 256, 0, stream>>>(qkv, sequences, list, pools,
                                               ssm_layer);
}

// ---------------------------------------------------------------------------
// Delta rule
// ---------------------------------------------------------------------------

__launch_bounds__(kSsmDim) __global__
    void KevDeltaPrepKernel(const float* __restrict__ conv_out,
                            float* __restrict__ qn, float* __restrict__ kn,
                            std::uint32_t k_heads, std::uint32_t channels) {
  const std::size_t t = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t d = threadIdx.x;
  const float* row = conv_out + t * channels;
  const float qv = row[h * kSsmDim + d];
  const float kv = row[k_heads * kSsmDim + h * kSsmDim + d];
  float q_sq = qv * qv;
  float k_sq = kv * kv;
  for (int off = 16; off > 0; off >>= 1) {
    q_sq += __shfl_xor(q_sq, off);
    k_sq += __shfl_xor(k_sq, off);
  }
  __shared__ float partial[2][kSsmDim / 32];
  if ((d & 31u) == 0) {
    partial[0][d / 32] = q_sq;
    partial[1][d / 32] = k_sq;
  }
  __syncthreads();
  float q_total = 0.0F;
  float k_total = 0.0F;
#pragma unroll
  for (std::uint32_t w = 0; w < kSsmDim / 32; ++w) {
    q_total += partial[0][w];
    k_total += partial[1][w];
  }
  const std::size_t o = (t * k_heads + h) * kSsmDim + d;
  qn[o] = qv * (1.0F / sqrtf(q_total + 1e-6F)) *
          (1.0F / sqrtf(static_cast<float>(kSsmDim)));
  kn[o] = kv * (1.0F / sqrtf(k_total + 1e-6F));
}

void LaunchDeltaPrep(const float* conv_out, float* qn, float* kn,
                     std::size_t tokens, std::uint32_t k_heads,
                     std::uint32_t v_heads, hipStream_t stream) {
  if (tokens == 0)
    return;
  const std::uint32_t channels = (2 * k_heads + v_heads) * kSsmDim;
  const dim3 grid(k_heads, static_cast<unsigned>(tokens));
  KevDeltaPrepKernel<<<grid, kSsmDim, 0, stream>>>(conv_out, qn, kn, k_heads,
                                                   channels);
}

constexpr std::uint32_t kDeltaLanes = 4;
constexpr std::uint32_t kDeltaRowsPerBlock = 32;
constexpr std::uint32_t kDeltaSlice = kSsmDim / kDeltaLanes;

__launch_bounds__(kDeltaLanes* kDeltaRowsPerBlock) __global__
    void KevDeltaRuleKernel(const float* __restrict__ conv_out,
                            const float* __restrict__ qn,
                            const float* __restrict__ kn,
                            const float* __restrict__ alpha,
                            const float* __restrict__ beta,
                            const float* __restrict__ a,
                            const float* __restrict__ dt,
                            const Sequence* __restrict__ sequences,
                            const std::uint32_t* __restrict__ list,
                            RecurrentPools pools, std::uint32_t ssm_layer,
                            std::uint32_t k_heads, float* __restrict__ raw) {
  const Sequence seq = sequences[list[blockIdx.z]];
  const std::uint32_t v_heads = pools.v_heads;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j =
      blockIdx.y * kDeltaRowsPerBlock + threadIdx.x / kDeltaLanes;
  const std::uint32_t lane = threadIdx.x % kDeltaLanes;
  const std::uint32_t i0 = lane * kDeltaSlice;
  const std::size_t head_elems = static_cast<std::size_t>(kSsmDim) * kSsmDim;
  const auto state_row = [&](std::int32_t slot) {
    return pools.state +
           ((static_cast<std::size_t>(slot) * pools.ssm_layers + ssm_layer) *
                v_heads +
            h) *
               head_elems +
           static_cast<std::size_t>(j) * kSsmDim + i0;
  };
  float row[kDeltaSlice];
  if (seq.init_slot != kNoSlot) {
    const float* s = state_row(seq.init_slot);
#pragma unroll
    for (std::uint32_t i = 0; i < kDeltaSlice; ++i)
      row[i] = s[i];
  } else {
#pragma unroll
    for (std::uint32_t i = 0; i < kDeltaSlice; ++i)
      row[i] = 0.0F;
  }
  const float a_h = a[h];
  const float dt_h = dt[h];
  const std::uint32_t channels = pools.channels;
  for (std::uint32_t step = 0; step < seq.length; ++step) {
    const std::size_t t = seq.offset + step;
    const float* q = qn + (t * k_heads + kh) * kSsmDim + i0;
    const float* k = kn + (t * k_heads + kh) * kSsmDim + i0;
    const float v = conv_out[t * channels + 2 * k_heads * kSsmDim +
                             h * kSsmDim + j];
    const float decay = expf(Softplus(alpha[t * v_heads + h] + dt_h) * a_h);
    const float b = 1.0F / (1.0F + expf(-beta[t * v_heads + h]));
    float kr[kDeltaSlice];
    float u = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < kDeltaSlice; ++i) {
      kr[i] = k[i];
      row[i] *= decay;
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kDeltaLanes / 2; off > 0; off >>= 1)
      u += __shfl_xor(u, off, kDeltaLanes);
    const float correction = (v - u) * b;
    float acc = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < kDeltaSlice; ++i) {
      row[i] = fmaf(correction, kr[i], row[i]);
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kDeltaLanes / 2; off > 0; off >>= 1)
      acc += __shfl_xor(acc, off, kDeltaLanes);
    if (lane == 0)
      raw[t * v_heads * kSsmDim + h * kSsmDim + j] = acc;
  }
  if (seq.final_slot != kNoSlot) {
    float* s = state_row(seq.final_slot);
#pragma unroll
    for (std::uint32_t i = 0; i < kDeltaSlice; ++i)
      s[i] = row[i];
  }
}

void LaunchDeltaRule(const float* conv_out, const float* qn, const float* kn,
                     const float* alpha, const float* beta, const float* a,
                     const float* dt, const Sequence* sequences,
                     const std::uint32_t* list, std::uint32_t count,
                     RecurrentPools pools, std::uint32_t ssm_layer,
                     std::uint32_t k_heads, float* raw, hipStream_t stream) {
  if (count == 0)
    return;
  const dim3 grid(pools.v_heads, kSsmDim / kDeltaRowsPerBlock, count);
  KevDeltaRuleKernel<<<grid, kDeltaLanes * kDeltaRowsPerBlock, 0, stream>>>(
      conv_out, qn, kn, alpha, beta, a, dt, sequences, list, pools, ssm_layer,
      k_heads, raw);
}

// ---------------------------------------------------------------------------
// Readout gather
// ---------------------------------------------------------------------------

__global__ void KevGatherRowsKernel(const float* __restrict__ hidden,
                                    const std::uint32_t* __restrict__ index,
                                    std::uint32_t width,
                                    float* __restrict__ rows) {
  const std::uint32_t r = blockIdx.x;
  const float* src = hidden + static_cast<std::size_t>(index[r]) * width;
  float* dst = rows + static_cast<std::size_t>(r) * width;
  for (std::uint32_t d = threadIdx.x; d < width; d += blockDim.x)
    dst[d] = src[d];
}

void LaunchGatherRows(const float* hidden, const std::uint32_t* index,
                      std::uint32_t count, std::uint32_t width, float* rows,
                      hipStream_t stream) {
  if (count == 0)
    return;
  KevGatherRowsKernel<<<count, 256, 0, stream>>>(hidden, index, width, rows);
}

}  // namespace gufo::models::kev::hip
#endif  // defined(ENGINE_ENABLE_HIP)
