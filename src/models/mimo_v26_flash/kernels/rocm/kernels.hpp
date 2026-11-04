#ifndef GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::mimo_v26_flash::rocm {

/// Tokens per full-attention KV page. A page holds K and V of every full
/// layer for kPageTokens consecutive positions of one session.
constexpr std::uint32_t kPageTokens = 256;

/// One run of consecutive query rows of one session inside a batch: rows
/// [row, row + count) hold positions [start, start + count).
struct AttnItem {
  std::int32_t row;
  std::int32_t count;
  std::int32_t start;
  std::int32_t session;  ///< index into the batch's session tables
};

/// Per-session device state the attention kernels read, one entry per batch
/// session (device arrays of these are uploaded per step).
struct SessionKv {
  const std::int32_t* pages;  ///< full-attention page table
  std::uint16_t* ring;        ///< sliding-window ring base (f16)
};

void EmbedQ8_0(const void* table, const std::int32_t* tokens, float* out, int n,
               hipStream_t stream);

void RmsNorm(const float* x, const float* weight, float* out, int n, float eps,
             hipStream_t stream);

/// Splits the fused projection [n x (q | k | v)], rotates q and k (NEOX over
/// the leading 64 dims) and writes q as F32 [n x 64 x 192] and k, v as F16
/// [n x kv_heads x 192 / 128].
void QkvRope(const float* qkv, const std::int32_t* positions, float* q,
             std::uint16_t* k, std::uint16_t* v, int n, int kv_heads,
             float rope_base, hipStream_t stream);

/// Stores the batch's k/v rows into each session's cache: full layers into
/// the page (`full_index` 0..8), sliding-window layers into the ring slot of
/// `swa_index` 0..38.
void KvCommit(const std::uint16_t* k, const std::uint16_t* v,
              const AttnItem* items, int item_count, int max_count,
              const SessionKv* sessions, std::uint16_t* page_pool,
              int full_index, int swa_index, int kv_heads, hipStream_t stream);

/// Attention of every item over its cached prefix plus its own rows (causal),
/// with a 128-key window and per-head sinks on sliding-window layers. `out` is
/// F32 [n x 64 x 128]. Scratch must hold AttentionScratchFloats(n, splits).
void Attention(const float* q, const std::uint16_t* k, const std::uint16_t* v,
               const AttnItem* items, int item_count, int max_count,
               int max_keys, const SessionKv* sessions,
               const std::uint16_t* page_pool, int full_index, int swa_index,
               const float* sinks, float* out, float* scratch, int splits,
               int n, hipStream_t stream);
[[nodiscard]] std::size_t AttentionScratchFloats(int n, int splits);

/// logits[n x 256] = x[n x 4096] . W^T (F32), for decode-width steps.
void RouterLogits(const float* x, const float* weight, float* logits, int n,
                  hipStream_t stream);

/// Sigmoid router: selection by score + bias, weights = normalised scores.
void RouterTopK(const float* logits, const float* bias, std::int32_t* ids,
                float* weights, int n, hipStream_t stream);

/// out[i] = silu(gate[i]) * up[i].
void SwiGlu(const float* gate, const float* up, float* out, std::int64_t count,
            hipStream_t stream);

/// hidden[t] += scale * x[t] over rows of `width`.
void AddScaled(float* hidden, const float* x, float scale, std::int64_t count,
               hipStream_t stream);

/// hidden[t] += sum_k weights[t, k] * y[t, k] over 8 expert rows of 4096.
void MoeCombine(float* hidden, const float* y, const float* weights, int n,
                hipStream_t stream);

/// dst[i] = src[rows[i]] for rows of 4096 floats.
void GatherRows(const float* src, const std::int32_t* rows, float* dst, int n,
                hipStream_t stream);

}  // namespace gufo::models::mimo_v26_flash::rocm

#endif  // GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_KERNELS_HPP_
