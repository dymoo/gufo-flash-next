#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_LFM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_LFM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

// LFM2 MoE (lfm2moe, LFM2.5-24B-A2B) layers on the Flash-Next engine:
// double-gated short convolutions, 64-wide GQA attention with four query
// heads per KV head, and sigmoid routing with a selection bias. The kernels
// are specialized to that geometry (Config::FromGguf rejects any other).
namespace gufo::models::qwen38_flash_next::rocm {

inline constexpr std::uint32_t kLfmHidden = 2048;
inline constexpr std::uint32_t kLfmHeads = 32;
inline constexpr std::uint32_t kLfmKvHeads = 8;
inline constexpr std::uint32_t kLfmHeadDim = 64;
/// Q, K and V rows of one token in the stacked projection output.
inline constexpr std::uint32_t kLfmQkvRow =
    (kLfmHeads + 2 * kLfmKvHeads) * kLfmHeadDim;

/// One session's rows [row, row + rows) of a multi-session short
/// convolution; `state` holds its last two B * x rows ([2][channels]).
struct LfmConvItem {
  float* state{nullptr};
  std::uint32_t row{0};
  std::uint32_t rows{0};
};

/// y[t] = C[t] * (w0 * s0 + w1 * s1 + w2 * B[t] * x[t]) per channel, where
/// s0, s1 are the two previous B * x rows (the state, then earlier rows of
/// this call). `bcx` rows are [B | C | x] (each `channels` wide) `stride`
/// floats apart; `w` is the GGUF [channels][3] tap layout. The state is
/// updated to the last two B * x rows.
void LfmShortConv(const float* bcx, std::uint32_t stride, const float* w,
                  float* state, float* y, std::uint32_t n_tokens,
                  std::uint32_t channels, hipStream_t stream);
void LfmShortConvBatch(const float* bcx, std::uint32_t stride, const float* w,
                       const LfmConvItem* items, std::uint32_t count,
                       std::uint32_t max_rows, float* y, std::uint32_t channels,
                       hipStream_t stream);

/// From stacked [q | k | v] rows: RMS-norms every q and k head with its
/// gamma, applies NEOX RoPE over all 64 dims at position *start_pos + t,
/// writes q to `q` ([t][32][64]) and stores k and v as F16 cache rows
/// ([position][8 * 64]).
void LfmPrepareAttention(const float* qkv, const float* q_gamma,
                         const float* k_gamma, float* q, const KvView& kv,
                         std::uint32_t n_tokens, const std::uint32_t* start_pos,
                         float theta, float eps, hipStream_t stream);
/// The same per item: rows [row, row + rows) at *position + i, into the
/// item's cache.
void LfmPrepareAttentionBatch(const float* qkv, const float* q_gamma,
                              const float* k_gamma, float* q,
                              KvCacheFormat format,
                              const AttentionBatchItem* items,
                              std::uint32_t count, std::uint32_t max_rows,
                              float theta, float eps, hipStream_t stream);

/// Causal attention of q rows ([t][32][64]) over an F16 cache: row t sees
/// keys [0, *start_pos + t]. Each work-group reads a key tile once for the
/// four query heads of one KV head. Few rows split the keys eight ways
/// through `partials` (kVecBatch * 32 * 8 * 66 floats) and merge.
void LfmAttention(const float* q, const KvView& kv, float* out, float* partials,
                  std::uint32_t n_tokens, const std::uint32_t* start_pos,
                  hipStream_t stream);
/// Multi-session decode rows: item rows split their keys item.splits ways
/// into partials + item.partial records, then merge.
/// `rows`: rows of `out` rotated back after a TurboQuant attention (the
/// decode rows the items cover).
void LfmAttentionBatch(const float* q, KvCacheFormat format,
                       const AttentionBatchItem* items, std::uint32_t count,
                       std::uint32_t max_rows, std::uint32_t rows, float* out,
                       float* partials, hipStream_t stream);

/// Router over 64 experts from the normalized rows x ([t][2048]) and the
/// F32 router matrix w ([64][2048]): sigmoid scores, top-k chosen on score +
/// bias (lowest index on ties), weights the chosen scores normalized to
/// sum 1.
void LfmRouter(const float* x, const float* w, const float* bias,
               std::int32_t* ids, float* weights, std::uint32_t n_tokens,
               std::uint32_t k, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_LFM_HPP_
