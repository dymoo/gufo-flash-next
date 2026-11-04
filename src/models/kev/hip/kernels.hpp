#ifndef GUFO_MODELS_KEV_HIP_KERNELS_HPP_
#define GUFO_MODELS_KEV_HIP_KERNELS_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Kernels for Kev's packed pass: every state and question row of a batch in
/// one varlen token buffer. Recurrent state lives in slots (one per cached or
/// in-flight state); attention keys and values live in a token-slot pool.
namespace gufo::models::kev::hip {

inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kQueryHeads = 16;
inline constexpr std::uint32_t kKvHeads = 4;
inline constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
/// Halves per KV slot per attention layer: K then V.
inline constexpr std::uint32_t kSlotHalves = 2 * kKvWidth;
inline constexpr std::uint32_t kSsmDim = 128;
inline constexpr std::uint32_t kConvTaps = 4;
inline constexpr std::int32_t kNoSlot = -1;

/// One conv/DeltaNet sequence of the packed buffer.
struct Sequence {
  std::uint32_t offset;  ///< First packed token.
  std::uint32_t length;
  std::int32_t init_slot;   ///< Recurrent slot read first, kNoSlot = zeros.
  std::int32_t final_slot;  ///< Recurrent slot written last, or kNoSlot.
};

/// A tile of at most 32 queries of one sequence. Keys are the prefix (all
/// visible) followed by the query's own sequence (causal).
struct AttentionTile {
  std::uint32_t q_packed;     ///< Packed index of the tile's first query.
  std::uint32_t n;            ///< Queries in the tile, 1..32.
  std::uint32_t prefix_slot;  ///< KV slot of prefix key 0.
  std::uint32_t prefix_len;
  std::uint32_t own_slot;   ///< KV slot of the sequence's own key 0.
  std::uint32_t own_first;  ///< Own index of the tile's first query.
};

inline constexpr std::uint32_t kAttentionTileQueries = 32;

struct RecurrentPools {
  float* conv;   ///< [slot][ssm_layer][channels][4]
  float* state;  ///< [slot][ssm_layer][v_heads][128][128]
  std::uint32_t ssm_layers;
  std::uint32_t channels;
  std::uint32_t v_heads;
};

/// Per-head q/k RMSNorm, RoPE at positions[t], K/V written to the slot pool
/// of `layer` (layer-major: [layer][slot][K|V][kv_head][dim], FP16). Q is
/// normalized and rotated in place.
void LaunchQkNormRopeKv(float* q, const float* k, const float* v,
                        const float* q_norm, const float* k_norm,
                        const std::uint32_t* positions,
                        const std::uint32_t* kv_slots, void* kv_pool,
                        std::size_t pool_slots, std::uint32_t layer,
                        std::size_t tokens, std::uint32_t rotary_dim,
                        float rope_theta, float eps, hipStream_t stream);

/// Block-causal attention with the sigmoid output gate:
/// out[t] = softmax(q k^T / 16) v * sigmoid(gate[t]).
void LaunchAttention(const float* q, const float* gate, const void* kv_pool,
                     std::size_t pool_slots, std::uint32_t layer,
                     const AttentionTile* tiles, std::uint32_t tile_count,
                     float* out, hipStream_t stream);

/// Causal depthwise conv (4 taps) + SiLU for the tokens of sequences whose
/// `phase` bit matches; history before a sequence comes from its init slot.
void LaunchConv(const float* qkv, const float* weights,
                const std::uint32_t* token_sequence, const Sequence* sequences,
                const std::uint8_t* sequence_phase, std::uint8_t phase,
                RecurrentPools pools, std::uint32_t ssm_layer,
                std::size_t tokens, float* out, hipStream_t stream);

/// Stores the last four conv inputs of each listed sequence into its final
/// slot.
void LaunchConvStore(const float* qkv, const Sequence* sequences,
                     const std::uint32_t* list, std::uint32_t count,
                     RecurrentPools pools, std::uint32_t ssm_layer,
                     hipStream_t stream);

/// L2-normalized q (times 1/sqrt(128)) and k for every token.
void LaunchDeltaPrep(const float* conv_out, float* qn, float* kn,
                     std::size_t tokens, std::uint32_t k_heads,
                     std::uint32_t v_heads, hipStream_t stream);

/// Gated delta rule for the listed sequences, one work group per (value head,
/// 32 state rows, sequence). Raw per-head outputs go to raw[t][v_heads][128].
void LaunchDeltaRule(const float* conv_out, const float* qn, const float* kn,
                     const float* alpha, const float* beta, const float* a,
                     const float* dt, const Sequence* sequences,
                     const std::uint32_t* list, std::uint32_t count,
                     RecurrentPools pools, std::uint32_t ssm_layer,
                     std::uint32_t k_heads, float* raw, hipStream_t stream);

/// rows[i] = hidden[index[i]].
void LaunchGatherRows(const float* hidden, const std::uint32_t* index,
                      std::uint32_t count, std::uint32_t width, float* rows,
                      hipStream_t stream);

}  // namespace gufo::models::kev::hip

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_KEV_HIP_KERNELS_HPP_
