#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/vision/rope.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
namespace gufo::models::qwen38_flash_next::rocm {

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
struct RollbackRows {
  float* rows[7]{};
};

// Recurrent rollback keeps the state after the first token, then exact
// FP32 keys/decays/betas/errors for later tokens. No inverse or quantization.
constexpr std::size_t GdnRollbackRowFloats(std::uint32_t row,
                                           std::uint32_t k_heads,
                                           std::uint32_t v_heads,
                                           std::uint32_t dim) {
  return row == 0 ? std::size_t{v_heads} * dim * dim
                  : std::size_t{k_heads} * dim + v_heads * (dim + 2);
}
/// Replays the rollback rows onto `state`, stored as `format`.
void RestoreGdnState(void* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t k_heads, std::uint32_t v_heads,
                     hipStream_t stream,
                     GdnStateFormat format = GdnStateFormat::kF32);

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kIQ3_XXS = 18,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ4_XS = 23,
  kBF16 = 30,
};

/// Re-encodes `blocks` Q4_0 blocks exactly as Q8_0 (dense matrices) or as
/// Q5_1 (routed down experts, m = -8 d); `out` holds blocks * 34 or 24 bytes.
void ReencodeQ4_0(const void* q4_0, void* out, std::size_t blocks, bool q5_1,
                  hipStream_t stream);
/// Rewrites `rows` Q6_K rows of `cols` elements as Q8_0 (one F16 scale per
/// 32 values, codes rounded to nearest) for the routed down kernels. Lossy
/// only by that rounding, well below Q6_K's own step.
void TranscodeQ6KToQ8_0(const void* q6k, void* q8, std::size_t rows,
                        std::size_t cols, hipStream_t stream);
/// Rewrites `rows` Q4_K rows of `cols` elements as Q5_1: each 32-value
/// sub-block keeps its codes, with d * scale and -dmin * min rounded to F16.
void TranscodeQ4KToQ5_1(const void* q4k, void* q5_1, std::size_t rows,
                        std::size_t cols, hipStream_t stream);

/// res[t][s][hidden] = table[tokens[t]] for every stream s (F32, BF16, F16,
/// Q8_0, Q4_K, Q5_K or Q6_K rows).
void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// Hyper-connection mix epilogue: mixed[t][i] = mean_s xn[t][s][i] *
/// sigmoid(gate[t][s][i]); with `inject_w` ([streams][hc_dim] f32) also
/// inject[t][s] = dot(inject_w[s], xn[t]).
/// Partial sums per inject logit the fused epilogue emits; `inject` holds
/// n_tokens * streams * HcInjectParts(hidden) floats.
std::uint32_t HcInjectParts(std::uint32_t hidden);
std::uint32_t HcInjectPartsVec4(std::uint32_t hidden);
/// A non-null `mixed_q81` also receives the mixed rows as Q8_1 blocks in the
/// vendored quantizer's layout (rows padded to 512 values; hidden % 32 == 0),
/// the input the decode GEMVs read. At most four streams.
void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream, void* mixed_q81 = nullptr);
/// Four adjacent hidden lanes per thread. The model's four-stream geometry
/// emits HcInjectPartsVec4(hidden) partial sums per inject logit.
void HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream);
/// The vec4 epilogue over an F16 `xn` (the F16 mixer input route); the
/// four-stream geometry is the caller's contract. Non-null `mixed_half` /
/// `mixed_q8` also receive the mixed rows as F16 and in the W8A8 tiled Q8
/// layout (K = hidden).
void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed,
                          __half* mixed_half, void* mixed_q8, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream);

/// res[t][s][i] += block_out[t][i] * 2*sigmoid(inject[t][s] / streams), and
/// when `gamma` is non-null also the next mixer's grouped RMSNorm of the
/// updated residual into `xn`.
/// `inject` is [t][s][inject_parts] partial sums (1 part for a plain GEMM).
/// Concatenated decode requests retain scalar reduction with `decode=true`.
/// A non-null `xn_q81` also receives the norm as Q8_1 blocks in the vendored
/// quantizer's layout (streams * hidden % 512 == 0), the decode GEMV input.
void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream,
               bool decode = false, void* xn_q81 = nullptr);
/// Lossy F16 residual storage for the wide prefill route (--prefill-residual).
/// mode 0: the F32 `res` (the default). mode 1: read the F32 `res`, write the
/// F16 `res_half`. mode 2: read and write `res_half`. A non-null `xn_scale`
/// also drops the F16 norm write (`xn` may then be null): the combine writes
/// the four per-stream rsqrt scales [t][4] instead, and the next mixer's
/// HcMixF16Gemm rebuilds xn = res_half * scale * gamma. Modes 1 and 2 track
/// the largest residual magnitude (HcResAbsMax) and clamp it to the F16 range.
struct HcResArgs {
  int mode{0};
  __half* res_half{nullptr};
  float* xn_scale{nullptr};
};
/// Largest |residual| stored in F16 since the last reset.
float HcResAbsMax(bool reset, hipStream_t stream);
/// res[i] = float(res_half[i]) over `count` elements.
void HcResHalfToF32(const __half* res_half, float* res, std::size_t count,
                    hipStream_t stream);

/// HcCombine writing the normalized output as F16, the width the next
/// mixer's projection and epilogue consume on the F16 mixer input route.
/// A non-null `xn_q8` also receives the norm quantized into the tiled Q8
/// layout (`Q8TiledBytes(n_tokens, streams * hidden)`) for the W8A8 down
/// projection; hidden must be a multiple of 32.
/// A non-null `inject_w` (the next mixer's F32 [streams][hc_dim] inject
/// weights) also writes that mixer's inject partial sums
/// ([t][streams][HcInjectPartsVec4(hidden)], bit-identical to
/// HcMixEpilogueVec4F16's) into `inject_next`; it may alias `inject` only
/// when inject_parts == HcInjectPartsVec4(hidden). The mixer then skips its
/// own inject pass.
void HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream,
                  const float* inject_w = nullptr, float* inject_next = nullptr,
                  const HcResArgs& res_args = {});
/// HcCombineF16 with the MoE epilogue fused in: the block output is
/// MoeEpilogueVec4F16's result over `expert_out` ([tokens][used][hidden]
/// F16 rows), `weights`, the gated shared expert; it is formed in
/// registers and never written. Returns false (launching nothing) for a
/// geometry the fused kernel does not cover (four 2,560-wide streams with a
/// norm and the tiled Q8 output are required).
/// Rows of a batch with their steering adapter: rows[t] is 0 for none or
/// i + 1 for adapter i, whose unit direction is directions[i * hidden...].
struct SteerRows {
  const std::uint8_t* rows{nullptr};
  const float* directions{nullptr};
  const float* alphas{nullptr};
  /// MoE outputs only (else null): per adapter, the layers whose routed
  /// experts are ablated (bit `layer`); in the others only the gated shared
  /// expert sigmoid(gate[t * gate_stride]) * shared[t] is.
  const std::uint64_t* routed_layers{nullptr};
  std::uint32_t layer{0};
  const float* shared{nullptr};
  const float* gate{nullptr};
  std::uint32_t gate_stride{0};
  /// Writer-class rows (embedding, PLE value, MTP blocks; else null): per
  /// adapter its SteeringAdapter::writers mask; rows of adapters without
  /// `writer` are untouched.
  const std::uint8_t* writers{nullptr};
  std::uint8_t writer{0};
};
/// Steering adapters' directional ablation of block-output rows:
/// y[t] -= alpha * dot(r, p[t]) * r, where p is y or, for an MoE output
/// outside the adapter's routed layers, its gated shared-expert part; rows
/// without an adapter are untouched. With `streams` > 1 each row holds that
/// many equal copies (stride hidden), all updated by stream 0's projection.
void Steer(float* y, const SteerRows& steer, std::uint32_t n_tokens,
           std::uint32_t hidden, hipStream_t stream, std::uint32_t streams = 1);
/// A non-null `steer` ablates adapter rows' block outputs (Steer) first.
bool HcCombineMoeF16(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, const float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t n_tokens,
    std::uint32_t hidden, std::uint32_t streams, float eps, hipStream_t stream,
    const SteerRows* steer = nullptr, const float* inject_w = nullptr,
    float* inject_next = nullptr, const HcResArgs& res_args = {});

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// out[i] = silu(gate[i]) * up[i] as F16 (`count` elements).
void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream);
/// silu(gate) * up over n_rows rows of k elements, written only into the
/// W8A8 tiled Q8 layout `out_q8`; false (nothing launched) unless k % 32 == 0.
bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, alpha/beta, indexer, inject) that the
/// quantized tier does not cover.
/// Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// W8A8 route for wide batches over Q8_0 weights: activations quantized per
/// 32-wide block into the tiled fragment layout (`Q8TiledBytes(batch, k)`
/// bytes, padded to the GEMM's 128-token macro tile), then an int8 WMMA
/// GEMM. out is [batch][m]. W8A8Gemm returns false, launching nothing, when
/// k is not a multiple of 32.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream);
/// Native wave64 launch, selected by W8A8Gemm for wide output projections.
void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

/// Q8_0 HC down projection [320,10240], SiLU(x / 4), then F16 output.
/// Matches the separate operators' rounding. Supports at least 96 tokens.
bool HcDownF16Gemm(const void* w, const void* x_tiled, __half* out,
                   std::uint32_t n_tokens, hipStream_t stream);

/// Attention KV cache layouts (KvCacheFormat). kF16 rows are
/// [ctx][kv_heads * d] halves.
/// TurboQuant formats hold, per (token, KV head), the 256-wide head row
/// rotated by R = P·H·diag(s)/16 (fixed ±1 signs s, Walsh-Hadamard H, and a
/// lane permutation P that makes storage positions contiguous per lane) as
/// integer Lloyd-Max codes plus one FP16 least-squares scale:
/// row ≈ R^T (scale · centroid[code]). Queries are rotated by the same R and
/// the attention output is rotated back before the sigmoid gate. Code rows
/// are [ctx][kv_heads][KvCodeBytes]; scales are [ctx][kv_heads] halves.
///   kTq4: 128 bytes; byte 4w + b holds lane w's positions 8w + b (low
///         nibble) and 8w + b + 4 (high nibble), b < 4.
///   kTq3: 96 bytes; a 64-byte plane of low bit pairs (lane w's u16: bits
///         4b..4b+1 position 8w + b, bits 4b+2.. position 8w + b + 4) and a
///         32-byte plane of high bits (lane w's byte: bit 2b position 8w + b,
///         bit 2b + 1 position 8w + b + 4).
///   kTq8: 256 bytes, one signed code per position (LFM2's 64-wide heads
///         only: lfm.hip.cpp).
/// Code bytes of one (token, KV head) row of 256 values; 0 for kF16.
constexpr std::uint32_t KvCodeBytes(KvCacheFormat format) {
  return format == KvCacheFormat::kTq4   ? 128
         : format == KvCacheFormat::kTq3 ? 96
         : format == KvCacheFormat::kTq8 ? 256
                                         : 0;
}
/// Device bytes of one layer's K and V for one token of kv_heads heads of d
/// values (TurboQuant requires d == 256).
constexpr std::size_t KvTokenBytes(KvCacheFormat format, std::uint32_t kv_heads,
                                   std::uint32_t d) {
  return format == KvCacheFormat::kF16
             ? std::size_t{2} * kv_heads * d * 2
             : std::size_t{2} * kv_heads * (KvCodeBytes(format) * d / 256 + 2);
}
/// Rotation signs: bit e % 32 of word e / 32 set means element e is negated
/// (splitmix64 seed 0x51A7C0DE).
inline constexpr std::uint32_t kTqSigns[8] = {
    0x1F672E1CU, 0x0141AC0CU, 0x62C1832EU, 0xE1575E0EU,
    0x31F568BCU, 0x302C9A51U, 0xFB589FA7U, 0x76A05942U};
/// Integer Lloyd-Max centroids for N(0, k^2) (positive half, ascending);
/// code 2^(b-1) + i is +centroid[i], code 2^(b-1) - 1 - i is -centroid[i].
/// k = 38.175 (4-bit) and 52.948 (3-bit) map unit variance to these units.
inline constexpr float kTq4Centroids[8] = {5, 15, 25, 36, 48, 62, 79, 104};
inline constexpr float kTq3Centroids[4] = {13, 40, 71, 114};
inline constexpr float kTq4Unit = 38.175F;
inline constexpr float kTq3Unit = 52.948F;

/// One layer's K and V caches. Scales are null for kF16.
struct KvView {
  KvCacheFormat format{KvCacheFormat::kF16};
  void* k{nullptr};
  void* v{nullptr};
  __half* k_scale{nullptr};
  __half* v_scale{nullptr};
};

/// One session's rows in a multi-session decode attention layer: rows of
/// the item are the batch rows [row, row + rows) of every packed input and
/// output (projections, queries, gates, indexer rows, masks, scores, ctx).
/// Items are device-accessible (mapped host memory); rows == 0 skips one.
struct AttentionBatchItem {
  KvView kv;
  float* index_k{nullptr};                    ///< raw indexer ring
  __half* block_k{nullptr};                   ///< pooled block keys
  const std::uint32_t* position{nullptr};     ///< device: first row position
  const std::uint32_t* first_block{nullptr};  ///< device: blocks pooled
  const qwen::vision::DeviceRope* rope{nullptr};
  std::uint32_t index_capacity{0};  ///< power-of-two ring rows
  std::uint32_t row{0};
  std::uint32_t rows{0};
  std::uint32_t pool{0};     ///< bound on blocks the rows complete
  std::uint32_t sparse{0};   ///< nonzero: select blocks and mask attention
  std::uint32_t splits{1};   ///< AttentionSplits(rows)
  std::uint32_t partial{0};  ///< first split record (rows * heads * splits)
};

/// Key splits of every narrow decode attention row. Fixed, so a row reduces
/// identically however many rows share the call (serial or batched decode,
/// verification or draft catch-up).
constexpr std::uint32_t kMaxAttentionSplits = 8;
constexpr std::uint32_t AttentionSplits(std::uint32_t /*rows*/) {
  return kMaxAttentionSplits;
}

/// Quantizes F32 K/V rows ([t][kv_heads * 256], already normalized and
/// rotated by RoPE) into a TurboQuant cache at positions *start_pos + t.
void QuantizeKv(const float* k, const float* v, const KvView& kv,
                std::uint32_t n_tokens, std::uint32_t kv_heads,
                const std::uint32_t* start_pos, hipStream_t stream);
/// In-place R over `rows` 256-wide rows (queries before TurboQuant attention).
void RotateHeads(float* x, std::uint32_t rows, hipStream_t stream);
/// In-place R^T over `rows` 256-wide rows, then x *= sigmoid(gate) when
/// `gate` (same layout) is non-null.
void UnrotateHeads(float* x, const float* gate, std::uint32_t rows,
                   hipStream_t stream);

/// Stacked Q8_0 QKV projection [13312,2560], head normalization and RoPE.
/// Writes Q/gates in F32 and K/V caches in F16, preserving separate rounding.
/// Fixed geometry: 24 query heads, two KV heads, 256 dimensions, 64 rotary.
/// Requires at least 1024 tokens; cache capacity must include position +
/// tokens. TurboQuant caches receive quantized rows instead.
bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, const KvView& kv, std::uint32_t n_tokens,
                      const std::uint32_t* position, float theta, float eps,
                      hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr);
inline bool AttentionF16Gemm(const void* weights, const __half* input,
                             const float* q_gamma, const float* k_gamma,
                             float* query, float* gate, __half* keys,
                             __half* values, std::uint32_t n_tokens,
                             const std::uint32_t* position, float theta,
                             float eps, hipStream_t stream,
                             const qwen::vision::DeviceRope* rope = nullptr) {
  return AttentionF16Gemm(weights, input, q_gamma, k_gamma, query, gate,
                          KvView{KvCacheFormat::kF16, keys, values}, n_tokens,
                          position, theta, eps, stream, rope);
}

/// Exact Q8_0 HC up projection and F16-input mixer for the Flash Next
/// 2560-hidden, rank-320 geometry. Returns false below 96 tokens or for other
/// shapes. low_rank must not overlap mixed_half; side outputs are optional.
bool HcMixF16Gemm(const void* up, const __half* low_rank, const __half* xn,
                  const float* inject_w, float* mixed, __half* mixed_half,
                  void* mixed_q8, float* inject, std::uint32_t n_tokens,
                  std::uint32_t hidden, std::uint32_t rank, hipStream_t stream,
                  const float* xn_scale = nullptr,
                  const float* xn_gamma = nullptr);

/// F16 activation rows [batch][k], Q8_0 weights dequantized to F16 in LDS,
/// F32 accumulation. out is [batch][m]. Unsupported shapes launch nothing.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream);

/// SSM Q8_0 projection fused with its four-tap convolution. Supports
/// [m=16384,k=2560,channels=10240] and at least 1024 tokens. qkvz retains
/// every Z row plus the QKV rows required for tile boundaries and rolling
/// history; other QKV rows are not written. convolved is [tokens][channels].
/// history is read only; pass convolved=true to GatedDeltaNet to consume
/// this result and update history. Speculative snapshots need all raw QKV
/// rows and must use DenseF16Gemm instead. Unsupported shapes launch nothing.
bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, 48-row token tile) as expert | tile << 16,
/// so empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q8_0.
/// Q4_K and Q5_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other types
/// and tile widths return false. `w` is in the packed routed layout
/// (PackRoutedExperts, moe.hpp), which needs m % 32 == 0.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Paired Q4_K/Q5_K gate/up GEMM with SwiGLU and F16 output. Uses the same
/// routing layout as RoutedF16Gemm, with 64 or 128 token rows per tile. The two
/// projections share one launch and never materialize the gate output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

/// F32/BF16/F16 projection in groups of up to eight token rows. Each token
/// keeps the same accumulation order at every batch width.
void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// PLE gate: for every stream s, sc = dot(key_n[t][s], query_n[t][s]) /
/// sqrt(hidden); gated[t][s][i] = value[t][i] * sigmoid(sgn(sc) *
/// sqrt(max(|sc|, 1e-6))).
void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream);

/// Dilated depthwise causal conv over tokens with SiLU:
/// out[t][c] = silu(sum_k w[c*kernel+k] * in[t - (kernel-1-k)*dilation][c]),
/// where tokens before the chunk come from `history` ([hist][channels],
/// oldest first). `history` is then advanced past the chunk.
/// `snapshots`, when populated, receives the history as it stands after
/// each proper prefix: [n_tokens-1][hist][channels]. The full batch's
/// history remains in `history`.
void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, RollbackRows snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream);

/// res[t][c] += gated[t][c] + conv[t][c] over hc_dim channels.
void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream);

/// Gated DeltaNet over a chunk of tokens for one layer. Runs the causal
/// conv (with rolling `conv_state`, [kernel-1][channels]) and the recurrence
/// on `state` ([v_heads][d][d]) sequentially over tokens, parallel over heads
/// and value dims. `alpha_beta` holds both projections per token:
/// [t][alpha(v_heads) | beta(v_heads)]. Outputs the normalized,
/// sigmoid-gated attention rows [tokens][v_heads*d] ready for the output
/// projection.
/// `state_snapshots` ([n_tokens-1][v_heads*d*d]) and `conv_snapshots`
/// ([n_tokens-1][(kernel-1)*channels]), when populated, receive the state after
/// each proper prefix. The live state already holds the full batch.
/// `state` is stored as `state_format`; snapshots stay F32.
/// Scratch: `conv_scratch` ((n_tokens + kernel) * channels), `qn`/`kn`
/// (n_tokens * k_heads * d), `raw` (n_tokens * v_heads * d).
/// `qkv` rows are `qkv_stride` floats apart and `z` rows `z_stride`, so a
/// stacked [qkv|z] projection feeds both without unpacking. A non-null
/// `out_q8` receives the rows quantized into the W8A8 tiled layout
/// (K = v_heads * d) instead of `out`. With null `out_q8`, a non-null
/// `out_half` receives F16 rows instead of `out`. `convolved` consumes
/// convolution rows already in conv_scratch; it requires null speculative
/// snapshots.
void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, void* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half = nullptr, bool silu_gate = false,
                   GdnStateFormat state_format = GdnStateFormat::kF32,
                   void* out_q81 = nullptr);

/// Private rows for one request in a decode batch. Scratch regions and all
/// recurrent/history/rollback buffers must be disjoint between requests.
struct GdnBatchItem {
  const float* qkv;
  const float* z;
  const float* alpha_beta;
  float* conv_state;
  float* conv_scratch;
  float* qn;
  float* kn;
  float* raw;
  void* state;  ///< stored as GatedDeltaNetBatch's `state_format`
  float* out;
  RollbackRows state_snapshots;
  RollbackRows conv_snapshots;
  std::uint32_t n_tokens;
};

/// Runs the decode arithmetic for independent requests of 1–8 rows,
/// 128-wide heads and four convolution taps. `items` is device-accessible;
/// only its first `active` entries run (the caller compacts active requests
/// and zeros the output rows of the others).
bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream,
                        bool silu_gate = false,
                        GdnStateFormat state_format = GdnStateFormat::kF32);

/// Splits the interleaved [q|gate] projection (rows `qg_stride` apart) into
/// q [t][heads][d] and gate [t][heads*d]. With non-null `k`, the row
/// continues with k and v (`kv_width` each), copied out contiguously.
void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream);

/// Unpacks a stacked Q/gate/K/V projection, normalizes and rotates Q/K,
/// and writes the caches (TurboQuant caches need d == 256). Returns false
/// for heads wider than 256.
bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, const KvView& kv, std::uint32_t n_tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t d, std::uint32_t rotary_dim,
                      const std::uint32_t* start_pos, float theta, float eps,
                      hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr,
                      bool prefill = false);
inline bool PrepareAttention(
    const float* packed, std::uint32_t stride, const float* q_gamma,
    const float* k_gamma, float* q, float* gate, __half* k_cache,
    __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t rotary_dim,
    const std::uint32_t* start_pos, float theta, float eps, hipStream_t stream,
    const qwen::vision::DeviceRope* rope = nullptr, bool prefill = false) {
  return PrepareAttention(packed, stride, q_gamma, k_gamma, q, gate,
                          KvView{KvCacheFormat::kF16, k_cache, v_cache},
                          n_tokens, heads, kv_heads, d, rotary_dim, start_pos,
                          theta, eps, stream, rope, prefill);
}

/// NEOX partial rotary on x [t][heads][d] at positions start_pos + t.
/// Positions are read from device memory (`start_pos` points at the
/// session's control block) so a captured decode graph replays at any
/// position.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope = nullptr);

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim].
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream);
/// Stores raw indexer rows in a power-of-two ring of `capacity` rows.
void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               std::uint32_t capacity, hipStream_t stream);

/// Pools raw indexer keys ([pos % capacity][dim] f32) of the blocks the batch
/// completes, [*first_block, (*start_pos + n_tokens) / ratio), into block
/// keys: mean over `ratio` positions, RMSNorm with gamma, rotary at the
/// block start position, then round once to F16 for scoring.
/// `grid_blocks` bounds how many blocks one launch
/// can pool (a device-side bound is not available at launch time).
void PoolIndexerBlocks(const float* raw_keys, const float* gamma,
                       __half* blocks, const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, std::uint32_t capacity, hipStream_t stream,
                       const qwen::vision::DeviceRope* rope = nullptr);

/// Per query t (position *start_pos + first_token + t): scores every
/// complete block below its own tail, keeps the `budget` highest, and
/// writes a visibility bitmap (`mask_words` uint32 per query, bit b = block
/// b visible). Every block is visible when the count fits the budget.
/// Queries retain F32 precision; pooled cache keys are F16. Scores
/// still accumulate in FP32. `scores` holds n_tokens * max_blocks floats.
void SelectBlocks(const float* q, const __half* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream);

/// Per-token attention (decode and narrow batches). With `partials`
/// (n_tokens * heads * splits * (d + 2) floats) the key tiles are split
/// across `splits` blocks per row and merged in a second launch.
/// TurboQuant caches take rotated queries and leave rotated outputs.
void Attention(const float* q, const KvView& kv, const std::uint32_t* mask,
               std::uint32_t mask_words, float* out, float* partials,
               std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream);
inline void Attention(const float* q, const __half* k_cache,
                      const __half* v_cache, const std::uint32_t* mask,
                      std::uint32_t mask_words, float* out, float* partials,
                      std::uint32_t splits, std::uint32_t n_tokens,
                      const std::uint32_t* start_pos, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t ratio, hipStream_t stream) {
  Attention(q,
            KvView{KvCacheFormat::kF16, const_cast<__half*>(k_cache),
                   const_cast<__half*>(v_cache)},
            mask, mask_words, out, partials, splits, n_tokens, start_pos, heads,
            kv_heads, d, ratio, stream);
}

/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` follows
/// Attention's block-selection contract (null for the dense window). Returns
/// false, launching nothing, when the geometry is not the model's 24 x 256
/// heads over two KV heads. `last_only` computes only the final dense query
/// tile, retaining its key sweep and leaving earlier output rows untouched.
/// TurboQuant caches take rotated queries; pass a null gate and apply
/// UnrotateHeads (with the gate) to the output.
bool WmmaCausalAttention(const float* q, const float* gate, const KvView& kv,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only = false);
inline bool WmmaCausalAttention(const float* q, const float* gate,
                                const __half* k_cache, const __half* v_cache,
                                const std::uint32_t* mask,
                                std::uint32_t mask_words, float* out,
                                std::uint32_t n_tokens, std::uint32_t start_pos,
                                std::uint32_t heads, std::uint32_t kv_heads,
                                std::uint32_t d, std::uint32_t ratio,
                                hipStream_t stream, bool last_only = false) {
  return WmmaCausalAttention(
      q, gate,
      KvView{KvCacheFormat::kF16, const_cast<__half*>(k_cache),
             const_cast<__half*>(v_cache)},
      mask, mask_words, out, n_tokens, start_pos, heads, kv_heads, d, ratio,
      stream, last_only);
}

/// Multi-session attention: each stage is one launch for all
/// `count` items (AttentionBatchItem, device-accessible; `max_rows` is the
/// largest item.rows), with the same per-row arithmetic as the
/// single-session calls above (Attention takes AttentionSplits(rows)).
/// Stages, in order: PrepareAttentionBatch (queries, gates, K/V caches from
/// the stacked projection rows), StoreIndexerBatch (raw indexer keys into
/// each ring, then pooling of the blocks each item completes, `max_pool`
/// >= item.pool), SelectBlocksBatch (rotary on the RMS-normalized indexer
/// queries, then top-k masks for sparse items), AttentionBatch (split-key
/// attention into `out`, `partials` holding every item's split records).
/// SelectBlocksBatch covers `packed_rows` rows (item.row + rows of every
/// item, up to a whole fused prompt step) but `scores` holds only
/// `score_rows` rows of max_blocks floats: the rows are scored and marked in
/// windows of score_rows. It returns false, launching nothing, when
/// score_rows is zero or the indexer geometry is not the model's.
void PrepareAttentionBatch(const float* packed, std::uint32_t stride,
                           const float* q_gamma, const float* k_gamma, float* q,
                           float* gate, KvCacheFormat format,
                           const AttentionBatchItem* items, std::uint32_t count,
                           std::uint32_t max_rows, std::uint32_t heads,
                           std::uint32_t kv_heads, std::uint32_t d,
                           std::uint32_t rotary_dim, float theta, float eps,
                           hipStream_t stream);
void StoreIndexerBatch(const float* keys, const float* gamma,
                       const AttentionBatchItem* items, std::uint32_t count,
                       std::uint32_t max_rows, std::uint32_t max_pool,
                       std::uint32_t ratio, std::uint32_t dim,
                       std::uint32_t rotary_dim, float theta, float eps,
                       hipStream_t stream);
bool SelectBlocksBatch(float* q, std::uint32_t* mask, float* scores,
                       const AttentionBatchItem* items, std::uint32_t count,
                       std::uint32_t max_rows, std::uint32_t packed_rows,
                       std::uint32_t score_rows, std::uint32_t heads,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       std::uint32_t ratio, std::uint32_t budget,
                       std::uint32_t mask_words, std::uint32_t max_blocks,
                       hipStream_t stream);
void AttentionBatch(const float* q, KvCacheFormat format,
                    const AttentionBatchItem* items, std::uint32_t count,
                    std::uint32_t max_rows, const std::uint32_t* mask,
                    std::uint32_t mask_words, float* out, float* partials,
                    std::uint32_t heads, std::uint32_t kv_heads,
                    std::uint32_t d, std::uint32_t ratio, hipStream_t stream);

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
/// `norm_k` > k normalizes over the top norm_k probabilities instead (the
/// first k slots keep their top-norm_k weights; norm_k <= 32).
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream, std::uint32_t norm_k = 0);

/// Rewrites RouterTopK's ids/weights for the whole batch in place with a
/// `--moe-routing` policy (`mode` is MoeRouting::Mode: 1 affinity, 2 mass,
/// 3 budget; `value` is its margin or mass target). Rows keep their first
/// `keep` experts; dropped slots repeat the row's top expert at weight 0.
/// One block, no host sync; n_experts <= 512, k <= 32. `norm_k` > 0: a
/// changed row's weights are its probabilities over the row's top norm_k
/// (RouterTopK's norm_k semantics) instead of renormalized over its slots.
void RoutePolicy(const float* logits, std::uint32_t stride, std::int32_t* ids,
                 float* weights, std::uint32_t n_tokens,
                 std::uint32_t n_experts, std::uint32_t k, std::uint32_t mode,
                 std::uint32_t keep, float value, std::uint32_t budget,
                 hipStream_t stream, std::uint32_t norm_k = 0);

/// Fast-model rows (`keep[t]` in 1..k-1) keep only their first keep[t]
/// slots: the rest repeat the row's top expert at weight 0, as RoutePolicy's
/// dropped slots do, so the batch's expert union shrinks and every expert
/// route reads the same ids/weights layout. Rows with keep 0 or >= k are
/// unchanged. Slots are RouterTopK's, in descending probability.
void TruncateRoutes(std::int32_t* ids, float* weights, const std::uint8_t* keep,
                    std::uint32_t n_tokens, std::uint32_t k,
                    hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// dst[t] = *row < 0 ? alt[t] : base[*row + t] (rows of `width` floats).
void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream);

/// Add the projected embedding once to each projected hidden branch.
void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream);

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};
/// Full-vocabulary argmax into out[t], with the lowest index winning ties.
/// Matches std::max_element, including NaNs. Scratch holds
/// n_tokens * kArgmaxParts candidates.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);

/// Gather selected IDs and their original logits from independent rows.
/// The caller checks finite values only for the verification prefix it visits.
void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// BF16 rows as Q8_0 (the DSpark drafter's projections).
void TranscodeBf16ToQ8_0(const void* bf16, void* q8, std::size_t rows,
                         std::size_t cols, hipStream_t stream);
/// One DSpark injection target: `rows` fused K/V rows from `row` of the
/// stacked projection, stored at positions pos.. of one session's caches.
struct DsparkStoreItem {
  std::uint32_t row{0};
  std::uint32_t rows{0};
  std::uint32_t pos{0};
  std::uint32_t pad{0};
  std::array<__half*, 8> k{};
  std::array<__half*, 8> v{};
};
/// For every item row and block: K = RMSNorm(k) * k_norms[block], NEOX
/// rotary over the whole 256-wide head at its position, then K and V as
/// F16 into that block's caches. `kv` rows are [k0 v0 k1 v1 ...]
/// (`stride` floats); `items` is device memory.
void DsparkStoreKv(const float* kv, std::uint32_t stride, const float* k_norms,
                   const DsparkStoreItem* items, std::uint32_t count,
                   std::uint32_t max_rows, std::uint32_t blocks,
                   std::uint32_t kv_heads, float theta, float eps,
                   hipStream_t stream);

/// MtpTopCandidates with the DSpark Markov shortlist width.
inline constexpr std::uint32_t kDsparkCandidates = 256;
std::uint32_t DsparkCandidateWorkspaceSize(std::uint32_t vocab);
void DsparkTopCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream);
/// DSpark Markov step of block row `i`: u = W1[chain[i]] (BF16 [vocab][rank]
/// rows, also written to `u_out`), then chain[i + 1] = the candidate with the
/// highest score + W2[id] . u (lowest ID on ties) among `count` <= 256 IDs.
void DsparkMarkovStep(const void* w1, const void* w2, std::uint32_t rank,
                      const std::uint32_t* ids, const float* scores,
                      std::uint32_t count, std::int32_t* chain, std::uint32_t i,
                      float* u_out, hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KERNELS_HPP_
