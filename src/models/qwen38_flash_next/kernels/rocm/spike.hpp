#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_HPP_

// Spike: lossy low-bit prefill emulation for the int4 (iu4) WMMA study.
// Nothing here changes production behaviour unless a
// GUFO_SPIKE_* variable is set; with none set every wrapper forwards to the
// real kernel and Init returns immediately.
//
// The emulation is arithmetic-exact for the integer part of a W4A4 GEMM:
// activations are quantized to an int4/int8 grid (per token or per group) and
// dequantized back to F16, weights are re-quantized to a 4-bit grid stored in
// the existing Q8_0 container, and the existing F16 WMMA GEMM then multiplies
// them. 4-bit codes times 4-bit codes accumulate exactly in fp32, so this
// matches an integer-WMMA kernel up to the order of the final scale
// multiplications. It measures quality, not speed.
//
//   GUFO_SPIKE_ACT      none|a8|a4|fp4        activation grid (default none);
//                                             fp4 = NVFP4: E2M1 codes, per-
//                                             group scales rounded to UE4M3,
//                                             times a per-row FP32 scale that
//                                             puts the row absmax at 6 x 448
//   GUFO_SPIKE_AGROUP   0 | 16 | 32 | ...     0 = one scale per token row;
//                                             default 16 for fp4, else 0
//   GUFO_SPIKE_CLIP     float, default 1      scale = clip * absmax / qmax
//   GUFO_SPIKE_ROUTED_ACT / _AGROUP / _CLIP   the recipe of the routed roles
//                                             (32, 64); each defaults to the
//                                             dense value above, so dense and
//                                             routed recipes stack in one run
//   GUFO_SPIKE_ACLIP_SEARCH 0|1               per group (or row) pick the clip
//                                             in {1, .95, ..., .65} with the
//                                             least squared error (replaces
//                                             CLIP for the scale choice)
//   GUFO_SPIKE_OUTLIERS K (0..256)            keep K channels in F16 (excluded
//                                             from the absmax), the rest on
//                                             the grid: an int4 GEMM plus a
//                                             K-wide F16 residual GEMM
//   GUFO_SPIKE_OUTLIER_MODE row|call          row: each row's K largest |x|
//                                             (after rotation); call: one set
//                                             per GEMM call, the K channels
//                                             with the largest column absmax
//   GUFO_SPIKE_MIN_ROWS N                     calls with fewer than N rows
//                                             (x_rows: tokens, token x slot
//                                             for routed down) keep the
//                                             original weights and activations
//   GUFO_SPIKE_MIN_ROWS_MODE all|act          all (default): as above. act:
//                                             below N rows, outside prefill
//                                             and in decode GEMVs the dense
//                                             W4 shadow (and its rotation) is
//                                             used with exact activations
//                                             (W4 always, A4 on big prefill)
//   GUFO_SPIKE_A8_LAYERS comma list           those layers use int8 instead of
//                                             the recipe's grid (same group)
//   GUFO_SPIKE_HAD      0 | 128 | 1           block Hadamard on activations
//                                             and (dense) weights; 1 rotates
//                                             the whole row: Kronecker
//                                             H_{k/b} x H_b (b = 128, or 32
//                                             for k = 640; Paley/Sylvester)
//   GUFO_SPIKE_WGROUP   0 | 1 | 32 | 64 | 128 requantize dense Q8_0 weights to
//                                             a 4-bit grid with this group
//                                             (1 = one scale per row)
//   GUFO_SPIKE_WBITS    4 | 5 | 6 | 8         weight grid of WGROUP (codes in
//                                             [-2^(b-1), 2^(b-1) - 1], same
//                                             group / SCALE6 handling)
//   Weights-only (decode) study: WGROUP + WBITS + MIN_ROWS_MODE=act and no
//   ACT: every prefill, small and decode-GEMV call of the targeted dense
//   families reads the shadow with unquantized activations.
//   GUFO_SPIKE_SCALE6   1 | 0                 6-bit group scales x row scale
//   GUFO_SPIKE_TARGETS  bitmask, default 127  1 ssm_in, 2 ssm_out, 4 attn_qkv,
//                                             8 attn_out, 16 shared expert,
//                                             32 routed gate/up, 64 routed down
//   GUFO_SPIKE_KEEP     comma list of layer indices left untouched
//   GUFO_SPIKE_REAL / _REAL_CHECK             integer iu4 kernels (dense,
//                                             per-row int4 only)
//   GUFO_SPIKE_ROUTED_REAL 0|1                prefill routed experts on the
//                                             integer iu4 kernels (Q4_K
//                                             gate/up exact; A4 per
//                                             ROUTED_AGROUP 0|32|64|128)
//   GUFO_SPIKE_ROUTED_DOWN4 0|1               Q5_1 down experts re-quantized
//                                             in place to 4-bit codes at Init
//                                             (every route; required by the
//                                             integer down; alone = emulation)
//   GUFO_SPIKE_ROUTED_CHECK N                 routed real mode also emulates
//                                             the first N gate/up and down
//                                             calls and logs the rel RMS
//   GUFO_SPIKE_ROUTED_TALL 0|1                routed real: 128 x 64 blocks
//                                             on 128-row tile maps
//   GUFO_SPIKE_ROUTED_HAD 0|128               the GGUF stores the routed
//                                             experts rotated (W R^T, R =
//                                             BlockHadamard(128)): every
//                                             routed call (prefill, decode,
//                                             real or emulated) rotates its
//                                             activations (x and the SwiGLU
//                                             rows)
//   GUFO_SPIKE_LOG      1 = print the registry summary
//
// Init prints the parsed recipe, parse warnings and any GUFO_SPIKE_* name
// this binary does not read. Parsing and the scalar numerics live in
// spike_knobs.hpp (CPU-tested by spike_knobs_test).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm::spike {

/// True when any GUFO_SPIKE_* variable asks for emulation.
[[nodiscard]] bool Enabled() noexcept;

/// GUFO_SPIKE_ROUTED_HAD: 128 when `w` is a main-layer routed expert matrix
/// stored rotated (its activations must be rotated on every route), else 0.
[[nodiscard]] int RoutedRotation(const void* w) noexcept;

/// Registers the model's prefill GEMM weights and builds requantized
/// shadows. Idempotent; safe to call with the spike disabled.
void Init(const DeviceModel& model, hipStream_t stream);

/// Roles for RegisterTensor (the GUFO_SPIKE_TARGETS bits).
enum Role : unsigned {
  kSsmIn = 1,
  kSsmOut = 2,
  kAttnQkv = 4,
  kAttnOut = 8,
  kShared = 16,
  kRoutedGateUp = 32,
  kRoutedDown = 64,
};

/// Registers one tensor (a dense Q8_0 projection or a routed expert matrix)
/// and builds its shadow when GUFO_SPIKE_WGROUP asks for one. Returns false
/// when the tensor is filtered out or does not fit. Used by Init and tests.
bool RegisterTensor(const DeviceTensor& t, unsigned role, int layer, bool dense,
                    hipStream_t stream);

/// The requantized shadow registered for `w`, or nullptr (tests).
[[nodiscard]] const void* ShadowOf(const void* w);

/// True when the integer (iu4) kernels own `w` (GUFO_SPIKE_REAL): the fused
/// projections that cannot take it must be skipped by the caller.
[[nodiscard]] bool Handles(const void* w);

/// Emulated W4A4 GEMM for `w` regardless of GUFO_SPIKE_REAL (needs
/// GUFO_SPIKE_REAL_CHECK so the shadow exists); tests compare it with the
/// integer kernel.
bool DenseF16GemmEmulated(const void* w, const __half* x, float* out,
                          std::size_t batch, std::size_t m, std::size_t k,
                          hipStream_t stream);

// Wrappers: `prefill` is the caller's prefill_phase. Outside prefill, or for
// unregistered weights, they forward unchanged.
bool DenseF16Gemm(bool prefill, const void* w, const __half* x, float* out,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream);
/// True when a GEMM over `w` with `rows` rows has spike work to do:
/// registered and, for a prefill call at or above MIN_ROWS, a shadow /
/// packed weight, rotation or quantizing recipe; for smaller or non-prefill
/// calls (MIN_ROWS_MODE=act only, not inside graph capture) a W4 shadow or
/// rotation. Callers whose int8 route would bypass the F16 wrappers
/// (attn_out 2560x6144, shared gate/up 640x2560) then call DenseF32Gemm.
[[nodiscard]] bool Covers(bool prefill, const void* w, std::size_t rows,
                          hipStream_t stream);
/// DenseF16Gemm for F32 activations: narrows them to F16 in spike scratch.
bool DenseF32Gemm(bool prefill, const void* w, const float* x, float* out,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream);
/// MIN_ROWS_MODE=act decode GEMV (<= 8 rows) over Q8_0 `w`: sets *w_out to
/// the registered W4 shadow (else `w`) and *x_out to the rows rotated like
/// that shadow (else `x`), in a buffer Init preallocated for `stream` (graph
/// capture safe). False only when a rotation is required but cannot run.
bool GemvOperands(const void* w, const float* x, std::uint32_t rows,
                  std::uint32_t k, const void** w_out, const float** x_out,
                  hipStream_t stream);
/// The W4 shadow GemvOperands would use for `w` (the fused gate of a gated
/// GEMV shares the up projection's rows), else `w`.
[[nodiscard]] const void* GemvWeight(const void* w);
bool DenseF16SsmGemm(bool prefill, const void* w, const __half* x,
                     const float* conv_w, const float* history, float* qkvz,
                     float* convolved, std::uint32_t n_tokens, std::uint32_t m,
                     std::uint32_t k, std::uint32_t channels,
                     std::uint32_t kernel, hipStream_t stream);
bool AttentionF16Gemm(bool prefill, const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, const KvView& kv, std::uint32_t n_tokens,
                      const std::uint32_t* position, float theta, float eps,
                      hipStream_t stream, const qwen::vision::DeviceRope* rope);
/// `x_rows` is the number of F16 rows `x` holds (tokens for gate/up, token x
/// used-slot rows for down).
bool RoutedGatedF16Gemm(bool prefill, std::size_t x_rows, const void* gate,
                        const void* up, WeightType type, const __half* x,
                        const std::int32_t* tiles, std::uint32_t n_tiles,
                        std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);
bool RoutedF16Gemm(bool prefill, std::size_t x_rows, const void* w,
                   WeightType type, const __half* x, const std::int32_t* tiles,
                   std::uint32_t n_tiles, std::uint32_t tile_rows,
                   const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                   const std::int32_t* rows_out, const float* swiglu_gate,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm::spike

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_HPP_
