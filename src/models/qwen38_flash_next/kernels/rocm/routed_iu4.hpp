#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_IU4_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_IU4_HPP_

// Spike: integer W4A4 routed-expert prefill GEMMs on the gfx1151
// iu4 WMMA (16x16x16, 2x the F16 instruction rate). Nothing here runs unless
// a spike knob routes a call to it.
//
// Gate/up (Q4_K, as shipped, exact): a Q4_K weight is d*sc_g*q - dmin*m_g
// with q unsigned 0..15, so the iu4 WMMA multiplies the signed int4
// activation codes with the unsigned weight codes exactly, the 6-bit group
// scales fold into an int32 sum per super-block (v_mad_i32_i24) and the
// min term  sum_g (dmin*m_g) * (sum of the dequantized activations of g)  is
// an F16 WMMA with K = groups (one per two super-blocks, ~6% of the work).
// The weights are read in place from the lane-interleaved routed layout
// (moe.hpp); each block repacks its nibbles once into natural K order.
//
// Down (Q5_1 container holding 4-bit codes, RequantDownQ4): the same
// structure with one F16 scale per 32-group (float epilogue per group).
//
// Activations: symmetric int4 [-8, 7] codes with
//   group 0   one scale per row,       s = rowmax / 7          (= FqRows a4)
//   group 128 s = sx * a_g, a_g = ceil(255 * groupmax / rowmax) per 128
//   group 64 / 32  the same per 64 / 32 elements
// so every group scale is the row scale times an 8-bit integer and the
// integer sums stay exact. Groups of 64/128 apply their multiplier once per
// 2/4 weight groups (cheaper than 32). With `had` = 128 the rows are first
// rotated by spike.hip.cpp's BlockHadamard(128) (for experts stored rotated,
// W' = W R^T: GUFO_SPIKE_ROUTED_HAD).
//
// Shapes: gate/up k = 2560 (k % 256 == 0), m % 64 == 0; down k % 128 == 0,
// k <= 2560, m % 128 == 0. Token tiles of 64 or 128 compact rows
// (`tile_rows`), with the routed tile map/pad_bounds/rows of RoutedF16Gemm.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm::routed_iu4 {

/// Quantized activation rows (device pointers into one scratch buffer).
struct QuantRows {
  std::uint8_t* x4{nullptr};  ///< [rows][k/2] signed nibbles, low = even k
  float* sx{nullptr};         ///< [rows] row scale
  std::uint8_t* a{nullptr};   ///< [rows][k/32] group multipliers (group > 0)
  __half* xs{nullptr};        ///< [rows][XsStride(k)] dequantized group sums
  std::size_t k{0};
};

/// Groups of 32 per row rounded up to whole 16-group F16 WMMA steps.
[[nodiscard]] std::size_t XsStride(std::size_t k);

/// Scratch bytes for `rows` quantized rows of `k` elements.
[[nodiscard]] std::size_t QuantBytes(std::size_t rows, std::size_t k);

/// Carves QuantRows out of `base` (at least QuantBytes(rows, k) bytes).
[[nodiscard]] QuantRows CarveQuant(void* base, std::size_t rows, std::size_t k);

/// Quantizes F16 rows to int4 (group 0, 32, 64 or 128; `had` 0 or 128
/// rotates first). With `xhat` non-null it also writes the dequantized F16
/// rows (the emulation of the same grid).
bool Quantize(const __half* x, std::size_t rows, std::size_t k, int group,
              int had, const QuantRows& q, __half* xhat, hipStream_t stream);

/// Block shapes of the GEMMs: kTile64 = 64 compact rows x 128 channels
/// (any 64/128-row map), kTile128 = 128 rows x 64 channels (128-row maps
/// only; half the weight staging per WMMA).
enum Shape : int { kTile64 = 0, kTile128 = 1 };

/// Paired gate/up: out[rows_out[c]][r] = silu(gate) * up as F16 for every
/// compact row c of the tiles (Q4_K, packed layout). `q` holds the token
/// rows that rows_in indexes; `tile_rows` is 64 or 128.
bool GateUp(const void* gate, const void* up, const QuantRows& q, int group,
            int shape, const std::int32_t* tiles, std::uint32_t n_tiles,
            std::uint32_t tile_rows, const std::int32_t* pad_bounds,
            const std::int32_t* rows_in, const std::int32_t* rows_out,
            __half* out, std::size_t m, std::size_t k, hipStream_t stream);

/// Down projection (packed Q5_1 container with 4-bit codes, qh == 0):
/// out[rows_out[c]][r] as F16.
bool Down(const void* w, const QuantRows& q, int group, int shape,
          const std::int32_t* tiles, std::uint32_t n_tiles,
          std::uint32_t tile_rows, const std::int32_t* pad_bounds,
          const std::int32_t* rows_in, const std::int32_t* rows_out,
          __half* out, std::size_t m, std::size_t k, hipStream_t stream);

/// Re-quantizes a packed routed Q5_1 matrix (`rows` = experts * rows per
/// expert, multiple of 32; `k` % 64 == 0) in place to 4-bit asymmetric codes
/// per 32-block: new F16 d and m from a small range search plus a
/// least-squares refit, qh = 0. The F16 route then reads exactly the values
/// the integer kernel uses.
bool RequantDownQ4(void* w, std::size_t rows, std::size_t k,
                   hipStream_t stream);

// ---------------------------------------------------------------------------
// Engine hook (called by the spike wrappers in spike.hip.cpp for prefill
// calls of registered routed tensors when GUFO_SPIKE_ROUTED_REAL is set).

struct HookOptions {
  int group{0};        ///< activation group: 0, 32, 64 or 128
  int check_calls{0};  ///< also emulate and log rel RMS for this many calls
  int had{0};          ///< 128: rotate activations (experts stored rotated)
  int shape{kTile64};  ///< kTile128 where the map has 128-row tiles
};

/// GUFO_SPIKE_ROUTED_DOWN4: re-quantizes one routed Q5_1 down matrix in
/// place (RequantDownQ4) and records it as eligible for the integer down.
/// Other types are left unchanged (false). Init-time only.
bool PrepareDown(void* w, WeightType type, std::size_t rows, std::size_t cols,
                 hipStream_t stream);

/// Paired gate/up: quantizes `x` (x_rows token rows) and runs GateUp into
/// `out`. Returns false (nothing launched) when the call is not eligible
/// (type other than Q4_K, tile rows other than 64/128, unsupported shape or
/// group); the caller then takes its F16 route. Records the tile map for
/// the following down call.
bool HookGateUp(std::size_t x_rows, const void* gate, const void* up,
                WeightType type, const __half* x, const std::int32_t* tiles,
                std::uint32_t n_tiles, std::uint32_t tile_rows,
                const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                const std::int32_t* rows_out, __half* out, std::size_t m,
                std::size_t k, const HookOptions& options, hipStream_t stream);

/// Down: quantizes the SwiGLU rows `x` (x_rows slot rows) and runs Down into
/// `out_half` on the 64/128-row map (its own, or the one the preceding
/// gate/up call recorded for the same pad_bounds). Eligible only for
/// matrices prepared by PrepareDown, F16 output and no fused SwiGLU.
bool HookDown(std::size_t x_rows, const void* w, WeightType type,
              const __half* x, const std::int32_t* tiles, std::uint32_t n_tiles,
              std::uint32_t tile_rows, const std::int32_t* pad_bounds,
              const std::int32_t* rows_in, const std::int32_t* rows_out,
              const float* swiglu_gate, float* out, __half* out_half,
              std::size_t m, std::size_t k, const HookOptions& options,
              hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm::routed_iu4

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_IU4_HPP_
