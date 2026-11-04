#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_PREFILL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_PREFILL_HPP_

// Integer (iu4 WMMA) routed-expert prefill GEMMs (prompt batches). Lossy:
// 4-bit activations, behind explicit options.
//
// The iu4 WMMA (16x16x16, twice the F16 rate) takes unsigned or signed int4
// operands. Two weight formats run on it:
//
//  * As shipped (`sym` false): Q4_K gate/up. A Q4_K weight is
//    d * sc_j * q - dmin * m_j, so per 32-group the integer dot products of a
//    tile are folded with the 6-bit scale (one v_mad_i32_i24 per
//    accumulator), once per activation group the integer sum is scaled into
//    F32, and the min term sum_j (dmin m_j) * S_j (S_j the dequantized
//    activation sum of group j) is one F16 WMMA per super-block. The scale
//    folds are most of the VALU work, which does not overlap WMMA.
//  * Symmetric (`sym`, RequantQ4KSym / RequantDownSym): one scale per 64 or
//    128 K and no minimum, stored in the same containers (a valid Q4_K /
//    Q5_1 block that every other reader decodes to the same values: Q4_K
//    dmin = 8 d, mins = scales, q = code + 8). The signed int4 dot products
//    of a whole group accumulate in the WMMA accumulator, and one F32 fold
//    per group applies d * sc * token scale: 3 VALU per accumulator element
//    per group instead of the per-32 folds, no min WMMA, no activation sums.
//
// Activations: symmetric int4 codes in [-7, 7] with one F32 scale per token
// and `group` elements (64, 128 or 256; sym needs 64 or 128 = the weight
// group).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm::routed_prefill {

/// Quantized token rows (device pointers into one scratch buffer).
struct A4Rows {
  std::uint8_t* codes{nullptr};  ///< [rows][k / 2] WMMA nibble order
  float* scale{nullptr};         ///< [rows][k / group]
  __half* sums{nullptr};         ///< [rows][k / 32] dequantized group sums
  int group{256};
  std::size_t k{0};
};

/// Scratch bytes for `rows` quantized rows.
[[nodiscard]] std::size_t A4Bytes(std::size_t rows, std::size_t k, int group);
/// Carves the rows out of `base` (at least A4Bytes bytes, 256-aligned).
[[nodiscard]] A4Rows CarveA4(void* base, std::size_t rows, std::size_t k,
                             int group);
/// Quantizes F16 rows (k % 256 == 0); false for an unsupported group. A null
/// `q.sums` skips the activation sums (symmetric weights need none).
bool QuantizeA4(const __half* x, std::size_t rows, const A4Rows& q,
                hipStream_t stream);

/// out[rows_out[c]][r] = silu(gate_e[r] . x) * (up_e[r] . x) as F16 for the
/// compact rows c of each tile (x row rows_in[c], a QuantizeA4 row). Q4_K,
/// k % 256 == 0, m % 128 == 0, 64-row tile maps. With `h4` (group 128,
/// k = m) the SwiGLU rows are written as int4 rows for DownIu4/DownSym
/// (indexed by rows_out) instead of F16 `out` (then null). `sym`: the weights
/// are RequantQ4KSym containers of the same group as the activations.
bool GateUpIu4(const void* gate, const void* up, WeightType type,
               const A4Rows& q, const std::int32_t* tiles,
               std::uint32_t n_tiles, std::uint32_t tile_rows,
               const std::int32_t* pad_bounds, const std::int32_t* rows_in,
               const std::int32_t* rows_out, __half* out, const A4Rows* h4,
               std::size_t m, bool sym, hipStream_t stream);

/// Re-quantizes a packed routed Q4_K matrix (rows = experts * m) in place to
/// the symmetric container with one scale per `group` (64 or 128) K: code
/// = clamp(rint(v / step), -8, 7) with the MSE-best step in [amax/8, amax/6],
/// d = max step / 63 and an integer 1..63 per group. Lossy for every route.
bool RequantQ4KSym(void* w, WeightType type, std::size_t rows, std::size_t k,
                   int group, hipStream_t stream);

/// Re-quantizes a packed routed Q5_1 down matrix in place to symmetric 4-bit
/// codes with one F16 d per 128 K (all four 32-blocks), m = -8 d, qh = 0.
/// Every route then reads the same lossy values. False for other types.
bool RequantDownSym(void* w, WeightType type, std::size_t rows, std::size_t k,
                    hipStream_t stream);

/// Re-quantizes a packed routed Q5_1 down matrix in place to 4-bit codes
/// (qh = 0, k % 128 == 0): per 32-block d = sc * 2^e with an integer sc and
/// one power of two per 128 elements, and a least-squares F16 m (the
/// asymmetric DownIu4 format). False for other types.
bool RequantDown4(void* w, WeightType type, std::size_t rows, std::size_t k,
                  hipStream_t stream);

/// Down projection on the iu4 WMMA over RequantDown4 weights, `h` the int4
/// SwiGLU rows of GateUpIu4 (group 128; rows_in indexes them).
/// m % 256 == 0, h.k % 128 == 0 and h.k <= 768, 64-row tile maps.
bool DownIu4(const void* w, WeightType type, const A4Rows& h,
             const std::int32_t* tiles, std::uint32_t n_tiles,
             std::uint32_t tile_rows, const std::int32_t* pad_bounds,
             const std::int32_t* rows_in, const std::int32_t* rows_out,
             __half* out, std::size_t m, hipStream_t stream);

/// DownIu4's contract over RequantDownSym weights (group 128 rows; needs no
/// activation sums).
bool DownSym(const void* w, WeightType type, const A4Rows& h,
             const std::int32_t* tiles, std::uint32_t n_tiles,
             std::uint32_t tile_rows, const std::int32_t* pad_bounds,
             const std::int32_t* rows_in, const std::int32_t* rows_out,
             __half* out, std::size_t m, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm::routed_prefill

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_ROUTED_PREFILL_HPP_
