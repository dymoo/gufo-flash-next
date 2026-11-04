#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_I4_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_I4_HPP_

// W4A4 dense prefill route on gfx1151's int4 matrix cores (iu4 WMMA, twice
// the F16 rate). Lossy: weights are a load-time 4-bit copy of the Q8_0
// projections, activations are quantized per call; both are rotated by the
// same orthogonal transform R first (y = W x = (W R^T)(R x)), which spreads
// outlier channels so one scale per weight row and one per token suffice.
// Integer sums over all of K are exact; one row x token scale per output.
//
// Layouts. An int4 matrix [rows][k] is fragment-major: 16-row tile t, 32-wide
// K chunk c is a 256-byte block at ((t * k/32) + c) * 256; within it row r
// (0..15) owns 16 bytes, byte b holding elements 2b (low nibble) and 2b + 1
// (high nibble), signed codes in [-8, 7]. A lane's 16 bytes are the two K16
// WMMA fragments of its row. Weight scales are tile-permuted: the scale of
// row 16t + 2l + h sits at (2t + h) * 8 + l, so a lane reads the eight rows
// of its accumulator column with two float4 loads. Token scales are plain.
// Weight rows are padded to kRowPad, tokens to kTokenPad (zero codes, zero
// scales), so the GEMM needs no bounds checks in its main loop.
//
// Rotation R (kBlock128): per 128-element block, signs D then the
// Walsh-Hadamard butterflies, scaled by 1/sqrt(128) (the spike's
// GUFO_SPIKE_HAD=128). kFull: the block transform with b = FullBlock(k), then
// H_{k/b} (Sylvester or Paley I) across the blocks, scaled by 1/sqrt(k/b)
// (GUFO_SPIKE_HAD=1). Both match the spike emulation up to float rounding.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm::dense_i4 {

constexpr std::uint32_t kRowPad = 256;
constexpr std::uint32_t kTokenPad = 128;

enum class Rotation { kBlock128, kFull };

/// Inner block of the kFull rotation for width k (largest of 128/64/32 whose
/// block count has a Hadamard order of at most 48), 0 if unsupported.
/// 2560 = 20 x 128, 6144 = 48 x 128, 640 = 20 x 32.
[[nodiscard]] std::uint32_t FullBlock(std::uint32_t k);

/// True when width k can be rotated and quantized (k % 32 == 0, Width(k) <=
/// 6144, and kFull has a block for Width(k)).
[[nodiscard]] bool Supported(std::uint32_t k, Rotation rotation);

[[nodiscard]] constexpr std::uint32_t Padded(std::uint32_t rows,
                                             std::uint32_t pad) {
  return (rows + pad - 1) / pad * pad;
}

/// Packed width of a k-wide operand: K slabs are 128 wide, so a narrower
/// tail (the HC up projection's 320) is zero-padded in both operands.
[[nodiscard]] constexpr std::uint32_t Width(std::uint32_t k) {
  return Padded(k, 128);
}

/// One int4 operand: codes then scales in one allocation.
struct Matrix {
  const void* codes{nullptr};
  const float* scale{nullptr};
  std::uint32_t rows{0};  ///< logical rows (weights m, activations tokens)
  std::uint32_t k{0};     ///< packed width, Width(source k)
  /// Activations: one scale per token and 128-wide K slab as an integer
  /// multiple (1..63) of `scale`, [token tile][slab][16 tokens]; null when
  /// `scale` alone scales each token. Weights: the same for each row and
  /// slab, [row tile][slab][half][8] (row 16t + 2l + half at (2t' + half) * 8
  /// + l within slab t'), relative to the row scale in `scale`.
  const std::uint8_t* group{nullptr};
};

/// Bytes of a weight matrix (codes + scales [+ slab scales], rows padded to
/// kRowPad) and of an activation matrix (rows padded to kTokenPad), for
/// source width k; views take the source width too.
[[nodiscard]] std::size_t WeightBytes(std::uint32_t m, std::uint32_t k,
                                      bool groups = false);
[[nodiscard]] std::size_t ActivationBytes(std::uint32_t rows, std::uint32_t k);
/// Views of such allocations.
[[nodiscard]] Matrix WeightView(const void* buffer, std::uint32_t m,
                                std::uint32_t k, bool groups = false);
[[nodiscard]] Matrix ActivationView(const void* buffer, std::uint32_t rows,
                                    std::uint32_t k, bool groups = false);

/// Load-time packing of a Q8_0 [m][k] matrix into `buffer` (WeightBytes):
/// dequantize, rotate, then round to nearest with one scale per row chosen
/// by a squared-error clip search over {1, 0.95, ..., 0.65} x absmax / 7
/// (the spike's RequantRow with GUFO_SPIKE_WGROUP=1); with `groups` one
/// scale per row and 128-wide slab instead (clips down to 0.6, scales
/// rounded to integer multiples 1..63 of the row's largest / 63).
/// `streams` > 1 interleaves the rows of that many stacked row blocks
/// (packed row r is source row (r % streams) * (m / streams) + r / streams),
/// the order the HC mixer epilogue reads.
bool PackQ8_0(const void* q8, std::uint32_t m, std::uint32_t k,
              Rotation rotation, void* buffer, hipStream_t stream,
              std::uint32_t streams = 1, bool groups = false);

/// Repacks an offline-quantized (already rotated) slab-grouped tensor into
/// rows [row0, row0 + rows) of a `groups` weight buffer of m_total rows
/// (cleared by the caller): device arrays codes [rows][k] int8 in [-8, 7],
/// sc [rows][k / 128] uint8 in [1, 63] and d [rows] F32 (w = d * sc * q),
/// the dense-w4-calib S4G128 layout. k % 128 == 0.
bool PackS4(const std::int8_t* codes, const std::uint8_t* sc, const float* d,
            std::uint32_t rows, std::uint32_t k, std::uint32_t row0,
            std::uint32_t m_total, void* buffer, hipStream_t stream);

/// Per-call activation quantization of F16 or F32 rows [rows][k] into
/// `buffer` (ActivationBytes): rotate, then one scale per token (absmax / 7),
/// or with `groups` one per token and 128-wide slab: the slab's absmax / 7
/// rounded up to an integer multiple (1..63) of the token's largest / 63.
/// Round to nearest even, codes clamped to [-8, 7].
bool Quantize(const __half* x, std::uint32_t rows, std::uint32_t k,
              Rotation rotation, void* buffer, hipStream_t stream,
              bool groups = false);
bool Quantize(const float* x, std::uint32_t rows, std::uint32_t k,
              Rotation rotation, void* buffer, hipStream_t stream,
              bool groups = false);

// GEMMs (kernels.hip.cpp, DenseF16GEMMKernel's kI4 mode; the epilogues are
// those of the F16 route). w.k == x.k; x.rows tokens.
/// out [x.rows][w.rows] F32. `variant` (benchmarks; 0 = the default for the
/// shape): 1/2 256x128 tiles with 128/256-wide K slabs, 3/4 128x128 tiles.
bool Gemm(const Matrix& w, const Matrix& x, float* out, hipStream_t stream,
          int variant = 0);
/// DenseF16SsmGemm with int4 operands (Flash-Next 16384 x 2560 only);
/// variant 2 uses 256-wide K slabs instead of 128.
bool SsmGemm(const Matrix& w, const Matrix& x, const float* conv_w,
             const float* history, float* qkvz, float* convolved,
             std::uint32_t channels, std::uint32_t kernel, hipStream_t stream,
             int variant = 0);
/// AttentionF16Gemm with int4 operands (13312 x 2560 only).
bool AttentionGemm(const Matrix& w, const Matrix& x, const float* q_gamma,
                   const float* k_gamma, float* query, float* gate,
                   const KvView& kv, const std::uint32_t* position, float theta,
                   float eps, hipStream_t stream,
                   const qwen::vision::DeviceRope* rope = nullptr);

}  // namespace gufo::models::qwen38_flash_next::rocm::dense_i4

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DENSE_I4_HPP_
