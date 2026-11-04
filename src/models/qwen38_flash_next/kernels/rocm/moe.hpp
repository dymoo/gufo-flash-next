#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Routed expert matrices with Q4_K/Q5_K gate/up and Q5_1/Q8_0 down, or
/// IQ3_S/IQ3_XXS/IQ4_XS gate/up and IQ4_NL/Q8_0 down (iq.hpp), live in a
/// lane-interleaved layout (the MTP block's all-Q8_0 experts stay GGUF):
/// the same bytes, permuted within each group of 32 consecutive rows so one
/// lane per row reads 16-byte pieces that are contiguous across the wave.
/// A row is a sequence of steps (a Q4_K/Q5_K super-block, or two Q5_1/Q8_0
/// blocks); a step's canonical bytes are
///   Q4_K/Q5_K: the super-block unchanged (9/11 pieces)
///   Q5_1: qs0 | qs1 | d0 m0 qh0 d1 m1 qh1 (3 pieces)
///   Q8_0: qs0 | qs1 (4 pieces), then the dword d0 | d1
/// and a 32-row group stores, per step, piece i of lane l at
/// (i * 32 + l) * 16, followed by the lanes' trailing dwords.
bool RoutedPacked(WeightType gate, WeightType down);

/// Permutes one routed expert matrix (`rows` = experts * rows per expert,
/// a multiple of 32; `cols` per row) from GGUF into the packed layout, in
/// place. Returns false for unsupported shapes.
bool PackRoutedExperts(void* data, WeightType type, std::size_t rows,
                       std::size_t cols, hipStream_t stream);

/// Device scratch for RoutedExperts over at most `rows` token rows.
std::size_t RoutedExpertsScratchBytes(std::uint32_t rows, std::uint32_t used);

/// Routed expert FFNs of decode/verification rows, planned on the device
/// (no host synchronization, graph-capturable): `out[t][s][2560]` =
/// down_e(swiglu(gate_e(x_t), up_e(x_t))) for e = ids[t][s]; negative ids
/// give zero rows. Every expert's weights stream once per call for all of
/// its rows. Activations use Q8_1 blocks and each 32-element block's integer
/// dot product is exact, then scaled and accumulated in one fixed FP32
/// order, so a row's result never depends on the batch around it.
/// Supports 640x2560 Q4_K/Q5_K/Q8_0/IQ4_XS/IQ3_S/IQ3_XXS gate/up and
/// 2560x640 Q5_1/Q8_0/IQ4_NL down matrices with at most 512 experts, packed
/// when RoutedPacked (IQ formats: iq.hpp; otherwise all GGUF Q8_0); returns
/// false for anything else. `hidden`/`ff` 2048/512 (qwen35moe) supports the
/// same K-quant and Q8_0 formats, without the IQ kernels.
/// `phases` (benchmarks) receives two events: planned, gate/up done.
/// `hadamard` 128 (spike, GUFO_SPIKE_ROUTED_HAD: experts stored rotated by
/// spike.hip.cpp's BlockHadamard): the Q8 token rows and the Q8 SwiGLU rows
/// are rotated (dequantize, rotate, requantize) before their dot products;
/// 2560x640 geometry only.
bool RoutedExperts(const void* gate, const void* up, WeightType gate_type,
                   const void* down, WeightType down_type,
                   std::uint32_t n_experts, const float* x,
                   const std::int32_t* ids, float* out, std::uint32_t rows,
                   std::uint32_t used, void* scratch, hipStream_t stream,
                   const hipEvent_t* phases = nullptr,
                   std::uint32_t hidden = 2560, std::uint32_t ff = 640,
                   int hadamard = 0);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_HPP_
