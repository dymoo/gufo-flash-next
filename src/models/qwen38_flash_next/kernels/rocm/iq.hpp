// Packed layout and block decoders of the IQ routed-expert formats of the
// smaller Unsloth quants (IQ3_S, IQ3_XXS, IQ4_XS gate/up; IQ4_NL down),
// shared by the decode kernels (moe_iq.inc) and the prefill WMMA GEMM
// (kernels.hip.cpp).
//
// Like the K-quant packing (moe.hpp), a row is a sequence of steps and each
// 32-row group stores, per step, piece i (16 bytes) of lane l at
// (i * 32 + l) * 16, then the lanes' tails at 512 * pieces + l * tail. A
// step's canonical bytes put the codes first so every code read is aligned:
//   IQ3_S   (256 values, 110 B): qs[64] signs[32] | qh[8] scales[4] d
//   IQ3_XXS (256 values,  98 B): qs[64] aux[32]   | d
//   IQ4_XS  (256 values, 136 B): qs[128]          | d scales_h scales_l[4]
//   IQ4_NL  (64 values,   36 B): qs0[16] qs1[16]  | d0 d1
// Every value is a signed int8 codebook code times one scale per 32 values.
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/ggml-common.h"

namespace gufo::models::qwen38_flash_next::rocm::iq {

struct Step {
  int bytes;   ///< per step (and per GGUF step)
  int pieces;  ///< 16-byte code pieces
  int tail;    ///< bytes after the pieces
  int values;  ///< weights per step
};

__host__ __device__ constexpr Step StepOf(WeightType t) {
  return t == WeightType::kIQ3_S     ? Step{110, 6, 14, 256}
         : t == WeightType::kIQ3_XXS ? Step{98, 6, 2, 256}
         : t == WeightType::kIQ4_XS  ? Step{136, 8, 8, 256}
         : t == WeightType::kIQ4_NL  ? Step{36, 2, 4, 64}
                                     : Step{0, 0, 0, 0};
}

/// GGUF byte of canonical step byte `b`.
__host__ __device__ constexpr int RawByte(WeightType t, int b) {
  switch (t) {
    case WeightType::kIQ3_S:  // d qs[64] qh[8] signs[32] scales[4]
      return b < 64    ? 2 + b
             : b < 96  ? 74 + (b - 64)
             : b < 104 ? 66 + (b - 96)
             : b < 108 ? 106 + (b - 104)
                       : b - 108;
    case WeightType::kIQ3_XXS:  // d qs[64] aux[32]
      return b < 96 ? 2 + b : b - 96;
    case WeightType::kIQ4_XS:  // d scales_h scales_l[4] qs[128]
      return b < 128 ? 8 + b : b - 128;
    case WeightType::kIQ4_NL:  // (d qs[16]) x 2
      return b < 16 ? 2 + b : b < 32 ? 20 + (b - 16) : b < 34 ? b - 32 : b - 16;
    default:
      return b;
  }
}

/// IQ4 codebook values of the four nibbles `q & 0x0F0F0F0F` (as bytes).
__device__ __forceinline__ unsigned Iq4Lookup(unsigned q) {
  constexpr unsigned t0 = 0xBFAD9881u;  // -127 -104 -83 -65
  constexpr unsigned t1 = 0xF6EADDCFu;  //  -49  -35 -22 -10
  constexpr unsigned t2 = 0x26190D01u;  //    1   13  25  38
  constexpr unsigned t3 = 0x71594535u;  //   53   69  89 113
  const unsigned lo = __builtin_amdgcn_perm(t1, t0, q & 0x07070707u);
  const unsigned hi = __builtin_amdgcn_perm(t3, t2, q & 0x07070707u);
  return __builtin_amdgcn_perm(hi, lo, 0x03020100u | ((q & 0x08080808u) >> 1));
}

/// 32 codes of an IQ4 block from its 16 nibble bytes (low nibbles are
/// values 0-15, high nibbles 16-31).
__device__ __forceinline__ void Iq4Codes(const unsigned (&qs)[4], unsigned* q) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = Iq4Lookup(qs[i] & 0x0F0F0F0Fu);
    q[i + 4] = Iq4Lookup((qs[i] >> 4) & 0x0F0F0F0Fu);
  }
}

/// Four grid magnitudes with the sign bits `s` (bits 0-3) applied.
__device__ __forceinline__ unsigned Signed4(unsigned g, unsigned s) {
  const unsigned m = ((s * 0x00204081u) & 0x01010101u) * 0xFFu;
  return (g ^ m) + (m & 0x01010101u);
}

__device__ __forceinline__ unsigned ByteOf(unsigned v, int i) {
  return (v >> (8 * i)) & 0xFFu;
}

/// IQ3_S block: grid indices qs (8 bytes), its qh byte, 32 sign bits.
__device__ __forceinline__ void Iq3sCodes(uint2 qs, unsigned qh, unsigned signs,
                                          const unsigned* grid, unsigned* q) {
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const unsigned w = l < 2 ? qs.x : qs.y;
    const unsigned i1 = ByteOf(w, 2 * (l % 2)) | ((qh << (8 - 2 * l)) & 256u);
    const unsigned i2 =
        ByteOf(w, 2 * (l % 2) + 1) | ((qh << (7 - 2 * l)) & 256u);
    const unsigned s = ByteOf(signs, l);
    q[2 * l] = Signed4(grid[i1], s & 15u);
    q[2 * l + 1] = Signed4(grid[i2], s >> 4);
  }
}

/// IQ3_XXS block: grid indices qs (8 bytes), its scale/sign word.
__device__ __forceinline__ void Iq3xxsCodes(uint2 qs, unsigned aux,
                                            const unsigned* grid, unsigned* q) {
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const unsigned w = l < 2 ? qs.x : qs.y;
    unsigned s = (aux >> (7 * l)) & 127u;
    s |= (__builtin_popcount(s) & 1u) << 7;  // the eighth sign is the parity
    q[2 * l] = Signed4(grid[ByteOf(w, 2 * (l % 2))], s & 15u);
    q[2 * l + 1] = Signed4(grid[ByteOf(w, 2 * (l % 2) + 1)], s >> 4);
  }
}

/// Block scales (exact in FP32).
__device__ __forceinline__ float Iq3sScale(float d, unsigned scales, int j) {
  return __fmul_rn(d, float(1 + 2 * int((scales >> (4 * j)) & 15u)));
}
__device__ __forceinline__ float Iq3xxsScale(float d, unsigned aux) {
  return __fmul_rn(__fmul_rn(d, float(2 * int(aux >> 28) + 1)), 0.25f);
}
/// `h` = d | scales_h << 16, `l` = scales_l[4].
__device__ __forceinline__ float Iq4xsScale(unsigned h, unsigned l, int j) {
  const unsigned ls =
      ((l >> (4 * j)) & 15u) | (((h >> (16 + 2 * j)) & 3u) << 4);
  return __fmul_rn(
      __half2float(__ushort_as_half(static_cast<unsigned short>(h))),
      float(int(ls) - 32));
}

}  // namespace gufo::models::qwen38_flash_next::rocm::iq

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_IQ_HPP_
