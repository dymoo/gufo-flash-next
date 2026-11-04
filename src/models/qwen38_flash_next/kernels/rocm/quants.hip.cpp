// Load-time weight transcoding for the smaller Unsloth quants.
#include <hip/hip_fp16.h>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// One thread per Q8_0 block: the 32 Q6_K values d * sc * (q - 32) are exact
/// in FP32 (11 + 13 significant bits), then quantized like ggml's
/// quantize_row_q8_0 against the stored F16 scale.
__global__ void TranscodeQ6KKernel(const std::uint8_t* __restrict__ src,
                                   std::uint8_t* __restrict__ dst,
                                   std::size_t blocks) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= blocks)
    return;
  const std::uint8_t* sb = src + (i / 8) * 210;  // ql[128] qh[64] sc[16] d
  const int b = static_cast<int>(i % 8);
  const int n = b / 4;        // 128-value half
  const int quarter = b % 4;  // q1..q4 of the ggml dequantizer
  const float d = __half2float(
      __ushort_as_half(static_cast<unsigned short>(sb[208] | sb[209] << 8)));
  float v[32];
  float amax = 0.0f;
  for (int l = 0; l < 32; ++l) {
    const int ql = sb[64 * n + l + (quarter & 1) * 32];
    const int lo = quarter < 2 ? ql & 15 : ql >> 4;
    const int hi = (sb[128 + 32 * n + l] >> (2 * quarter)) & 3;
    const int sc =
        static_cast<std::int8_t>(sb[192 + 8 * n + l / 16 + 2 * quarter]);
    v[l] = d * static_cast<float>(sc * ((lo | hi << 4) - 32));
    amax = fmaxf(amax, fabsf(v[l]));
  }
  const __half h = __float2half_rn(amax / 127.0f);
  const float scale = __half2float(h);
  std::uint8_t* out = dst + i * 34;
  const unsigned short bits = __half_as_ushort(h);
  out[0] = static_cast<std::uint8_t>(bits);
  out[1] = static_cast<std::uint8_t>(bits >> 8);
  for (int l = 0; l < 32; ++l) {
    const float q = scale == 0.0f ? 0.0f : rintf(v[l] / scale);
    out[2 + l] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(fminf(127.0f, fmaxf(-127.0f, q))));
  }
}

/// One thread per 32-value Q4_K sub-block: d * scale and dmin * min become
/// the F16 d and -m of a Q5_1 block (rounded to nearest); the nibbles move
/// unchanged and the fifth bits stay zero.
__global__ void TranscodeQ4KKernel(const std::uint8_t* __restrict__ src,
                                   std::uint8_t* __restrict__ dst,
                                   std::size_t blocks) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= blocks)
    return;
  const std::uint8_t* sb = src + (i / 8) * 144;  // d dmin scales[12] qs[128]
  const int j = static_cast<int>(i % 8);
  const std::uint8_t* sc = sb + 4;
  const int scale =
      j < 4 ? sc[j] & 63 : (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
  const int min =
      j < 4 ? sc[j + 4] & 63 : (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
  const float d = __half2float(
      __ushort_as_half(static_cast<unsigned short>(sb[0] | sb[1] << 8)));
  const float dmin = __half2float(
      __ushort_as_half(static_cast<unsigned short>(sb[2] | sb[3] << 8)));
  const unsigned short hd =
      __half_as_ushort(__float2half_rn(d * static_cast<float>(scale)));
  const unsigned short hm =
      __half_as_ushort(__float2half_rn(-(dmin * static_cast<float>(min))));
  std::uint8_t* out = dst + i * 24;  // d m qh[4] qs[16]
  out[0] = static_cast<std::uint8_t>(hd);
  out[1] = static_cast<std::uint8_t>(hd >> 8);
  out[2] = static_cast<std::uint8_t>(hm);
  out[3] = static_cast<std::uint8_t>(hm >> 8);
  for (int b = 4; b < 8; ++b)
    out[b] = 0;
  const std::uint8_t* qs = sb + 16 + (j / 2) * 32;
  const int shift = 4 * (j & 1);
  for (int e = 0; e < 16; ++e)
    out[8 + e] = static_cast<std::uint8_t>(((qs[e] >> shift) & 15) |
                                           ((qs[e + 16] >> shift) & 15) << 4);
}

/// One thread per Q4_0 block (F16 d; element j in the low and j + 16 in the
/// high nibble of byte j; value d * (q - 8)), re-encoded exactly: Q8_0 codes
/// q - 8 under the same d, or Q5_1 codes q with m = -8 d and no fifth bits.
template<bool kQ5_1>
__global__ void ReencodeQ4_0Kernel(const std::uint8_t* __restrict__ src,
                                   std::uint8_t* __restrict__ dst,
                                   std::size_t blocks) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= blocks)
    return;
  const std::uint8_t* in = src + i * 18;
  if constexpr (kQ5_1) {
    std::uint8_t* out = dst + i * 24;
    const __half d =
        __ushort_as_half(static_cast<unsigned short>(in[0] | in[1] << 8));
    const unsigned short m = __half_as_ushort(__hmul(d, __float2half(-8.0f)));
    out[0] = in[0];
    out[1] = in[1];
    out[2] = static_cast<std::uint8_t>(m);
    out[3] = static_cast<std::uint8_t>(m >> 8);
    for (int b = 4; b < 8; ++b)
      out[b] = 0;
    for (int j = 0; j < 16; ++j)
      out[8 + j] = in[2 + j];
  } else {
    std::uint8_t* out = dst + i * 34;
    out[0] = in[0];
    out[1] = in[1];
    for (int j = 0; j < 16; ++j) {
      out[2 + j] = static_cast<std::uint8_t>((in[2 + j] & 15) - 8);
      out[18 + j] = static_cast<std::uint8_t>((in[2 + j] >> 4) - 8);
    }
  }
}

/// One thread per Q8_0 block of BF16 values, quantized like the Q6_K
/// transcode above against the stored F16 scale.
__global__ void TranscodeBf16Kernel(const std::uint16_t* __restrict__ src,
                                    std::uint8_t* __restrict__ dst,
                                    std::size_t blocks) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= blocks)
    return;
  float v[32];
  float amax = 0.0f;
  for (int l = 0; l < 32; ++l) {
    v[l] = __uint_as_float(static_cast<std::uint32_t>(src[i * 32 + l]) << 16);
    amax = fmaxf(amax, fabsf(v[l]));
  }
  const __half h = __float2half_rn(amax / 127.0f);
  const float scale = __half2float(h);
  std::uint8_t* out = dst + i * 34;
  const unsigned short bits = __half_as_ushort(h);
  out[0] = static_cast<std::uint8_t>(bits);
  out[1] = static_cast<std::uint8_t>(bits >> 8);
  for (int l = 0; l < 32; ++l) {
    const float q = scale == 0.0f ? 0.0f : rintf(v[l] / scale);
    out[2 + l] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(fminf(127.0f, fmaxf(-127.0f, q))));
  }
}

}  // namespace

void TranscodeBf16ToQ8_0(const void* bf16, void* q8, std::size_t rows,
                         std::size_t cols, hipStream_t stream) {
  const std::size_t blocks = rows * (cols / 32);
  TranscodeBf16Kernel<<<(blocks + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint16_t*>(bf16), static_cast<std::uint8_t*>(q8),
      blocks);
}

void ReencodeQ4_0(const void* q4_0, void* out, std::size_t blocks, bool q5_1,
                  hipStream_t stream) {
  const auto* in = static_cast<const std::uint8_t*>(q4_0);
  auto* o = static_cast<std::uint8_t*>(out);
  const unsigned grid = static_cast<unsigned>((blocks + 255) / 256);
  if (q5_1)
    ReencodeQ4_0Kernel<true><<<grid, 256, 0, stream>>>(in, o, blocks);
  else
    ReencodeQ4_0Kernel<false><<<grid, 256, 0, stream>>>(in, o, blocks);
}

void TranscodeQ6KToQ8_0(const void* q6k, void* q8, std::size_t rows,
                        std::size_t cols, hipStream_t stream) {
  const std::size_t blocks = rows * (cols / 32);
  TranscodeQ6KKernel<<<(blocks + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(q6k), static_cast<std::uint8_t*>(q8),
      blocks);
}

void TranscodeQ4KToQ5_1(const void* q4k, void* q5_1, std::size_t rows,
                        std::size_t cols, hipStream_t stream) {
  const std::size_t blocks = rows * (cols / 32);
  TranscodeQ4KKernel<<<(blocks + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(q4k), static_cast<std::uint8_t*>(q5_1),
      blocks);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
