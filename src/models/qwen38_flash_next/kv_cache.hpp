#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// Attention KV cache storage (`--kv-cache`). f16 is exact; tq4/tq3 are
/// lossy TurboQuant formats (see rocm/kernels.hpp) about 3.9x/5.2x smaller;
/// tq8 (LFM2 only) is the rotated int8 form, about 1.9x smaller.
enum class KvCacheFormat : std::uint8_t { kF16, kTq4, kTq3, kTq8 };

constexpr std::string_view KvCacheFormatName(KvCacheFormat format) {
  return format == KvCacheFormat::kTq4   ? "tq4"
         : format == KvCacheFormat::kTq3 ? "tq3"
         : format == KvCacheFormat::kTq8 ? "tq8"
                                         : "f16";
}

constexpr std::optional<KvCacheFormat> ParseKvCacheFormat(
    std::string_view name) {
  for (auto format : {KvCacheFormat::kF16, KvCacheFormat::kTq4,
                      KvCacheFormat::kTq3, KvCacheFormat::kTq8}) {
    if (name == KvCacheFormatName(format))
      return format;
  }
  return std::nullopt;
}

/// Storage of the GatedDeltaNet recurrent state (`--gdn-state`): f32 is the
/// exact default; f16 (saturating) and bf16 halve the state every decode step
/// reads and writes. Arithmetic stays F32 in registers; the state is rounded
/// once at the end of each forward call. Lossy and opt-in.
enum class GdnStateFormat : std::uint8_t { kF32, kF16, kBF16 };

constexpr std::string_view GdnStateFormatName(GdnStateFormat format) {
  return format == GdnStateFormat::kF16    ? "f16"
         : format == GdnStateFormat::kBF16 ? "bf16"
                                           : "f32";
}

constexpr std::optional<GdnStateFormat> ParseGdnStateFormat(
    std::string_view name) {
  for (auto format :
       {GdnStateFormat::kF32, GdnStateFormat::kF16, GdnStateFormat::kBF16}) {
    if (name == GdnStateFormatName(format))
      return format;
  }
  return std::nullopt;
}

constexpr std::size_t GdnStateElementBytes(GdnStateFormat format) {
  return format == GdnStateFormat::kF32 ? 4 : 2;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
