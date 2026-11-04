#ifndef GUFO_MODELS_MIMO_V26_FLASH_CONFIG_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::mimo_v26_flash {

/// MiMo-V2.6-Flash (`mimo2`) trunk geometry. The kernels are specialised to
/// these values; Parse rejects any artifact that differs.
struct Config {
  static constexpr std::uint32_t kHidden = 4096;
  static constexpr std::uint32_t kLayers = 48;  ///< trunk, without MTP
  static constexpr std::uint32_t kHeads = 64;
  static constexpr std::uint32_t kFullKvHeads = 4;
  static constexpr std::uint32_t kSwaKvHeads = 8;
  static constexpr std::uint32_t kHeadDimQk = 192;
  static constexpr std::uint32_t kHeadDimV = 128;
  static constexpr std::uint32_t kRopeDims = 64;  ///< NEOX, leading dims
  static constexpr std::uint32_t kWindow = 128;   ///< keys incl. self
  static constexpr std::uint32_t kExperts = 256;
  static constexpr std::uint32_t kExpertsUsed = 8;
  static constexpr std::uint32_t kExpertFf = 2048;
  static constexpr std::uint32_t kDenseFf = 16384;
  static constexpr std::uint32_t kQDim = kHeads * kHeadDimQk;       // 12288
  static constexpr std::uint32_t kAttnOutDim = kHeads * kHeadDimV;  // 8192

  std::uint32_t vocab = 0;
  std::uint32_t native_context = 0;
  float rms_eps = 1e-6f;
  float rope_base_full = 1e7f;
  float rope_base_swa = 1e4f;
  float value_scale = 0.707f;
  /// Per trunk layer: true for the 128-token sliding-window layers.
  std::vector<bool> swa;
  /// Per trunk layer: true for the dense (layer 0) feed-forward.
  std::vector<bool> dense_ffn;

  [[nodiscard]] std::uint32_t KvHeads(std::uint32_t layer) const noexcept {
    return swa[layer] ? kSwaKvHeads : kFullKvHeads;
  }
  /// Rows of the fused QKV projection of one layer.
  [[nodiscard]] std::uint32_t QkvRows(std::uint32_t layer) const noexcept {
    return kQDim + KvHeads(layer) * (kHeadDimQk + kHeadDimV);
  }
  [[nodiscard]] std::uint32_t FullLayerCount() const noexcept;

  [[nodiscard]] static std::optional<Config> Parse(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::mimo_v26_flash

#endif  // GUFO_MODELS_MIMO_V26_FLASH_CONFIG_HPP_
