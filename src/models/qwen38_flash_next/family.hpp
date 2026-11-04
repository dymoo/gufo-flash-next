#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_FAMILY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_FAMILY_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// The model families this engine hosts (CONTEXT.md: Model family). Config
/// parsing sets one from `general.architecture`; consumers ask its traits
/// and per-layer kinds instead of probing derived fields or empty weights.
enum class Family : std::uint8_t {
  kQwen4Exp,   ///< qwen4exp: Qwen3.8-Flash-Next
  kQwen35Moe,  ///< qwen35moe: MiniCPM-V 4.7, Qwen3.6
  kLfm2Moe,    ///< lfm2moe: LFM2.5
};

/// A layer's token mixer.
enum class MixerKind : std::uint8_t {
  kGatedDeltaNet,   ///< linear attention: causal conv, delta-rule state
  kShortConv,       ///< double-gated short convolution (in_proj -> B, C, x)
  kGatedAttention,  ///< GQA with a sigmoid output gate (q|gate rows)
  kAttention,       ///< plain GQA: no output gate, no indexer
};

/// A layer's feed-forward block.
enum class FfnKind : std::uint8_t {
  kRoutedMoe,  ///< routed experts, plus the family's shared expert if any
  kDense,      ///< SwiGLU
};

/// Output gate of the gated DeltaNet's RMSNorm: sigmoid(z) or silu(z).
enum class GdnGate : std::uint8_t { kSigmoid, kSilu };

/// The image encoder an artifact pairs with.
enum class VisionStack : std::uint8_t {
  kNone,  ///< text only
  kQwen,  ///< qwen::vision::Encoder
  kMtmd,  ///< llama.cpp mtmd projector (MtmdVision)
};

/// Everything fixed per family. Each row is that family's adapter; hot
/// paths read these as plain values, resolved once per model.
struct FamilyTraits {
  Family family;
  std::string_view architecture;  ///< GGUF general.architecture
  MixerKind linear_mixer;         ///< mixer of the non-attention layers
  MixerKind attention_mixer;      ///< mixer of the attention layers
  /// hc_count > 1 parallel residual streams joined by learned mixers;
  /// otherwise one stream with a plain RMS pre-norm.
  bool hyper_connections;
  /// QSA block top-k indexer. Without it attention is always dense and
  /// compress_ratio is 0.
  bool indexer;
  /// Softmax-routed experts beside an always-on, sigmoid-gated shared
  /// expert. Without it experts route on sigmoid scores plus a selection
  /// bias (expert_bias) and the router stays F32.
  bool shared_expert;
  GdnGate gdn_gate;
  VisionStack vision;
};

inline constexpr std::array<FamilyTraits, 3> kFamilies{{
    {.family = Family::kQwen4Exp,
     .architecture = "qwen4exp",
     .linear_mixer = MixerKind::kGatedDeltaNet,
     .attention_mixer = MixerKind::kGatedAttention,
     .hyper_connections = true,
     .indexer = true,
     .shared_expert = true,
     .gdn_gate = GdnGate::kSigmoid,
     .vision = VisionStack::kQwen},
    {.family = Family::kQwen35Moe,
     .architecture = "qwen35moe",
     .linear_mixer = MixerKind::kGatedDeltaNet,
     .attention_mixer = MixerKind::kGatedAttention,
     .hyper_connections = false,
     .indexer = false,
     .shared_expert = true,
     .gdn_gate = GdnGate::kSilu,
     .vision = VisionStack::kMtmd},
    {.family = Family::kLfm2Moe,
     .architecture = "lfm2moe",
     .linear_mixer = MixerKind::kShortConv,
     .attention_mixer = MixerKind::kAttention,
     .hyper_connections = false,
     .indexer = false,
     .shared_expert = false,
     .gdn_gate = GdnGate::kSigmoid,
     .vision = VisionStack::kNone},
}};

[[nodiscard]] constexpr const FamilyTraits& TraitsOf(Family family) noexcept {
  return kFamilies[static_cast<std::size_t>(family)];
}

/// The family hosting `architecture`, if this engine runs it.
[[nodiscard]] constexpr std::optional<Family> FamilyOf(
    std::string_view architecture) noexcept {
  for (const auto& row : kFamilies)
    if (row.architecture == architecture)
      return row.family;
  return std::nullopt;
}

[[nodiscard]] constexpr bool IsLinearMixer(MixerKind mixer) noexcept {
  return mixer == MixerKind::kGatedDeltaNet || mixer == MixerKind::kShortConv;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_FAMILY_HPP_
