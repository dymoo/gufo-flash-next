#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/family.hpp"

namespace gufo::models::qwen38_flash_next {

/// Architecture parameters of a Flash-Next engine artifact (`qwen4exp`,
/// `qwen35moe` or `lfm2moe`). Every value is read from the file; the fixed
/// limits below only bound what this runtime was written and validated for.
struct Config {
  static constexpr std::uint32_t kMaxPleHeads = 16;
  static constexpr std::uint32_t kMaxPleNgram = 3;

  /// Set once from general.architecture; Traits() answers what it implies.
  Family family{Family::kQwen4Exp};

  std::uint32_t num_layers{0};  ///< Trunk layers (MTP block excluded).
  /// File index of trunk layer 0 (non-zero only for a diagnostic slice).
  std::uint32_t first_layer{0};
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP block.
  std::uint32_t hidden_size{0};     ///< n_embd, 2560.
  std::uint32_t vocab_size{0};      ///< 248320.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Hyper-connections: hc_count parallel residual streams of hidden_size.
  std::uint32_t hc_count{0};     ///< 4
  std::uint32_t hc_low_rank{0};  ///< 320

  // Full (gated GQA) attention on every full_attention_interval-th layer.
  std::uint32_t full_attention_interval{0};  ///< 4
  std::uint32_t num_heads{0};                ///< 24
  std::uint32_t num_kv_heads{0};             ///< 2
  std::uint32_t head_dim{0};                 ///< 256
  std::uint32_t rotary_dim{0};               ///< 64
  float rope_theta{0.0F};                    ///< 1e7
  std::array<std::uint32_t, 4> rope_sections{};

  // Qwen Sparse Attention indexer (block top-k selection).
  std::uint32_t indexer_heads{0};     ///< 4
  std::uint32_t indexer_head_dim{0};  ///< 128
  std::uint32_t indexer_top_k{0};     ///< 2048 tokens
  /// 4 tokens per scored block; 0 exactly when !HasIndexer().
  std::uint32_t compress_ratio{0};

  // Gated DeltaNet linear attention.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{0};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{0};  ///< 48 (time_step_rank)
  std::uint32_t ssm_inner_size{0};   ///< 6144

  // LFM2 MoE (`lfm2moe`, e.g. LFM2.5-24B-A2B): the linear layers are
  // double-gated short convolutions (in_proj -> B, C, x; y = C * conv(B * x))
  // whose state is the last conv_cache - 1 rows of B * x, attention is plain
  // GQA (no output gate, no indexer), the first dense_layers FFNs are dense
  // SwiGLU and the routed experts use sigmoid scores with a selection bias.
  std::uint64_t attention_layers{0};  ///< lfm2: bit l marks attention
  std::uint32_t conv_cache{0};        ///< lfm2: short conv window (3)
  std::uint32_t dense_layers{0};      ///< lfm2: leading dense FFN layers
  std::uint32_t dense_ff{0};          ///< lfm2: their SwiGLU width

  // Mixture of experts with one always-on shared expert.
  std::uint32_t num_experts{0};       ///< 512
  std::uint32_t num_experts_used{0};  ///< 10
  std::uint32_t expert_ff{0};         ///< 640
  std::uint32_t shared_expert_ff{0};  ///< 640

  // PLE n-gram hash embedding, injected into one linear-attention layer.
  std::int32_t ple_layer{-1};       ///< -1 when the artifact carries no PLE.
  std::uint32_t ple_ngram_size{0};  ///< 3: bigrams and trigrams
  std::uint32_t ple_heads_per_ngram{0};  ///< 8
  std::uint32_t ple_heads{0};            ///< (ngram_size - 1) * heads_per_ngram
  std::uint32_t ple_head_dim{0};     ///< 160 (embedding_length_per_layer_input)
  std::uint32_t ple_conv_kernel{0};  ///< 4
  std::uint32_t ple_eos_token{0};    ///< resets the n-gram window
  std::array<std::uint64_t, kMaxPleNgram> ple_multipliers{};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_offsets{};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_vocab{};
  std::uint64_t ple_rows{0};  ///< max(offset + vocab) over heads

  // Speculative MTP block (`nextn`), present in the draft sidecar only.
  std::uint32_t nextn_layers{0};

  [[nodiscard]] const FamilyTraits& Traits() const noexcept {
    return TraitsOf(family);
  }
  /// Without them (qwen35moe, lfm2moe) there is one plain residual stream:
  /// hc_count 1 and no mixer projections.
  [[nodiscard]] bool HyperConnections() const noexcept {
    return Traits().hyper_connections;
  }
  [[nodiscard]] bool HasIndexer() const noexcept { return Traits().indexer; }

  [[nodiscard]] std::uint32_t HcDim() const noexcept {
    return hc_count * hidden_size;
  }
  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  /// Channels of the fused q|k|v projection and its causal convolution
  /// (short convolution: the B * x rows).
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return Traits().linear_mixer == MixerKind::kShortConv
               ? hidden_size
               : 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t PleEmbeddingDim() const noexcept {
    return ple_heads * ple_head_dim;
  }
  /// Tokens of PLE conv history kept per sequence: (kernel - 1) * dilation.
  [[nodiscard]] std::uint32_t PleConvHistory() const noexcept {
    return (ple_conv_kernel - 1) * ple_ngram_size;
  }

  /// lfm2moe marks its attention layers; the Qwen families place one on
  /// every full_attention_interval-th layer.
  [[nodiscard]] bool IsLinearLayer(std::uint32_t layer) const noexcept {
    if (family == Family::kLfm2Moe)
      return layer < num_layers &&
             ((attention_layers >> (first_layer + layer)) & 1U) == 0;
    return layer < num_layers &&
           ((first_layer + layer + 1) % full_attention_interval) != 0;
  }
  /// Bit l set for trunk attention layers (l < 64).
  [[nodiscard]] std::uint64_t AttentionMask() const noexcept {
    std::uint64_t mask = 0;
    for (std::uint32_t l = 0; l < num_layers && l < 64; ++l)
      mask |= std::uint64_t{!IsLinearLayer(l)} << l;
    return mask;
  }
  [[nodiscard]] std::uint32_t AttentionLayerCount() const noexcept {
    std::uint32_t n = 0;
    for (std::uint32_t l = 0; l < num_layers; ++l)
      n += IsLinearLayer(l) ? 0 : 1;
    return n;
  }
  [[nodiscard]] MixerKind MixerOf(std::uint32_t layer) const noexcept {
    return IsLinearLayer(layer) ? Traits().linear_mixer
                                : Traits().attention_mixer;
  }
  /// Leading dense_layers (lfm2moe only) are dense SwiGLU, the rest routed.
  [[nodiscard]] FfnKind FfnOf(std::uint32_t layer) const noexcept {
    return first_layer + layer < dense_layers ? FfnKind::kDense
                                              : FfnKind::kRoutedMoe;
  }
  [[nodiscard]] bool IsPleLayer(std::uint32_t layer) const noexcept {
    return ple_layer >= 0 && layer == static_cast<std::uint32_t>(ple_layer);
  }

  /// The draft executes with trunk constants and shared vocabulary weights.
  /// PLE/SSM parameters are intentionally excluded: the draft is full
  /// attention, and indexed (its positions advance in compress_ratio blocks).
  [[nodiscard]] bool MtpMatches(const Config& trunk) const noexcept {
    return nextn_layers == 1 && family == trunk.family && HasIndexer() &&
           num_layers == trunk.num_layers && hidden_size == trunk.hidden_size &&
           context_length == trunk.context_length && rms_eps == trunk.rms_eps &&
           hc_count == trunk.hc_count && hc_low_rank == trunk.hc_low_rank &&
           num_heads == trunk.num_heads && num_kv_heads == trunk.num_kv_heads &&
           head_dim == trunk.head_dim && rotary_dim == trunk.rotary_dim &&
           rope_theta == trunk.rope_theta &&
           rope_sections == trunk.rope_sections &&
           indexer_heads == trunk.indexer_heads &&
           indexer_head_dim == trunk.indexer_head_dim &&
           indexer_top_k == trunk.indexer_top_k &&
           compress_ratio == trunk.compress_ratio &&
           num_experts == trunk.num_experts &&
           num_experts_used == trunk.num_experts_used &&
           expert_ff == trunk.expert_ff &&
           // A healed trunk may widen its shared experts; scratch follows
           // the trunk and the draft binds its own width.
           shared_expert_ff <= trunk.shared_expert_ff;
  }

  /// Reads and validates the `<architecture>.*` metadata. `require_trunk`
  /// rejects draft-only sidecars; the MTP sidecar is read with it false. A
  /// non-zero `experts_used` lowers the routed experts per token (lossy).
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, bool require_trunk,
      std::string* error_msg = nullptr, std::uint32_t experts_used = 0);
};

/// `--moe-routing` for decode/verify steps (prefill always routes exactly).
/// Batch-aware policies trade routing fidelity for a smaller per-layer union
/// of experts read by the batch, so a row's output depends on its batch:
///   exact              top-k per row (default)
///   affinity:KEEP:M    rows keep their top KEEP; the other slots prefer
///                      experts other rows keep, within M logits
///   mass:KEEP:T        rows drop slots after the exact top-k mass reaches T
///   budget:KEEP:N      at most N distinct experts per layer-step (the
///                      most demanded), beyond every row's top KEEP
struct MoeRouting {
  enum class Mode : std::uint8_t { kExact, kAffinity, kMass, kBudget };
  Mode mode{Mode::kExact};
  std::uint32_t keep{0};
  float value{0.0F};  ///< affinity margin or mass target
  std::uint32_t budget{0};

  /// Parses the option; KEEP must be 1..`used` (the model's top-k).
  [[nodiscard]] static std::optional<MoeRouting> Parse(std::string_view spec,
                                                       std::uint32_t used,
                                                       std::string* error_msg);
};

/// Residual-stream writer classes a steering adapter projects (bit mask).
namespace steer_writer {
/// Main-layer block outputs: attention o_proj/out_proj and the whole MoE
/// output (routed + shared expert down projections). Every adapter has it.
inline constexpr std::uint8_t kBlocks = 1;
/// Token embedding rows (E' = E - (E r) r^T), trunk and MTP draft input.
inline constexpr std::uint8_t kEmbedding = 2;
/// The PLE value projection output.
inline constexpr std::uint8_t kPle = 4;
/// The MTP draft block's attention and MoE outputs.
inline constexpr std::uint8_t kMtp = 8;
}  // namespace steer_writer

/// A steering adapter (`--adapter NAME=PATH.json`): directional ablation
/// y <- y - alpha * dot(r, y) * r of the residual-stream writers in
/// `writers`, which equals serving weights W' = W - alpha r (r^T W).
/// v1 metadata covers the main-layer block outputs only; v2 names its
/// writers (embedding rows, PLE value, MTP block outputs as well).
struct SteeringAdapter {
  std::string name;
  float alpha{0.0F};
  std::vector<float> direction;  ///< unit vector, hidden_size floats
  /// SHA-256 of alpha, routed layers, direction and (beyond the v1 block
  /// outputs) the writers: part of every cache key.
  std::string digest;
  /// Trunk layers whose MoE output is ablated whole (bit i: layer i); the
  /// others ablate only their gated shared expert. Optional
  /// `routed_expert_layers` in the metadata; all layers by default.
  std::uint64_t routed_layers{~std::uint64_t{0}};
  /// steer_writer bits; always includes kBlocks.
  std::uint8_t writers{steer_writer::kBlocks};

  /// Reads `gufo steering adapter v1` or `v2` metadata and the little-endian
  /// float32 direction file it names (relative to the JSON's directory).
  [[nodiscard]] static std::optional<SteeringAdapter> Load(
      std::string name, const std::string& json_path, std::string* error_msg);
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_
