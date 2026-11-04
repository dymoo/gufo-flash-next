#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Optional host destinations for independent predictor qualification.
/// Every populated span contains the final complete row of its stage.
struct MtpTrace {
  std::span<float> normalized_hidden;
  std::span<float> fused;
  std::span<float> attention;
  std::span<float> hidden;
  std::span<float> head;
  std::span<float> ffn_input;
  std::span<float> ffn_output;
  [[nodiscard]] bool Valid(std::size_t H, std::size_t D) const noexcept {
    for (const auto row : {normalized_hidden, fused, attention, hidden})
      if (!row.empty() && row.size() != D)
        return false;
    for (const auto row : {head, ffn_input, ffn_output})
      if (!row.empty() && row.size() != H)
        return false;
    return true;
  }
};

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Two hyper-connection mixers per layer plus the output head mixer share
/// this shape; `inject` is absent on the head mixer.
struct HcMixer {
  TensorRef norm;    ///< [hc_dim] grouped RMSNorm gamma, (1 + w) folded.
  TensorRef down;    ///< [hc_dim -> low_rank]
  TensorRef up;      ///< [low_rank -> hc_dim]
  TensorRef inject;  ///< [hc_dim -> hc_count], empty on the head mixer.
};

/// One layer's tensors, bound by its family (weights.cpp) into the slots its
/// kinds use; the slots of other kinds stay empty.
struct LayerWeights {
  MixerKind mixer{MixerKind::kGatedAttention};
  FfnKind ffn{FfnKind::kRoutedMoe};
  [[nodiscard]] bool linear() const noexcept { return IsLinearMixer(mixer); }
  HcMixer hc_attn;
  HcMixer hc_ffn;

  // Gated DeltaNet (linear layers). A short convolution binds in_proj as
  // ssm_qkv, its [cache, hidden] taps as ssm_conv1d and out_proj as ssm_out.
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, channels]
  TensorRef ssm_alpha;   ///< [hidden -> v_heads]
  TensorRef ssm_beta;    ///< [hidden -> v_heads]
  TensorRef ssm_dt;      ///< [v_heads] softplus bias
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log)
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim -> hidden]

  // Gated GQA + QSA indexer (attention layers).
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim], q|gate per head
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;     ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]
  TensorRef indexer_q;    ///< [hidden -> indexer_heads * indexer_dim]
  TensorRef indexer_k;    ///< [hidden -> indexer_dim]
  TensorRef indexer_q_norm;
  TensorRef indexer_k_norm;

  // PLE n-gram injection (one linear layer).
  TensorRef ple_key;         ///< [ple_dim -> hc_dim]
  TensorRef ple_value;       ///< [ple_dim -> hidden]
  TensorRef ple_norm_key;    ///< [hc_dim]
  TensorRef ple_norm_query;  ///< [hc_dim]
  TensorRef ple_norm_conv;   ///< [hc_dim]
  TensorRef ple_conv1d;      ///< [conv_kernel, hc_dim]

  // Mixture of experts. A dense FFN binds its SwiGLU as shexp_gate/up/down.
  TensorRef router;          ///< [hidden -> num_experts] F32
  TensorRef expert_bias;     ///< lfm2: [num_experts] selection bias F32
  TensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  TensorRef shexp_gate;      ///< [hidden -> shared_ff]
  TensorRef shexp_up;        ///< [hidden -> shared_ff]
  TensorRef shexp_down;      ///< [shared_ff -> hidden]

  // Speculative `nextn` block only.
  TensorRef nextn_enorm;  ///< [hidden]
  TensorRef nextn_hnorm;  ///< [hc_dim], one RMS denominator across all streams
  TensorRef nextn_eh_proj;  ///< GGUF rows: [fc_embedding | fc_hidden]
  HcMixer nextn_head;       ///< Output mixer of the draft block.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;  ///< [hidden -> vocab]
  TensorRef output;      ///< [hidden -> vocab]
  HcMixer hc_head;       ///< Final mixer, doubles as the output norm.
  TensorRef ple_table;   ///< [ple_head_dim, ple_rows], read from disk only.
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  /// A non-zero `layer_count` binds file layers [first_layer, first_layer +
  /// layer_count) as the whole trunk: a diagnostic slice whose output is
  /// not a language model, for device checks within a small memory budget.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr,
      std::uint32_t first_layer = 0, std::uint32_t layer_count = 0,
      std::uint32_t experts_used = 0);
};

/// The MTP draft block from the `shared-*` sidecar: one attention layer with
/// its own hyper-connection mixers and experts; token embedding and LM head
/// are borrowed from the trunk.
struct MtpWeights {
  Config config;
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr, std::uint32_t experts_used = 0);
};

/// OpenBMB's DSpark drafter (`dflash` GGUF): a five-block non-causal
/// transformer over [anchor, MASK...] that attends to target features
/// (the inputs of `target_layers`, fused by `fc`) injected as K/V, plus a
/// low-rank Markov head chaining the block and a confidence head. Token
/// embedding and LM head are the target's.
struct DsparkWeights {
  struct Block {
    TensorRef attn_norm, q, k, v, q_norm, k_norm, out, ffn_norm, gate, up, down;
  };
  std::uint32_t block_size{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t ffn{0};
  std::uint32_t rank{0};  ///< Markov rank
  std::int32_t mask_token{0};
  float rope_theta{0};
  float rms_eps{0};
  std::vector<std::uint32_t> target_layers;
  TensorRef fc, enc_norm, out_norm, markov_w1, markov_w2, conf_w, conf_b;
  std::vector<Block> blocks;

  [[nodiscard]] static std::optional<DsparkWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
