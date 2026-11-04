#ifndef GUFO_MODELS_MIMO_V26_FLASH_WEIGHTS_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo_v26_flash/config.hpp"

namespace gufo::models::mimo_v26_flash {

/// Non-owning view of one GGUF tensor: `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1]) the output dimension and `experts`
/// (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};
  std::uint32_t shard{0};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
};

struct LayerWeights {
  TensorRef attn_norm;   ///< [hidden] F32
  TensorRef attn_qkv;    ///< [hidden -> q | k | v] Q6_K (Q8_0 in MTP)
  TensorRef attn_out;    ///< [heads * v_dim -> hidden]
  TensorRef attn_sinks;  ///< [heads] F32, sliding-window layers only
  TensorRef ffn_norm;    ///< [hidden] F32
  // Dense feed-forward (layer 0).
  TensorRef ffn_gate, ffn_up, ffn_down;
  // Mixture of experts.
  TensorRef router;       ///< [hidden -> experts] F32
  TensorRef router_bias;  ///< [experts] F32, selection only
  TensorRef gate_exps, up_exps, down_exps;
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab] Q8_0
  TensorRef output_norm;  ///< [hidden] F32
  TensorRef output;       ///< [hidden -> vocab] Q6_K
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk. The MTP blocks are not bound.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
  [[nodiscard]] std::size_t TrunkBytes() const noexcept;
};

}  // namespace gufo::models::mimo_v26_flash

#endif  // GUFO_MODELS_MIMO_V26_FLASH_WEIGHTS_HPP_
