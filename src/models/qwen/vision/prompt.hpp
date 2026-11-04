#ifndef GUFO_MODELS_QWEN_VISION_PROMPT_HPP_
#define GUFO_MODELS_QWEN_VISION_PROMPT_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::uint32_t kImageToken = 248056;
inline constexpr std::uint32_t kPatchSize = 16;
inline constexpr std::uint32_t kMergeSize = 2;

struct ImageGrid {
  std::uint32_t offset{0};  ///< physical first image-pad token
  std::uint32_t height{0};  ///< merged language-token grid, not ViT patches
  std::uint32_t width{0};
  bool operator==(const ImageGrid&) const = default;
};

/// Compact request-owned position state, sufficient to restore RoPE without
/// retaining pixels or encoder scratch in a continuation snapshot.
struct RopeLayout {
  std::vector<ImageGrid> images;
  /// Externally planned prompts (MiniCPM-V 4.7 through llama.cpp mtmd) give
  /// every prefix token its own [T,H,W] RoPE triple; later tokens continue
  /// at physical + explicit_delta. Empty for Qwen image grids.
  std::vector<std::array<std::int32_t, 3>> explicit_positions;
  std::int32_t explicit_delta{0};
  [[nodiscard]] std::array<std::int32_t, 3> Position(
      std::uint32_t physical) const;
  [[nodiscard]] std::int32_t Delta() const;
  [[nodiscard]] std::uint32_t PrefixLength() const;
  void Validate(std::uint32_t max_context) const;
  bool operator==(const RopeLayout&) const = default;
};

struct PreparedImage {
  core::Image pixels;  ///< resized RGB8; temporal repetition happens on GPU
  ImageGrid grid;
};

/// Encoder output computed before the session runs (MiniCPM-V via mtmd):
/// `rows` language-width embeddings replace the tokens at `offset`.
struct ExternalEmbedding {
  std::uint32_t offset{0};
  std::uint32_t rows{0};
  std::shared_ptr<const std::vector<float>> data;
};

struct Prompt {
  std::vector<tokenization::TokenId> tokens;
  RopeLayout rope;
  std::vector<PreparedImage> images;
  std::vector<ExternalEmbedding> external;
  /// SHA-256 covers decoded pixels, grid placement, preprocessing version,
  /// and the model-specific encoder identity. Tokens remain a separate key.
  std::vector<std::uint8_t> cache_identity;
};

[[nodiscard]] core::Image ResizeImage(const core::Image& image);
[[nodiscard]] Prompt Prepare(
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    const tokenization::ChatTemplateOptions& options,
    std::string_view encoder_identity, std::uint32_t max_context);
/// Plans an already rendered prompt; `offsets` holds the byte offset of each
/// image's <|image_pad|> in `rendered`, in message order.
[[nodiscard]] Prompt Prepare(
    const tokenization::QwenTokenizer& tokenizer, std::string_view rendered,
    std::span<const std::size_t> offsets,
    std::span<const tokenization::ChatMessage> messages,
    std::string_view encoder_identity, std::uint32_t max_context);

}  // namespace gufo::models::qwen::vision
#endif
