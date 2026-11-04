#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTMD_VISION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTMD_VISION_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/vision/prompt.hpp"

struct llama_model;
struct mtmd_context;

namespace gufo::models::qwen38_flash_next {

/// MiniCPM-V 4.7 image input through llama.cpp's mtmd (a pinned upstream
/// build linked with GUFO_LLAMA_CPP_DIR): image decoding, LLaVA-UHD slicing,
/// the SigLIP2 encoder with its merger and projector, the special-token
/// layout and the canvas M-RoPE positions all come from mtmd. Gufo runs the
/// language model on the returned embeddings and positions.
class MtmdVision {
public:
  /// Media marker the rendered prompt carries, once per image in order.
  static constexpr std::string_view kMarker = "<__media__>";

  /// A prompt ready for the language model: token ids (image rows hold a
  /// placeholder), explicit [z, y, x] RoPE triples through the last image
  /// (the qwen35moe "zyxt" section order), and the image embeddings.
  struct Plan {
    std::vector<std::int32_t> tokens;
    qwen::vision::RopeLayout rope;
    std::vector<qwen::vision::ExternalEmbedding> external;
    std::vector<std::uint8_t> identity;
  };

  /// `text_model` supplies mtmd's vocabulary and RoPE type (loaded
  /// vocabulary-only); `mmproj` is the MiniCPM-V projector GGUF.
  [[nodiscard]] static std::unique_ptr<MtmdVision> Open(
      const std::string& text_model, const std::string& mmproj,
      std::uint32_t hidden_size, std::string* error);
  ~MtmdVision();
  MtmdVision(const MtmdVision&) = delete;
  MtmdVision& operator=(const MtmdVision&) = delete;

  /// Plans `prompt` (rendered chat text with one kMarker per image) with the
  /// encoded image files `images` (PNG/JPEG/... bytes). Throws
  /// std::invalid_argument for undecodable images or marker mismatches.
  [[nodiscard]] Plan Prepare(
      std::string_view prompt,
      std::span<const std::shared_ptr<const std::vector<std::uint8_t>>> images)
      const;

  [[nodiscard]] std::string_view identity() const noexcept { return identity_; }

private:
  MtmdVision() = default;

  llama_model* model_{nullptr};
  mtmd_context* ctx_{nullptr};
  std::uint32_t hidden_size_{0};
  std::string identity_;
  /// The encoder context is not reentrant.
  mutable std::mutex mutex_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTMD_VISION_HPP_
