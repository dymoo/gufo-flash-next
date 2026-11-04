#ifndef GUFO_MODELS_MIMO_V26_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::mimo_v26_flash {

/// Compiled form of the MiMo-V2.6-Flash chat template (SHA-256 pinned).
/// Assistant turns render `<think>reasoning</think>content` and tool calls as
/// `<tool_call><function=name><parameter=k>v</parameter></function>
/// </tool_call>`; turns are joined without newlines.
struct ChatTemplate {
  [[nodiscard]] static constexpr std::string_view Sha256() noexcept {
    return "11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059";
  }
  [[nodiscard]] static bool Validate(const core::GgufReader& reader,
                                     std::string* error);

  [[nodiscard]] static std::string Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, bool enable_thinking,
      bool add_generation_prompt = true);

  /// The open assistant turn appended after the stable conversation.
  [[nodiscard]] static std::string_view GenerationPrompt(bool enable_thinking);
};

}  // namespace gufo::models::mimo_v26_flash

#endif  // GUFO_MODELS_MIMO_V26_FLASH_CHAT_TEMPLATE_HPP_
