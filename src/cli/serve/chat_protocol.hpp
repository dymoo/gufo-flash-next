#ifndef GUFO_SERVER_CHAT_PROTOCOL_HPP_
#define GUFO_SERVER_CHAT_PROTOCOL_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/model_admission.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

/// Model admission assigns each architecture its template family
/// (core::LoadPlan::template_family). One chat protocol adapter serves each.
using core::TemplateFamily;

/// Encodes rendered prompt text: special tokens parsed, no BOS or EOS added.
using PromptTokenizer =
    std::function<std::vector<tokenization::TokenId>(std::string_view text)>;

struct RenderedPrompt {
  std::vector<tokenization::TokenId> tokens;
  /// Snapshot point before mutable assistant framing; zero for none.
  std::size_t cache_prefix_tokens{0};
};

/// One generation split into the OpenAI message fields. Tool call ids are
/// left empty for the caller to assign.
struct ParsedOutput {
  std::string content;
  std::string reasoning;
  std::vector<tokenization::ChatMessage::ToolCall> tool_calls;
};

class ChatProtocol;

/// Per-request output parser. Push streams the text that is unambiguously
/// reasoning or content (tool markup is held back); Finish parses the whole
/// output, including its tool calls.
class OutputParser {
public:
  /// Receives visible text; returning false stops the parser (disconnect).
  using Emit = std::function<bool(std::string_view text, bool reasoning)>;

  /// Feeds raw model bytes, which may split UTF-8 sequences and markers.
  bool Push(std::string_view bytes, const Emit& emit);
  /// Emits the remaining visible text and returns the parsed output, or
  /// nullopt once `emit` failed. With `enforce_required`, a required tool
  /// choice without a declared call throws kToolChoiceUnsatisfied.
  std::optional<ParsedOutput> Finish(bool enforce_required, const Emit& emit);

private:
  friend class ChatProtocol;
  enum class State : std::uint8_t {
    kInitial,
    kThinking,
    kContent,
  };

  OutputParser(const ChatProtocol& protocol, const ChatRequest& request);
  bool Feed(std::string_view bytes, bool final, const Emit& emit);
  [[nodiscard]] ParsedOutput Parse(bool enforce_required, bool* hide) const;
  [[nodiscard]] std::size_t EarliestMarker(std::string_view text) const;
  [[nodiscard]] std::size_t HeldMarkerPrefix(std::string_view text) const;

  const ChatProtocol* protocol_;
  std::span<const std::string_view> markers_;
  TextGenerationBackend::InitialOutputState start_;
  std::vector<tokenization::ChatTool> tools_;
  ChatRequest::ToolChoice choice_;
  std::string function_;
  core::Utf8Decoder decoder_;
  std::string raw_;
  std::string pending_;
  std::string hidden_;
  State state_{State::kInitial};
  bool tool_mode_{false};
  bool trim_reasoning_separator_{false};
};

/// A template family's chat protocol: how a chat request renders into a
/// prompt and how the model's output parses back into reasoning, content and
/// tool calls. Adapters are stateless singletons from ProtocolFor().
class ChatProtocol {
public:
  virtual ~ChatProtocol() = default;

  /// Prompt text including the generation prompt. Image parts render as the
  /// family's placeholders only when `image_offsets` collects their byte
  /// offsets; otherwise, and for text-only families, images throw
  /// std::invalid_argument, as does any request the template rejects.
  [[nodiscard]] virtual std::string RenderText(
      const ChatRequest& request,
      std::vector<std::size_t>* image_offsets = nullptr) const = 0;

  /// Tokens of a text-only request and their stable cache prefix.
  [[nodiscard]] RenderedPrompt Render(const ChatRequest& request,
                                      const PromptTokenizer& tokenize) const;

  /// Stable prefix of a rendered prompt (image planners reuse it): the
  /// tokens a follow-up turn keeps, before the open assistant framing.
  [[nodiscard]] virtual std::size_t CachePrefix(
      const ChatRequest& request, std::span<const tokenization::TokenId> prompt,
      const PromptTokenizer& tokenize) const = 0;

  /// Where generated text starts: inside reasoning, in content, or either.
  [[nodiscard]] virtual TextGenerationBackend::InitialOutputState ParserStart(
      const ChatRequest& request) const = 0;

  [[nodiscard]] OutputParser NewParser(const ChatRequest& request) const;

  /// How the family writes one tool call, for grammar-constrained tool
  /// choices: `open` + name + `name_end`, the arguments as an xgrammar
  /// `parameter_style` object, then `close`.
  struct ToolCallSyntax {
    /// Text that starts a call; never valid inside arguments.
    std::string_view marker;
    std::string_view open;
    std::string_view name_end;
    std::string_view close;
    std::string_view parameter_style;
  };
  /// Nullopt: tool choices on this family stay unconstrained.
  [[nodiscard]] virtual std::optional<ToolCallSyntax> ToolSyntax()
      const noexcept {
    return std::nullopt;
  }
  /// Text the template writes between the closing </think> and the answer.
  [[nodiscard]] virtual std::string_view ReasoningSeparator() const noexcept {
    return {};
  }

private:
  friend class OutputParser;
  /// Text that opens tool-call markup in this family's output.
  [[nodiscard]] virtual std::span<const std::string_view> ToolMarkers()
      const noexcept = 0;
  /// Appends complete calls found in `text`, which starts at a marker.
  virtual void ParseToolCalls(
      std::string_view text, std::span<const tokenization::ChatTool> tools,
      std::vector<tokenization::ChatMessage::ToolCall>* calls) const = 0;
};

[[nodiscard]] const ChatProtocol& ProtocolFor(TemplateFamily family);

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_PROTOCOL_HPP_
