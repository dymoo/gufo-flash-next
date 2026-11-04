#ifndef GUFO_MODELS_KEV_ENCODING_HPP_
#define GUFO_MODELS_KEV_ENCODING_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kev/systemone.hpp"
#include "src/models/qwen/tokenizer.hpp"

/// Kev's token layout (kev/model.py encode):
///   <|fim_prefix|> state
///   per question: <|fim_middle|> instructions
///                 per option: <|box_start|> text <|box_end|>
///                 <|fim_suffix|>
/// Every question runs as a row that continues the state; its positions
/// continue from the state length.
namespace gufo::models::kev {

using tokenization::TokenId;

struct Limits {
  std::size_t max_state{32768};  ///< State tokens, <|fim_prefix|> included.
  std::size_t max_row{32768 + 8192};  ///< State plus one question's branch.
  std::size_t max_branch{8192};       ///< One question's branch alone.
};

struct Row {
  std::vector<TokenId> ids;
  std::uint32_t decide{0};                 ///< Offset of <|fim_suffix|>.
  std::vector<std::uint32_t> option_ends;  ///< Offsets of each <|box_end|>.
};

struct Encoding {
  std::vector<TokenId> state;
  std::vector<Row> rows;
  bool state_truncated{false};

  [[nodiscard]] std::size_t Tokens() const noexcept {
    std::size_t n = state.size();
    for (const Row& row : rows)
      n += row.ids.size();
    return n;
  }
};

class Encoder {
public:
  [[nodiscard]] static std::unique_ptr<Encoder> Create(
      const core::GgufReader& reader, std::string* error);

  /// kev.model.user_tokens: caller text can never produce a delimiter.
  [[nodiscard]] std::vector<TokenId> UserTokens(std::string_view text) const;

  /// Truncates an overlong state like the reference; a branch that does not
  /// fit its row is an error (the 422 detail).
  [[nodiscard]] std::optional<Encoding> Encode(const Request& request,
                                               const Limits& limits,
                                               std::string* error) const;

  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }

private:
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  TokenId state_{}, question_{}, option_open_{}, option_close_{}, decide_{};
  /// Added tokens the rewrite leaves matchable, with their ids.
  std::vector<std::pair<std::string, TokenId>> added_;
};

}  // namespace gufo::models::kev

#endif  // GUFO_MODELS_KEV_ENCODING_HPP_
