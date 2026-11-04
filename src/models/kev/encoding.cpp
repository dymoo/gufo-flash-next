#include "src/models/kev/encoding.hpp"

#include <algorithm>
#include <cctype>

namespace gufo::models::kev {
namespace {

// kev.model._SPECIAL_RE: <|name|> -> <¦name¦> (U+00A6), leftmost first.
std::string RewriteDelimiters(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    if (text.compare(i, 2, "<|") == 0) {
      std::size_t j = i + 2;
      while (j < text.size() &&
             (std::isalnum(static_cast<unsigned char>(text[j])) != 0 ||
              text[j] == '_'))
        ++j;
      if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
        out += "<\xC2\xA6";
        out.append(text.substr(i + 2, j - i - 2));
        out += "\xC2\xA6>";
        i = j + 2;
        continue;
      }
    }
    out += text[i++];
  }
  return out;
}

}  // namespace

std::unique_ptr<Encoder> Encoder::Create(const core::GgufReader& reader,
                                         std::string* error) {
  auto encoder = std::unique_ptr<Encoder>(new Encoder());
  encoder->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(reader, error);
  if (encoder->tokenizer_ == nullptr)
    return nullptr;
  const auto find = [&](std::string_view name, TokenId* id) {
    const auto found = encoder->tokenizer_->FindSpecialToken(name);
    if (!found.has_value()) {
      if (error != nullptr)
        *error = "Kev tokenizer lacks " + std::string(name);
      return false;
    }
    *id = *found;
    return true;
  };
  if (!find("<|fim_prefix|>", &encoder->state_) ||
      !find("<|fim_middle|>", &encoder->question_) ||
      !find("<|box_start|>", &encoder->option_open_) ||
      !find("<|box_end|>", &encoder->option_close_) ||
      !find("<|fim_suffix|>", &encoder->decide_))
    return nullptr;
  // The added tokens of the Kev checkpoint's tokenizer outside the <|name|>
  // form: the Hugging Face tokenizer still splits these out of caller text.
  for (const char* name :
       {"<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<think>", "</think>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
        "<tts_text_bos_single>"}) {
    TokenId id = 0;
    if (!find(name, &id))
      return nullptr;
    encoder->added_.emplace_back(name, id);
  }
  return encoder;
}

std::vector<TokenId> Encoder::UserTokens(std::string_view text) const {
  const std::string rewritten = RewriteDelimiters(text);
  const std::string_view s = rewritten;
  std::vector<TokenId> out;
  const auto encode = [&](std::string_view segment) {
    const auto ids =
        tokenizer_->Encode(segment, {.parse_special_tokens = false});
    out.insert(out.end(), ids.begin(), ids.end());
  };
  std::size_t start = 0;
  std::size_t i = 0;
  while (i < s.size()) {
    // Leftmost-longest, as the Hugging Face added-token matcher.
    std::size_t length = 0;
    TokenId id = 0;
    if (s[i] == '<') {
      for (const auto& [name, token] : added_) {
        if (name.size() > length && s.compare(i, name.size(), name) == 0) {
          length = name.size();
          id = token;
        }
      }
    }
    if (length == 0) {
      ++i;
      continue;
    }
    encode(s.substr(start, i - start));
    out.push_back(id);
    i += length;
    start = i;
  }
  encode(s.substr(start));
  return out;
}

std::optional<Encoding> Encoder::Encode(const Request& request,
                                        const Limits& limits,
                                        std::string* error) const {
  Encoding encoding;
  const auto state_tokens = UserTokens(request.state);
  const std::size_t kept = std::min(state_tokens.size(), limits.max_state - 1);
  encoding.state_truncated = kept < state_tokens.size();
  encoding.state.reserve(kept + 1);
  encoding.state.push_back(state_);
  encoding.state.insert(
      encoding.state.end(), state_tokens.begin(),
      state_tokens.begin() + static_cast<std::ptrdiff_t>(kept));
  for (const Question& q : request.questions) {
    Row row;
    row.ids.push_back(question_);
    const auto instructions = UserTokens(q.instructions);
    row.ids.insert(row.ids.end(), instructions.begin(), instructions.end());
    for (const std::string& option : q.options) {
      row.ids.push_back(option_open_);
      const auto text = UserTokens(option);
      row.ids.insert(row.ids.end(), text.begin(), text.end());
      row.option_ends.push_back(static_cast<std::uint32_t>(row.ids.size()));
      row.ids.push_back(option_close_);
    }
    row.decide = static_cast<std::uint32_t>(row.ids.size());
    row.ids.push_back(decide_);
    if (row.ids.size() > limits.max_row - encoding.state.size() ||
        row.ids.size() > limits.max_branch) {
      if (error != nullptr)
        *error = "branch too long: " + std::to_string(row.ids.size()) +
                 " tokens with a " + std::to_string(encoding.state.size()) +
                 "-token state (row limit " + std::to_string(limits.max_row) +
                 ")";
      return std::nullopt;
    }
    encoding.rows.push_back(std::move(row));
  }
  return encoding;
}

}  // namespace gufo::models::kev
