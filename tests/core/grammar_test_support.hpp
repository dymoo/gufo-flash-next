#ifndef GUFO_TESTS_CORE_GRAMMAR_TEST_SUPPORT_HPP_
#define GUFO_TESTS_CORE_GRAMMAR_TEST_SUPPORT_HPP_

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/grammar/grammar.hpp"
#include "src/core/sampling.hpp"

/// A small byte-level vocabulary with Qwen-style control tokens, and a
/// seeded random "model" that samples through a grammar constraint.
namespace gufo::grammar::test {

using sampling::TokenId;

constexpr TokenId kStop = 256;
constexpr TokenId kThinkStart = 257;
constexpr TokenId kThinkEnd = 258;
constexpr TokenId kDoubleNewline = 259;
constexpr TokenId kToolCall = 260;
constexpr TokenId kToolCallEnd = 261;
constexpr TokenId kParameterEnd = 271;
constexpr std::size_t kLogitsWidth = 280;

inline const std::vector<std::pair<TokenId, std::string>>& Words() {
  static const std::vector<std::pair<TokenId, std::string>> words{
      {kThinkStart, "<think>"},
      {kThinkEnd, "</think>"},
      {kDoubleNewline, "\n\n"},
      {kToolCall, "<tool_call>"},
      {kToolCallEnd, "</tool_call>"},
      {262, "{\""},
      {263, "\": "},
      {264, "true"},
      {265, "false"},
      {266, "null"},
      {268, "function="},
      {269, "parameter="},
      {270, "</function>"},
      {kParameterEnd, "</parameter>"},
  };
  return words;
}

inline grammar::Vocabulary MakeVocabulary() {
  grammar::Vocabulary vocabulary;
  vocabulary.tokens.resize(272);
  for (std::size_t byte = 0; byte < 256; ++byte)
    vocabulary.tokens[byte] = std::string(1, static_cast<char>(byte));
  for (const auto& [id, text] : Words())
    vocabulary.tokens[id] = text;
  // 256 (<|im_end|>) stops; 267 (<|endoftext|>) is a blank control token.
  vocabulary.stop_tokens = {static_cast<std::int32_t>(kStop)};
  vocabulary.logits_width = kLogitsWidth;
  return vocabulary;
}

inline std::string Text(TokenId token) {
  if (token < 256)
    return std::string(1, static_cast<char>(token));
  for (const auto& [id, text] : Words())
    if (id == token)
      return text;
  return {};
}

/// Longest-match tokenization over the control words, else bytes.
inline std::vector<TokenId> Tokenize(std::string_view text) {
  std::vector<TokenId> tokens;
  while (!text.empty()) {
    std::size_t best = 0;
    TokenId token = static_cast<unsigned char>(text.front());
    for (const auto& [id, word] : Words()) {
      if (word.size() > best && text.starts_with(word) && id >= kThinkStart &&
          id <= kToolCallEnd) {
        best = word.size();
        token = id;
      }
    }
    tokens.push_back(token);
    text.remove_prefix(best == 0 ? 1 : best);
  }
  return tokens;
}

inline std::vector<float> FlatRow() {
  return std::vector<float>(kLogitsWidth, 0.0F);
}

inline std::vector<TokenId> Allowed(const std::vector<float>& row) {
  std::vector<TokenId> allowed;
  for (std::size_t token = 0; token < row.size(); ++token)
    if (row[token] > -std::numeric_limits<float>::infinity())
      allowed.push_back(static_cast<TokenId>(token));
  return allowed;
}

/// Platform-independent pseudo-random logits.
inline float NextLogit(std::uint64_t* state) {
  std::uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<float>(z >> 40) * 0x1.0p-24F * 2.0F;
}

/// Random logits biased towards closing structure so outputs stay short.
inline std::vector<float> RandomRow(std::uint64_t* state) {
  std::vector<float> row(kLogitsWidth);
  for (auto& logit : row)
    logit = NextLogit(state);
  row[kStop] += 1.5F;
  row['"'] += 0.8F;
  row['}'] += 0.5F;
  row[']'] += 0.5F;
  row[kParameterEnd] += 1.0F;
  row[kToolCallEnd] += 1.0F;
  row['e'] -= 1.0F;  // Rare exponents: huge ones overflow strict parsers.
  row['E'] -= 1.0F;
  row['\\'] -= 8.0F;
  // xgrammar admits UTF-8-encoded surrogates (0xED 0xA0-0xBF ..), which
  // strict JSON parsers reject; byte-level models practically never emit
  // them, so the random model avoids their lead byte.
  row[0xED] = -std::numeric_limits<float>::infinity();
  return row;
}

/// Samples until the stop token; the constraint sees every committed token.
inline std::string Generate(const Engine& engine, const GrammarSource& source,
                            const Gating& gating, std::uint64_t seed,
                            std::vector<TokenId> prefix = {}) {
  auto constraint = engine.NewConstraint(source, gating);
  // Greedy over fresh random rows: random trajectories without exp(), so
  // every platform draws the same tokens.
  sampling::SamplingConfig config{.temperature = 0.0F,
                                  .seed = static_cast<std::int64_t>(seed)};
  config.constraint = constraint;
  sampling::SamplerState sampler(config);
  std::uint64_t state = seed * 7919;
  std::string text;
  for (const TokenId token : prefix) {
    constraint->Accept(token);
    text += Text(token);
  }
  for (int step = 0; step < 20000; ++step) {
    const auto row = RandomRow(&state);
    const TokenId token = sampler.Sample(row);
    if (token == kStop)
      return text;
    sampler.Accept(token);
    constraint->Accept(token);
    text += Text(token);
  }
  throw std::runtime_error("constrained generation did not finish: " + text);
}

}  // namespace gufo::grammar::test

#endif  // GUFO_TESTS_CORE_GRAMMAR_TEST_SUPPORT_HPP_
