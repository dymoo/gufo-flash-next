#ifndef GUFO_SERVER_STRUCTURED_OUTPUT_HPP_
#define GUFO_SERVER_STRUCTURED_OUTPUT_HPP_

#include <cstddef>
#include <memory>
#include <optional>

#include "src/cli/serve/chat_protocol.hpp"
#include "src/core/grammar/grammar.hpp"
#include "src/models/qwen/tokenizer.hpp"

/// Structured outputs for chat requests: `response_format` json_object and
/// json_schema, and `tool_choice` required or named on families with a known
/// tool-call syntax, enforced by grammar-constrained sampling.
namespace gufo::server {

/// The grammar that constrains `request`'s answer, or nullopt when the
/// request asks for none (or asks only for a tool choice the family cannot
/// express, which keeps the unconstrained path and its 502 check). Throws
/// grammar::GrammarError for combinations the protocol cannot serve.
[[nodiscard]] std::optional<grammar::GrammarSource> OutputGrammar(
    const ChatRequest& request, const ChatProtocol& protocol);

/// Reasoning gating for `request` under its template family.
[[nodiscard]] grammar::Gating OutputGating(const ChatRequest& request,
                                           const ChatProtocol& protocol,
                                           const PromptTokenizer& tokenize);

/// The request's constraint on `engine`, or nullptr when OutputGrammar is
/// empty. A null `engine` (a model without grammar support) rejects
/// response_format with GrammarError and leaves tool choices unconstrained.
[[nodiscard]] std::shared_ptr<sampling::TokenConstraint> NewOutputConstraint(
    const ChatRequest& request, const ChatProtocol& protocol,
    const grammar::Engine* engine, const PromptTokenizer& tokenize);

/// A Qwen-tokenizer vocabulary as grammars see it: raw token bytes, with
/// control tokens blanked except the reasoning and tool-call delimiters the
/// templates generate as single tokens.
[[nodiscard]] grammar::Vocabulary QwenGrammarVocabulary(
    const tokenization::QwenTokenizer& tokenizer, std::size_t logits_width);

}  // namespace gufo::server

#endif  // GUFO_SERVER_STRUCTURED_OUTPUT_HPP_
