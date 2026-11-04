#include "src/cli/serve/structured_output.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::server {
namespace {

using ToolChoice = ChatRequest::ToolChoice;

constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";

/// Control tokens the templates generate as single tokens and grammars
/// spell out; every other control token is blanked (never generated).
constexpr std::array<std::string_view, 4> kTextualControlTokens{
    "<think>", "</think>", "<tool_call>", "</tool_call>"};

json::Value ParseSchema(const std::string& text, std::string_view what) {
  json::Value schema;
  try {
    schema = json::parse(text);
  } catch (const std::exception& error) {
    throw grammar::GrammarError("invalid_response_format",
                                std::string(what) + ": " + error.what());
  }
  if (!schema.is_object())
    throw grammar::GrammarError("invalid_response_format",
                                std::string(what) + " must be a JSON object");
  return schema;
}

json::Value ToolTag(const ChatProtocol::ToolCallSyntax& syntax,
                    const tokenization::ChatTool& tool) {
  json::Value content = json::Value::object();
  content["type"] = "json_schema";
  content["json_schema"] =
      ParseSchema(tool.parameters_json.empty() ? "{}" : tool.parameters_json,
                  "function parameters");
  content["style"] = std::string(syntax.parameter_style);
  // The output parser reads a call marker or reasoning tag inside an
  // argument as structure; never let the grammar write one there.
  json::Value excludes = json::Value::array();
  excludes.push_back(std::string(syntax.marker));
  excludes.push_back(std::string(kThinkStart));
  excludes.push_back(std::string(kThinkEnd));
  content["excludes"] = std::move(excludes);
  json::Value tag = json::Value::object();
  tag["type"] = "tag";
  tag["begin"] =
      std::string(syntax.open) + tool.name + std::string(syntax.name_end);
  tag["content"] = std::move(content);
  tag["end"] = std::string(syntax.close);
  return tag;
}

/// Tool calls only: at least one call of a declared function (or exactly
/// the named one), as the family writes them.
json::Value ToolCalls(const ChatRequest& request,
                      const ChatProtocol::ToolCallSyntax& syntax) {
  if (!request.tool_choice_function.empty()) {
    const auto tool =
        std::ranges::find(request.tools, request.tool_choice_function,
                          &tokenization::ChatTool::name);
    if (tool == request.tools.end())
      throw grammar::GrammarError(
          "invalid_tool_choice",
          "'tool_choice' names a function that is not in 'tools'");
    return ToolTag(syntax, *tool);
  }
  json::Value tags = json::Value::array();
  for (const auto& tool : request.tools)
    tags.push_back(ToolTag(syntax, tool));
  json::Value triggers = json::Value::array();
  triggers.push_back(std::string(syntax.open));
  json::Value excludes = json::Value::array();
  excludes.push_back(std::string(kThinkStart));
  excludes.push_back(std::string(kThinkEnd));
  json::Value calls = json::Value::object();
  calls["type"] = "triggered_tags";
  calls["triggers"] = std::move(triggers);
  calls["tags"] = std::move(tags);
  calls["excludes"] = std::move(excludes);
  calls["at_least_one"] = true;
  calls["stop_after_first"] = !request.parallel_tool_calls;
  return calls;
}

grammar::GrammarSource StructuralTag(json::Value format) {
  json::Value tag = json::Value::object();
  tag["type"] = "structural_tag";
  tag["format"] = std::move(format);
  return {.kind = grammar::GrammarSource::Kind::kStructuralTag,
          .text = tag.dump()};
}

std::string ResponseSchema(const ChatRequest& request) {
  if (request.response_format.type == ResponseFormat::Type::kJsonObject)
    return R"({"type":"object"})";
  return request.response_format.schema;
}

}  // namespace

std::optional<grammar::GrammarSource> OutputGrammar(
    const ChatRequest& request, const ChatProtocol& protocol) {
  const auto syntax = protocol.ToolSyntax();
  const bool tools_allowed =
      request.tool_choice != ToolChoice::kNone && !request.tools.empty();
  if (request.tool_choice == ToolChoice::kRequired && tools_allowed) {
    if (!syntax)
      return std::nullopt;
    return StructuralTag(ToolCalls(request, *syntax));
  }
  if (request.response_format.type == ResponseFormat::Type::kText)
    return std::nullopt;
  const auto schema = ResponseSchema(request);
  if (!tools_allowed) {
    (void)ParseSchema(schema, "'response_format.json_schema.schema'");
    return grammar::GrammarSource{
        .kind = grammar::GrammarSource::Kind::kJsonSchema, .text = schema};
  }
  // With tools on offer the answer is either the requested JSON or calls.
  if (!syntax)
    throw grammar::GrammarError(
        "unsupported_response_format",
        "'response_format' cannot be combined with tools for this model");
  json::Value answer = json::Value::object();
  answer["type"] = "json_schema";
  answer["json_schema"] =
      ParseSchema(schema, "'response_format.json_schema.schema'");
  json::Value elements = json::Value::array();
  elements.push_back(std::move(answer));
  elements.push_back(ToolCalls(request, *syntax));
  json::Value either = json::Value::object();
  either["type"] = "or";
  either["elements"] = std::move(elements);
  return StructuralTag(std::move(either));
}

grammar::Gating OutputGating(const ChatRequest& request,
                             const ChatProtocol& protocol,
                             const PromptTokenizer& tokenize) {
  using Start = grammar::Gating::Start;
  using OutputStart = TextGenerationBackend::InitialOutputState;
  grammar::Gating gating;
  switch (protocol.ParserStart(request)) {
    case OutputStart::kContent:
      return gating;
    case OutputStart::kReasoning:
      gating.start = Start::kReasoning;
      break;
    case OutputStart::kAuto:
      gating.start = Start::kEither;
      break;
  }
  gating.reasoning_end = tokenize(kThinkEnd);
  if (const auto start = tokenize(kThinkStart); start.size() == 1)
    gating.reasoning_start = start.front();
  else if (gating.start == Start::kEither)
    gating.start = Start::kAnswer;  // Cannot follow a multi-token opener.
  if (const auto separator = protocol.ReasoningSeparator(); !separator.empty())
    gating.separator = tokenize(separator);
  return gating;
}

std::shared_ptr<sampling::TokenConstraint> NewOutputConstraint(
    const ChatRequest& request, const ChatProtocol& protocol,
    const grammar::Engine* engine, const PromptTokenizer& tokenize) {
  if (engine == nullptr) {
    if (request.response_format.type != ResponseFormat::Type::kText)
      throw grammar::GrammarError(
          "unsupported_response_format",
          "this model does not support structured outputs");
    return nullptr;
  }
  auto source = OutputGrammar(request, protocol);
  if (!source)
    return nullptr;
  if (!request.forced_completion.empty())
    throw grammar::GrammarError(
        "invalid_response_format",
        "'forced_completion' cannot be combined with structured outputs");
  return engine->NewConstraint(*source,
                               OutputGating(request, protocol, tokenize));
}

grammar::Vocabulary QwenGrammarVocabulary(
    const tokenization::QwenTokenizer& tokenizer, std::size_t logits_width) {
  grammar::Vocabulary vocabulary;
  const std::size_t size = tokenizer.GetVocabSize();
  vocabulary.tokens.reserve(size);
  for (std::size_t id = 0; id < size; ++id) {
    const auto token = static_cast<tokenization::TokenId>(id);
    if (tokenizer.IsStopToken(token))
      vocabulary.stop_tokens.push_back(static_cast<std::int32_t>(id));
    auto text = tokenizer.DecodeTokenCopy(token);
    if (tokenizer.IsSpecialToken(token) &&
        std::ranges::find(kTextualControlTokens, text) ==
            kTextualControlTokens.end())
      text.clear();
    vocabulary.tokens.push_back(std::move(text));
  }
  vocabulary.logits_width = std::max(logits_width, size);
  return vocabulary;
}

}  // namespace gufo::server
