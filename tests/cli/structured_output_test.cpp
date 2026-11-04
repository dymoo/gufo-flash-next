#include "src/cli/serve/structured_output.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "tests/core/grammar_test_support.hpp"

namespace {

using gufo::grammar::Engine;
using gufo::grammar::Gating;
using gufo::grammar::GrammarError;
using gufo::grammar::GrammarSource;
using gufo::server::ChatRequest;
using gufo::server::ProtocolFor;
using gufo::server::ResponseFormat;
using gufo::server::TemplateFamily;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTool;
namespace test = gufo::grammar::test;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

ChatTool Tool(std::string name, std::string parameters) {
  ChatTool tool;
  tool.name = std::move(name);
  tool.parameters_json = std::move(parameters);
  return tool;
}

ChatRequest Request(bool thinking) {
  ChatRequest request({ChatMessage(ChatRole::kUser, "weather in Rome?")});
  request.reasoning.enabled = thinking;
  request.tools = {
      Tool("get_weather", R"({"type":"object","properties":{
        "city":{"type":"string"},"days":{"type":"integer"}},
        "required":["city","days"]})"),
      Tool("get_time", R"({"type":"object","properties":{
        "zone":{"type":"string","enum":["utc","local"]}},
        "required":["zone"]})"),
  };
  return request;
}

std::string ErrorCode(const ChatRequest& request, TemplateFamily family,
                      const Engine* engine) {
  try {
    (void)gufo::server::NewOutputConstraint(request, ProtocolFor(family),
                                            engine, test::Tokenize);
  } catch (const GrammarError& error) {
    return error.code();
  }
  return {};
}

void TestGrammarSelection() {
  const auto& qwen = ProtocolFor(TemplateFamily::kQwen38);
  auto request = Request(false);
  Expect(!gufo::server::OutputGrammar(request, qwen),
         "plain requests are unconstrained");

  request.response_format = {.type = ResponseFormat::Type::kJsonObject};
  request.tool_choice = ChatRequest::ToolChoice::kNone;
  auto source = gufo::server::OutputGrammar(request, qwen);
  Expect(source && source->kind == GrammarSource::Kind::kJsonSchema &&
             source->text == R"({"type":"object"})",
         "json_object constrains to any object");

  request.tool_choice = ChatRequest::ToolChoice::kAuto;
  source = gufo::server::OutputGrammar(request, qwen);
  Expect(source && source->kind == GrammarSource::Kind::kStructuralTag &&
             source->text.find(R"("type":"or")") != std::string::npos,
         "with tools on offer the answer is JSON or tool calls");

  request.response_format = {};
  request.tool_choice = ChatRequest::ToolChoice::kRequired;
  source = gufo::server::OutputGrammar(request, qwen);
  Expect(source && source->text.find("triggered_tags") != std::string::npos &&
             source->text.find(R"("at_least_one":true)") != std::string::npos &&
             source->text.find("qwen_xml") != std::string::npos,
         "a required tool choice constrains to declared calls");

  request.tool_choice_function = "get_time";
  source = gufo::server::OutputGrammar(request, qwen);
  Expect(source && source->text.find("get_time") != std::string::npos &&
             source->text.find("get_weather") == std::string::npos,
         "a named tool choice constrains to that function only");

  // LFM2.5 writes Pythonic calls: no grammar, the legacy 502 path remains.
  Expect(
      !gufo::server::OutputGrammar(request, ProtocolFor(TemplateFamily::kLfm2)),
      "families without a tool syntax keep tool choices unconstrained");
  request.tool_choice_function.clear();
  request.tool_choice = ChatRequest::ToolChoice::kAuto;
  request.response_format = {.type = ResponseFormat::Type::kJsonObject};
  Expect(ErrorCode(request, TemplateFamily::kLfm2, nullptr) ==
             "unsupported_response_format",
         "response_format needs grammar support");

  const Engine engine(test::MakeVocabulary());
  Expect(ErrorCode(request, TemplateFamily::kLfm2, &engine) ==
             "unsupported_response_format",
         "response_format with tools needs a tool syntax");
  request.forced_completion = "{}";
  request.tools.clear();
  Expect(ErrorCode(request, TemplateFamily::kQwen38, &engine) ==
             "invalid_response_format",
         "teacher forcing cannot be constrained");
  request.forced_completion.clear();
  request.response_format = {.type = ResponseFormat::Type::kJsonSchema,
                             .schema = "[1]"};
  Expect(ErrorCode(request, TemplateFamily::kQwen38, &engine) ==
             "invalid_response_format",
         "a schema must be a JSON object");

  auto tools_only = Request(false);
  tools_only.tool_choice = ChatRequest::ToolChoice::kRequired;
  Expect(gufo::server::NewOutputConstraint(tools_only, qwen, nullptr,
                                           test::Tokenize) == nullptr,
         "without grammar support a tool choice stays unconstrained");
}

void TestGatingFollowsTheTemplate() {
  auto request = Request(true);
  const auto qwen = gufo::server::OutputGating(
      request, ProtocolFor(TemplateFamily::kQwen38), test::Tokenize);
  Expect(qwen.start == Gating::Start::kReasoning &&
             qwen.reasoning_end == std::vector{test::kThinkEnd} &&
             qwen.separator == std::vector{test::kDoubleNewline},
         "Qwen reasons first and writes a blank line before the answer");
  const auto mimo = gufo::server::OutputGating(
      request, ProtocolFor(TemplateFamily::kMimo), test::Tokenize);
  Expect(mimo.start == Gating::Start::kEither &&
             mimo.reasoning_start == test::kThinkStart &&
             mimo.separator.empty(),
         "MiMo opens its own reasoning");
  request.reasoning.enabled = false;
  Expect(gufo::server::OutputGating(
             request, ProtocolFor(TemplateFamily::kQwen38), test::Tokenize)
                 .start == Gating::Start::kAnswer,
         "without thinking the grammar starts at the first token");
}

/// Required and named tool choices always produce a parseable declared call:
/// the unsatisfied-tool-choice 502 can no longer happen on Qwen and MiMo.
void TestToolChoiceAlwaysCalls() {
  const Engine engine(test::MakeVocabulary());
  const auto schema = gufo::json::parse(
      R"({"city":"string","days":"integer","zone":"string"})");
  for (const auto family : {TemplateFamily::kQwen38, TemplateFamily::kMimo}) {
    const auto& protocol = ProtocolFor(family);
    for (const bool named : {false, true}) {
      for (const bool thinking : {false, true}) {
        for (std::uint64_t seed = 1; seed <= 6; ++seed) {
          auto request = Request(thinking);
          request.tool_choice = ChatRequest::ToolChoice::kRequired;
          if (named)
            request.tool_choice_function = "get_weather";
          request.parallel_tool_calls = seed % 2 == 0;
          const auto source = gufo::server::OutputGrammar(request, protocol);
          const auto gating =
              gufo::server::OutputGating(request, protocol, test::Tokenize);
          std::vector<gufo::sampling::TokenId> reasoning;
          if (gating.start == Gating::Start::kReasoning) {
            reasoning = {'h', 'm', test::kThinkEnd};
          } else if (gating.start == Gating::Start::kEither) {
            reasoning = {test::kThinkStart, 'h', 'm', test::kThinkEnd};
          }
          const auto text =
              test::Generate(engine, *source, gating, seed, reasoning);
          auto output = protocol.NewParser(request);
          (void)output.Push(text, {});
          const auto result = output.Finish(true, {});
          Expect(result.has_value() && !result->tool_calls.empty(),
                 "a constrained tool choice always calls a tool: " + text);
          if (!request.parallel_tool_calls)
            Expect(result->tool_calls.size() == 1,
                   "parallel_tool_calls=false stops after one call");
          for (const auto& call : result->tool_calls) {
            Expect(!named || call.name == "get_weather",
                   "a named choice calls that function");
            for (const auto& argument : call.arguments) {
              const auto type = schema.member_str(argument.name);
              Expect(!type.empty(), "arguments are declared parameters");
              if (type == "integer") {
                const auto value = gufo::json::parse(argument.value);
                Expect(!argument.is_string && value.is_number() &&
                           std::floor(value.as_double()) == value.as_double(),
                       "integer arguments are JSON integers");
              } else {
                Expect(argument.is_string, "string arguments stay strings");
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  TestGrammarSelection();
  TestGatingFollowsTheTemplate();
  TestToolChoiceAlwaysCalls();
  std::cout << "All structured output tests passed\n";
  return 0;
}
