#include "src/cli/serve/chat_protocol.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace {

using gufo::json::Value;
using gufo::server::ChatProtocol;
using gufo::server::ChatRequest;
using gufo::server::OutputParser;
using gufo::server::ParsedOutput;
using gufo::server::ProtocolFor;
using gufo::server::TemplateFamily;
using gufo::server::TextGenerationBackend;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::TokenId;
using OutputStart = TextGenerationBackend::InitialOutputState;
using ToolCall = ChatMessage::ToolCall;

constexpr TemplateFamily kFamilies[] = {
    TemplateFamily::kQwen38, TemplateFamily::kQwen35, TemplateFamily::kLfm2,
    TemplateFamily::kDeepSeekV4, TemplateFamily::kMimo};

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

/// One token per byte: cache prefixes become byte offsets in the text.
std::vector<TokenId> Bytes(std::string_view text) {
  return {text.begin(), text.end()};
}

gufo::tokenization::ChatTool Tool(std::string name, std::string properties) {
  const std::string parameters =
      R"({"type":"object","properties":)" + properties + "}";
  return {
      .name = name,
      .description = "Get weather",
      .parameters_json = parameters,
      .definition_json = R"({"type":"function","function":{"name":")" + name +
                         R"(","description":"Get weather","parameters":)" +
                         parameters + "}}",
  };
}

/// System prompt, user turn, an assistant tool call and its result.
ChatRequest WeatherRequest(std::optional<bool> thinking) {
  ChatMessage assistant(ChatRole::kAssistant, "", {}, "Need data.");
  assistant.tool_calls.push_back(
      {.id = "call_1",
       .name = "get_weather",
       .arguments = {{.name = "city", .value = "Rome", .is_string = true},
                     {.name = "days", .value = "3", .is_string = false}}});
  ChatMessage result(ChatRole::kTool, R"({"temp": 21})");
  result.tool_call_id = "call_1";
  ChatRequest request({ChatMessage(ChatRole::kSystem, "Be brief."),
                       ChatMessage(ChatRole::kUser, "Weather in Rome?"),
                       assistant, result});
  request.tools.push_back(
      Tool("get_weather",
           R"({"city":{"type":"string"},"days":{"type":"integer"}})"));
  request.reasoning.enabled = thinking;
  return request;
}

const std::string kToolJson =
    R"({"type": "function", "function": {"name": "get_weather", )"
    R"("description": "Get weather", "parameters": {"type": "object", )"
    R"("properties": {"city": {"type": "string"}, "days": )"
    R"({"type": "integer"}}}}})";

const std::string kQwenTools =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
    kToolJson +
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
    "following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call "
    "in natural language BEFORE the function call, but NOT after\n- If "
    "there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function "
    "calls\n</IMPORTANT>";

const std::string kQwenTurns =
    "Be brief.<|im_end|>\n<|im_start|>user\nWeather in Rome?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nNeed data.\n</think>\n\n<tool_call>\n"
    "<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response>"
    "<|im_end|>\n<|im_start|>assistant\n";

const std::string kDeepSeekTools =
    "## Tools\n\nYou have access to a set of tools to help answer the user's "
    "question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" "
    "block like the following:\n\n<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter "
    "name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE"
    "</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke "
    "name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\n"
    "String parameters should be specified as is and set `string=\"true\"`. "
    "For all other types (numbers, booleans, arrays, objects), pass the value "
    "in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is "
    "enabled (triggered by <think>), you MUST output your complete reasoning "
    "inside <think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final "
    "response.\n\n### Available Tool Schemas\n\n"
    R"({"name": "get_weather", "description": "Get weather", "parameters": )"
    R"({"type": "object", "properties": {"city": {"type": "string"}, )"
    R"("days": {"type": "integer"}}}})"
    "\n\nYou MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls.\n";

std::string DeepSeekPrompt(std::string_view history_think,
                           std::string_view generation) {
  return "<｜begin▁of▁sentence｜>Be brief.\n\n" + kDeepSeekTools +
         "<｜User｜>Weather in Rome?<｜Assistant｜>" +
         std::string(history_think) +
         "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
         "<｜DSML｜parameter name=\"city\" string=\"true\">Rome"
         "</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" "
         "string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
         "</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>"
         "{\"temp\": 21}</tool_result>" +
         std::string(generation);
}

void TestRenderGoldens() {
  const std::string xhigh =
      "Reasoning effort is set to xhigh. Please think carefully through the "
      "task, validate key assumptions, consider plausible alternatives, and "
      "prioritize correctness, consistency, and clarity in the final answer.";
  const std::string lfm =
      "<|startoftext|><|im_start|>system\nBe brief.\nList of tools: [" +
      kToolJson +
      "]<|im_end|>\n<|im_start|>user\nWeather in Rome?<|im_end|>\n"
      "<|im_start|>assistant\n<think>Need data.</think><|tool_call_start|>"
      "[get_weather(city='Rome', days=3)]<|tool_call_end|><|im_end|>\n"
      "<|im_start|>tool\n{\"temp\": 21}<|im_end|>\n<|im_start|>assistant\n"
      "<think>";
  const std::string mimo =
      "<|im_start|>system\nYou are provided with the following tools:\n\n"
      "<tools>\n" +
      kToolJson +
      "\n</tools><|im_end|><|im_start|>system\nBe brief.<|im_end|>"
      "<|im_start|>user\nWeather in Rome?<|im_end|><|im_start|>assistant\n"
      "<think>Need data.</think><tool_call><function=get_weather>"
      "<parameter=city>Rome</parameter><parameter=days>3</parameter>"
      "</function></tool_call><|im_end|><|im_start|>tool\n{\"temp\": 21}"
      "<|im_end|><|im_start|>assistant\n";
  const std::string qwen_off = "<|im_start|>system\n" + kQwenTools + "\n\n" +
                               kQwenTurns + "<think>\n\n</think>\n\n";
  struct Golden {
    TemplateFamily family;
    bool thinking;
    std::string text;
  };
  const Golden goldens[] = {
      {TemplateFamily::kQwen38, true,
       "<|im_start|>system\n" + xhigh + "\n\n" + kQwenTools + "\n\n" +
           kQwenTurns + "<think>\n"},
      // Qwen3.5 has no effort instruction (native medium).
      {TemplateFamily::kQwen35, true,
       "<|im_start|>system\n" + kQwenTools + "\n\n" + kQwenTurns + "<think>\n"},
      {TemplateFamily::kQwen38, false, qwen_off},
      {TemplateFamily::kQwen35, false, qwen_off},
      // LFM2.5 always opens <think>.
      {TemplateFamily::kLfm2, true, lfm},
      {TemplateFamily::kLfm2, false, lfm},
      {TemplateFamily::kDeepSeekV4, true,
       DeepSeekPrompt("<think>Need data.</think>", "<｜Assistant｜><think>")},
      {TemplateFamily::kDeepSeekV4, false,
       DeepSeekPrompt("</think>", "<｜Assistant｜></think>")},
      // With thinking on, MiMo leaves <think> to the model.
      {TemplateFamily::kMimo, true, mimo},
      {TemplateFamily::kMimo, false, mimo + "<think></think>"},
  };
  for (const auto& golden : goldens) {
    const auto rendered =
        ProtocolFor(golden.family).RenderText(WeatherRequest(golden.thinking));
    if (rendered != golden.text)
      std::cerr << "rendered:\n" << rendered << "\nexpected:\n" << golden.text;
    Expect(rendered == golden.text, "rendered prompt matches its golden");
  }

  // tool_choice=none hides the tools; required asks for a call.
  auto request = WeatherRequest(true);
  request.tool_choice = ChatRequest::ToolChoice::kNone;
  for (const auto family : kFamilies) {
    Expect(ProtocolFor(family).RenderText(request).find("Get weather") ==
               std::string::npos,
           "tool_choice none omits tool definitions");
  }
  request.tool_choice = ChatRequest::ToolChoice::kRequired;
  Expect(ProtocolFor(TemplateFamily::kQwen38)
                 .RenderText(request)
                 .find("You must call at least one available function.") !=
             std::string::npos,
         "required tool choice reaches the Qwen template");

  // Reasoning effort: Qwen3.8 writes its instruction, Qwen3.5 never does.
  auto low = WeatherRequest(true);
  low.reasoning.effort = gufo::ReasoningEffort::kLow;
  Expect(ProtocolFor(TemplateFamily::kQwen38)
                 .RenderText(low)
                 .starts_with("<|im_start|>system\nReasoning effort is set to "
                              "low.") &&
             ProtocolFor(TemplateFamily::kQwen35)
                 .RenderText(low)
                 .starts_with("<|im_start|>system\n# Tools"),
         "only Qwen3.8 renders the reasoning effort");
}

void TestLfmHistoryMatchesOfficialTemplate() {
  // Structured arguments render as Python literals (True/False/None, tojson
  // containers); stale reasoning in content is cut and trimmed.
  ChatMessage call(ChatRole::kAssistant, "");
  call.tool_calls.push_back(
      {.id = "a",
       .name = "f",
       .arguments = {
           {.name = "s", .value = "it's\n", .is_string = true},
           {.name = "b", .value = "true", .is_string = false},
           {.name = "n", .value = "null", .is_string = false},
           {.name = "o", .value = R"({"k":[1,2]})", .is_string = false},
           {.name = "x", .value = "1.5", .is_string = false}}});
  ChatRequest request(
      {ChatMessage(ChatRole::kUser, "Q1"),
       ChatMessage(ChatRole::kAssistant, "<think>old</think>\n A1 \n"),
       ChatMessage(ChatRole::kUser, "Q2"), call});
  Expect(ProtocolFor(TemplateFamily::kLfm2).RenderText(request) ==
             "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n"
             "<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2"
             "<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>"
             "[f(s='it\\'s\\n', b=True, n=None, o={\"k\": [1, 2]}, x=1.5)]"
             "<|tool_call_end|><|im_end|>\n<|im_start|>assistant\n<think>",
         "LFM history follows LFM2.5's chat_template.jinja");
}

void TestParserStart() {
  ChatRequest request;
  struct Case {
    TemplateFamily family;
    OutputStart unset, on, off;
  };
  for (const auto& item : {
           Case{TemplateFamily::kQwen38, OutputStart::kReasoning,
                OutputStart::kReasoning, OutputStart::kContent},
           Case{TemplateFamily::kQwen35, OutputStart::kReasoning,
                OutputStart::kReasoning, OutputStart::kContent},
           // The LFM2.5 prompt always opens <think>.
           Case{TemplateFamily::kLfm2, OutputStart::kReasoning,
                OutputStart::kReasoning, OutputStart::kReasoning},
           Case{TemplateFamily::kDeepSeekV4, OutputStart::kContent,
                OutputStart::kReasoning, OutputStart::kContent},
           // MiMo writes its own <think> when thinking.
           Case{TemplateFamily::kMimo, OutputStart::kAuto, OutputStart::kAuto,
                OutputStart::kContent},
       }) {
    const auto& protocol = ProtocolFor(item.family);
    request.reasoning.enabled.reset();
    Expect(protocol.ParserStart(request) == item.unset, "default start");
    request.reasoning.enabled = true;
    Expect(protocol.ParserStart(request) == item.on, "thinking start");
    request.reasoning.enabled = false;
    Expect(protocol.ParserStart(request) == item.off, "thinking-off start");
  }
}

void TestCachePrefix() {
  auto request = WeatherRequest(true);
  struct Case {
    TemplateFamily family;
    std::string_view generation;
  };
  for (const auto& item : {
           Case{TemplateFamily::kQwen38, "<|im_start|>assistant\n<think>\n"},
           Case{TemplateFamily::kLfm2, "<|im_start|>assistant\n<think>"},
           Case{TemplateFamily::kDeepSeekV4, "<｜Assistant｜><think>"},
           Case{TemplateFamily::kMimo, "<|im_start|>assistant\n"},
       }) {
    const auto prompt = ProtocolFor(item.family).Render(request, Bytes);
    Expect(prompt.tokens == Bytes(ProtocolFor(item.family).RenderText(request)),
           "Render tokenizes the rendered text");
    Expect(prompt.cache_prefix_tokens ==
               prompt.tokens.size() - item.generation.size(),
           "cache prefix stops before the open assistant turn");
  }
  // Qwen keeps the whole prompt when reasoning is preserved and no tools can
  // make an agent drop the next assistant turn.
  request.tools.clear();
  Expect(ProtocolFor(TemplateFamily::kQwen38)
                 .Render(request, Bytes)
                 .cache_prefix_tokens == 0,
         "preserved Qwen reasoning without tools caches the full prompt");
  request.reasoning.preserve_thinking = false;
  Expect(ProtocolFor(TemplateFamily::kQwen38)
                 .Render(request, Bytes)
                 .cache_prefix_tokens != 0,
         "dropped Qwen reasoning caches before the generation prompt");
}

struct Streamed {
  std::string reasoning;
  std::string content;
  ParsedOutput parsed;
};

/// Streams `output` in `chunk`-byte pieces, splitting markers and UTF-8.
Streamed Stream(TemplateFamily family, const ChatRequest& request,
                std::string_view output, std::size_t chunk) {
  auto parser = ProtocolFor(family).NewParser(request);
  Streamed run;
  const OutputParser::Emit emit = [&](std::string_view text, bool reasoning) {
    (reasoning ? run.reasoning : run.content).append(text);
    return true;
  };
  for (std::size_t at = 0; at < output.size(); at += chunk)
    Expect(parser.Push(output.substr(at, chunk), emit), "push succeeds");
  auto parsed = parser.Finish(true, emit);
  Expect(parsed.has_value(), "finish succeeds");
  run.parsed = std::move(*parsed);
  return run;
}

bool SameArguments(const ToolCall& left, const ToolCall& right) {
  if (left.name != right.name ||
      left.arguments.size() != right.arguments.size())
    return false;
  for (std::size_t i = 0; i < left.arguments.size(); ++i) {
    const auto& a = left.arguments[i];
    const auto& b = right.arguments[i];
    if (a.name != b.name || a.is_string != b.is_string)
      return false;
    if (a.is_string ? a.value != b.value
                    : gufo::json::parse(a.value).dump() !=
                          gufo::json::parse(b.value).dump())
      return false;
  }
  return true;
}

ChatRequest ParseRequest(std::optional<bool> thinking) {
  ChatRequest request({ChatMessage(ChatRole::kUser, "go")});
  request.reasoning.enabled = thinking;
  request.tools.push_back(
      Tool("get_weather",
           R"({"city":{"type":"string"},"days":{"type":"integer"},)"
           R"("units":{"type":"object"},"live":{"type":"boolean"}})"));
  request.tools.push_back(Tool("get_time", R"({"zone":{"type":"string"}})"));
  return request;
}

void TestStreamingParse() {
  const ToolCall weather{
      .id = {},
      .name = "get_weather",
      .arguments = {{.name = "city", .value = "Rome", .is_string = true},
                    {.name = "days", .value = "3", .is_string = false}}};
  const ToolCall time{
      .id = {},
      .name = "get_time",
      .arguments = {
          {.name = "zone", .value = "Europe/Rome", .is_string = true}}};
  struct Case {
    TemplateFamily family;
    std::optional<bool> thinking;
    std::string output;
    std::string reasoning;
    std::string content;
    std::vector<ToolCall> calls;
  };
  const Case cases[] = {
      {TemplateFamily::kQwen38,
       true,
       "Check ┌ the city.</think>\n\nLooking it up.\n\n<tool_call>\n"
       "<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
       "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
       "Check ┌ the city.",
       "Looking it up.\n\n",
       {weather}},
      {TemplateFamily::kQwen35,
       false,
       "Plain answer ┌.",
       "",
       "Plain answer ┌.",
       {}},
      // LFM2.5 thinks even when the request turns thinking off, and writes
      // parallel calls as one Python list.
      {TemplateFamily::kLfm2,
       false,
       "Two lookups.</think>\n\n<|tool_call_start|>[get_weather("
       "city=\"Rome\", days=3), get_time(zone='Europe/Rome')]"
       "<|tool_call_end|>",
       "Two lookups.",
       "",
       {weather, time}},
      {TemplateFamily::kLfm2,
       true,
       "Easy ┌.</think>\n\nIt is sunny ┌.",
       "Easy ┌.",
       "It is sunny ┌.",
       {}},
      {TemplateFamily::kDeepSeekV4,
       true,
       "Plan ┌.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke "
       "name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" "
       "string=\"true\">Rome</｜DSML｜parameter>\n<｜DSML｜parameter "
       "name=\"days\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>"
       "\n</｜DSML｜tool_calls>",
       "Plan ┌.",
       "",
       {weather}},
      {TemplateFamily::kMimo,
       true,
       "<think>Plan ┌.</think><tool_call><function=get_time>"
       "<parameter=zone>Europe/Rome</parameter></function></tool_call>",
       "Plan ┌.",
       "",
       {time}},
      {TemplateFamily::kMimo, false, "Direct ┌.", "", "Direct ┌.", {}},
  };
  for (const auto& item : cases) {
    const auto request = ParseRequest(item.thinking);
    for (const std::size_t chunk : {1UL, 2UL, 3UL, 5UL, 64UL, 4096UL}) {
      const auto run = Stream(item.family, request, item.output, chunk);
      Expect(run.parsed.reasoning == item.reasoning &&
                 run.parsed.content == item.content,
             "output splits into reasoning and content");
      Expect(run.reasoning == item.reasoning && run.content == item.content,
             "streamed deltas match the parse and hide tool markup");
      Expect(run.parsed.tool_calls.size() == item.calls.size(),
             "every complete tool call is parsed");
      for (std::size_t i = 0; i < item.calls.size(); ++i)
        Expect(SameArguments(run.parsed.tool_calls[i], item.calls[i]),
               "tool call names and typed arguments survive");
    }
  }
}

void TestPythonicCalls() {
  const auto request = ParseRequest(true);
  // Python literals, JSON literals, escapes, nested containers, a bare call.
  auto run = Stream(TemplateFamily::kLfm2, request,
                    "</think><|tool_call_start|>[get_weather(city='It\\'s "
                    "\\u00e9\\n', units={'a': [1, 2.5, None], \"b\": (True,)},"
                    " live=False, days=-2,)]<|tool_call_end|>",
                    7);
  Expect(run.parsed.tool_calls.size() == 1, "Python literal call parses");
  const auto& arguments = run.parsed.tool_calls[0].arguments;
  Expect(arguments.size() == 4 && arguments[0].value == "It's \xC3\xA9\n" &&
             arguments[0].is_string &&
             arguments[1].value == R"({"a":[1,2.5,null],"b":[true]})" &&
             !arguments[1].is_string && arguments[2].value == "false" &&
             arguments[3].value == "-2",
         "Python values convert to JSON arguments");
  run = Stream(TemplateFamily::kLfm2, request,
               "</think><|tool_call_start|>get_time(zone=\"UTC\")"
               "<|tool_call_end|>",
               4096);
  Expect(run.parsed.tool_calls.size() == 1, "a bare call parses");
  // Undeclared and malformed calls never become tool calls; a complete call
  // before an interrupted one survives.
  for (const auto* output :
       {"</think><|tool_call_start|>[rm(path='/')]<|tool_call_end|>",
        "</think><|tool_call_start|>[get_time(zone='UTC'",
        "</think><|tool_call_start|>[get_time(zone=UTC)]<|tool_call_end|>",
        "</think><|tool_call_start|>[get_time('UTC')]<|tool_call_end|>"}) {
    auto parser = ProtocolFor(TemplateFamily::kLfm2).NewParser(request);
    parser.Push(output, {});
    const auto parsed = parser.Finish(false, {});
    Expect(parsed && parsed->tool_calls.empty() && parsed->content.empty(),
           "invalid calls are dropped and their markup hidden");
  }
  run = Stream(TemplateFamily::kLfm2, request,
               "</think><|tool_call_start|>[get_time(zone='UTC'), "
               "get_weather(city='Ro",
               3);
  Expect(run.parsed.tool_calls.size() == 1 &&
             run.parsed.tool_calls[0].name == "get_time",
         "calls completed before an interruption survive");
}

void TestFamiliesKeepTheirOwnSyntax() {
  // Another family's markup is ordinary text.
  const auto request = ParseRequest(false);
  const auto qwen_call =
      "<tool_call><function=get_time><parameter=zone>UTC</parameter>"
      "</function></tool_call>";
  for (const auto family :
       {TemplateFamily::kDeepSeekV4, TemplateFamily::kLfm2}) {
    auto parser = ProtocolFor(family).NewParser(request);
    parser.Push(std::string("</think>") + qwen_call, {});
    const auto parsed = parser.Finish(false, {});
    Expect(parsed && parsed->tool_calls.empty() &&
               parsed->content.find("<tool_call>") != std::string::npos,
           "foreign tool syntax is not parsed");
  }
}

void TestRequiredToolChoice() {
  auto request = ParseRequest(false);
  request.tool_choice = ChatRequest::ToolChoice::kRequired;
  for (const auto family : kFamilies) {
    auto parser = ProtocolFor(family).NewParser(request);
    parser.Push("</think>No tools today.", {});
    bool threw = false;
    try {
      (void)parser.Finish(true, {});
    } catch (const gufo::server::TextGenerationError& error) {
      threw = error.code() ==
              gufo::server::TextGenerationErrorCode::kToolChoiceUnsatisfied;
    }
    Expect(threw, "required tool choice rejects text-only output");
  }
}

void TestRenderParseRoundTrip() {
  // Each family parses the tool calls its own template writes into history.
  const ToolCall call{
      .id = "call_1",
      .name = "get_weather",
      .arguments = {
          {.name = "city", .value = "Rome \"Lazio\"", .is_string = true},
          {.name = "days", .value = "3", .is_string = false},
          {.name = "units",
           .value = R"({"t":"c","n":[1,2]})",
           .is_string = false},
          {.name = "live", .value = "true", .is_string = false}}};
  const ToolCall second{
      .id = "call_2",
      .name = "get_time",
      .arguments = {
          {.name = "zone", .value = "Europe/Rome", .is_string = true}}};
  for (const auto family : kFamilies) {
    auto request = ParseRequest(false);
    ChatMessage assistant(ChatRole::kAssistant, "");
    assistant.tool_calls = {call, second};
    request.messages.push_back(assistant);
    const auto rendered = ProtocolFor(family).RenderText(request);
    const std::string_view end_of_turn = family == TemplateFamily::kDeepSeekV4
                                             ? "<｜end▁of▁sentence｜>"
                                             : "<|im_end|>";
    const auto assistant_turn = rendered.rfind(
        family == TemplateFamily::kDeepSeekV4 ? "<｜Assistant｜>"
                                              : "<|im_start|>assistant",
        rendered.rfind(end_of_turn));
    const auto end = rendered.rfind(end_of_turn);
    std::string history = rendered.substr(assistant_turn, end - assistant_turn);
    history = history.substr(history.find_first_of('<', 1));
    // Strip the empty reasoning the templates write before the calls.
    for (const std::string_view framing :
         {"<think>\n\n</think>\n\n", "<think></think>", "</think>\n\n"})
      if (history.starts_with(framing))
        history.erase(0, framing.size());
    // LFM2.5 output starts inside reasoning.
    if (ProtocolFor(family).ParserStart(request) == OutputStart::kReasoning)
      history.insert(0, "</think>");
    const auto run = Stream(family, request, history, 3);
    Expect(run.parsed.tool_calls.size() == 2 &&
               SameArguments(run.parsed.tool_calls[0], call) &&
               SameArguments(run.parsed.tool_calls[1], second),
           "rendered tool calls parse back to the same calls");
    Expect(run.content.empty() && run.parsed.content.empty(),
           "tool markup never reaches content");
  }
}

}  // namespace

int main() {
  TestRenderGoldens();
  TestLfmHistoryMatchesOfficialTemplate();
  TestParserStart();
  TestCachePrefix();
  TestStreamingParse();
  TestPythonicCalls();
  TestFamiliesKeepTheirOwnSyntax();
  TestRequiredToolChoice();
  TestRenderParseRoundTrip();
  std::cout << "All chat protocol tests passed\n";
  return 0;
}
