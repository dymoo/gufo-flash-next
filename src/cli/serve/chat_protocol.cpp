#include "src/cli/serve/chat_protocol.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "src/core/json.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/mimo_v26_flash/chat_template.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {
namespace {

using OutputStart = TextGenerationBackend::InitialOutputState;
using tokenization::ChatMessage;
using tokenization::ChatRole;
using tokenization::ChatTool;
using tokenization::TokenId;
using ToolCall = ChatMessage::ToolCall;

constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";

std::span<const ChatTool> PromptTools(const ChatRequest& request) {
  return request.tool_choice == ChatRequest::ToolChoice::kNone
             ? std::span<const ChatTool>{}
             : std::span<const ChatTool>{request.tools};
}

bool HasImages(const ChatRequest& request) {
  return std::ranges::any_of(request.messages,
                             [](const auto& m) { return !m.images.empty(); });
}

std::size_t StablePromptPrefix(std::span<const TokenId> tokens,
                               std::span<const TokenId> generation) {
  if (generation.empty() || tokens.size() <= generation.size() ||
      !std::ranges::equal(tokens.last(generation.size()), generation))
    return 0;
  return tokens.size() - generation.size();
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

// ---- Qwen and MiMo tool calls: <tool_call><function=...> XML or JSON.

// Qwen writes a parameter as "<parameter=name>\nVALUE\n</parameter>": one
// newline on each side is framing, everything else (a file's final newline,
// indentation, blank lines) belongs to the value.
std::string_view StripFramingNewlines(std::string_view value) {
  if (value.starts_with("\r\n"))
    value.remove_prefix(2);
  else if (value.starts_with('\n'))
    value.remove_prefix(1);
  if (value.ends_with("\r\n"))
    value.remove_suffix(2);
  else if (value.ends_with('\n'))
    value.remove_suffix(1);
  return value;
}

// Qwen's XML-like calls look like Python keyword arguments, and the models
// sometimes write Python's True, False and None where the schema asks for a
// JSON boolean or null. Rewrites those words outside JSON string literals.
std::string PythonLiteralsToJson(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  bool in_string = false;
  for (std::size_t i = 0; i < value.size();) {
    const char c = value[i];
    if (in_string) {
      const std::size_t length = c == '\\' && i + 1 < value.size() ? 2 : 1;
      out += value.substr(i, length);
      in_string = c != '"';
      i += length;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) == 0) {
      out += c;
      in_string = c == '"';
      ++i;
      continue;
    }
    std::size_t word_end = i;
    while (word_end < value.size() &&
           (std::isalnum(static_cast<unsigned char>(value[word_end])) != 0 ||
            value[word_end] == '_')) {
      ++word_end;
    }
    const auto word = value.substr(i, word_end - i);
    out += word == "True"    ? std::string_view{"true"}
           : word == "False" ? std::string_view{"false"}
           : word == "None"  ? std::string_view{"null"}
                             : word;
    i = word_end;
  }
  return out;
}

// Qwen's XML-like arguments carry no type marker. The advertised schema is
// needed to distinguish a string such as 42 from the JSON number 42.
bool SchemaAccepts(const json::Value& schema, const json::Value& value) {
  const auto matches = [&](std::string_view type) {
    return (type == "string" && value.is_string()) ||
           (type == "number" && value.is_number()) ||
           (type == "integer" && value.is_number() &&
            std::floor(value.as_double()) == value.as_double()) ||
           (type == "boolean" && value.is_bool()) ||
           (type == "null" && value.is_null()) ||
           (type == "array" && value.is_array()) ||
           (type == "object" && value.is_object());
  };
  if (const auto* type = schema.find("type")) {
    if (type->is_string())
      return matches(type->get_str());
    if (type->is_array())
      return std::ranges::any_of(type->items(), [&](const auto& item) {
        return item.is_string() && matches(item.get_str());
      });
    return false;
  }
  for (const auto* name : {"anyOf", "oneOf"}) {
    if (const auto* choices = schema.find(name); choices && choices->is_array())
      return std::ranges::any_of(choices->items(), [&](const auto& item) {
        return SchemaAccepts(item, value);
      });
  }
  return true;
}

void ParseQwenCalls(std::string_view text, std::span<const ChatTool> tools,
                    std::vector<ToolCall>* calls) {
  constexpr std::string_view start = "<tool_call>";
  constexpr std::string_view end = "</tool_call>";
  std::size_t cursor = 0;
  while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
    const auto begin = cursor + start.size();
    cursor = begin;  // A malformed call may be followed by a valid call.
    auto body = text.substr(begin);
    const auto consume = [&](std::string_view tag) {
      body = Trim(body);
      if (!body.starts_with(tag))
        return false;
      body.remove_prefix(tag.size());
      return true;
    };
    body = Trim(body);
    ToolCall call;
    bool complete = false;
    if (consume("<function=")) {
      const auto name_end = body.find('>');
      if (name_end == std::string_view::npos)
        continue;
      call.name = std::string(Trim(body.substr(0, name_end)));
      body.remove_prefix(name_end + 1);
      std::optional<json::Value> schema;
      for (const auto& tool : tools) {
        if (tool.name == call.name) {
          schema = TryParseJson(tool.parameters_json);
          break;
        }
      }
      if (call.name.empty() || (!tools.empty() && !schema))
        continue;
      bool valid = true;
      while (consume("<parameter=")) {
        const auto name_end = body.find('>');
        if (name_end == std::string_view::npos) {
          valid = false;
          break;
        }
        const std::string name(Trim(body.substr(0, name_end)));
        body.remove_prefix(name_end + 1);
        // Outer closing tags inside a parameter are data, not structure.
        const auto close = body.find("</parameter>");
        if (close == std::string_view::npos || body.find(start) < close ||
            name.empty() ||
            std::ranges::any_of(call.arguments, [&](const auto& arg) {
              return arg.name == name;
            })) {
          valid = false;
          break;
        }
        const auto value = StripFramingNewlines(body.substr(0, close));
        const auto* properties = schema ? schema->find("properties") : nullptr;
        const auto* property = properties ? properties->find(name) : nullptr;
        const bool string_allowed =
            !property ||
            SchemaAccepts(*property, json::Value(std::string(value)));
        // Prefer text if the schema permits it; parsing ambiguous scalars
        // as JSON would silently change a caller's declared string type.
        const bool is_string = string_allowed;
        std::string raw(is_string ? value : Trim(value));
        if (!is_string) {
          auto parsed = TryParseJson(raw);
          if (!parsed || !SchemaAccepts(*property, *parsed)) {
            raw = PythonLiteralsToJson(raw);
            parsed = TryParseJson(raw);
          }
          if (!parsed || !SchemaAccepts(*property, *parsed)) {
            valid = false;
            break;
          }
        }
        call.arguments.push_back(
            {.name = name, .value = std::move(raw), .is_string = is_string});
        body.remove_prefix(close + std::string_view{"</parameter>"}.size());
      }
      complete = valid && consume("</function>") && consume(end);
    } else {
      // JSON calls already encode their argument types. Try closing markers
      // until the preceding payload is complete JSON (a marker in a quoted
      // string cannot terminate the call).
      std::size_t close = 0;
      while ((close = body.find(end, close)) != std::string_view::npos) {
        const auto parsed = TryParseJson(Trim(body.substr(0, close)));
        if (parsed && parsed->is_object()) {
          call.name = parsed->member_str("name");
          const auto* arguments = parsed->find("arguments");
          if (!call.name.empty() && arguments && arguments->is_object()) {
            for (const auto& [name, value] : arguments->members())
              call.arguments.push_back(
                  {.name = name,
                   .value = value.is_string() ? value.get_str() : value.dump(),
                   .is_string = value.is_string()});
            complete = true;
            body.remove_prefix(close + end.size());
          }
          break;
        }
        close += end.size();
      }
    }
    if (complete && !tools.empty() &&
        std::ranges::none_of(
            tools, [&](const auto& tool) { return tool.name == call.name; }))
      complete = false;
    if (complete) {
      calls->push_back(std::move(call));
      cursor = text.size() - body.size();
    }
  }
}

// ---- DeepSeek tool calls: DSML invoke blocks.

std::optional<std::string> Attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const std::size_t start = tag.find(prefix);
  if (start == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = tag.find('"', value_start);
  if (end == std::string_view::npos) {
    return std::nullopt;
  }
  return std::string(tag.substr(value_start, end - value_start));
}

void ParseDsmlCalls(std::string_view text, std::vector<ToolCall>* calls) {
  constexpr std::array<std::string_view, 4> kInvokeStarts{
      "<｜DSML｜invoke",
      "<DSML｜invoke",
      "<｜DS｜invoke",
      "<DS｜invoke",
  };
  constexpr std::array<std::string_view, 4> kInvokeEnds{
      "</｜DSML｜invoke>",
      "</DSML｜invoke>",
      "</｜DS｜invoke>",
      "</DS｜invoke>",
  };
  constexpr std::array<std::string_view, 4> kParameterStarts{
      "<｜DSML｜parameter",
      "<DSML｜parameter",
      "<｜DS｜parameter",
      "<DS｜parameter",
  };
  constexpr std::array<std::string_view, 4> kParameterEnds{
      "</｜DSML｜parameter>",
      "</DSML｜parameter>",
      "</｜DS｜parameter>",
      "</DS｜parameter>",
  };

  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t invoke_start = std::string_view::npos;
    std::size_t syntax = 0;
    for (std::size_t index = 0; index < kInvokeStarts.size(); ++index) {
      const std::size_t position = text.find(kInvokeStarts[index], cursor);
      if (position < invoke_start) {
        invoke_start = position;
        syntax = index;
      }
    }
    if (invoke_start == std::string_view::npos) {
      break;
    }
    const std::size_t tag_end = text.find('>', invoke_start);
    const std::size_t invoke_end = text.find(kInvokeEnds[syntax], tag_end);
    if (tag_end == std::string_view::npos ||
        invoke_end == std::string_view::npos) {
      break;
    }
    ToolCall call;
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < invoke_end) {
      const std::size_t parameter_start =
          text.find(kParameterStarts[syntax], parameter_cursor);
      if (parameter_start == std::string_view::npos ||
          parameter_start >= invoke_end) {
        break;
      }
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      const std::size_t parameter_end =
          text.find(kParameterEnds[syntax], parameter_tag_end);
      if (parameter_tag_end == std::string_view::npos ||
          parameter_end == std::string_view::npos ||
          parameter_end > invoke_end) {
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      if (parameter_name.has_value()) {
        call.arguments.push_back({
            .name = *parameter_name,
            .value = std::string(Trim(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))),
            .is_string = string_value.value_or("true") != "false",
        });
      }
      parameter_cursor = parameter_end + kParameterEnds[syntax].size();
    }
    if (!call.name.empty()) {
      calls->push_back(std::move(call));
    }
    cursor = invoke_end + kInvokeEnds[syntax].size();
  }
}

// ---- LFM2.5 tool calls: a Python list of calls between special tokens,
// <|tool_call_start|>[name(arg='v', n=1), other(flag=True)]<|tool_call_end|>.

constexpr std::string_view kLfmCallStart = "<|tool_call_start|>";
constexpr std::string_view kLfmGeneration = "<|im_start|>assistant\n<think>";

void AppendUtf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// Function and keyword names: anything but Python call punctuation.
bool IsCallName(std::string_view name) {
  return !name.empty() &&
         name.find_first_of(" \t\r\n'\"()[]{},=<>|") == std::string_view::npos;
}

struct PythonCursor {
  std::string_view text;
  std::size_t at{0};

  void Space() {
    while (at < text.size() &&
           std::isspace(static_cast<unsigned char>(text[at])) != 0)
      ++at;
  }
  bool Eat(char c) {
    Space();
    if (at >= text.size() || text[at] != c)
      return false;
    ++at;
    return true;
  }
  [[nodiscard]] bool Peek(char c) const {
    return at < text.size() && text[at] == c;
  }
};

std::optional<std::uint32_t> HexCodePoint(PythonCursor& c, std::size_t digits) {
  std::uint32_t value = 0;
  if (c.text.size() - c.at < digits)
    return std::nullopt;
  const auto* first = c.text.data() + c.at;
  const auto [end, error] = std::from_chars(first, first + digits, value, 16);
  if (error != std::errc{} || end != first + digits)
    return std::nullopt;
  c.at += digits;
  return value;
}

/// A single- or double-quoted Python string literal, unescaped.
std::optional<std::string> PythonString(PythonCursor& c) {
  c.Space();
  if (!c.Peek('\'') && !c.Peek('"'))
    return std::nullopt;
  const char quote = c.text[c.at++];
  std::string out;
  while (c.at < c.text.size()) {
    const char ch = c.text[c.at++];
    if (ch == quote)
      return out;
    if (ch != '\\') {
      out += ch;
      continue;
    }
    if (c.at >= c.text.size())
      break;
    const char escape = c.text[c.at++];
    switch (escape) {
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case '\\':
      case '\'':
      case '"':
        out += escape;
        break;
      case '\n':  // Line continuation.
        break;
      case 'x':
      case 'u':
      case 'U': {
        auto cp = HexCodePoint(c, escape == 'x' ? 2 : escape == 'u' ? 4 : 8);
        if (!cp)
          return std::nullopt;
        // JSON-style surrogate pairs join; Python writes \U for these.
        if (*cp >= 0xD800 && *cp <= 0xDBFF && c.text.substr(c.at, 2) == "\\u") {
          c.at += 2;
          const auto low = HexCodePoint(c, 4);
          if (!low || *low < 0xDC00 || *low > 0xDFFF)
            return std::nullopt;
          *cp = 0x10000 + ((*cp - 0xD800) << 10) + (*low - 0xDC00);
        }
        if (*cp > 0x10FFFF || (*cp >= 0xD800 && *cp <= 0xDFFF))
          return std::nullopt;
        AppendUtf8(out, *cp);
        break;
      }
      default:  // Python keeps unknown escapes verbatim.
        out += '\\';
        out += escape;
    }
  }
  return std::nullopt;
}

/// A Python (or JSON) literal: string, number, True/False/None, list, tuple
/// or dict with string keys.
// NOLINTNEXTLINE(misc-no-recursion)
std::optional<json::Value> PythonValue(PythonCursor& c, int depth) {
  c.Space();
  if (depth > 64 || c.at >= c.text.size())
    return std::nullopt;
  if (c.Peek('\'') || c.Peek('"')) {
    auto text = PythonString(c);
    if (!text)
      return std::nullopt;
    return json::Value(std::move(*text));
  }
  if (c.Peek('[') || c.Peek('(')) {
    const char close = c.text[c.at++] == '[' ? ']' : ')';
    auto array = json::Value::array();
    while (!c.Eat(close)) {
      auto item = PythonValue(c, depth + 1);
      if (!item)
        return std::nullopt;
      array.push_back(std::move(*item));
      if (!c.Eat(',') && !c.Peek(close))
        return std::nullopt;
    }
    return array;
  }
  if (c.Peek('{')) {
    ++c.at;
    auto object = json::Value::object();
    while (!c.Eat('}')) {
      auto key = PythonString(c);
      if (!key || !c.Eat(':'))
        return std::nullopt;
      auto value = PythonValue(c, depth + 1);
      if (!value)
        return std::nullopt;
      object[*key] = std::move(*value);
      if (!c.Eat(',') && !c.Peek('}'))
        return std::nullopt;
    }
    return object;
  }
  std::size_t end = c.at;
  while (end < c.text.size() &&
         (std::isalnum(static_cast<unsigned char>(c.text[end])) != 0 ||
          std::string_view("_.+-").find(c.text[end]) != std::string_view::npos))
    ++end;
  const std::string word(c.text.substr(c.at, end - c.at));
  c.at = end;
  if (word == "True" || word == "true")
    return json::Value(true);
  if (word == "False" || word == "false")
    return json::Value(false);
  if (word == "None" || word == "null")
    return json::Value();
  if (auto number = TryParseJson(word);
      number && number->is_number() && std::isfinite(number->as_double()))
    return number;
  char* parsed_end = nullptr;
  const double number = std::strtod(word.c_str(), &parsed_end);
  if (word.empty() || parsed_end != word.c_str() + word.size() ||
      !std::isfinite(number))
    return std::nullopt;
  return json::Value(number);
}

/// `name(key=value, ...)`; false when the call is malformed or incomplete.
bool PythonCall(PythonCursor& c, ToolCall* call) {
  c.Space();
  const auto open = c.text.find('(', c.at);
  if (open == std::string_view::npos)
    return false;
  const auto name = Trim(c.text.substr(c.at, open - c.at));
  if (!IsCallName(name))
    return false;
  call->name = std::string(name);
  c.at = open + 1;
  while (!c.Eat(')')) {
    const auto equals = c.text.find('=', c.at);
    if (equals == std::string_view::npos)
      return false;
    const std::string key(Trim(c.text.substr(c.at, equals - c.at)));
    if (!IsCallName(key) ||
        std::ranges::any_of(call->arguments,
                            [&](const auto& arg) { return arg.name == key; }))
      return false;
    c.at = equals + 1;
    c.Space();
    const bool is_string = c.Peek('\'') || c.Peek('"');
    auto value = PythonValue(c, 0);
    if (!value)
      return false;
    call->arguments.push_back(
        {.name = key,
         .value = is_string ? value->get_str() : value->dump(),
         .is_string = is_string});
    if (!c.Eat(',') && !c.Peek(')'))
      return false;
  }
  return true;
}

void ParsePythonicCalls(std::string_view text, std::vector<ToolCall>* calls) {
  std::size_t cursor = 0;
  while ((cursor = text.find(kLfmCallStart, cursor)) !=
         std::string_view::npos) {
    cursor += kLfmCallStart.size();
    PythonCursor c{.text = text.substr(cursor)};
    // The model writes a list (parallel calls); accept one bare call too.
    const bool list = c.Eat('[');
    do {
      ToolCall call;
      // An interrupted call is dropped; the complete ones before it remain.
      if (!PythonCall(c, &call))
        break;
      calls->push_back(std::move(call));
    } while (list && c.Eat(','));
  }
}

/// Python literal for a structured argument, as LFM2.5's template writes it:
/// quoted strings, True/False/None, and containers through `tojson`.
void AppendPythonArgument(std::string& out,
                          const ChatMessage::ToolArgument& arg) {
  if (arg.is_string) {
    out += "'";
    for (const char ch : arg.value) {
      switch (ch) {
        case '\\':
          out += "\\\\";
          break;
        case '\'':
          out += "\\'";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        default:
          out += ch;
      }
    }
    out += "'";
    return;
  }
  const auto value = TryParseJson(arg.value);
  if (!value)
    out += arg.value;
  else if (value->is_bool())
    out += value->as_bool() ? "True" : "False";
  else if (value->is_null())
    out += "None";
  else if (value->is_array() || value->is_object())
    out += tokenization::PythonJsonSpacing(value->dump());
  else
    out += arg.value;
}

// ---- Adapters.

const std::array<std::string_view, 1> kQwenMarkers{"<tool_call>"};

class QwenProtocol final : public ChatProtocol {
public:
  explicit QwenProtocol(bool qwen35) : qwen35_(qwen35) {}

  [[nodiscard]] std::string RenderText(
      const ChatRequest& request,
      std::vector<std::size_t>* image_offsets) const override {
    if (image_offsets == nullptr && HasImages(request))
      throw std::invalid_argument(
          "image messages require the model's vision prompt preparation");
    std::string error;
    auto text = tokenization::QwenChatTemplate::Render(
        request.messages, PromptTools(request), Options(request), &error,
        image_offsets);
    if (!text)
      throw std::invalid_argument(error);
    return std::move(*text);
  }

  [[nodiscard]] std::size_t CachePrefix(
      const ChatRequest& request, std::span<const TokenId> prompt,
      const PromptTokenizer& tokenize) const override {
    // Agent clients may discard an interrupted assistant entirely and append
    // the next user turn directly after tool results. Preserve a checkpoint
    // before the generation suffix even when reasoning itself is retained.
    const auto options = Options(request);
    if (options.preserve_thinking && request.tools.empty())
      return 0;
    return StablePromptPrefix(
        prompt,
        tokenize(tokenization::GenerationPrompt(options.enable_thinking)));
  }

  [[nodiscard]] OutputStart ParserStart(
      const ChatRequest& request) const override {
    return Options(request).enable_thinking ? OutputStart::kReasoning
                                            : OutputStart::kContent;
  }

private:
  /// Qwen3.5 has the same ChatML, thinking and tool format as Qwen3.8 but no
  /// reasoning-effort instruction, which equals the native medium effort.
  [[nodiscard]] tokenization::ChatTemplateOptions Options(
      const ChatRequest& request) const {
    auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                        request.add_vision_id);
    if (qwen35_)
      options.reasoning_effort = tokenization::QwenReasoningEffort::kMedium;
    options.require_tool_call =
        request.tool_choice == ChatRequest::ToolChoice::kRequired;
    return options;
  }

  [[nodiscard]] std::optional<ToolCallSyntax> ToolSyntax()
      const noexcept override {
    return ToolCallSyntax{.marker = "<tool_call>",
                          .open = "<tool_call>\n<function=",
                          .name_end = ">\n",
                          .close = "\n</function>\n</tool_call>",
                          .parameter_style = "qwen_xml"};
  }
  [[nodiscard]] std::string_view ReasoningSeparator() const noexcept override {
    return "\n\n";
  }

  [[nodiscard]] std::span<const std::string_view> ToolMarkers()
      const noexcept override {
    return kQwenMarkers;
  }
  void ParseToolCalls(std::string_view text, std::span<const ChatTool> tools,
                      std::vector<ToolCall>* calls) const override {
    ParseQwenCalls(text, tools, calls);
  }

  bool qwen35_;
};

/// LFM2 ChatML, as LFM2.5's chat_template.jinja renders it: BOS, a system
/// block that also lists the tools, thinking kept only on turns after the
/// last user message, Pythonic tool calls, and a generation prompt that opens
/// <think>: the model always reasons, whatever the request asks.
class LfmProtocol final : public ChatProtocol {
public:
  [[nodiscard]] std::string RenderText(
      const ChatRequest& request, std::vector<std::size_t>*) const override {
    if (HasImages(request))
      throw std::invalid_argument("lfm2moe serves text only");
    const auto& messages = request.messages;
    std::string out = "<|startoftext|>";
    std::string system;
    std::size_t first = 0;
    if (!messages.empty() && (messages[0].role == ChatRole::kSystem ||
                              messages[0].role == ChatRole::kDeveloper)) {
      system = messages[0].content;
      first = 1;
    }
    const auto tools = PromptTools(request);
    if (!tools.empty()) {
      system += system.empty() ? "List of tools: [" : "\nList of tools: [";
      for (std::size_t i = 0; i < tools.size(); ++i) {
        if (i != 0)
          system += ", ";
        system += tokenization::PythonJsonSpacing(Definition(tools[i]));
      }
      system += "]";
    }
    if (!system.empty())
      out += "<|im_start|>system\n" + system + "<|im_end|>\n";
    std::size_t last_user = messages.size();
    for (std::size_t i = first; i < messages.size(); ++i)
      if (messages[i].role == ChatRole::kUser)
        last_user = i;
    for (std::size_t i = first; i < messages.size(); ++i) {
      const auto& m = messages[i];
      out += "<|im_start|>";
      out += tokenization::ToString(m.role);
      out += "\n";
      if (m.role != ChatRole::kAssistant) {
        out += m.content + "<|im_end|>\n";
        continue;
      }
      const bool keep = last_user == messages.size() || i > last_user;
      std::string_view content = m.content;
      if (keep && !m.thought.empty())
        out += "<think>" + m.thought + "</think>";
      if (const auto end = content.rfind(kThinkEnd);
          !keep && end != std::string_view::npos)
        content = Trim(content.substr(end + kThinkEnd.size()));
      out += content;
      if (!m.tool_calls.empty()) {
        out += kLfmCallStart;
        out += "[";
        for (std::size_t c = 0; c < m.tool_calls.size(); ++c) {
          const auto& call = m.tool_calls[c];
          if (c != 0)
            out += ", ";
          out += call.name + "(";
          for (std::size_t a = 0; a < call.arguments.size(); ++a) {
            if (a != 0)
              out += ", ";
            out += call.arguments[a].name + "=";
            AppendPythonArgument(out, call.arguments[a]);
          }
          out += ")";
        }
        out += "]<|tool_call_end|>";
      }
      out += "<|im_end|>\n";
    }
    return out + std::string(kLfmGeneration);
  }

  [[nodiscard]] std::size_t CachePrefix(
      const ChatRequest&, std::span<const TokenId> prompt,
      const PromptTokenizer& tokenize) const override {
    return StablePromptPrefix(prompt, tokenize(kLfmGeneration));
  }

  [[nodiscard]] OutputStart ParserStart(const ChatRequest&) const override {
    return OutputStart::kReasoning;
  }

private:
  /// The tool object the template's `tojson` sees.
  static std::string Definition(const ChatTool& tool) {
    if (!tool.definition_json.empty())
      return tool.definition_json;
    auto function = json::Value::object();
    function["name"] = tool.name;
    function["description"] = tool.description;
    function["parameters"] =
        TryParseJson(tool.parameters_json).value_or(json::Value::object());
    auto definition = json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function);
    return definition.dump();
  }

  [[nodiscard]] std::span<const std::string_view> ToolMarkers()
      const noexcept override {
    static constexpr std::array<std::string_view, 1> kMarkers{kLfmCallStart};
    return kMarkers;
  }
  void ParseToolCalls(std::string_view text, std::span<const ChatTool>,
                      std::vector<ToolCall>* calls) const override {
    ParsePythonicCalls(text, calls);
  }
};

class DeepSeekProtocol final : public ChatProtocol {
public:
  [[nodiscard]] std::string RenderText(
      const ChatRequest& request, std::vector<std::size_t>*) const override {
    namespace ds4 = models::deepseek_v4_flash;
    if (HasImages(request))
      throw std::invalid_argument("model does not support image input");
    std::vector<ds4::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      ds4::ChatMessage converted{
          .role = std::string(RoleName(message.role)),
          .content = message.content,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        ds4::ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<ds4::ChatTool> tools;
    for (const auto& tool : PromptTools(request)) {
      tools.push_back({
          .name = tool.name,
          .description = tool.description,
          .parameters_json = tool.parameters_json,
          .definition_json =
              tool.definition_json.empty()
                  ? std::string{}
                  : json::parse(tool.definition_json)["function"].dump(),
      });
    }
    return ds4::RenderChat(
        messages, tools,
        ds4::ChatTemplateOptions{
            .enable_thinking = Thinking(request),
            .reasoning_effort =
                request.reasoning.effort.value_or(ReasoningEffort::kLow),
            .preserve_thinking =
                request.reasoning.preserve_thinking.value_or(false),
            .tools_present = !PromptTools(request).empty(),
            .require_tool_call =
                request.tool_choice == ChatRequest::ToolChoice::kRequired,
        });
  }

  [[nodiscard]] std::size_t CachePrefix(
      const ChatRequest& request, std::span<const TokenId> prompt,
      const PromptTokenizer& tokenize) const override {
    // DeepSeek joins adjacent user/tool messages into one user block. A new
    // user turn after a discarded assistant can therefore also change the
    // final BPE token before the generation suffix (e.g. ">\n\n"). Find the
    // exact stable frontier by rendering that continuation instead of
    // assuming the suffix alone is the only changed part of the prompt.
    auto continuation = request;
    continuation.messages.emplace_back(ChatRole::kUser, "");
    const auto continued = tokenize(RenderText(continuation, nullptr));
    return static_cast<std::size_t>(
        std::ranges::mismatch(prompt, continued).in1 - prompt.begin());
  }

  [[nodiscard]] OutputStart ParserStart(
      const ChatRequest& request) const override {
    return Thinking(request) ? OutputStart::kReasoning : OutputStart::kContent;
  }

private:
  static bool Thinking(const ChatRequest& request) {
    return request.reasoning.enabled.value_or(false);
  }

  static std::string_view RoleName(ChatRole role) {
    // The DeepSeek encoder keeps developer messages distinct.
    return role == ChatRole::kDeveloper ? "developer"
                                        : tokenization::ToString(role);
  }

  [[nodiscard]] std::span<const std::string_view> ToolMarkers()
      const noexcept override {
    static constexpr std::array<std::string_view, 6> kMarkers{
        "<｜DSML｜tool_calls｜>", "<｜DSML｜tool_calls>",
        "<DSML｜tool_calls｜>",   "<DSML｜tool_calls>",
        "<tool_calls｜>",         "<tool_calls>",
    };
    return kMarkers;
  }
  void ParseToolCalls(std::string_view text, std::span<const ChatTool>,
                      std::vector<ToolCall>* calls) const override {
    ParseDsmlCalls(text, calls);
  }
};

/// MiMo's template writes Qwen-style XML tool calls without newlines; with
/// thinking on its generation prompt leaves <think> to the model.
class MimoProtocol final : public ChatProtocol {
public:
  using Template = models::mimo_v26_flash::ChatTemplate;

  [[nodiscard]] std::string RenderText(
      const ChatRequest& request, std::vector<std::size_t>*) const override {
    if (HasImages(request))
      throw std::invalid_argument("MiMo serving is text-only");
    return Template::Render(request.messages, PromptTools(request),
                            Thinking(request));
  }

  [[nodiscard]] std::size_t CachePrefix(
      const ChatRequest& request, std::span<const TokenId> prompt,
      const PromptTokenizer& tokenize) const override {
    // Agents drop earlier reasoning, so the next turn diverges inside the
    // open assistant turn: checkpoint just before it.
    return StablePromptPrefix(
        prompt, tokenize(Template::GenerationPrompt(Thinking(request))));
  }

  [[nodiscard]] OutputStart ParserStart(
      const ChatRequest& request) const override {
    return Thinking(request) ? OutputStart::kAuto : OutputStart::kContent;
  }

private:
  static bool Thinking(const ChatRequest& request) {
    return request.reasoning.enabled.value_or(true);
  }

  [[nodiscard]] std::optional<ToolCallSyntax> ToolSyntax()
      const noexcept override {
    return ToolCallSyntax{.marker = "<tool_call>",
                          .open = "<tool_call><function=",
                          .name_end = ">",
                          .close = "</function></tool_call>",
                          .parameter_style = "qwen_xml"};
  }

  [[nodiscard]] std::span<const std::string_view> ToolMarkers()
      const noexcept override {
    return kQwenMarkers;
  }
  void ParseToolCalls(std::string_view text, std::span<const ChatTool> tools,
                      std::vector<ToolCall>* calls) const override {
    ParseQwenCalls(text, tools, calls);
  }
};

bool Send(const OutputParser::Emit& emit, std::string_view text,
          bool reasoning) {
  return !emit || emit(text, reasoning);
}

}  // namespace

RenderedPrompt ChatProtocol::Render(const ChatRequest& request,
                                    const PromptTokenizer& tokenize) const {
  RenderedPrompt prompt{.tokens = tokenize(RenderText(request))};
  prompt.cache_prefix_tokens = CachePrefix(request, prompt.tokens, tokenize);
  return prompt;
}

OutputParser ChatProtocol::NewParser(const ChatRequest& request) const {
  return OutputParser(*this, request);
}

const ChatProtocol& ProtocolFor(TemplateFamily family) {
  static const QwenProtocol qwen38(false);
  static const QwenProtocol qwen35(true);
  static const LfmProtocol lfm;
  static const DeepSeekProtocol deepseek;
  static const MimoProtocol mimo;
  switch (family) {
    case TemplateFamily::kQwen38:
      return qwen38;
    case TemplateFamily::kQwen35:
      return qwen35;
    case TemplateFamily::kLfm2:
      return lfm;
    case TemplateFamily::kDeepSeekV4:
      return deepseek;
    case TemplateFamily::kMimo:
      return mimo;
  }
  return qwen38;
}

OutputParser::OutputParser(const ChatProtocol& protocol,
                           const ChatRequest& request)
    : protocol_(&protocol),
      markers_(protocol.ToolMarkers()),
      start_(protocol.ParserStart(request)),
      tools_(request.tools),
      choice_(request.tool_choice),
      function_(request.tool_choice_function) {
  if (start_ == OutputStart::kReasoning)
    state_ = State::kThinking;
  else if (start_ == OutputStart::kContent)
    state_ = State::kContent;
}

std::size_t OutputParser::EarliestMarker(std::string_view text) const {
  std::size_t earliest = std::string_view::npos;
  for (const auto candidate : markers_)
    earliest = std::min(earliest, text.find(candidate));
  return earliest;
}

std::size_t OutputParser::HeldMarkerPrefix(std::string_view text) const {
  std::size_t maximum_marker = 0;
  for (const auto marker : markers_)
    maximum_marker = std::max(maximum_marker, marker.size());
  const std::size_t maximum =
      std::min(text.size(), maximum_marker > 0 ? maximum_marker - 1 : 0);
  for (std::size_t length = maximum; length > 0; --length) {
    const std::string_view suffix = text.substr(text.size() - length);
    if (std::ranges::any_of(markers_, [&](std::string_view marker) {
          return marker.starts_with(suffix);
        }))
      return length;
  }
  return 0;
}

bool OutputParser::Push(std::string_view bytes, const Emit& emit) {
  return Feed(bytes, false, emit);
}

bool OutputParser::Feed(std::string_view bytes, bool final, const Emit& emit) {
  const auto piece = decoder_.Push(bytes, final);
  raw_.append(piece);
  if (tool_mode_) {
    hidden_.append(piece);
    return true;
  }
  pending_.append(piece);

  if (state_ == State::kInitial) {
    std::string_view view = pending_;
    while (!view.empty() &&
           std::isspace(static_cast<unsigned char>(view.front())) != 0)
      view.remove_prefix(1);
    if (view.empty())
      return true;
    if (view.starts_with(kThinkStart)) {
      // Text after <think> in the same piece is reasoning too.
      pending_.erase(0, pending_.size() - view.size() + kThinkStart.size());
      state_ = State::kThinking;
    } else if (kThinkStart.starts_with(view)) {
      return true;
    } else {
      state_ = State::kContent;
    }
  }

  if (state_ == State::kThinking) {
    const std::size_t end_pos = pending_.find(kThinkEnd);
    const auto marker = EarliestMarker(pending_);
    if (marker < end_pos) {
      if (marker > 0 && !Send(emit, pending_.substr(0, marker), true))
        return false;
      hidden_ = pending_.substr(marker);
      pending_.clear();
      tool_mode_ = true;
      return true;
    }
    if (end_pos == std::string::npos) {
      std::size_t held = HeldMarkerPrefix(pending_);
      for (std::size_t len = std::min(pending_.size(), kThinkEnd.size() - 1);
           len > 0; --len) {
        if (kThinkEnd.starts_with(pending_.substr(pending_.size() - len))) {
          held = std::max(held, len);
          break;
        }
      }
      const std::size_t ready = pending_.size() - held;
      if (ready > 0 && !Send(emit, pending_.substr(0, ready), true))
        return false;
      pending_.erase(0, ready);
      return true;
    }
    if (end_pos > 0 && !Send(emit, pending_.substr(0, end_pos), true))
      return false;
    pending_.erase(0, end_pos + kThinkEnd.size());
    state_ = State::kContent;
    trim_reasoning_separator_ = true;
  }

  if (trim_reasoning_separator_) {
    const auto first = pending_.find_first_not_of("\r\n");
    if (first == std::string::npos) {
      pending_.clear();
      return true;
    }
    pending_.erase(0, first);
    trim_reasoning_separator_ = false;
  }
  const std::size_t marker = EarliestMarker(pending_);
  if (marker != std::string::npos) {
    if (marker > 0 && !Send(emit, pending_.substr(0, marker), false))
      return false;
    hidden_ = pending_.substr(marker);
    pending_.clear();
    tool_mode_ = true;
    return true;
  }
  const std::size_t ready = pending_.size() - HeldMarkerPrefix(pending_);
  if (ready > 0 && !Send(emit, pending_.substr(0, ready), false))
    return false;
  pending_.erase(0, ready);
  return true;
}

std::optional<ParsedOutput> OutputParser::Finish(bool enforce_required,
                                                 const Emit& emit) {
  if (!Feed({}, true, emit))
    return std::nullopt;
  bool hide_tool_markup = false;
  auto parsed = Parse(enforce_required, &hide_tool_markup);
  if (!tool_mode_) {
    if (!pending_.empty() && !Send(emit, pending_, state_ == State::kThinking))
      return std::nullopt;
    pending_.clear();
  } else if (!hide_tool_markup && !hidden_.empty() &&
             !Send(emit, hidden_, false)) {
    return std::nullopt;
  }
  return parsed;
}

ParsedOutput OutputParser::Parse(bool enforce_required, bool* hide) const {
  ParsedOutput parsed;
  std::string_view content = raw_;
  *hide = false;
  if (start_ == OutputStart::kReasoning) {
    if (content.starts_with(kThinkStart))
      content.remove_prefix(kThinkStart.size());
    std::size_t think_end = content.find(kThinkEnd);
    if (EarliestMarker(content) < think_end)
      think_end = std::string_view::npos;
    if (think_end == std::string_view::npos) {
      const auto marker = EarliestMarker(content);
      parsed.reasoning = std::string(Trim(content.substr(0, marker)));
      if (marker == std::string_view::npos) {
        if (enforce_required && choice_ == ChatRequest::ToolChoice::kRequired)
          throw TextGenerationError(
              TextGenerationErrorCode::kToolChoiceUnsatisfied,
              "model did not produce a declared tool call");
        return parsed;
      }
      parsed.content = std::string(content.substr(marker));
    } else {
      parsed.reasoning = std::string(Trim(content.substr(0, think_end)));
      content.remove_prefix(think_end + kThinkEnd.size());
      while (!content.empty() &&
             (content.front() == '\n' || content.front() == '\r'))
        content.remove_prefix(1);
      parsed.content = std::string(content);
    }
  } else {
    const std::size_t think_start = content.find(kThinkStart);
    if (think_start != std::string_view::npos) {
      const std::size_t think_content_start = think_start + kThinkStart.size();
      std::size_t think_end = content.find(kThinkEnd, think_content_start);
      const auto marker = EarliestMarker(content.substr(think_content_start));
      if (marker != std::string_view::npos &&
          think_content_start + marker < think_end)
        think_end = std::string_view::npos;
      if (think_end != std::string_view::npos) {
        parsed.reasoning = std::string(Trim(content.substr(
            think_content_start, think_end - think_content_start)));
        std::string_view remaining =
            content.substr(think_end + kThinkEnd.size());
        if (remaining.starts_with("\n"))
          remaining.remove_prefix(1);
        parsed.content = std::string(content.substr(0, think_start)) +
                         std::string(remaining);
      } else {
        const auto remaining = content.substr(think_content_start);
        const auto marker = EarliestMarker(remaining);
        parsed.reasoning = std::string(Trim(remaining.substr(0, marker)));
        parsed.content = std::string(content.substr(0, think_start));
        if (marker != std::string_view::npos)
          parsed.content += remaining.substr(marker);
      }
    } else {
      parsed.content = std::string(content);
    }
  }

  const std::size_t marker = EarliestMarker(parsed.content);
  if (choice_ != ChatRequest::ToolChoice::kNone && !tools_.empty() &&
      marker != std::string_view::npos) {
    protocol_->ParseToolCalls(std::string_view(parsed.content).substr(marker),
                              tools_, &parsed.tool_calls);
    std::erase_if(parsed.tool_calls, [&](const auto& call) {
      return std::ranges::none_of(
          tools_, [&](const auto& tool) { return tool.name == call.name; });
    });
    // An explicit stop can interrupt a call before its closing tags. Keep
    // complete calls, but do not expose an unfinished call as ordinary text.
    if (!parsed.tool_calls.empty() || !enforce_required) {
      parsed.content.resize(marker);
      *hide = true;
    }
  }
  if (enforce_required && choice_ == ChatRequest::ToolChoice::kRequired &&
      (parsed.tool_calls.empty() ||
       (!function_.empty() &&
        std::ranges::none_of(parsed.tool_calls, [&](const auto& call) {
          return call.name == function_;
        }))))
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not produce a declared tool call");
  return parsed;
}

}  // namespace gufo::server
