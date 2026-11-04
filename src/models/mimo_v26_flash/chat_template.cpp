#include "src/models/mimo_v26_flash/chat_template.hpp"

#include <cstdint>

#include "src/core/crypto/sha256.hpp"

namespace gufo::models::mimo_v26_flash {
namespace {

using tokenization::ChatMessage;
using tokenization::ChatRole;
using tokenization::ChatTool;

/// Jinja `tojson` separators (", " and ": ") applied to compact JSON.
std::string PythonJsonSpacing(std::string_view value) {
  std::string out;
  out.reserve(value.size() + value.size() / 8);
  bool in_string = false, escaped = false;
  for (const char c : value) {
    out.push_back(c);
    if (in_string) {
      if (escaped)
        escaped = false;
      else if (c == '\\')
        escaped = true;
      else if (c == '"')
        in_string = false;
      continue;
    }
    if (c == '"')
      in_string = true;
    else if (c == ':' || c == ',')
      out.push_back(' ');
  }
  return out;
}

void AppendJsonString(std::string& out, std::string_view value) {
  out.push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        out.append("\\\"");
        break;
      case '\\':
        out.append("\\\\");
        break;
      case '\b':
        out.append("\\b");
        break;
      case '\f':
        out.append("\\f");
        break;
      case '\n':
        out.append("\\n");
        break;
      case '\r':
        out.append("\\r");
        break;
      case '\t':
        out.append("\\t");
        break;
      default:
        out.push_back(static_cast<char>(c));
    }
  }
  out.push_back('"');
}

void AppendTools(std::string& out, std::span<const ChatTool> tools) {
  out.append("You are provided with the following tools:\n\n<tools>");
  for (const auto& tool : tools) {
    out.push_back('\n');
    if (!tool.definition_json.empty()) {
      out.append(PythonJsonSpacing(tool.definition_json));
      continue;
    }
    out.append("{\"type\": \"function\", \"function\": {\"name\": ");
    AppendJsonString(out, tool.name);
    out.append(", \"description\": ");
    AppendJsonString(out, tool.description);
    out.append(", \"parameters\": ");
    out.append(tool.parameters_json.empty()
                   ? "{}"
                   : PythonJsonSpacing(tool.parameters_json));
    out.append("}}");
  }
  out.append("\n</tools>");
}

}  // namespace

bool ChatTemplate::Validate(const core::GgufReader& reader,
                            std::string* error) {
  const auto text = reader.GetMetadataString("tokenizer.chat_template");
  if (!text) {
    if (error)
      *error = "MiMo GGUF has no tokenizer.chat_template";
    return false;
  }
  const auto sha = crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text->data()), text->size()));
  if (sha != Sha256()) {
    if (error)
      *error = "MiMo chat template SHA-256 " + sha + " is not the pinned " +
               std::string(Sha256());
    return false;
  }
  return true;
}

std::string_view ChatTemplate::GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<|im_start|>assistant\n"
                         : "<|im_start|>assistant\n<think></think>";
}

std::string ChatTemplate::Render(std::span<const ChatMessage> messages,
                                 std::span<const ChatTool> tools,
                                 bool enable_thinking,
                                 bool add_generation_prompt) {
  std::string out;
  if (!tools.empty()) {
    out.append("<|im_start|>system\n");
    AppendTools(out, tools);
    out.append("<|im_end|>");
  }
  for (const auto& m : messages) {
    if (m.role == ChatRole::kAssistant) {
      out.append("<|im_start|>assistant\n<think>");
      out.append(m.thought);
      out.append("</think>");
      out.append(m.content);
      // Structured arguments render as parameters (the model's own format);
      // a string value is written raw and anything else as tojson.
      for (const auto& call : m.tool_calls) {
        out.append("<tool_call><function=");
        out.append(call.name);
        out.push_back('>');
        for (const auto& arg : call.arguments) {
          out.append("<parameter=");
          out.append(arg.name);
          out.push_back('>');
          out.append(arg.is_string ? arg.value : PythonJsonSpacing(arg.value));
          out.append("</parameter>");
        }
        out.append("</function></tool_call>");
      }
      out.append("<|im_end|>");
      continue;
    }
    out.append("<|im_start|>");
    // The template writes message.role verbatim (Qwen's maps developer to
    // system).
    out.append(m.role == ChatRole::kDeveloper ? std::string_view("developer")
                                              : tokenization::ToString(m.role));
    out.push_back('\n');
    out.append(m.content);
    out.append("<|im_end|>");
  }
  if (add_generation_prompt)
    out.append(GenerationPrompt(enable_thinking));
  return out;
}

}  // namespace gufo::models::mimo_v26_flash
