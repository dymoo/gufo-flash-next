#include "src/cli/serve/openai_chat.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/chat_protocol.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/grammar/grammar.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

struct ParsedChatRequest {
  ChatRequest chat;
  std::string model;
  std::size_t max_tokens{0};
  sampling::SamplingConfig sampling;
  bool stream{false};
  bool include_usage{false};
};

long long Now() {
  return static_cast<long long>(std::time(nullptr));
}

std::string RandomId(std::string_view prefix) {
  static constexpr std::string_view kCharacters =
      "abcdefghijklmnopqrstuvwxyz0123456789";
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<std::size_t> distribution{
      0, kCharacters.size() - 1};

  std::string result(prefix);
  result.reserve(prefix.size() + 20);
  for (int index = 0; index < 20; ++index) {
    result.push_back(kCharacters[distribution(generator)]);
  }
  return result;
}

HttpResponse Error(int status, const char* reason, std::string message,
                   const char* code) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = code;
  response["error"] = std::move(error);
  return {.status = status, .reason = reason, .body = response.dump()};
}

const char* StatusReason(int status) noexcept {
  switch (status) {
    case 408:
      return "Request Timeout";
    case 429:
      return "Too Many Requests";
    case 503:
      return "Service Unavailable";
    case 502:
      return "Bad Gateway";
    default:
      return "Internal Server Error";
  }
}

HttpResponse GenerationError(const TextGenerationError& exception) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = exception.what();
  error["type"] = "server_error";
  error["code"] = exception.stable_code();
  response["error"] = std::move(error);
  HttpResponse output{
      .status = exception.http_status(),
      .reason = StatusReason(exception.http_status()),
      .body = response.dump(),
      .headers = {},
      .streaming_body = {},
  };
  if (exception.retryable()) {
    output.headers.emplace_back("Retry-After", "1");
  }
  return output;
}

bool IsKnownRole(std::string_view role) {
  return role == "system" || role == "developer" || role == "user" ||
         role == "assistant" || role == "tool";
}

tokenization::ChatRole ParseRole(std::string_view role) {
  if (role == "system") {
    return tokenization::ChatRole::kSystem;
  }
  if (role == "developer") {
    return tokenization::ChatRole::kDeveloper;
  }
  if (role == "assistant") {
    return tokenization::ChatRole::kAssistant;
  }
  if (role == "tool") {
    return tokenization::ChatRole::kTool;
  }
  return tokenization::ChatRole::kUser;
}

bool ParseContent(const json::Value* content,
                  tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error) {
  auto* output = &message->content;
  if (content == nullptr || content->is_null()) {
    return true;
  }
  if (content->is_string()) {
    *output = content->get_str();
    return true;
  }
  if (!content->is_array()) {
    *error = "message content must be a string, null, or content-part array";
    return false;
  }

  for (const auto& part : content->items()) {
    if (!part.is_object()) {
      *error = "message content parts must be objects";
      return false;
    }
    const std::string type = part.member_str("type", "text");
    if (type == "image_url") {
      const auto* image = part.find("image_url");
      const auto* url =
          image != nullptr && image->is_object() ? image->find("url") : nullptr;
      if (message->role != tokenization::ChatRole::kUser || url == nullptr ||
          !url->is_string() || message->images.size() >= 16) {
        *error =
            "image_url requires a user message and a string URL (at most 16 "
            "images)";
        return false;
      }
      // Resolution is model-owned; accept only the automatic policy rather
      // than silently ignoring a requested low/high preprocessing policy.
      const auto* detail = image->find("detail");
      if (detail != nullptr &&
          (!detail->is_string() || detail->get_str() != "auto")) {
        *error = "image_url.detail supports only 'auto'";
        return false;
      }
      try {
        message->images.push_back(
            {output->size(), std::make_shared<const std::vector<std::uint8_t>>(
                                 core::ReadImageUrl(url->get_str(), budget))});
      } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
      }
      continue;
    }
    if (type != "text" && type != "input_text") {
      *error = "message content parts must use text or image_url";
      return false;
    }
    const json::Value* text = part.find("text");
    if (text == nullptr || !text->is_string()) {
      *error = "text message content parts require a string 'text'";
      return false;
    }
    output->append(text->get_str());
  }
  return true;
}

bool ParseArguments(std::string_view arguments,
                    std::vector<tokenization::ChatMessage::ToolArgument>* out,
                    std::string* error) {
  json::Value parsed;
  try {
    parsed = json::parse(arguments);
  } catch (const std::exception& exception) {
    *error =
        std::string("tool arguments are not valid JSON: ") + exception.what();
    return false;
  }
  if (!parsed.is_object()) {
    *error = "tool arguments must encode a JSON object";
    return false;
  }
  for (const auto& [name, value] : parsed.members()) {
    out->push_back({
        .name = name,
        .value = value.is_string() ? value.get_str() : value.dump(),
        .is_string = value.is_string(),
    });
  }
  return true;
}

bool ParseMessage(const json::Value& value, tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error) {
  if (!value.is_object()) {
    *error = "each message must be an object";
    return false;
  }
  const std::string role = value.member_str("role");
  if (!IsKnownRole(role)) {
    *error = "message role must be system, developer, user, assistant, or tool";
    return false;
  }
  message->role = ParseRole(role);
  message->name = value.member_str("name");
  message->tool_call_id = value.member_str("tool_call_id");
  if (!ParseContent(value.find("content"), message, budget, error)) {
    return false;
  }
  if (const json::Value* reasoning = value.find("reasoning_content");
      reasoning != nullptr && !reasoning->is_null()) {
    if (message->role != tokenization::ChatRole::kAssistant ||
        !reasoning->is_string()) {
      *error =
          "'reasoning_content' is only valid as a string on assistant "
          "messages";
      return false;
    }
    message->thought = reasoning->get_str();
  }

  const json::Value* tool_calls = value.find("tool_calls");
  if (tool_calls == nullptr) {
    return true;
  }
  if (message->role != tokenization::ChatRole::kAssistant ||
      !tool_calls->is_array()) {
    *error = "'tool_calls' is only valid as an array on assistant messages";
    return false;
  }
  for (const auto& item : tool_calls->items()) {
    if (!item.is_object() ||
        item.member_str("type", "function") != "function") {
      *error = "only function tool calls are supported";
      return false;
    }
    const json::Value* function = item.find("function");
    if (function == nullptr || !function->is_object()) {
      *error = "assistant tool calls require a function object";
      return false;
    }
    tokenization::ChatMessage::ToolCall call;
    call.id = item.member_str("id");
    call.name = function->member_str("name");
    const std::string arguments = function->member_str("arguments");
    if (call.name.empty() || arguments.empty() ||
        !ParseArguments(arguments, &call.arguments, error)) {
      if (error->empty()) {
        *error = "assistant tool calls require a name and JSON arguments";
      }
      return false;
    }
    message->tool_calls.push_back(std::move(call));
  }
  return true;
}

bool ParseTools(const json::Value* tools,
                std::vector<tokenization::ChatTool>* output,
                std::string* error) {
  if (tools == nullptr || tools->is_null()) {
    return true;
  }
  if (!tools->is_array()) {
    *error = "'tools' must be an array";
    return false;
  }
  for (const auto& item : tools->items()) {
    if (!item.is_object()) {
      *error = "'tools' entries must be objects";
      return false;
    }
    if (item.member_str("type") != "function") {
      *error = "only function tools are supported";
      return false;
    }

    const json::Value* function = item.find("function");
    if (function != nullptr && !function->is_object()) {
      *error = "'function' must be an object";
      return false;
    }
    const json::Value* src = function != nullptr ? function : &item;
    // OpenAI uses "parameters"; some agent clients send parametersJsonSchema.
    const json::Value* params_src = src->find("parameters");
    if (params_src == nullptr || params_src->is_null()) {
      params_src = src->find("parametersJsonSchema");
    }

    tokenization::ChatTool tool;
    tool.name = src->member_str("name");
    tool.description = src->member_str("description");
    if (tool.name.empty()) {
      *error = "function tools require a nonempty name";
      return false;
    }
    if (params_src != nullptr && !params_src->is_null() &&
        !params_src->is_object()) {
      *error = "function tools require an object parameters schema";
      return false;
    }
    // Preserve nested definitions and their field order. For flat tools, move
    // the complete function body (including strict) under "function".
    json::Value function_obj = json::Value::object();
    for (const auto& [key, value] : src->members()) {
      if (key == "parametersJsonSchema" ||
          (function == nullptr && key == "type")) {
        continue;
      }
      function_obj.append_member(key, value);
    }
    function_obj["parameters"] =
        params_src != nullptr && params_src->is_object()
            ? *params_src
            : json::Value::object();
    tool.parameters_json = function_obj.find("parameters")->dump();
    json::Value definition = function != nullptr ? item : json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function_obj);
    tool.definition_json = definition.dump();
    output->push_back(std::move(tool));
  }
  return true;
}

bool ParseToolChoice(const json::Value* value, ParsedChatRequest* request,
                     std::string* error) {
  if (value == nullptr || value->is_null()) {
    return true;
  }
  if (value->is_string()) {
    const std::string choice = value->get_str();
    if (choice == "auto") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kAuto;
      return true;
    }
    if (choice == "none") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kNone;
      return true;
    }
    if (choice == "required") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kRequired;
      return true;
    }
  }
  // Named choice: {"type": "function", "function": {"name": "..."}}.
  if (value->is_object() && value->member_str("type") == "function") {
    const json::Value* function = value->find("function");
    const std::string name = function != nullptr && function->is_object()
                                 ? function->member_str("name")
                                 : std::string{};
    if (!name.empty()) {
      request->chat.tool_choice = ChatRequest::ToolChoice::kRequired;
      request->chat.tool_choice_function = name;
      return true;
    }
  }
  *error = "'tool_choice' must be auto, none, required, or a named function";
  return false;
}

/// OpenAI `response_format`: text, json_object, or json_schema with a
/// `json_schema.schema` object (`name`, `description` and `strict` are
/// accepted; both strict modes enforce the schema).
std::optional<std::string> ParseResponseFormat(const json::Value* value,
                                               ResponseFormat* format) {
  if (value == nullptr || value->is_null())
    return std::nullopt;
  if (!value->is_object())
    return "'response_format' must be an object";
  const std::string type = value->member_str("type");
  if (type == "text")
    return std::nullopt;
  if (type == "json_object") {
    format->type = ResponseFormat::Type::kJsonObject;
    return std::nullopt;
  }
  if (type != "json_schema")
    return "'response_format.type' must be text, json_object, or json_schema";
  const json::Value* spec = value->find("json_schema");
  if (spec == nullptr || !spec->is_object())
    return "'response_format.json_schema' must be an object";
  const json::Value* schema = spec->find("schema");
  if (schema == nullptr || !schema->is_object())
    return "'response_format.json_schema.schema' must be a JSON Schema object";
  if (const json::Value* strict = spec->find("strict");
      strict != nullptr && !strict->is_null() && !strict->is_bool())
    return "'response_format.json_schema.strict' must be a boolean";
  if (const json::Value* name = spec->find("name");
      name != nullptr && !name->is_null() && !name->is_string())
    return "'response_format.json_schema.name' must be a string";
  format->type = ResponseFormat::Type::kJsonSchema;
  format->schema = schema->dump();
  return std::nullopt;
}

std::optional<ReasoningEffort> ParseReasoningEffortName(
    std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

bool AssignReasoningEnabled(ReasoningOptions* options, bool enabled,
                            std::string* error) {
  if (options->enabled.has_value() && *options->enabled != enabled) {
    *error = "reasoning controls disagree about whether thinking is enabled";
    return false;
  }
  options->enabled = enabled;
  return true;
}

bool AssignReasoningEffort(ReasoningOptions* options, std::string_view value,
                           std::string* error, bool enable_thinking = true) {
  if (value == "off" || value == "none") {
    return AssignReasoningEnabled(options, false, error);
  }
  const auto effort = ParseReasoningEffortName(value);
  if (!effort.has_value()) {
    *error =
        "reasoning_effort must be off, minimal, low, medium, high, xhigh, or "
        "max";
    return false;
  }
  if (options->effort.has_value() && options->effort != effort) {
    *error = "top-level and chat_template_kwargs reasoning_effort disagree";
    return false;
  }
  options->effort = effort;
  return !enable_thinking || AssignReasoningEnabled(options, true, error);
}

bool ParseReasoningOptions(const json::Value& body, ReasoningOptions* options,
                           std::string* error) {
  if (const json::Value* thinking = body.find("thinking")) {
    if (!thinking->is_object()) {
      *error = "'thinking' must be an object";
      return false;
    }
    const json::Value* type = thinking->find("type");
    if (type == nullptr || !type->is_string()) {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    const std::string value = type->get_str();
    if (value != "enabled" && value != "disabled") {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    if (!AssignReasoningEnabled(options, value == "enabled", error)) {
      return false;
    }
  }

  if (const json::Value* effort = body.find("reasoning_effort");
      effort != nullptr && !effort->is_null()) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error)) {
      if (error->empty()) {
        *error = "'reasoning_effort' must be a string";
      }
      return false;
    }
  }

  const json::Value* kwargs = body.find("chat_template_kwargs");
  if (kwargs == nullptr) {
    return true;
  }
  if (!kwargs->is_object()) {
    *error = "'chat_template_kwargs' must be an object";
    return false;
  }
  if (const json::Value* enabled = kwargs->find("enable_thinking")) {
    if (!enabled->is_bool() ||
        !AssignReasoningEnabled(options, enabled->as_bool(), error)) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.enable_thinking' must be a boolean";
      }
      return false;
    }
  }
  if (const json::Value* mode = kwargs->find("thinking_mode")) {
    if (!mode->is_string()) {
      *error = "'chat_template_kwargs.thinking_mode' must be a string";
      return false;
    }
    const std::string value = mode->get_str();
    if (value != "auto") {
      const bool enabled = value == "thinking" || value == "on";
      if ((!enabled && value != "chat" && value != "off" && value != "none") ||
          !AssignReasoningEnabled(options, enabled, error)) {
        if (error->empty()) {
          *error =
              "'chat_template_kwargs.thinking_mode' must be auto, thinking, "
              "or chat";
        }
        return false;
      }
    }
  }
  if (const json::Value* effort = kwargs->find("reasoning_effort")) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error,
                               options->enabled.value_or(true))) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.reasoning_effort' must be a string";
      }
      return false;
    }
  }
  if (const json::Value* preserve = kwargs->find("preserve_thinking")) {
    if (!preserve->is_bool()) {
      *error = "'chat_template_kwargs.preserve_thinking' must be a boolean";
      return false;
    }
    options->preserve_thinking = preserve->as_bool();
  }
  return true;
}

std::optional<HttpResponse> ParseRequest(const HttpRequest& request,
                                         TextGenerationBackend& backend,
                                         ParsedChatRequest* output) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& exception) {
    return Error(400, "Bad Request", exception.what(), "parse_error");
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }

  output->model = body.member_str("model");
  if (output->model.empty()) {
    return Error(400, "Bad Request", "'model' is required", "missing_model");
  }
  if (!backend.serves_model(output->model)) {
    return Error(404, "Not Found",
                 "model '" + output->model + "' is not served by this process",
                 "model_not_found");
  }
  output->chat.model = output->model;

  output->chat.client_id = request.client_id;
  if (const auto error =
          ParseServiceTier(body, backend.is_flex_model(output->model),
                           &output->chat.service_tier)) {
    return Error(400, "Bad Request", *error, "invalid_service_tier");
  }
  output->chat.no_queue = request.header("X-Gufo-No-Queue") == "1";
  output->chat.queue = request.header("X-Gufo-Queue") == "1";
  if (const auto* cache_prompt = body.find("cache_prompt")) {
    if (!cache_prompt->is_bool()) {
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    }
    output->chat.cache_prompt = cache_prompt->as_bool();
  }
  if (const auto error =
          ParseStopSequences(body.find("stop"), StopSequenceFormat::kOpenAi,
                             &output->chat.stop_sequences)) {
    return Error(400, "Bad Request", *error, "invalid_stop");
  }

  const json::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() || messages->empty()) {
    return Error(400, "Bad Request", "'messages' must be a non-empty array",
                 "missing_messages");
  }
  core::ImageReadBudget image_budget;
  for (const auto& item : messages->items()) {
    tokenization::ChatMessage message;
    std::string parse_error;
    if (!ParseMessage(item, &message, image_budget, &parse_error)) {
      return Error(400, "Bad Request", std::move(parse_error),
                   "invalid_messages");
    }
    output->chat.messages.push_back(std::move(message));
  }

  std::string parse_error;
  if (!ParseTools(body.find("tools"), &output->chat.tools, &parse_error) ||
      !ParseToolChoice(body.find("tool_choice"), output, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error), "invalid_tools");
  }
  if (output->chat.tool_choice == ChatRequest::ToolChoice::kRequired &&
      output->chat.tools.empty()) {
    return Error(400, "Bad Request",
                 "'tool_choice' cannot be required without tools",
                 "invalid_tool_choice");
  }
  if (const auto& name = output->chat.tool_choice_function;
      !name.empty() &&
      std::ranges::none_of(output->chat.tools, [&](const auto& tool) {
        return tool.name == name;
      })) {
    return Error(
        400, "Bad Request",
        "'tool_choice' names function '" + name + "', which is not in 'tools'",
        "invalid_tool_choice");
  }
  if (const json::Value* parallel = body.find("parallel_tool_calls");
      parallel != nullptr && !parallel->is_null()) {
    if (!parallel->is_bool()) {
      return Error(400, "Bad Request",
                   "'parallel_tool_calls' must be a boolean", "invalid_tools");
    }
    output->chat.parallel_tool_calls = parallel->as_bool();
  }
  if (const auto error = ParseResponseFormat(body.find("response_format"),
                                             &output->chat.response_format)) {
    return Error(400, "Bad Request", *error, "invalid_response_format");
  }

  if (!ParseReasoningOptions(body, &output->chat.reasoning, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error),
                 "invalid_reasoning");
  }
  if (const auto* kwargs = body.find("chat_template_kwargs")) {
    if (const auto* vision_id = kwargs->find("add_vision_id")) {
      if (!vision_id->is_bool())
        return Error(400, "Bad Request",
                     "'chat_template_kwargs.add_vision_id' must be a boolean",
                     "invalid_template_options");
      output->chat.add_vision_id = vision_id->as_bool();
    }
  }
  const ReasoningOptions defaults = backend.reasoning_defaults();
  if (!output->chat.reasoning.enabled.has_value()) {
    output->chat.reasoning.enabled = defaults.enabled;
  }
  if (!output->chat.reasoning.effort.has_value()) {
    output->chat.reasoning.effort = defaults.effort;
  }
  if (!output->chat.reasoning.preserve_thinking.has_value()) {
    output->chat.reasoning.preserve_thinking = defaults.preserve_thinking;
  }

  if (const json::Value* stream = body.find("stream");
      stream != nullptr && !stream->is_null()) {
    if (!stream->is_bool()) {
      return Error(400, "Bad Request", "'stream' must be a boolean",
                   "invalid_stream");
    }
    output->stream = stream->as_bool();
  }
  if (const json::Value* options = body.find("stream_options");
      options != nullptr && !options->is_null()) {
    if (!options->is_object()) {
      return Error(400, "Bad Request", "'stream_options' must be an object",
                   "invalid_stream_options");
    }
    if (const json::Value* include_usage = options->find("include_usage")) {
      if (!include_usage->is_bool()) {
        return Error(400, "Bad Request",
                     "'stream_options.include_usage' must be a boolean",
                     "invalid_stream_options");
      }
      output->include_usage = include_usage->as_bool();
    }
  }

  const json::Value* max_tokens = body.find("max_completion_tokens");
  if (max_tokens == nullptr || max_tokens->is_null()) {
    max_tokens = body.find("max_tokens");
  }
  if (max_tokens != nullptr && !max_tokens->is_null()) {
    const double value =
        max_tokens->is_number() ? max_tokens->as_double() : 0.0;
    if (!max_tokens->is_number() || !std::isfinite(value) ||
        std::floor(value) != value || value < 1.0 ||
        value >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
      return Error(400, "Bad Request",
                   "'max_tokens' must be a positive integer",
                   "invalid_max_tokens");
    }
    output->max_tokens = max_tokens->as_size();
  }

  sampling::SamplingConfig parsed_sampling;
  if (const auto sampling_error =
          ParseSamplingConfig(body, output->sampling, &parsed_sampling)) {
    return Error(400, "Bad Request", sampling_error->message,
                 sampling_error->code.c_str());
  }
  if (parsed_sampling.temperature > 2.0F) {
    return Error(400, "Bad Request", "'temperature' must be between 0 and 2",
                 "invalid_temperature");
  }
  output->sampling = parsed_sampling;

  // Evaluation probes: raw-model logprobs and teacher-forced completions.
  if (const json::Value* logprobs = body.find("logprobs");
      logprobs != nullptr && !logprobs->is_null()) {
    if (!logprobs->is_bool()) {
      return Error(400, "Bad Request", "'logprobs' must be a boolean",
                   "invalid_logprobs");
    }
    output->sampling.top_logprobs = logprobs->as_bool() ? 0 : -1;
  }
  if (const json::Value* top = body.find("top_logprobs");
      top != nullptr && !top->is_null()) {
    const double value = top->is_number() ? top->as_double() : -1.0;
    if (value < 0.0 || value > 20.0 || std::floor(value) != value) {
      return Error(400, "Bad Request",
                   "'top_logprobs' must be an integer from 0 to 20",
                   "invalid_logprobs");
    }
    if (output->sampling.top_logprobs < 0) {
      return Error(400, "Bad Request",
                   "'top_logprobs' requires 'logprobs': true",
                   "invalid_logprobs");
    }
    output->sampling.top_logprobs = static_cast<std::int32_t>(value);
  }
  if (const json::Value* forced = body.find("forced_completion");
      forced != nullptr && !forced->is_null()) {
    bool valid = forced->is_string() ? !forced->str().empty()
                                     : forced->is_array() && !forced->empty();
    if (forced->is_string()) {
      output->chat.forced_completion = forced->str();
    } else if (valid) {
      for (const auto& item : forced->items()) {
        const double id = item.is_number() ? item.as_double() : -1.0;
        valid = valid && id >= 0.0 && std::floor(id) == id &&
                id <= std::numeric_limits<std::uint32_t>::max();
        output->sampling.forced_tokens.push_back(
            valid ? static_cast<sampling::TokenId>(id) : 0);
      }
    }
    if (!valid) {
      return Error(400, "Bad Request",
                   "'forced_completion' must be nonempty text or token IDs",
                   "invalid_forced_completion");
    }
  }
  if (output->stream &&
      (output->sampling.probing() || !output->chat.forced_completion.empty())) {
    return Error(400, "Bad Request",
                 "'logprobs' and 'forced_completion' require stream=false",
                 "unsupported_stream");
  }

  if (const json::Value* choices = body.find("n");
      choices != nullptr && !choices->is_null() &&
      (!choices->is_number() || choices->as_double() != 1.0)) {
    return Error(400, "Bad Request", "only n=1 is supported", "unsupported_n");
  }
  for (const std::string_view unsupported : {"modalities", "audio"}) {
    if (const auto* value = body.find(std::string(unsupported));
        value != nullptr && !value->is_null()) {
      if (unsupported == "modalities" && value->is_array() &&
          value->size() == 1 && value->items().front().is_string() &&
          value->items().front().str() == "text")
        continue;
      return Error(
          400, "Bad Request",
          "request field '" + std::string(unsupported) + "' is not implemented",
          "unsupported_field");
    }
  }
  return std::nullopt;
}

std::string ArgumentsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> arguments) {
  json::Value object = json::Value::object();
  for (const auto& argument : arguments) {
    if (argument.is_string) {
      object[argument.name] = argument.value;
      continue;
    }
    try {
      object[argument.name] = json::parse(argument.value);
    } catch (...) {
      object[argument.name] = argument.value;
    }
  }
  return object.dump();
}

const char* FinishReason(const TextGenerationBackend::Result& result,
                         bool has_tool_calls) {
  if (result.finish_reason ==
      TextGenerationBackend::FinishReason::kStopSequence)
    return "stop";
  if (has_tool_calls) {
    return "tool_calls";
  }
  if (result.finish_reason == TextGenerationBackend::FinishReason::kLength) {
    return "length";
  }
  return "stop";
}

json::Value Usage(const TextGenerationBackend::Result& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  json::Value prompt_details = json::Value::object();
  prompt_details["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_details"] = std::move(prompt_details);

  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double predicted_per_second =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  usage["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_per_second"] = prompt_per_second;
  usage["completion_tokens_per_second"] = predicted_per_second;
  usage["draft_tokens"] = result.draft_tokens;
  usage["draft_tokens_accepted"] = result.draft_accepted_tokens;

  json::Value metrics = json::Value::object();
  metrics["cache_hit"] = result.cache_hit;
  // How far a miss matched cached entries stays in the server log: the cache
  // is shared across clients, so reporting it would let one client probe
  // another's prompts.
  if (!result.cache_miss_reason.empty())
    metrics["cache_miss_reason"] = result.cache_miss_reason;
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_queued_bytes"] = result.cache_disk_queued_bytes;
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
  metrics["cache_shared_prefix_snapshots"] =
      result.cache_shared_prefix_snapshots;
  metrics["cache_shared_prefix_bytes"] = result.cache_shared_prefix_bytes;
  metrics["cache_shared_prefix_ms"] = result.cache_shared_prefix_ms;
  metrics["prefill_tokens"] = result.prefill_tokens;
  metrics["prefill_chunks"] = result.prefill_chunks;
  metrics["active_decode_prefill_chunks"] = result.active_decode_prefill_chunks;
  metrics["max_prefill_chunk_tokens"] = result.max_prefill_chunk_tokens;
  metrics["queue_depth_at_submit"] = result.queue_depth_at_submit;
  metrics["client_queue_depth_at_submit"] = result.client_queue_depth_at_submit;
  metrics["resident_requests_at_admission"] =
      result.resident_requests_at_admission;
  metrics["requested_logical_concurrency"] =
      result.requested_logical_concurrency;
  metrics["physical_execution_width"] = result.physical_execution_width;
  metrics["queue_ms"] = result.queue_ms;
  metrics["prefill_ms"] = result.prefill_ms;
  metrics["decode_ms"] = result.decode_ms;
  metrics["ttft_ms"] = result.ttft_ms;
  metrics["mean_inter_token_ms"] = result.mean_inter_token_ms;
  metrics["max_inter_token_ms"] = result.max_inter_token_ms;
  metrics["execution_plan"] = result.execution_plan;
  usage["gufo"] = std::move(metrics);
  return usage;
}

json::Value ToolCallsJson(
    std::span<const tokenization::ChatMessage::ToolCall> calls) {
  json::Value output = json::Value::array();
  for (const auto& call : calls) {
    json::Value item = json::Value::object();
    item["id"] = RandomId("call_");
    item["type"] = "function";
    json::Value function = json::Value::object();
    function["name"] = call.name;
    function["arguments"] = ArgumentsJson(call.arguments);
    item["function"] = std::move(function);
    output.push_back(std::move(item));
  }
  return output;
}

std::string Sse(const json::Value& value) {
  return "data: " + value.dump() + "\n\n";
}

json::Value BaseChunk(std::string_view id, long long created,
                      std::string_view model, const char* service_tier) {
  json::Value chunk = json::Value::object();
  chunk["id"] = std::string(id);
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = std::string(model);
  chunk["service_tier"] = service_tier;
  return chunk;
}

json::Value ChoiceChunk(std::string_view id, long long created,
                        std::string_view model, const char* service_tier,
                        json::Value delta,
                        const char* finish_reason = nullptr) {
  json::Value chunk = BaseChunk(id, created, model, service_tier);
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = std::move(delta);
  if (finish_reason == nullptr) {
    choice["finish_reason"] = json::Value();
  } else {
    choice["finish_reason"] = finish_reason;
  }
  choices.push_back(std::move(choice));
  chunk["choices"] = std::move(choices);
  return chunk;
}

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Build the same output items for streaming and buffered responses.
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, const char* service_tier,
                  HttpResponse::BodyWriter writer)
      : writer_(std::move(writer)) {
    response_ = json::Value::object();
    response_["id"] = RandomId("resp_");
    response_["object"] = "response";
    response_["created_at"] = Now();
    response_["model"] = std::move(model);
    response_["service_tier"] = service_tier;
    response_["status"] = "in_progress";
    response_["error"] = json::Value();
    response_["incomplete_details"] = json::Value();
    response_["usage"] = json::Value();
    response_["output"] = json::Value::array();
    response_["store"] = false;
    response_["parallel_tool_calls"] = false;
    response_["tool_choice"] = "none";
    response_["tools"] = json::Value::array();
  }

  bool Begin() {
    return Lifecycle("response.created") && Lifecycle("response.in_progress");
  }

  bool Append(std::string_view text, bool reasoning) {
    if (text.empty())
      return true;
    if (!active_ || reasoning_ != reasoning) {
      if (!CloseItem("completed"))
        return false;
      reasoning_ = reasoning;
      active_ = true;
      item_ = json::Value::object();
      item_["id"] = RandomId(reasoning ? "rs_" : "msg_");
      item_["type"] = reasoning ? "reasoning" : "message";
      item_["status"] = "in_progress";
      item_[reasoning ? "summary" : "content"] = json::Value::array();
      if (!reasoning)
        item_["role"] = "assistant";
      auto added = IndexedEvent("response.output_item.added");
      added["item"] = item_;
      if (!Emit(std::move(added)))
        return false;
      text_.clear();
      auto part = PartEvent(reasoning ? "response.reasoning_summary_part.added"
                                      : "response.content_part.added");
      part["part"] = Part();
      if (!Emit(std::move(part)))
        return false;
    }
    text_.append(text);
    auto delta = PartEvent(reasoning ? "response.reasoning_summary_text.delta"
                                     : "response.output_text.delta");
    delta["delta"] = std::string(text);
    if (!reasoning)
      delta["logprobs"] = json::Value::array();
    return Emit(std::move(delta));
  }

  json::Value Complete(const TextGenerationBackend::Result& result) {
    const bool limited =
        result.finish_reason == TextGenerationBackend::FinishReason::kLength;
    CloseItem(limited ? "incomplete" : "completed");
    response_["status"] = limited ? "incomplete" : "completed";
    if (limited)
      response_["incomplete_details"]["reason"] = "max_output_tokens";
    auto usage = json::Value::object();
    usage["input_tokens"] = result.prompt_tokens;
    usage["input_tokens_details"]["cached_tokens"] =
        result.cached_prompt_tokens;
    usage["input_tokens_details"]["cache_write_tokens"] = result.prefill_tokens;
    usage["output_tokens"] = result.completion_tokens;
    usage["output_tokens_details"]["reasoning_tokens"] =
        result.reasoning_tokens;
    usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
    response_["usage"] = std::move(usage);
    response_["timings"] = GenerationTimings(result);
    Lifecycle(limited ? "response.incomplete" : "response.completed");
    return response_;
  }

  bool Fail(std::string_view message) {
    response_["status"] = "failed";
    // Responses defines a closed error-code enum. Keep the specific runtime
    // code in the request log, rather than emitting an invalid wire value.
    response_["error"]["code"] = "server_error";
    response_["error"]["message"] = std::string(message);
    return Lifecycle("response.failed");
  }

private:
  json::Value IndexedEvent(std::string_view type) const {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["output_index"] = output_index_;
    return event;
  }

  json::Value PartEvent(std::string_view type) const {
    auto event = IndexedEvent(type);
    event["item_id"] = item_.member_str("id");
    event[reasoning_ ? "summary_index" : "content_index"] = 0;
    return event;
  }

  json::Value Part() const {
    auto part = json::Value::object();
    part["type"] = reasoning_ ? "summary_text" : "output_text";
    part["text"] = text_;
    if (!reasoning_) {
      part["annotations"] = json::Value::array();
      part["logprobs"] = json::Value::array();
    }
    return part;
  }

  bool CloseItem(const char* status) {
    if (!active_)
      return connected_;
    auto done = PartEvent(reasoning_ ? "response.reasoning_summary_text.done"
                                     : "response.output_text.done");
    done["text"] = text_;
    if (!reasoning_)
      done["logprobs"] = json::Value::array();
    if (!Emit(std::move(done)))
      return false;
    auto part = Part();
    auto part_done =
        PartEvent(reasoning_ ? "response.reasoning_summary_part.done"
                             : "response.content_part.done");
    part_done["part"] = part;
    if (!Emit(std::move(part_done)))
      return false;
    item_[reasoning_ ? "summary" : "content"].push_back(std::move(part));
    item_["status"] = status;
    auto item_done = IndexedEvent("response.output_item.done");
    item_done["item"] = item_;
    response_["output"].push_back(item_);
    active_ = false;
    ++output_index_;
    return Emit(std::move(item_done));
  }

  bool Lifecycle(std::string_view type) {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["response"] = response_;
    return Emit(std::move(event));
  }

  bool Emit(json::Value event) {
    if (!writer_)
      return true;
    if (!connected_)
      return false;
    event["sequence_number"] = sequence_++;
    connected_ =
        writer_("event: " + event.member_str("type") + "\n" + Sse(event));
    return connected_;
  }

  HttpResponse::BodyWriter writer_;
  json::Value response_;
  json::Value item_;
  std::string text_;
  std::size_t sequence_{0};
  std::size_t output_index_{0};
  bool active_{false};
  bool reasoning_{false};
  bool connected_{true};
};

HttpResponse NonStreamingResponse(
    const ParsedChatRequest& request,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>& generation,
    const ChatProtocol& protocol) {
  const auto result = generation->Wait();
  auto parser = protocol.NewParser(request.chat);
  parser.Push(result.text, {});
  const ParsedOutput generated =
      *parser.Finish(result.finish_reason !=
                         TextGenerationBackend::FinishReason::kStopSequence,
                     {});

  json::Value response = json::Value::object();
  response["id"] = RandomId("chatcmpl-");
  response["object"] = "chat.completion";
  response["created"] = Now();
  response["model"] = request.model;
  response["service_tier"] = ServiceTierName(request.chat.service_tier);
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  if (!generated.reasoning.empty()) {
    message["reasoning_content"] = generated.reasoning;
  }
  if (generated.content.empty() && !generated.tool_calls.empty()) {
    message["content"] = json::Value();
  } else {
    message["content"] = generated.content;
  }
  if (!generated.tool_calls.empty()) {
    message["tool_calls"] = ToolCallsJson(generated.tool_calls);
  }
  choice["message"] = std::move(message);
  choice["finish_reason"] = FinishReason(result, !generated.tool_calls.empty());
  if (request.sampling.top_logprobs >= 0) {
    if (result.logprobs.size() != result.tokens.size()) {
      return Error(500, "Internal Server Error",
                   "logprobs are unavailable on this decode path",
                   "logprobs_unavailable");
    }
    // OpenAI shape plus the token "id"; -9999 stands in for -inf.
    const auto entry = [](const TokenLogprob& item) {
      json::Value value = json::Value::object();
      value["id"] = static_cast<std::size_t>(item.token);
      value["token"] = core::Utf8Decoder{}.Push(item.text, true);
      value["logprob"] = std::isfinite(item.logprob)
                             ? std::max(item.logprob, -9999.0)
                             : -9999.0;
      json::Value bytes = json::Value::array();
      for (const unsigned char byte : item.text) {
        bytes.push_back(static_cast<std::size_t>(byte));
      }
      value["bytes"] = std::move(bytes);
      return value;
    };
    json::Value content = json::Value::array();
    for (const auto& item : result.logprobs) {
      json::Value value = entry(item);
      json::Value top = json::Value::array();
      for (const auto& alternative : item.top) {
        top.push_back(entry(alternative));
      }
      value["top_logprobs"] = std::move(top);
      content.push_back(std::move(value));
    }
    json::Value logprobs = json::Value::object();
    logprobs["content"] = std::move(content);
    choice["logprobs"] = std::move(logprobs);
  }
  choices.push_back(std::move(choice));
  response["choices"] = std::move(choices);
  response["usage"] = Usage(result);
  response["timings"] = GenerationTimings(result);
  RecordServerMetrics(result);

  std::ostringstream timing;
  timing << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
         << ", inter_token;dur=" << result.mean_inter_token_ms
         << ", max_inter_token;dur=" << result.max_inter_token_ms;

  return {
      .status = 200,
      .reason = "OK",
      .body = response.dump(),
      .headers = {{"Server-Timing", timing.str()}},
      .streaming_body = {},
      .log_details = GenerationLogDetails(result),
  };
}

HttpResponse StreamingResponse(
    const ParsedChatRequest& request,
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation,
    const ChatProtocol& protocol) {
  const std::string id = RandomId("chatcmpl-");
  const long long created = Now();
  const std::string model = request.model;
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  return {
      .status = 200,
      .reason = "OK",
      .body = {},
      .headers =
          {
              {"Content-Type", "text/event-stream"},
              {"Cache-Control", "no-cache"},
              {"X-Accel-Buffering", "no"},
          },
      .streaming_body =
          [request, generation = std::move(generation), id, created, model,
           protocol = &protocol, stream_log,
           tier = ServiceTierName(request.chat.service_tier)](
              const HttpResponse::BodyWriter& writer) {
            json::Value role_delta = json::Value::object();
            role_delta["role"] = "assistant";
            if (!writer(Sse(ChoiceChunk(id, created, model, tier,
                                        std::move(role_delta))))) {
              generation->Cancel();
              return;
            }

            bool connected = true;
            auto parser = protocol->NewParser(request.chat);
            const OutputParser::Emit emit = [&](std::string_view text,
                                                bool is_reasoning) {
              if (text.empty()) {
                return true;
              }
              json::Value delta = json::Value::object();
              if (is_reasoning) {
                delta["reasoning_content"] = std::string(text);
              } else {
                delta["content"] = std::string(text);
              }
              connected = writer(
                  Sse(ChoiceChunk(id, created, model, tier, std::move(delta))));
              return connected;
            };

            try {
              const auto result = generation->Wait([&](std::string_view piece) {
                return connected && parser.Push(piece, emit);
              });
              stream_log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (!connected || result.cancelled) {
                return;
              }
              const auto generated = parser.Finish(
                  result.finish_reason !=
                      TextGenerationBackend::FinishReason::kStopSequence,
                  emit);
              if (!generated) {
                return;
              }
              for (std::size_t index = 0; index < generated->tool_calls.size();
                   ++index) {
                const auto& call = generated->tool_calls[index];
                json::Value delta = json::Value::object();
                json::Value tool_calls = json::Value::array();
                json::Value item = json::Value::object();
                item["index"] = index;
                item["id"] = RandomId("call_");
                item["type"] = "function";
                json::Value function = json::Value::object();
                function["name"] = call.name;
                function["arguments"] = ArgumentsJson(call.arguments);
                item["function"] = std::move(function);
                tool_calls.push_back(std::move(item));
                delta["tool_calls"] = std::move(tool_calls);
                if (!writer(Sse(ChoiceChunk(id, created, model, tier,
                                            std::move(delta))))) {
                  return;
                }
              }

              json::Value terminal_delta = json::Value::object();
              auto terminal_chunk = ChoiceChunk(
                  id, created, model, tier, std::move(terminal_delta),
                  FinishReason(result, !generated->tool_calls.empty()));
              // llama.cpp reports timings on the terminal choice regardless
              // of the optional OpenAI usage chunk. Proxies need these even
              // when a client does not request stream_options.include_usage.
              terminal_chunk["timings"] = GenerationTimings(result);
              if (!writer(Sse(terminal_chunk))) {
                return;
              }
              if (request.include_usage) {
                json::Value usage_chunk = BaseChunk(id, created, model, tier);
                usage_chunk["choices"] = json::Value::array();
                usage_chunk["usage"] = Usage(result);
                usage_chunk["timings"] = GenerationTimings(result);
                if (!writer(Sse(usage_chunk))) {
                  return;
                }
              }
              (void)writer("data: [DONE]\n\n");
            } catch (const TextGenerationError& exception) {
              stream_log->error_code = exception.stable_code();
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = exception.what();
              detail["type"] = "server_error";
              detail["code"] = exception.stable_code();
              error["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            } catch (const std::exception&) {
              stream_log->error_code = "generation_failed";
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = "generation failed";
              detail["type"] = "server_error";
              detail["code"] = "generation_failed";
              error["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            }
          },
      .stream_log = std::move(stream_log),
  };
}

}  // namespace

HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream) {
  const auto parser = backend.chat_protocol().NewParser(chat);
  auto generation = backend.start_chat(chat, max_tokens, sampling,
                                       request.is_cancelled, stream);
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  auto timing = std::make_shared<std::string>();
  const auto run = [generation, parser,
                    model =
                        chat.model.empty() ? backend.model_id() : chat.model,
                    stream_log, timing,
                    tier = ServiceTierName(chat.service_tier)](
                       const HttpResponse::BodyWriter& writer) {
    ResponsesOutput output(model, tier, writer);
    if (!output.Begin()) {
      generation->Cancel();
      return json::Value();
    }
    auto filter = parser;
    const OutputParser::Emit emit = [&](std::string_view piece,
                                        bool reasoning) {
      if (output.Append(piece, reasoning))
        return true;
      generation->Cancel();
      return false;
    };
    try {
      const auto result = writer
                              ? generation->Wait([&](std::string_view piece) {
                                  return filter.Push(piece, emit);
                                })
                              : generation->Wait();
      stream_log->details = GenerationLogDetails(result);
      RecordServerMetrics(result);
      if (!writer) {
        std::ostringstream value;
        value << std::fixed << std::setprecision(3)
              << "ttft;dur=" << result.ttft_ms
              << ", inter_token;dur=" << result.mean_inter_token_ms
              << ", max_inter_token;dur=" << result.max_inter_token_ms;
        *timing = value.str();
      }
      if (result.cancelled)
        return json::Value();
      if (!writer)
        filter.Push(result.text, emit);
      if (!filter.Finish(false, emit)) {
        generation->Cancel();
        return json::Value();
      }
      return output.Complete(result);
    } catch (const std::exception& error) {
      if (!writer)
        throw;
      const auto* generation_error =
          dynamic_cast<const TextGenerationError*>(&error);
      stream_log->error_code = generation_error
                                   ? generation_error->stable_code()
                                   : "generation_failed";
      stream_log->error_event_sent =
          output.Fail(generation_error ? error.what() : "generation failed");
      generation->Cancel();
      return json::Value();
    }
  };
  if (stream) {
    return {.status = 200,
            .reason = "OK",
            .body = {},
            .headers = {{"Content-Type", "text/event-stream"},
                        {"Cache-Control", "no-cache"},
                        {"X-Accel-Buffering", "no"}},
            .streaming_body =
                [run](const HttpResponse::BodyWriter& writer) {
                  (void)run(writer);
                },
            .stream_log = std::move(stream_log)};
  }
  auto response = run({});
  return {.status = 200,
          .reason = "OK",
          .body = response.dump(),
          .headers = {{"Server-Timing", *timing}},
          .log_details = stream_log->details};
}

std::optional<std::string> ParseServiceTier(const json::Value& body,
                                            bool flex_model,
                                            ServiceTier* tier) {
  const json::Value* value = body.find("service_tier");
  *tier = flex_model ? ServiceTier::kFlex : ServiceTier::kDefault;
  if (value == nullptr || value->is_null()) {
    return std::nullopt;
  }
  if (value->is_string()) {
    const std::string& name = value->str();
    if (name == "flex" || name == "auto") {
      *tier = name == "flex" ? ServiceTier::kFlex : *tier;
      return std::nullopt;
    }
    if (name == "default" || name == "scale" || name == "priority") {
      if (flex_model)
        return "the flex model id runs at service_tier 'flex' only";
      return std::nullopt;
    }
  }
  return "'service_tier' must be one of auto, default, flex, scale or priority";
}

HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend) {
  ParsedChatRequest parsed;
  const auto defaults = backend.sampling_defaults();
  parsed.max_tokens = defaults.max_tokens;
  parsed.sampling = defaults.sampling;
  if (auto error = ParseRequest(request, backend, &parsed); error.has_value()) {
    return std::move(*error);
  }
  try {
    const auto& protocol = backend.chat_protocol();
    auto generation =
        backend.start_chat(parsed.chat, parsed.max_tokens, parsed.sampling,
                           request.is_cancelled, parsed.stream);
    if (parsed.stream) {
      return StreamingResponse(parsed, std::move(generation), protocol);
    }
    return NonStreamingResponse(parsed, generation, protocol);
  } catch (const TextGenerationError& exception) {
    return GenerationError(exception);
  } catch (const grammar::GrammarError& exception) {
    return Error(400, "Bad Request", exception.what(),
                 exception.code().c_str());
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

}  // namespace gufo::server
