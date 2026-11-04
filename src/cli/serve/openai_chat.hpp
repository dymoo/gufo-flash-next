#ifndef GUFO_SERVER_OPENAI_CHAT_HPP_
#define GUFO_SERVER_OPENAI_CHAT_HPP_

#include <optional>
#include <string>

#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// consume scheduler-published token pieces from HttpResponse::streaming_body;
/// socket writes never own or execute model state.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

/// Reads OpenAI `service_tier`: "flex" selects ServiceTier::kFlex; "auto",
/// "default", "scale", "priority", null or absent run as default. A
/// `flex_model` (TextGenerationBackend::is_flex_model) is flex and accepts
/// only "flex", "auto", null or absent. Returns the error message otherwise.
std::optional<std::string> ParseServiceTier(const json::Value& body,
                                            bool flex_model, ServiceTier* tier);

/// Responses text output uses the same reasoning/UTF-8 filter and scheduler
/// as Chat Completions, including streaming cancellation and cache retention.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream);

}  // namespace gufo::server

#endif  // GUFO_SERVER_OPENAI_CHAT_HPP_
