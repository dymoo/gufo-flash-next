#ifndef GUFO_MODELS_KEV_SYSTEMONE_HPP_
#define GUFO_MODELS_KEV_SYSTEMONE_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

/// TypeSafe System One requests and answers, as the Kev reference maps them
/// (jaredpalmer/kev kev/api.py). Rendering and number spelling follow Python
/// exactly, because they decide the tokens the model sees.
namespace gufo::models::kev {

enum class QuestionType { kNoul, kChoice, kScore };

struct Question {
  std::string id;
  QuestionType type{QuestionType::kNoul};
  std::string instructions;          ///< Rendered.
  std::vector<std::string> options;  ///< Rendered option texts, in order.
  std::vector<std::string> keys;     ///< Probability keys, one per option.
};

struct Request {
  std::string model{"kev-latest"};
  std::string state;  ///< Rendered.
  std::vector<Question> questions;
};

inline constexpr std::size_t kMaxOptions = 255;
/// Questions per request. The reference sets no limit; this bounds the work
/// one request can queue.
inline constexpr std::size_t kMaxQuestions = 128;

/// Parses and validates a request body; on failure `error` is the 422
/// detail.
[[nodiscard]] std::optional<Request> ParseRequest(std::string_view body,
                                                  std::string* error);

/// kev.api.render: flattens a JSON value into the text the model reads.
[[nodiscard]] std::string Render(const json::Value& value);

/// Python repr() of a float (shortest round trip, Python's layout).
[[nodiscard]] std::string PythonFloatRepr(double value);

/// Python round(value, 4).
[[nodiscard]] double RoundProbability(double value);

enum class JsonStyle {
  kFastApi,     ///< Compact separators, UTF-8 kept (the response body).
  kPythonDumps  ///< json.dumps defaults: ", " / ": ", ASCII escapes.
};

/// The `answers` object for per-question option probabilities.
[[nodiscard]] std::string AnswersJson(
    const Request& request, std::span<const std::vector<double>> probabilities,
    JsonStyle style);

/// Python-compatible JSON string literal.
[[nodiscard]] std::string PythonJsonString(std::string_view text,
                                           JsonStyle style);

}  // namespace gufo::models::kev

#endif  // GUFO_MODELS_KEV_SYSTEMONE_HPP_
