#include "src/models/kev/systemone.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <system_error>

namespace gufo::models::kev {
namespace {

using json::Value;

// Python str.isspace() code points, for the str.lstrip() in render().
bool IsPythonSpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

// Decodes one code point of valid UTF-8 (every string here came through the
// JSON parser, which validates it). Returns its byte length.
std::size_t DecodeUtf8(std::string_view s, std::size_t i, char32_t* out) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    *out = b0;
    return 1;
  }
  const std::size_t n = b0 >= 0xF0 ? 4 : b0 >= 0xE0 ? 3 : 2;
  char32_t c = b0 & (0x7F >> n);
  for (std::size_t k = 1; k < n; ++k)
    c = (c << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  *out = c;
  return n;
}

std::string PythonLstrip(std::string text) {
  std::size_t i = 0;
  while (i < text.size()) {
    char32_t c = 0;
    const std::size_t n = DecodeUtf8(text, i, &c);
    if (!IsPythonSpace(c))
      break;
    i += n;
  }
  text.erase(0, i);
  return text;
}

// str() of a JSON number as pydantic hands it to render(): an integer
// spelling becomes a Python int, anything with a fraction or exponent a float.
std::string PythonNumber(const Value& v) {
  const std::string& text = v.number_text();
  if (text.empty())
    return PythonFloatRepr(v.as_double());
  if (text.find_first_of(".eE") == std::string::npos)
    return text == "-0" ? "0" : text;
  return PythonFloatRepr(v.as_double());
}

// Rendering pads every line by its depth, so nested input grows. Bound the
// depth and each rendered value before a request reaches the tokenizer.
constexpr std::size_t kMaxRenderDepth = 32;
constexpr std::size_t kMaxRenderedBytes = 4 * 1024 * 1024;

struct RenderLimit : std::length_error {
  RenderLimit()
      : std::length_error(
            "input is too large or too deeply nested to render (limits: 32 "
            "levels, 4 MiB)") {}
};

// NOLINTNEXTLINE(misc-no-recursion)
std::string RenderAt(const Value& v, std::size_t indent) {
  if (indent > kMaxRenderDepth)
    throw RenderLimit();
  switch (v.type()) {
    case Value::Type::kNull:
      return "";
    case Value::Type::kString:
      return v.str();
    case Value::Type::kBool:
      return v.as_bool() ? "True" : "False";
    case Value::Type::kNumber:
      return PythonNumber(v);
    case Value::Type::kArray: {
      const std::string pad(2 * indent, ' ');
      std::string out;
      bool first = true;
      for (const auto& item : v.items()) {
        if (!first)
          out += '\n';
        first = false;
        out += pad + "- " + PythonLstrip(RenderAt(item, indent + 1));
        if (out.size() > kMaxRenderedBytes)
          throw RenderLimit();
      }
      return out;
    }
    case Value::Type::kObject: {
      const std::string pad(2 * indent, ' ');
      std::string out;
      bool first = true;
      for (const auto& [key, item] : v.members()) {
        if (!first)
          out += '\n';
        first = false;
        if (item.is_object() || item.is_array())
          out += pad + key + ":\n" + RenderAt(item, indent + 1);
        else
          out += pad + key + ": " + RenderAt(item, 0);
        if (out.size() > kMaxRenderedBytes)
          throw RenderLimit();
      }
      return out;
    }
  }
  return "";
}

// kev.api.option_text: "name" or "name: description".
std::string OptionText(const std::string& name, const Value* description) {
  if (description == nullptr || description->is_null() ||
      (description->is_string() && description->str().empty()))
    return name;
  return name + ": " + Render(*description);
}

std::vector<double> Normalized(const std::vector<double>& p) {
  double total = 0.0;
  for (const double x : p)
    total += x;
  std::vector<double> out(p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    out[i] = total == 0.0 ? 1.0 / static_cast<double>(p.size()) : p[i] / total;
  return out;
}

std::size_t FirstArgmax(const std::vector<double>& p) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < p.size(); ++i)
    if (p[i] > p[best])
      best = i;
  return best;
}

double ChoiceConfidence(const std::vector<double>& p) {
  const auto k = static_cast<double>(p.size());
  if (p.size() == 1)
    return 1.0;
  const auto n = Normalized(p);
  return (*std::max_element(n.begin(), n.end()) - 1.0 / k) / (1.0 - 1.0 / k);
}

double ScoreConfidence(const std::vector<double>& p) {
  const std::size_t levels = p.size();
  if (levels == 1)
    return 1.0;
  const auto n = Normalized(p);
  const std::size_t mode = FirstArgmax(n);
  const double center = (static_cast<double>(levels) - 1.0) / 2.0;
  double spread = 0.0;
  for (std::size_t i = 0; i < levels; ++i)
    spread += std::fabs(static_cast<double>(i) - center);
  spread /= static_cast<double>(levels);
  double distance = 0.0;
  for (std::size_t i = 0; i < levels; ++i)
    distance +=
        n[i] * std::fabs(static_cast<double>(i) - static_cast<double>(mode));
  return std::max(0.0, 1.0 - distance / spread);
}

struct Writer {
  JsonStyle style;
  std::string out;

  void String(std::string_view s) { out += PythonJsonString(s, style); }
  void Number(double x) { out += PythonFloatRepr(RoundProbability(x)); }
  void Raw(std::string_view s) { out += s; }
  void Key(std::string_view key, bool first) {
    if (!first)
      out += style == JsonStyle::kPythonDumps ? ", " : ",";
    String(key);
    out += style == JsonStyle::kPythonDumps ? ": " : ":";
  }
};

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool ParseQuestion(const std::string& id, const Value& q, Question* out,
                   std::string* error) {
  const std::string where = "questions." + id;
  if (!q.is_object())
    return Fail(error, where + ": question must be an object");
  const Value* type = q.find("type");
  if (type == nullptr || !type->is_string())
    return Fail(error, where + ".type: must be one of noul, choice, score");
  out->id = id;
  const Value* instructions = q.find("instructions");
  out->instructions = instructions == nullptr ? "" : Render(*instructions);
  const Value* criteria = q.find("criteria");
  if (type->str() == "noul") {
    out->type = QuestionType::kNoul;
    if (criteria != nullptr && !criteria->is_null() && !criteria->is_object())
      return Fail(error, where + ".criteria: must be an object or null");
    const Value* no = criteria == nullptr ? nullptr : criteria->find("false");
    const Value* yes = criteria == nullptr ? nullptr : criteria->find("true");
    out->options = {OptionText("no", no), OptionText("yes", yes)};
    out->keys = {"false", "true"};
    return true;
  }
  if (type->str() == "choice") {
    out->type = QuestionType::kChoice;
    if (criteria == nullptr || !criteria->is_object())
      return Fail(error, where + ".criteria: must be an object");
    if (criteria->members().empty() || criteria->members().size() > kMaxOptions)
      return Fail(error, where + ".criteria: must have 1..255 options");
    for (const auto& [name, description] : criteria->members()) {
      out->options.push_back(OptionText(name, &description));
      out->keys.push_back(name);
    }
    return true;
  }
  if (type->str() == "score") {
    out->type = QuestionType::kScore;
    if (criteria == nullptr || !criteria->is_array())
      return Fail(error, where + ".criteria: must be an array");
    if (criteria->items().empty() || criteria->items().size() > kMaxOptions)
      return Fail(error, where + ".criteria: must have 1..255 levels");
    for (std::size_t i = 0; i < criteria->items().size(); ++i) {
      out->options.push_back(Render(criteria->items()[i]));
      out->keys.push_back(std::to_string(i));
    }
    return true;
  }
  return Fail(error, where + ".type: must be one of noul, choice, score");
}

}  // namespace

std::string Render(const Value& value) {
  return RenderAt(value, 0);
}

std::string PythonFloatRepr(double value) {
  if (std::isnan(value))
    return "NaN";
  if (std::isinf(value))
    return value > 0 ? "Infinity" : "-Infinity";
  const std::string sign = std::signbit(value) ? "-" : "";
  if (value == 0.0)
    return sign + "0.0";
  char buffer[64];
  const auto [end, ec] =
      std::to_chars(buffer, buffer + sizeof(buffer), std::fabs(value),
                    std::chars_format::scientific);
  if (ec != std::errc{})
    throw std::runtime_error("float formatting failed");
  const std::string_view text(buffer, static_cast<std::size_t>(end - buffer));
  const std::size_t e = text.find('e');
  std::string digits;
  for (const char c : text.substr(0, e))
    if (c != '.')
      digits += c;
  const int exponent = std::atoi(std::string(text.substr(e + 1)).c_str());
  const auto n = static_cast<int>(digits.size());
  std::string out = sign;
  if (exponent >= -4 && exponent < 16) {
    const int point = exponent + 1;
    if (point <= 0)
      out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    else if (point >= n)
      out +=
          digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
    else
      out += digits.substr(0, static_cast<std::size_t>(point)) + "." +
             digits.substr(static_cast<std::size_t>(point));
    return out;
  }
  out += digits[0];
  if (n > 1)
    out += "." + digits.substr(1);
  out += exponent < 0 ? "e-" : "e+";
  const int magnitude = std::abs(exponent);
  if (magnitude < 10)
    out += '0';
  out += std::to_string(magnitude);
  return out;
}

double RoundProbability(double value) {
  char buffer[64];
  const auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                       std::chars_format::fixed, 4);
  double rounded = 0.0;
  if (ec != std::errc{} ||
      std::from_chars(buffer, end, rounded).ec != std::errc{})
    throw std::runtime_error("probability rounding failed");
  return rounded;
}

std::string PythonJsonString(std::string_view text, JsonStyle style) {
  std::string out = "\"";
  std::size_t i = 0;
  while (i < text.size()) {
    char32_t c = 0;
    const std::size_t n = DecodeUtf8(text, i, &c);
    const auto escape = [&](char32_t unit) {
      char hex[8];
      (void)std::snprintf(hex, sizeof(hex), "\\u%04x",
                          static_cast<unsigned>(unit));
      out += hex;
    };
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          escape(c);
        } else if (style == JsonStyle::kPythonDumps && c > 0x7E) {
          if (c > 0xFFFF) {
            const char32_t v = c - 0x10000;
            escape(0xD800 + (v >> 10));
            escape(0xDC00 + (v & 0x3FF));
          } else {
            escape(c);
          }
        } else {
          out.append(text.substr(i, n));
        }
    }
    i += n;
  }
  out += '"';
  return out;
}

std::string AnswersJson(const Request& request,
                        std::span<const std::vector<double>> probabilities,
                        JsonStyle style) {
  if (probabilities.size() != request.questions.size())
    throw std::invalid_argument("one probability row per question");
  Writer w{style, {}};
  w.Raw("{");
  for (std::size_t qi = 0; qi < request.questions.size(); ++qi) {
    const Question& q = request.questions[qi];
    const std::vector<double>& p = probabilities[qi];
    if (p.size() != q.options.size())
      throw std::invalid_argument("one probability per option");
    w.Key(q.id, qi == 0);
    w.Raw("{");
    switch (q.type) {
      case QuestionType::kNoul:
        w.Key("type", true);
        w.String("noul");
        w.Key("noul", false);
        w.Number(p[1]);
        break;
      case QuestionType::kChoice:
        w.Key("type", true);
        w.String("choice");
        w.Key("choice", false);
        w.String(q.keys[FirstArgmax(p)]);
        w.Key("confidence", false);
        w.Number(ChoiceConfidence(p));
        w.Key("probabilities", false);
        w.Raw("{");
        for (std::size_t i = 0; i < p.size(); ++i) {
          w.Key(q.keys[i], i == 0);
          w.Number(p[i]);
        }
        w.Raw("}");
        break;
      case QuestionType::kScore: {
        double score = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i)
          score += static_cast<double>(i) * p[i];
        w.Key("type", true);
        w.String("score");
        w.Key("score", false);
        w.Number(score);
        w.Key("legend", false);
        w.Raw("{");
        for (std::size_t i = 0; i < p.size(); ++i) {
          w.Key(q.keys[i], i == 0);
          w.String(q.options[i]);
        }
        w.Raw("}");
        w.Key("probabilities", false);
        w.Raw("{");
        for (std::size_t i = 0; i < p.size(); ++i) {
          w.Key(q.keys[i], i == 0);
          w.Number(p[i]);
        }
        w.Raw("}");
        w.Key("confidence", false);
        w.Number(ScoreConfidence(p));
        break;
      }
    }
    w.Raw("}");
  }
  w.Raw("}");
  return w.out;
}

std::optional<Request> ParseRequest(std::string_view body, std::string* error) {
  Value root;
  try {
    root = json::parse(body);
  } catch (const std::exception& e) {
    Fail(error, e.what());
    return std::nullopt;
  }
  if (!root.is_object()) {
    Fail(error, "request body must be a JSON object");
    return std::nullopt;
  }
  Request request;
  if (const Value* model = root.find("model"); model != nullptr) {
    if (!model->is_string()) {
      Fail(error, "model: must be a string");
      return std::nullopt;
    }
    request.model = model->str();
  }
  const Value* state = root.find("state");
  if (state == nullptr) {
    Fail(error, "state: field required");
    return std::nullopt;
  }
  const Value* questions = root.find("questions");
  if (questions == nullptr || !questions->is_object() ||
      questions->members().empty()) {
    Fail(error, "questions: must be an object with at least one question");
    return std::nullopt;
  }
  if (questions->members().size() > kMaxQuestions) {
    Fail(error, "questions: at most " + std::to_string(kMaxQuestions) +
                    " per request");
    return std::nullopt;
  }
  try {
    request.state = Render(*state);
    for (const auto& [id, q] : questions->members()) {
      Question parsed;
      if (!ParseQuestion(id, q, &parsed, error))
        return std::nullopt;
      request.questions.push_back(std::move(parsed));
    }
  } catch (const RenderLimit& e) {
    Fail(error, e.what());
    return std::nullopt;
  }
  return request;
}

}  // namespace gufo::models::kev
