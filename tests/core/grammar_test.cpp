#include "src/core/grammar/grammar.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/core/json.hpp"
#include "tests/core/grammar_test_support.hpp"

namespace {

using gufo::grammar::Engine;
using gufo::grammar::Gating;
using gufo::grammar::GrammarError;
using gufo::grammar::GrammarSource;
using gufo::grammar::Limits;
using gufo::json::Value;
using gufo::sampling::TokenId;
namespace test = gufo::grammar::test;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

GrammarSource Schema(std::string text) {
  return {.kind = GrammarSource::Kind::kJsonSchema, .text = std::move(text)};
}

/// Minimal JSON Schema check for the keywords these tests use.
// NOLINTNEXTLINE(misc-no-recursion)
bool Valid(const Value& schema, const Value& value) {
  if (const auto* options = schema.find("enum")) {
    return std::ranges::any_of(options->items(), [&](const Value& option) {
      return option.dump() == value.dump();
    });
  }
  if (const auto* type = schema.find("type")) {
    const auto matches = [&](const std::string& name) {
      return (name == "object" && value.is_object()) ||
             (name == "array" && value.is_array()) ||
             (name == "string" && value.is_string()) ||
             (name == "number" && value.is_number()) ||
             (name == "integer" && value.is_number() &&
              std::floor(value.as_double()) == value.as_double()) ||
             (name == "boolean" && value.is_bool()) ||
             (name == "null" && value.is_null());
    };
    const bool typed =
        type->is_string()
            ? matches(type->str())
            : std::ranges::any_of(type->items(), [&](const Value& item) {
                return matches(item.str());
              });
    if (!typed)
      return false;
  }
  if (value.is_object()) {
    const auto* properties = schema.find("properties");
    if (const auto* required = schema.find("required")) {
      for (const auto& name : required->items())
        if (value.find(name.str()) == nullptr)
          return false;
    }
    const auto* additional = schema.find("additionalProperties");
    for (const auto& [name, member] : value.members()) {
      const auto* property =
          properties != nullptr ? properties->find(name) : nullptr;
      if (property == nullptr) {
        if (additional != nullptr && additional->is_bool() &&
            !additional->as_bool())
          return false;
        continue;
      }
      if (!Valid(*property, member))
        return false;
    }
  }
  if (value.is_array()) {
    if (const auto* most = schema.find("maxItems");
        most != nullptr && value.size() > most->as_size())
      return false;
    if (const auto* least = schema.find("minItems");
        least != nullptr && value.size() < least->as_size())
      return false;
    if (const auto* items = schema.find("items")) {
      for (const auto& item : value.items())
        if (!Valid(*items, item))
          return false;
    }
  }
  return true;
}

constexpr std::string_view kNestedSchema = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "score": {"type": "number"},
    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b", "c"]},
             "maxItems": 4},
    "address": {
      "type": "object",
      "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    },
    "kind": {"enum": ["x", "y", 3]},
    "flag": {"type": "boolean"},
    "maybe": {"type": ["string", "null"]},
    "items": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
      "type": "object",
      "properties": {"id": {"type": "integer"}, "ok": {"type": "boolean"}},
      "required": ["id", "ok"],
      "additionalProperties": false
    }}
  },
  "required": ["name", "age", "address", "kind", "items"],
  "additionalProperties": false
})";

void TestSchemasAlwaysProduceValidJson() {
  const Engine engine(test::MakeVocabulary());
  const auto schema = gufo::json::parse(kNestedSchema);
  for (std::uint64_t seed = 1; seed <= 12; ++seed) {
    const auto text =
        test::Generate(engine, Schema(std::string(kNestedSchema)), {}, seed);
    Value value;
    try {
      value = gufo::json::parse(text);
    } catch (const std::exception& error) {
      Expect(false, std::string("constrained output parses as JSON (") +
                        error.what() + "): " + text);
    }
    Expect(Valid(schema, value), "constrained output satisfies the schema");
    Expect(value.find("kind") != nullptr && value.find("address") != nullptr,
           "required fields are always generated");
  }
  for (std::uint64_t seed = 1; seed <= 8; ++seed) {
    const auto text =
        test::Generate(engine, Schema(R"({"type":"object"})"), {}, seed);
    // Free-form objects may repeat a key (valid JSON grammar, which no
    // grammar can rule out); Gufo's strict parser rejects only that.
    try {
      Expect(gufo::json::parse(text).is_object(),
             "json_object output is always a JSON object");
    } catch (const std::runtime_error& error) {
      Expect(std::string_view(error.what()).find("duplicate object key") !=
                     std::string_view::npos &&
                 text.find_first_not_of(" \t\r\n") == text.find('{'),
             "json_object output is a JSON object: " + text);
    }
  }
}

void TestReasoningRunsUnconstrained() {
  const Engine engine(test::MakeVocabulary());
  const Gating gating{.start = Gating::Start::kReasoning,
                      .reasoning_end = {test::kThinkEnd},
                      .reasoning_start = test::kThinkStart,
                      .separator = {test::kDoubleNewline}};
  auto constraint =
      engine.NewConstraint(Schema(R"({"type":"object"})"), gating);
  Expect(!constraint->Restricts() && constraint->Ready(),
         "reasoning starts unconstrained");
  for (const TokenId token : {TokenId{'x'}, TokenId{'['}, test::kThinkStart})
    constraint->Accept(token);
  Expect(!constraint->Restricts(), "any reasoning token is accepted");
  constraint->Accept(test::kThinkEnd);
  Expect(constraint->Restricts(), "the grammar applies after </think>");
  auto row = test::FlatRow();
  constraint->Mask(row);
  Expect(test::Allowed(row) == std::vector<TokenId>{test::kDoubleNewline},
         "the template separator is forced first");
  bool rejected = false;
  try {
    constraint->Accept('{');
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "skipping the separator is rejected");

  constraint = engine.NewConstraint(Schema(R"({"type":"object"})"), gating);
  constraint->Accept(test::kThinkEnd);
  constraint->Accept(test::kDoubleNewline);
  row = test::FlatRow();
  constraint->Mask(row);
  const auto allowed = test::Allowed(row);
  Expect(std::ranges::find(allowed, TokenId{'{'}) != allowed.end() &&
             std::ranges::find(allowed, TokenId{'x'}) == allowed.end() &&
             std::ranges::find(allowed, test::kStop) == allowed.end(),
         "the answer opens the JSON object");
}

void TestMultiTokenReasoningEnd() {
  const Engine engine(test::MakeVocabulary());
  // "</</>" overlaps itself: the matcher must not lose a partial close.
  const std::vector<TokenId> close{'<', '/', '<', '/', '>'};
  auto constraint = engine.NewConstraint(
      Schema(R"({"type":"object"})"),
      {.start = Gating::Start::kReasoning, .reasoning_end = close});
  for (const char c : std::string_view("<</</")) {
    constraint->Accept(static_cast<TokenId>(c));
    Expect(!constraint->Restricts(), "a partial close keeps reasoning free");
  }
  constraint->Accept('<');
  constraint->Accept('/');
  Expect(!constraint->Restricts(), "still a partial close");
  constraint->Accept('>');
  Expect(constraint->Restricts(), "the overlapping close is recognized");
}

void TestOptionalReasoning() {
  const Engine engine(test::MakeVocabulary());
  const Gating either{.start = Gating::Start::kEither,
                      .reasoning_end = {test::kThinkEnd},
                      .reasoning_start = test::kThinkStart};
  auto constraint =
      engine.NewConstraint(Schema(R"({"type":"object"})"), either);
  auto row = test::FlatRow();
  constraint->Mask(row);
  auto allowed = test::Allowed(row);
  Expect(std::ranges::find(allowed, test::kThinkStart) != allowed.end() &&
             std::ranges::find(allowed, TokenId{'{'}) != allowed.end() &&
             std::ranges::find(allowed, TokenId{'x'}) == allowed.end(),
         "the first token may open reasoning or the answer");
  constraint->Accept(test::kThinkStart);
  Expect(!constraint->Restricts(), "opened reasoning runs free");
  constraint->Accept('x');
  constraint->Accept(test::kThinkEnd);
  Expect(constraint->Restricts(), "the answer follows reasoning");

  constraint = engine.NewConstraint(Schema(R"({"type":"object"})"), either);
  constraint->Accept('{');
  row = test::FlatRow();
  constraint->Mask(row);
  allowed = test::Allowed(row);
  Expect(std::ranges::find(allowed, test::kThinkStart) == allowed.end(),
         "after a direct answer reasoning cannot open");
}

void TestGrammarRejectsViolations() {
  const Engine engine(test::MakeVocabulary());
  auto constraint = engine.NewConstraint(Schema(R"({"type":"object"})"), {});
  bool rejected = false;
  try {
    constraint->Accept('x');
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "a token outside the grammar fails the request");
}

std::string LargeEnumSchema(std::size_t count) {
  std::string schema = R"({"type":"object","properties":{"v":{"enum":[)";
  for (std::size_t i = 0; i < count; ++i) {
    if (i != 0)
      schema += ',';
    schema += "\"value-" + std::to_string(i * 7919) + "\"";
  }
  return schema + R"(]}},"required":["v"]})";
}

std::string ErrorCode(const Engine& engine, const GrammarSource& source) {
  try {
    (void)engine.NewConstraint(source, {});
  } catch (const GrammarError& error) {
    return error.code();
  }
  return {};
}

void TestCompileFailuresAreClientErrors() {
  Limits small;
  small.max_source_bytes = 1024;
  const Engine bounded(test::MakeVocabulary(), small);
  Expect(
      ErrorCode(bounded, Schema(LargeEnumSchema(200))) == "grammar_too_large",
      "an oversized schema is rejected before compiling");

  Limits instant;
  instant.compile_timeout = std::chrono::milliseconds(0);
  const Engine hurried(test::MakeVocabulary(), instant);
  Expect(ErrorCode(hurried, Schema(LargeEnumSchema(4000))) == "grammar_timeout",
         "a grammar that misses the compile limit is rejected");

  const Engine engine(test::MakeVocabulary());
  for (const char* invalid :
       {R"({"type":"object","properties":{"a":{"$ref":"#/$defs/missing"}}})",
        R"({"type":"nonsense"})", "not json"}) {
    Expect(ErrorCode(engine, Schema(invalid)) == "invalid_grammar",
           "an unsupported schema is a client error");
  }
  Expect(ErrorCode(engine, {.kind = GrammarSource::Kind::kStructuralTag,
                            .text = R"({"type":"structural_tag"})"}) ==
             "invalid_grammar",
         "a malformed structural tag is a client error");
  Expect(ErrorCode(engine, Schema(R"({"type":"object"})")).empty(),
         "the engine keeps serving after failures");
  Expect(engine.CacheBytes() <= engine.limits().cache_bytes,
         "the compiled-grammar cache stays under its cap");
}

void TestConcurrentCompilation() {
  const Engine engine(test::MakeVocabulary());
  std::vector<std::thread> threads;
  std::atomic<int> compiled{0};
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] {
      const auto schema = i % 2 == 0 ? std::string(kNestedSchema)
                                     : std::string(R"({"type":"object"})");
      if (engine.NewConstraint(Schema(schema), {}) != nullptr)
        compiled.fetch_add(1);
    });
  }
  for (auto& thread : threads)
    thread.join();
  Expect(compiled == 8, "concurrent requests share one compiler and cache");
}

}  // namespace

int main() {
  TestSchemasAlwaysProduceValidJson();
  TestReasoningRunsUnconstrained();
  TestMultiTokenReasoningEnd();
  TestOptionalReasoning();
  TestGrammarRejectsViolations();
  TestCompileFailuresAreClientErrors();
  TestConcurrentCompilation();
  std::cout << "All grammar tests passed\n";
  return 0;
}
