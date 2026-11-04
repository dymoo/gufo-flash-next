// Expected strings come from jaredpalmer/kev kev/api.py (render, to_answers,
// round_prob) run through pydantic, as kev.serve does.
#include "src/models/kev/systemone.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace kev = gufo::models::kev;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL kev_systemone_test: " << message << '\n';
    std::exit(1);
  }
}

void CheckEq(const std::string& got, const std::string& want,
             std::string_view what) {
  if (got != want) {
    std::cerr << "FAIL kev_systemone_test: " << what << "\n  got:  " << got
              << "\n  want: " << want << '\n';
    std::exit(1);
  }
}

void TestRender() {
  const std::string body =
      R"({"questions": {"q": {"type": "noul"}}, "state": )"
      R"({"a": 1, "b": 1.0, "c": -0, "d": 1e2, "e": 1e-05,)"
      R"( "f": 12345678901234567890123, "g": 0.1, "h": 1e16,)"
      R"( "i": 123456789012345678.0, "j": true, "k": null,)"
      R"( "l": [1, {"x": "  y"}, [2, 3], null, "\u3000 lead"],)"
      R"( "m": {}, "n": [], "o": {"p": {"q": false}},)"
      R"( "r": -0.0, "s": 1.5e300, "t": 0.0001, "u": 1E+2}})";
  std::string error;
  const auto request = kev::ParseRequest(body, &error);
  Check(request.has_value(), "parse: " + error);
  CheckEq(request->state,
          "a: 1\nb: 1.0\nc: 0\nd: 100.0\ne: 1e-05\nf: "
          "12345678901234567890123\ng: 0.1\nh: 1e+16\ni: "
          "1.2345678901234568e+17\nj: True\nk: \nl:\n  - 1\n  - x:   y\n  - - "
          "2\n    - 3\n  - \n  - lead\nm:\n\nn:\n\no:\n  p:\n    q: False\nr: "
          "-0.0\ns: 1.5e+300\nt: 0.0001\nu: 100.0",
          "render");
}

void TestRounding() {
  const std::vector<std::pair<double, std::string>> cases = {
      {0.5, "0.5"},     {1.0, "1.0"},          {0.123456, "0.1235"},
      {0.99995, "1.0"}, {5e-05, "0.0001"},     {0.00015, "0.0001"},
      {2.5e-05, "0.0"}, {1.0 / 3.0, "0.3333"}, {0.66665, "0.6666"}};
  for (const auto& [x, want] : cases)
    CheckEq(kev::PythonFloatRepr(kev::RoundProbability(x)), want,
            "round_prob " + std::to_string(x));
}

void TestAnswers() {
  const std::string body =
      R"({"state": "s", "questions": {"c": {"type": "choice", "criteria": )"
      R"({"\u00e9": null, "b\"q": "x"}}, "s": {"type": "score", "criteria": )"
      R"(["lo", "hi \u2615"]}, "n": {"type": "noul"}}})";
  std::string error;
  const auto request = kev::ParseRequest(body, &error);
  Check(request.has_value(), "parse answers request: " + error);
  const std::vector<std::vector<double>> p = {
      {0.25, 0.75}, {0.4, 0.6}, {0.3, 0.7}};
  CheckEq(kev::AnswersJson(*request, p, kev::JsonStyle::kPythonDumps),
          R"({"c": {"type": "choice", "choice": "b\"q", "confidence": 0.5, )"
          R"("probabilities": {"\u00e9": 0.25, "b\"q": 0.75}}, "s": {"type": )"
          R"("score", "score": 0.6, "legend": {"0": "lo", "1": "hi \u2615"}, )"
          R"("probabilities": {"0": 0.4, "1": 0.6}, "confidence": 0.2}, "n": )"
          R"({"type": "noul", "noul": 0.7}})",
          "json.dumps answers");
  CheckEq(kev::AnswersJson(*request, p, kev::JsonStyle::kFastApi),
          "{\"c\":{\"type\":\"choice\",\"choice\":\"b\\\"q\",\"confidence\":0."
          "5,\"probabilities\":{\"\xC3\xA9\":0.25,\"b\\\"q\":0.75}},\"s\":{"
          "\"type\":\"score\",\"score\":0.6,\"legend\":{\"0\":\"lo\",\"1\":"
          "\"hi \xE2\x98\x95\"},\"probabilities\":{\"0\":0.4,\"1\":0.6},"
          "\"confidence\":0.2},\"n\":{\"type\":\"noul\",\"noul\":0.7}}",
          "response answers");
}

void TestOptions() {
  const std::string body =
      R"({"state": null, "model": "jev-latest", "questions": {)"
      R"("n": {"type": "noul", "instructions": {"ask": "x"}, "criteria": )"
      R"({"true": "Money", "false": ""}}, "c": {"type": "choice", "criteria": )"
      R"({"a": {"k": [1, 2]}, "b": [], "c": null}}}})";
  std::string error;
  const auto request = kev::ParseRequest(body, &error);
  Check(request.has_value(), "parse options request: " + error);
  CheckEq(request->model, "jev-latest", "model");
  CheckEq(request->state, "", "null state");
  CheckEq(request->questions[0].instructions, "ask: x", "instructions");
  CheckEq(request->questions[0].options[0], "no", "noul empty description");
  CheckEq(request->questions[0].options[1], "yes: Money", "noul description");
  CheckEq(request->questions[1].options[0], "a: k:\n  - 1\n  - 2",
          "choice object description");
  CheckEq(request->questions[1].options[1], "b: ", "choice empty list");
  CheckEq(request->questions[1].options[2], "c", "choice null description");
}

void TestLimits() {
  std::string error;
  std::string nested = "1";
  for (int i = 0; i < 40; ++i)
    nested = "[" + nested + "]";
  Check(!kev::ParseRequest(R"({"state": )" + nested +
                               R"(, "questions": {"q": {"type": "noul"}}})",
                           &error),
        "40-level state accepted");
  Check(error.find("nested") != std::string::npos, "depth error: " + error);
  std::string wide = R"({"state": "s", "questions": {)";
  for (int i = 0; i < 129; ++i)
    wide += (i ? ", " : "") + std::string("\"q") + std::to_string(i) +
            R"(": {"type": "noul"})";
  Check(!kev::ParseRequest(wide + "}}", &error), "129 questions accepted");
}

void TestRejections() {
  std::string error;
  Check(!kev::ParseRequest(R"({"questions": {"q": {"type": "noul"}}})", &error),
        "missing state accepted");
  Check(!kev::ParseRequest(R"({"state": "", "questions": {}})", &error),
        "empty questions accepted");
  Check(
      !kev::ParseRequest(
          R"({"state": "", "questions": {"q": {"type": "choice", "criteria": {}}}})",
          &error),
      "empty choice accepted");
  Check(!kev::ParseRequest(
            R"({"state": "", "questions": {"q": {"type": "maybe"}}})", &error),
        "unknown type accepted");
  Check(
      !kev::ParseRequest(
          R"({"state": "", "model": 3, "questions": {"q": {"type": "noul"}}})",
          &error),
      "numeric model accepted");
}

}  // namespace

int main() {
  TestRender();
  TestRounding();
  TestAnswers();
  TestOptions();
  TestRejections();
  TestLimits();
  std::cout << "PASS kev_systemone_test\n";
  return 0;
}
