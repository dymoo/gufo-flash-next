// Kev parity against golden answers from tools/kev/reference_oracle.py.
//   KEV_GGUF=kev-4b-q8_0.gguf KEV_GOLDEN=kev4b-golden.jsonl kev_parity_test
// Token ids must match the reference exactly. Probabilities are reported as
// the largest absolute difference and argmax flips; KEV_MAX_ABS_DIFF sets the
// failure bound (default 0.05). The packed pass must also give the same
// answers alone, batched with every other case, and from cached states
// (KEV_BATCH_TOLERANCE, default 1e-3). KEV_BASELINE=1 adds the unbatched
// reference executor.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/kev/model.hpp"
#include "src/models/kev/systemone.hpp"

namespace kev = gufo::models::kev;
namespace json = gufo::json;

namespace {

using Probabilities = std::vector<std::vector<double>>;

std::size_t Argmax(const std::vector<double>& p) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < p.size(); ++i)
    if (p[i] > p[best])
      best = i;
  return best;
}

double MaxDiff(const Probabilities& a, const Probabilities& b,
               std::size_t* flips = nullptr) {
  double worst = 0.0;
  for (std::size_t q = 0; q < a.size(); ++q) {
    for (std::size_t i = 0; i < a[q].size(); ++i)
      worst = std::max(worst, std::fabs(a[q][i] - b[q][i]));
    if (flips != nullptr)
      *flips += Argmax(a[q]) != Argmax(b[q]);
  }
  return worst;
}

double Ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

struct Case {
  std::string name;
  kev::Request request;
  kev::Encoding encoding;
  Probabilities want;
  std::string answers_json;
};

}  // namespace

int main() {
  const char* gguf = std::getenv("KEV_GGUF");
  const char* golden = std::getenv("KEV_GOLDEN");
  if (gguf == nullptr || golden == nullptr) {
    std::cout << "SKIP kev_parity_test: set KEV_GGUF and KEV_GOLDEN\n";
    return 77;
  }
  const auto env_double = [](const char* name, double fallback) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::atof(v) : fallback;
  };
  const double bound = env_double("KEV_MAX_ABS_DIFF", 0.05);
  const double batch_tolerance = env_double("KEV_BATCH_TOLERANCE", 1e-3);
  const bool baseline = std::getenv("KEV_BASELINE") != nullptr;
  std::string error;
  kev::ModelOptions options;
  options.limits.max_state = 16384;
  options.limits.max_row = 16384 + 8192;
  auto model = kev::Model::Load(gguf, options, &error);
  if (model == nullptr) {
    std::cerr << "FAIL kev_parity_test: " << error << '\n';
    return 1;
  }
  std::cout << "packed bank and pass buffers: "
            << model->ReservedBytes() / (1024 * 1024) << " MiB\n";

  bool failed = false;
  std::vector<Case> cases;
  std::ifstream in(golden);
  std::string line;
  while (std::getline(in, line)) {
    const json::Value g = json::parse(line);
    Case c;
    c.name = g.member_str("name");
    auto request = kev::ParseRequest(g.find("request")->str(), &error);
    if (!request) {
      std::cerr << c.name << ": parse failed: " << error << '\n';
      failed = true;
      continue;
    }
    auto encoding = model->encoder().Encode(*request, model->limits(), &error);
    if (!encoding) {
      std::cerr << c.name << ": encode failed: " << error << '\n';
      failed = true;
      continue;
    }
    std::vector<std::uint32_t> ids = encoding->state;
    for (const auto& row : encoding->rows)
      ids.insert(ids.end(), row.ids.begin(), row.ids.end());
    const auto& want_ids = g.find("encoding")->find("ids")->items();
    std::size_t first = 0;
    while (first < ids.size() && first < want_ids.size() &&
           static_cast<std::uint32_t>(want_ids[first].as_double()) ==
               ids[first])
      ++first;
    if (first != ids.size() || ids.size() != want_ids.size()) {
      std::cerr << c.name << ": token ids differ (got " << ids.size()
                << ", want " << want_ids.size() << ", first difference at "
                << first << ")\n";
      failed = true;
      continue;
    }
    for (const auto& q : g.find("probabilities")->items()) {
      std::vector<double> row;
      for (const auto& v : q.items())
        row.push_back(v.as_double());
      c.want.push_back(std::move(row));
    }
    if (const json::Value* answers = g.find("answers_json"))
      c.answers_json = answers->str();
    c.request = std::move(*request);
    c.encoding = std::move(*encoding);
    cases.push_back(std::move(c));
  }

  // Each case alone, cold.
  double worst = 0.0;
  std::size_t flips_total = 0;
  std::size_t questions_total = 0;
  std::vector<Probabilities> alone;
  for (const Case& c : cases) {
    const auto start = std::chrono::steady_clock::now();
    alone.push_back(model->Score(c.encoding));
    const double ms = Ms(start);
    std::size_t flips = 0;
    const double diff = MaxDiff(c.want, alone.back(), &flips);
    if (!c.answers_json.empty() &&
        kev::AnswersJson(c.request, c.want, kev::JsonStyle::kPythonDumps) !=
            c.answers_json) {
      std::cerr << c.name << ": answers formatting differs\n";
      failed = true;
    }
    worst = std::max(worst, diff);
    flips_total += flips;
    questions_total += c.want.size();
    failed |= diff > bound;
    std::cout << std::left << std::setw(28) << c.name << std::right
              << std::setw(7) << c.encoding.Tokens() << " tokens  max|dp| "
              << std::fixed << std::setprecision(5) << diff << "  flips "
              << flips << "/" << c.want.size() << "  " << std::setprecision(1)
              << ms << " ms\n";
  }
  std::cout << "reference: worst max|dp| " << std::setprecision(5) << worst
            << ", argmax flips " << flips_total << "/" << questions_total
            << '\n';

  // Everything in one batch, then again from cached states.
  std::vector<const kev::Encoding*> batch;
  std::size_t batch_tokens = 0;
  for (const Case& c : cases) {
    batch.push_back(&c.encoding);
    batch_tokens += c.encoding.Tokens();
  }
  for (const char* label : {"batch (cold)", "batch (cached)"}) {
    const auto before = model->stats();
    const auto start = std::chrono::steady_clock::now();
    const auto together = model->ScoreBatch(batch);
    const double ms = Ms(start);
    const auto after = model->stats();
    double diff = 0.0;
    for (std::size_t i = 0; i < cases.size(); ++i)
      diff = std::max(diff, MaxDiff(alone[i], together[i]));
    failed |= diff > batch_tolerance;
    std::cout << label << ": " << cases.size() << " requests, " << batch_tokens
              << " tokens, " << (after.passes - before.passes) << " passes of "
              << (after.pass_tokens - before.pass_tokens) << " tokens, "
              << (after.cache_hits - before.cache_hits) << " cache hits, "
              << std::setprecision(1) << ms << " ms, max|dp| vs alone "
              << std::setprecision(6) << diff << '\n';
  }

  if (baseline) {
    double diff = 0.0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
      if (cases[i].encoding.state.size() > 12000)
        continue;
      diff = std::max(
          diff, MaxDiff(alone[i], model->ScoreBaseline(cases[i].encoding)));
    }
    std::cout << "baseline executor: max|dp| vs packed " << std::setprecision(6)
              << diff << '\n';
  }
  std::cout << (failed ? "FAIL" : "PASS") << " kev_parity_test\n";
  return failed ? 1 : 0;
}
