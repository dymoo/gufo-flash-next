#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckLengthController() {
  qfn::MtpLengthController controller(7);
  const auto initial = controller.Choose(7);
  Require(controller.Choose(0) == 0, "empty draft budget was ignored");
  Require(controller.Choose(1) == 1, "one-draft budget was ignored");
  controller.Observe(0, 0);
  Require(controller.Choose(7) == initial, "empty cycle changed policy");

  for (unsigned i = 0; i < 12; ++i) {
    const auto length = controller.Choose(7);
    controller.Observe(length, length);
  }
  Require(controller.Choose(7) == 7,
          "fully accepted prefixes did not grow to the maximum length");
  Require(controller.Choose(2) == 2, "remaining output budget was ignored");
  for (unsigned i = 0; i < 12; ++i) {
    controller.Observe(0, controller.Choose(7));
  }
  Require(controller.Choose(7) == 0,
          "unprofitable speculation did not back off to AR");
  qfn::MtpLengthController backed_off(7);
  Require(backed_off.Restore(controller.State()), "backoff restore failed");
  for (unsigned i = 0; i < 16; ++i) {
    Require(controller.Choose(7) == 0 && backed_off.Choose(7) == 0,
            "AR interval ended early");
    controller.ObserveArToken();
    backed_off.ObserveArToken();
  }
  Require(controller.Choose(7) == 1 && backed_off.Choose(7) == 1,
          "AR backoff did not retry a proposal");
  controller.Reset();
  Require(controller.Choose(7) == initial, "reset retained old acceptance");
  controller.Observe(5, 6);
  qfn::MtpLengthController restored(7);
  Require(restored.Restore(controller.State()), "controller restore failed");
  for (unsigned budget = 0; budget <= 8; ++budget) {
    Require(restored.Choose(budget) == controller.Choose(budget),
            "restored policy changes draft widths");
  }
  auto invalid = controller.State();
  invalid.successes[0] = -1.0F;
  Require(!restored.Restore(invalid) && !restored.Restore({}),
          "invalid controller state accepted");

  // The unverified suffix is not additional evidence of failure.
  qfn::MtpLengthController short_tail(7);
  qfn::MtpLengthController long_tail(7);
  for (unsigned accepted : {2U, 0U, 1U, 2U, 1U, 0U}) {
    short_tail.Observe(accepted, accepted + 1);
    long_tail.Observe(accepted, 7);
    for (unsigned budget = 0; budget <= 9; ++budget) {
      Require(short_tail.Choose(budget) == long_tail.Choose(budget),
              "unverified suffix biased the acceptance estimate");
    }
  }
  for (unsigned limit = 1; limit <= 7; ++limit) {
    qfn::MtpLengthController a(limit);
    qfn::MtpLengthController b(limit);
    for (unsigned cycle = 0; cycle < 32; ++cycle) {
      const auto budget = cycle % 10;
      const auto length = a.Choose(budget);
      Require(length == b.Choose(budget) && length <= budget && length <= limit,
              "controller violated a limit or changed on replay");
      const unsigned accepted = length == 0 ? 0 : cycle % (length + 1);
      a.Observe(accepted, length);
      b.Observe(accepted, length);
      if (length == 0) {
        a.ObserveArToken();
        b.ObserveArToken();
      }
    }
  }
  qfn::MtpLengthController early_failure(7), late_failure(7);
  for (unsigned i = 0; i < 12; ++i) {
    early_failure.Observe(0, 7);
    late_failure.Observe(4, 7);
  }
  Require(early_failure.Choose(7) == 0 && late_failure.Choose(7) > 1,
          "conditional acceptance by depth was collapsed to one probability");
}

void CheckCompactProposals() {
  const std::array<float, 5> target_logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    sampling::SamplerState sampler(test.config, history);
    const auto p = sampler.Distribution(target_logits);
    for (std::uint64_t seed = 0; seed < 16; ++seed) {
      auto rng = seed;
      auto replay_rng = seed;
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &rng);
      const auto replay =
          qfn::SampleMtpProposal(candidates, sampler, &replay_rng);
      Require(q.token == replay.token && q.ids == replay.ids &&
                  q.probabilities == replay.probabilities && rng == replay_rng,
              "compact proposal is not reproducible");
      double sum = 0.0;
      std::array<double, 5> draft{};
      for (std::size_t i = 0; i < q.size; ++i) {
        const double mass = q.probabilities[i] * double(1U << 24);
        Require(mass >= 0.0 && mass == std::floor(mass),
                "proposal probability is not an exact discrete mass");
        draft[q.ids[i]] += q.probabilities[i];
        sum += q.probabilities[i];
      }
      Require(
          sum == 1.0 && q.probability > 0.0F && draft[q.token] == q.probability,
          "sampled q differs from the exported distribution");
      double accepted = 0.0, residual = 0.0;
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        accepted += std::min(probability, draft[i]);
        residual += std::max(probability - draft[i], 0.0);
      }
      Require(std::abs(accepted + residual - 1.0) < 1e-12,
              "acceptance and residual mass do not conserve the target");
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        const double correction =
            residual > 0.0
                ? (1.0 - accepted) * std::max(probability - draft[i], 0.0) /
                      residual
                : 0.0;
        Require(std::abs(std::min(probability, draft[i]) + correction -
                         probability) < 1e-12,
                "bounded draft support changes target output probability");
      }
    }
  }
  sampling::SamplerState counted({.temperature = 1,
                                  .repeat_penalty = 1.5F,
                                  .repeat_last_n = 1,
                                  .frequency_penalty = 0.25F,
                                  .presence_penalty = 0.5F},
                                 std::array<sampling::TokenId, 2>{1, 1});
  counted.Accept(std::array<sampling::TokenId, 2>{3, 3});
  counted.Accept(std::vector<sampling::TokenId>(80, 0));
  counted.Accept(4);
  std::uint64_t rng = 17;
  const auto proposal = qfn::SampleMtpProposal(candidates, counted, &rng);
  // Token 3 is outside the repetition window but keeps both generated counts.
  // Token 1 occurs only in the prompt. Token 4 gets all three penalties.
  const std::array<double, 3> adjusted{1.2 - 2 * 0.25 - 0.5, 0.1,
                                       -0.7 * 1.5 - 0.25 - 0.5};
  double sum = 0;
  for (auto logit : adjusted)
    sum += std::exp(logit);
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto index = std::find(candidates.ids.begin(),
                                 candidates.ids.begin() + 3, proposal.ids[i]) -
                       candidates.ids.begin();
    Require(index < 3 &&
                std::abs(proposal.probabilities[i] -
                         std::exp(adjusted[index]) / sum) < 2.0 / (1U << 24),
            "compact MTP proposal lost full generated penalty counts");
  }
}

void CheckCalibratedCosts() {
  qfn::MtpLengthController single(7, 1), concurrent(7, 32);
  auto state = single.State();
  state.successes.fill(6.0F);
  state.failures.fill(4.0F);
  state.explored_depth = 7;
  state.failed_depths = 127;
  Require(single.Restore(state) && concurrent.Restore(state),
          "calibrated controller state rejected");
  Require(single.Choose(7, 0) > concurrent.Choose(7, 0),
          "concurrent verification cost did not shorten the draft chain");
  single.Observe(0, 1);
  concurrent.Observe(0, 1);
  Require(single.Choose(7) > 0 && concurrent.Choose(7) == 0,
          "measured verification cost did not change the AR break-even point");

  for (unsigned c : {1, 2, 4, 8, 16, 32, 64}) {
    qfn::MtpLengthController perfect(7, c);
    for (unsigned cycle = 0; cycle < 8; ++cycle) {
      const auto length = perfect.Choose(7);
      perfect.Observe(length, length);
    }
    Require(perfect.Choose(7) == 7,
            "perfect acceptance did not retain the fastest measured width");
    for (unsigned depth : {0, 4096, 32768, 131072, 262144}) {
      const auto costs = qfn::MtpCycleCosts(depth, c);
      Require(costs[0] > 0, "AR baseline cost is not positive");
      for (std::size_t i = 1; i < costs.size(); ++i)
        Require(std::isfinite(costs[i]) && costs[i] > costs[i - 1],
                "calibrated costs do not increase with verification work");
    }
  }
}

qfn::MtpLengthController Trained(double acceptance) {
  qfn::MtpLengthController controller(7);
  // Seeded chains with the requested per-depth acceptance.
  std::uint64_t rng = 11;
  for (unsigned cycle = 0; cycle < 64; ++cycle) {
    std::size_t accepted = 0;
    while (accepted < 7 && gufo::sampling::Uniform(&rng) < acceptance)
      ++accepted;
    controller.Observe(accepted, 7);
  }
  return controller;
}

using Row = qfn::MtpBatchController::Row;

double Throughput(const qfn::StepCostModel& model, std::span<const Row> rows,
                  std::span<const qfn::DraftPlan> plans) {
  qfn::StepShape shape;
  double tokens = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& plan = plans[i];
    shape.sessions += 1;
    shape.rows += 1 + plan.drafts;
    if (!plan.lookup) {
      shape.mtp_drafts += plan.drafts;
      shape.rounds = std::max(shape.rounds, plan.drafts);
    }
    tokens += plan.drafts == 0 ? 1.0
              : plan.lookup    ? rows[i].lookup->ExpectedTokens(plan.drafts)
                               : rows[i].mtp->ExpectedTokens(plan.drafts);
  }
  return tokens / model.Predict(shape);
}

void CheckStepModel() {
  // A synthetic machine: the online fit must recover it from mixed shapes.
  const qfn::StepFeatures truth{20.0, 0.3, 1.0, 0.4, 0.2, 3.0, 2.0};
  qfn::StepCostModel model;
  std::uint64_t rng = 7;
  for (unsigned step = 0; step < 4000; ++step) {
    const auto b =
        1 + static_cast<std::uint32_t>(gufo::sampling::Uniform(&rng) * 64);
    const auto w =
        static_cast<std::uint32_t>(gufo::sampling::Uniform(&rng) * 4);
    const auto lookups =
        static_cast<std::uint32_t>(gufo::sampling::Uniform(&rng) * b * w / 2);
    const qfn::StepShape shape{
        .sessions = b,
        .rows = b * (1 + w),
        .mtp_drafts = b * w - lookups,
        .rounds = w,
        .row_context = static_cast<float>(gufo::sampling::Uniform(&rng) * 8)};
    const double noise = (gufo::sampling::Uniform(&rng) - 0.5) * 2.0;
    model.Observe(shape, static_cast<float>(qfn::PredictStep(truth, shape) +
                                            (step == 100 ? 5000 : noise)));
  }
  for (std::uint32_t b : {1U, 8U, 32U, 64U})
    for (std::uint32_t w : {0U, 2U}) {
      const qfn::StepShape shape{.sessions = b,
                                 .rows = b * (1 + w),
                                 .mtp_drafts = b * w,
                                 .rounds = w,
                                 .row_context = 1};
      const double want = qfn::PredictStep(truth, shape);
      Require(std::abs(model.Predict(shape) - want) < 0.05 * want,
              "online step model did not converge to the measured machine");
    }
  model.Observe({.sessions = 0}, 1.0F);
  model.Observe({.sessions = 2, .rows = 2}, std::nanf(""));
  Require(std::isfinite(model.Predict({.sessions = 2, .rows = 2})),
          "invalid step timing corrupted the model");
}

void CheckBatchAllocation() {
  const auto good = Trained(0.97), bad = Trained(0.1),
             lookup_good = Trained(0.97);
  const qfn::MtpLengthController fresh(7);
  // Prior costs at C32: drafts pay only for high-acceptance sessions.
  qfn::MtpBatchController policy;
  std::vector<Row> rows;
  for (unsigned i = 0; i < 32; ++i)
    rows.push_back({i % 2 ? &good : &bad, &fresh, 7, 0, 0});
  std::vector<qfn::DraftPlan> plans(rows.size());
  policy.Choose(rows, plans);  // cycle 1
  for (std::size_t i = 0; i < rows.size(); ++i) {
    Require(i % 2 ? plans[i].drafts > 0 : plans[i].drafts == 0,
            "draft allocation ignored per-session acceptance");
    Require(!plans[i].lookup, "lookup drafts without a match");
  }
  Require(Throughput(policy.Model(), rows, plans) >
              Throughput(policy.Model(), rows,
                         std::vector<qfn::DraftPlan>(rows.size())),
          "chosen drafts do not beat plain decoding under the model");

  // Never below AR under the model, for any acceptance mix and occupancy.
  std::uint64_t rng = 3;
  std::vector<qfn::MtpLengthController> mixes;
  for (unsigned i = 0; i < 64; ++i)
    mixes.push_back(Trained(gufo::sampling::Uniform(&rng)));
  for (std::size_t b : {1, 2, 8, 16, 32, 64}) {
    std::vector<Row> mixed;
    for (std::size_t i = 0; i < b; ++i)
      mixed.push_back({&mixes[i], &mixes[63 - i], 7,
                       static_cast<std::uint32_t>(i % 8), 4096});
    std::vector<qfn::DraftPlan> chosen(b);
    qfn::MtpBatchController sized;
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
      sized.Choose(mixed, chosen);
      for (std::size_t i = 0; i < b; ++i)
        Require(
            chosen[i].drafts <= 7 &&
                (!chosen[i].lookup || chosen[i].drafts <= mixed[i].lookup_size),
            "allocation exceeded a draft budget or lookup match");
      Require(
          Throughput(policy.Model(), mixed, chosen) >=
              Throughput(policy.Model(), mixed, std::vector<qfn::DraftPlan>(b)),
          "allocation predicted fewer tokens per ms than AR");
    }
  }

  // Lookup continuations win when they are accepted more often.
  qfn::MtpBatchController hybrid;
  const std::array<Row, 2> pair{
      {{&bad, &lookup_good, 7, 7, 0}, {&good, &bad, 7, 7, 0}}};
  std::array<qfn::DraftPlan, 2> picked{};
  hybrid.Choose(pair, picked);
  Require(picked[0].lookup && picked[0].drafts > 0 && !picked[1].lookup,
          "hybrid allocation did not pick the better draft source");
  const std::array<Row, 1> mtp_only{{{&good, &bad, 7, 7, 0}}};
  std::array<qfn::DraftPlan, 1> single{};
  hybrid.Choose(mtp_only, single);
  Require(!single[0].lookup && single[0].drafts > 0,
          "hybrid allocation preferred a poorly accepted lookup");
  const std::array<Row, 1> short_match{{{&bad, &lookup_good, 7, 2, 0}}};
  std::array<qfn::DraftPlan, 1> one{};
  hybrid.Choose(short_match, one);
  Require(!one[0].lookup || one[0].drafts <= 2, "lookup exceeded its match");

  // Sampled rows keep their fixed plan; budgets bind.
  const std::array<Row, 2> fixed{
      {{&good, &fresh, 7, 0, 0, qfn::DraftPlan{3, false}},
       {&good, &fresh, 1, 0, 0}}};
  hybrid.Choose(fixed, picked);
  Require(picked[0].drafts == 3 && picked[1].drafts <= 1,
          "fixed plan or budget ignored");

  // Measured expensive drafts turn speculation off; cheap ones widen it.
  qfn::MtpBatchController measured;
  std::vector<Row> cohort;
  for (unsigned i = 0; i < 8; ++i)
    cohort.push_back({&good, &fresh, 7, 0, 0});
  std::vector<qfn::DraftPlan> out(cohort.size());
  const auto train = [&](double per_draft_ms, unsigned steps) {
    for (unsigned step = 0; step < steps; ++step) {
      const std::uint32_t w = step % 4, b = 1 + step % 64;
      const qfn::StepShape shape{
          .sessions = b, .rows = b * (1 + w), .mtp_drafts = b * w, .rounds = w};
      measured.Observe(shape,
                       static_cast<float>(30 + 2.0 * b + per_draft_ms * b * w));
    }
  };
  train(40.0, 400);
  measured.Choose(cohort, out);
  for (const auto& plan : out)
    Require(plan.drafts == 0, "unprofitable measured drafts were chosen");
  train(0.05, 3000);
  unsigned total = 0;
  for (unsigned cycle = 0; cycle < 3; ++cycle) {
    measured.Choose(cohort, out);
    for (const auto& plan : out)
      total += plan.drafts;
  }
  Require(total >= 3 * 8 * 4, "cheap measured drafts were not extended");
  // Periodic plain control steps and exploration of idle sessions.
  bool control = false, explored = false;
  for (unsigned cycle = 0; cycle < 40; ++cycle) {
    measured.Choose(cohort, out);
    control = control || std::ranges::all_of(out, [](const auto& plan) {
                return plan.drafts == 0;
              });
  }
  qfn::MtpBatchController idle_policy;
  const std::array<Row, 1> idle{{{&bad, &fresh, 7, 0, 0}}};
  for (unsigned cycle = 0; cycle < 40; ++cycle) {
    idle_policy.Choose(idle, one);
    explored = explored || one[0].drafts > 0;
  }
  Require(control && explored, "controller never measured AR or re-probed");
}

void CheckSessionDraftChoice() {
  const auto good = Trained(0.9), bad = Trained(0.1);
  const auto a = qfn::ChooseSessionDraft(bad, good, 7, 5, 0, 1);
  Require(a.lookup && a.drafts > 0 && a.drafts <= 5,
          "session plan did not prefer the accepted lookup source");
  const auto b = qfn::ChooseSessionDraft(good, bad, 7, 5, 0, 1);
  Require(!b.lookup && b.drafts > 0, "session plan preferred a poor lookup");
  const auto c = qfn::ChooseSessionDraft(good, good, 7, 0, 0, 1);
  Require(!c.lookup, "session plan used a missing lookup");
  const auto again = qfn::ChooseSessionDraft(bad, good, 7, 5, 0, 1);
  Require(again.drafts == a.drafts && again.lookup == a.lookup,
          "session plan is not deterministic");
}

void CheckSampledOutputFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  constexpr unsigned trials = 100000;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 73});
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &draft_rng);
      const auto token = qfn::VerifyMtpProposal(logits, q, sampler).token;
      ++counts[token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      Require(std::abs(double(counts[token]) / trials - p.probability(token)) <
                  0.008,
              "proposal and correction draws distort target frequencies");
    }
  }
}

void CheckVerificationBoundaries() {
  const std::array<float, 3> logits{0, -1, -2};
  const std::array<sampling::TokenId, 3> history{1, 1, 2};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    auto config = test.config;
    config.temperature = 0.5F;
    config.seed = 6;
    // top-p and min-p cutoffs immediately around the finite FP64 boundary.
    for (const float top_p : {0.8F, 0.8668133F, 0.8668134F, 1.0F}) {
      config.top_p = top_p;
      for (const float min_p : {0.0F, 0.13533527F, 0.13533530F}) {
        config.min_p = min_p;
        sampling::SamplerState sampler(config, history);
        sampler.Accept(1);
        const auto target = sampler.Distribution(logits);
        for (unsigned draft = 0; draft < logits.size(); ++draft) {
          qfn::MtpProposal proposal;
          proposal.size = 1;
          proposal.ids[0] = proposal.token = draft;
          proposal.probabilities[0] = proposal.probability = 1;
          for (unsigned draw = 0; draw < 16; ++draw) {
            auto reference = sampler;
            auto replay = sampler;
            const bool accepted =
                reference.Uniform() < target.probability(draft);
            const auto expected =
                accepted ? draft
                         : target.SampleResidual(
                               std::span(proposal.ids).first(1),
                               std::span(proposal.probabilities).first(1),
                               reference.mutable_rng_state());
            const auto actual =
                qfn::VerifyMtpProposal(logits, proposal, sampler);
            const auto repeated =
                qfn::VerifyMtpProposal(logits, proposal, replay);
            Require(actual.accepted == accepted && actual.token == expected &&
                        sampler.rng_state() == reference.rng_state() &&
                        actual.token == repeated.token &&
                        actual.accepted == repeated.accepted &&
                        sampler.rng_state() == replay.rng_state(),
                    "MTP acceptance/residual must use the canonical AR "
                    "distribution exactly");
          }
        }
      }
    }
  }
}

int main() {
  try {
    CheckVerificationBoundaries();
    CheckLengthController();
    CheckCalibratedCosts();
    CheckStepModel();
    CheckBatchAllocation();
    CheckSessionDraftChoice();
    CheckCompactProposals();
    CheckSampledOutputFrequencies();
    const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
    const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
    // Deterministic (greedy or prompt-lookup) drafts: every sampler, random
    // ones included, reproduces the AR tokens, RNG and history exactly.
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      for (unsigned accepted = 0; accepted < 4; ++accepted) {
        sampling::SamplerState ar(test.config, history);
        auto speculative = ar;
        for (unsigned i = 0; i < accepted; ++i) {
          const auto token = ar.Sample(logits);
          ar.Accept(token);
          Require(qfn::VerifyDraft(logits, static_cast<std::int32_t>(token),
                                   speculative, [](auto) { return false; }) ==
                      qfn::DraftDecision::kAccept,
                  "matching target sample was rejected");
        }
        const auto before = speculative.rng_state();
        const auto saved_history = std::vector(speculative.history().begin(),
                                               speculative.history().end());
        const auto token = ar.Sample(logits);
        Require(qfn::VerifyDraft(logits, (token + 1) % logits.size(),
                                 speculative, [](auto) { return false; }) ==
                    qfn::DraftDecision::kReject,
                "mismatching target sample was accepted");
        Require(speculative.rng_state() == before &&
                    std::vector(speculative.history().begin(),
                                speculative.history().end()) == saved_history,
                "rejected suffix consumed RNG or history");
        Require(speculative.Sample(logits) == token,
                "deferred rejection did not reproduce the AR token");
        ar.Accept(token);
        speculative.Accept(token);
        Require(ar.rng_state() == speculative.rng_state() &&
                    std::vector(ar.history().begin(), ar.history().end()) ==
                        std::vector(speculative.history().begin(),
                                    speculative.history().end()),
                "accepted prefix differs from AR state");
        auto stop_reference = speculative;
        const auto stop_token = stop_reference.Sample(logits);
        Require(qfn::VerifyDraft(
                    logits, (stop_token + 1) % logits.size(), speculative,
                    [&](auto t) {
                      return t == static_cast<std::int32_t>(stop_token);
                    }) == qfn::DraftDecision::kStop,
                "target stop must end verification even when draft differs");
        Require(speculative.rng_state() == stop_reference.rng_state(),
                "stop draw was discarded");
      }
    }
    std::cout
        << "MTP policy replay, greedy prefix/RNG/stops and compact proposal "
           "probabilities pass all sampling configurations\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
