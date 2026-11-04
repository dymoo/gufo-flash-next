#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestGreedySelectsFiniteArgmaxWithoutAdvancingRng() {
  const std::array<float, 4> logits = {
      -1.0F, 5.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  std::uint64_t rng_state = 1234;

  const auto token = gufo::sampling::SampleLogits(logits, 0.0F, &rng_state);

  Expect(token == 1, "greedy sampling selects the finite argmax");
  Expect(rng_state == 1234, "greedy sampling does not consume RNG state");
}

void TestLogprobProbeAndForcedTokens() {
  const std::array<float, 5> logits = {
      1.0F, 3.0F, -std::numeric_limits<float>::infinity(), 2.0F, 0.0F};
  const double normalizer =
      std::log(std::exp(1.0) + std::exp(3.0) + std::exp(2.0) + std::exp(0.0));
  gufo::sampling::SamplerState sampler(
      {.top_logprobs = 2, .forced_tokens = {4, 3}});

  Expect(sampler.Sample(logits) == 4, "first forced token overrides greedy");
  const auto first = sampler.TakeLogprobs();
  Expect(first && first->token == 4 &&
             std::abs(first->logprob - (0.0 - normalizer)) < 1e-9,
         "forced token logprob is the raw log-softmax");
  Expect(first->top.size() == 2 && first->top[0].first == 1 &&
             first->top[1].first == 3 &&
             std::abs(first->top[0].second - (3.0 - normalizer)) < 1e-9,
         "top alternatives are sorted by logprob");
  Expect(!sampler.TakeLogprobs(), "logprobs are taken once");
  auto preview = sampler;
  Expect(preview.Sample(logits) == 3, "copies continue the forced sequence");
  Expect(sampler.Sample(logits) == 3, "second forced token");
  Expect(sampler.Sample(logits) == 1, "greedy resumes after forced tokens");
  Expect(sampler.TakeLogprobs()->top.size() == 2, "sampled tokens also report");

  gufo::sampling::SamplerState silent;
  (void)silent.Sample(logits);
  Expect(!silent.TakeLogprobs(), "reporting is opt-in");
  bool rejected = false;
  try {
    gufo::sampling::SamplerState bad({.forced_tokens = {9}});
    (void)bad.Sample(logits);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "forced tokens must be inside the vocabulary");
}

void TestDefaultConfigPreservesGreedyDecoding() {
  const std::array<float, 4> logits = {-1.0F, 5.0F, 3.0F, 2.0F};
  gufo::sampling::SamplerState sampler;

  const auto before = sampler.rng_state();
  const auto token = sampler.Sample(logits);

  Expect(token == 1, "default sampler is greedy");
  Expect(sampler.rng_state() == before,
         "default sampler does not consume RNG state");
}

void TestTemperatureSamplingIsDeterministicAndNonGreedy() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  bool saw_first = false;
  bool saw_non_first = false;

  for (std::uint64_t seed = 1; seed <= 128; ++seed) {
    auto left_state = seed;
    auto right_state = seed;
    const auto left = gufo::sampling::SampleLogits(logits, 1.0F, &left_state);
    const auto right = gufo::sampling::SampleLogits(logits, 1.0F, &right_state);
    Expect(left == right && left_state == right_state,
           "temperature sampling is reproducible from the same RNG state");
    saw_first = saw_first || left == 0;
    saw_non_first = saw_non_first || left != 0;
  }

  Expect(saw_first && saw_non_first,
         "temperature sampling does not collapse to greedy argmax");
}

void TestTopKFiltersTheCandidateSet() {
  const std::array<float, 5> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-k retains exactly k candidates");
  Expect(distribution.probability(0) > 0.0 && distribution.probability(1) > 0.0,
         "top-k retains the highest-logit candidates");
  Expect(distribution.probability(2) == 0.0,
         "top-k removes lower-logit candidates");
}

void TestTopPFiltersByCumulativeProbability() {
  const std::array<float, 3> logits = {3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_p = 0.7F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-p retains the smallest cumulative-probability prefix");
  Expect(distribution.probability(2) == 0.0,
         "top-p removes the low-probability tail");
}

void TestMinPFiltersRelativeToTheBestToken() {
  const std::array<float, 3> logits = {3.0F, 2.4F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "min-p retains candidates above the relative threshold");
  Expect(distribution.probability(2) == 0.0,
         "min-p removes candidates far below the best token");
}

void TestMinKeepProvidesAFilterFloor() {
  const std::array<float, 4> logits = {5.0F, 4.0F, 3.0F, 2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 1;
  config.top_p = 0.1F;
  config.min_p = 0.99F;
  config.min_keep = 3;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 3,
         "min-keep prevents enabled filters from shrinking below its floor");
}

void TestTemperatureBeforeProbabilityFilters() {
  // Independent closed-form expectations for softmax([0,-1,-2] / T).
  // Also exercise the bounded (>1024 vocabulary) candidate selector.
  for (const std::size_t size : {3U, 2048U}) {
    std::vector<float> logits(size, -1000.0F);
    logits[0] = 0.0F;
    logits[1] = -1.0F;
    logits[2] = -2.0F;
    for (const bool nucleus : {false, true}) {
      gufo::sampling::SamplingConfig config{.temperature = 0.5F, .seed = 42};
      if (nucleus)
        config.top_p = 0.8F;
      else
        config.min_p = 0.2F;
      gufo::sampling::SamplerState sampler(config);
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            sampler.Distribution(logits)}) {
        Expect(distribution.entries().size() == 1 &&
                   distribution.probability(0) == 1.0,
               "temperature sharpens probabilities before filtering");
      }
      for (unsigned draw = 0; draw < 32; ++draw)
        Expect(sampler.Sample(logits) == 0,
               "ordinary decode uses the tempered candidate set");

      config.temperature = 2.0F;
      config.top_p = nucleus ? 0.8F : 1.0F;
      config.min_p = nucleus ? 0.0F : 0.5F;
      const double first = 1.0 / (1.0 + std::exp(-0.5));
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            gufo::sampling::SamplerState(config).Distribution(logits)}) {
        Expect(
            distribution.entries().size() == 2 &&
                std::abs(distribution.probability(0) - first) < 1e-12 &&
                std::abs(distribution.probability(1) - (1.0 - first)) < 1e-12,
            "temperature flattens probabilities before filtering");
      }
    }
  }
}

void TestRepeatPenaltyUsesCommittedHistory() {
  const std::array<float, 3> logits = {2.0F, 1.0F, 0.0F};
  const std::array<gufo::sampling::TokenId, 1> history = {0};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 3.0F;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 1,
         "repeat penalty can demote a recently used token");
}

void TestFrequencyAndPresencePenaltiesUseCounts() {
  const std::array<float, 2> logits = {2.0F, 1.4F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 0};

  gufo::sampling::SamplingConfig frequency_config;
  frequency_config.frequency_penalty = 0.4F;
  const auto frequency_distribution =
      gufo::sampling::BuildDistribution(logits, frequency_config, {}, history);
  Expect(frequency_distribution.best_token() == 1,
         "frequency penalty scales with occurrence count");

  gufo::sampling::SamplingConfig presence_config;
  presence_config.presence_penalty = 0.7F;
  const auto presence_distribution =
      gufo::sampling::BuildDistribution(logits, presence_config, {}, history);
  Expect(presence_distribution.best_token() == 1,
         "presence penalty applies once when a token is present");
}

void TestRepeatWindowIsBounded() {
  const std::array<float, 2> logits = {2.0F, 2.1F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 1};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 1;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 0,
         "repeat-last-n excludes tokens outside the configured window");
}

void TestFastGreedyMatchesNormalizedPenaltyRoute() {
  const std::array<float, 5> logits = {3.0F, 2.9F, 2.8F, 2.7F, 2.6F};
  const std::array<gufo::sampling::TokenId, 4> history = {0, 0, 1, 3};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 1.5F;
  config.frequency_penalty = 0.2F;
  config.presence_penalty = 0.1F;
  gufo::sampling::SamplerState sampler(config, history);

  const auto expected =
      gufo::sampling::BuildDistribution(logits, config, history).best_token();
  const auto actual = sampler.Sample(logits);

  Expect(actual == expected,
         "linear greedy route matches the normalized penalty route");
}

void TestFastTopKSamplingNeverEscapesSelectedSet() {
  const std::array<float, 6> logits = {8.0F, 7.0F, 1.0F, 0.0F, -1.0F, -2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  bool saw_first = false;
  bool saw_second = false;
  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    const auto token = sampler.Sample(logits);
    Expect(token <= 1, "top-k fast route retains only selected candidates");
    saw_first = saw_first || token == 0;
    saw_second = saw_second || token == 1;
  }
  Expect(saw_first && saw_second,
         "top-k fast route samples the retained distribution");
}

void TestFastMinPSamplingNeverEscapesRelativeThreshold() {
  const std::array<float, 4> logits = {3.0F, 2.5F, 1.0F, -4.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    Expect(sampler.Sample(logits) <= 1,
           "min-p linear route rejects tokens below the relative threshold");
  }
}

void TestSeedAndStateAreRequestLocal() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.seed = 42;
  config.repeat_last_n = 2;
  gufo::sampling::SamplerState left(config);
  gufo::sampling::SamplerState right(config);

  for (int sample = 0; sample < 16; ++sample) {
    Expect(left.Sample(logits) == right.Sample(logits),
           "equal seeds produce equal token sequences");
  }
  left.Accept(1);
  left.Accept(2);
  left.Accept(0);
  Expect(left.history().size() == 2 && left.history()[0] == 2 &&
             left.history()[1] == 0,
         "sampler retains only its bounded committed history");
  Expect(right.history().empty(), "sampler histories are request-local");
}

void TestRngIsKeyedByCommittedPosition() {
  const std::array<float, 4> logits = {0.3F, 0.1F, -0.2F, 0.0F};
  const gufo::sampling::SamplingConfig config{.temperature = 1.0F, .seed = 42};
  gufo::sampling::SamplerState plain(config);
  gufo::sampling::SamplerState busy(config);
  std::vector<std::uint64_t> keys;
  for (gufo::sampling::TokenId step = 0; step < 64; ++step) {
    // A batch or draft path that draws extra words at this position must
    // not shift any later draw.
    for (gufo::sampling::TokenId extra = 0; extra < step % 3; ++extra)
      (void)busy.Uniform();
    const auto token = plain.Sample(logits);
    plain.Accept(token);
    busy.Accept(token);
    keys.push_back(plain.rng_state());
    Expect(plain.rng_state() == busy.rng_state() &&
               plain.Sample(logits) == busy.Sample(logits),
           "draws depend on seed and position, not on earlier draw counts");
    busy = plain;
  }
  std::ranges::sort(keys);
  Expect(std::ranges::adjacent_find(keys) == keys.end(),
         "each position starts a distinct draw stream");

  gufo::sampling::SamplerState stepwise(config);
  gufo::sampling::SamplerState spanwise(config);
  stepwise.Accept(1);
  stepwise.Accept(2);
  const std::array<gufo::sampling::TokenId, 2> both{1, 2};
  spanwise.Accept(both);
  Expect(stepwise.rng_state() == spanwise.rng_state(),
         "a committed span keys the same position as single accepts");
  stepwise.ResetHistory({});
  Expect(
      stepwise.rng_state() == gufo::sampling::SamplerState(config).rng_state(),
      "a new request restarts at position zero");
  gufo::sampling::SamplerState other_seed({.temperature = 1.0F, .seed = 43});
  Expect(other_seed.rng_state() !=
             gufo::sampling::SamplerState(config).rng_state(),
         "seeds key independent streams");
}

void TestDistributionIsNormalized() {
  const std::array<float, 4> logits = {2.0F, 1.0F, 0.0F, -1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.8F;
  const auto distribution = gufo::sampling::BuildDistribution(logits, config);
  double sum = 0.0;
  for (const auto& entry : distribution.entries()) {
    sum += entry.value;
  }
  Expect(std::abs(sum - 1.0) < 1e-12,
         "post-filter probabilities are normalized");
}

void TestSamplingFailsClosedOnInvalidInputs() {
  bool empty_rejected = false;
  try {
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits({}, 1.0F, &rng_state);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  Expect(empty_rejected, "empty logits are rejected");

  bool missing_rng_rejected = false;
  try {
    const std::array<float, 2> logits = {0.0F, 1.0F};
    (void)gufo::sampling::SampleLogits(logits, 1.0F, nullptr);
  } catch (const std::invalid_argument&) {
    missing_rng_rejected = true;
  }
  Expect(missing_rng_rejected, "sampled decoding requires RNG state");

  bool non_finite_rejected = false;
  try {
    const std::array<float, 2> logits = {
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::infinity()};
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits(logits, 1.0F, &rng_state);
  } catch (const std::runtime_error&) {
    non_finite_rejected = true;
  }
  Expect(non_finite_rejected, "an all-non-finite distribution is rejected");

  bool invalid_config_rejected = false;
  try {
    gufo::sampling::SamplingConfig config;
    config.top_p = 0.0F;
    config.Validate();
  } catch (const std::invalid_argument&) {
    invalid_config_rejected = true;
  }
  Expect(invalid_config_rejected, "invalid sampling controls are rejected");
}

void TestZeroDrawAndNonFiniteCandidates() {
  // Inverse xorshift state whose next 53-bit uniform is exactly zero.
  constexpr std::uint64_t zero_draw_state = UINT64_C(0x98d76a164d99a710);
  const std::array<float, 3> cold_logits{-1000.0F, 0.0F, -1.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 0});
  sampler.SetRngState(zero_draw_state);
  Expect(sampler.Sample(cold_logits) == 1,
         "zero draw must not select underflowed zero probability");
  const std::array<float, 5> masked_logits{
      std::numeric_limits<float>::quiet_NaN(), 4.0F,
      std::numeric_limits<float>::infinity(), 0.0F,
      -std::numeric_limits<float>::infinity()};
  for (const auto seed : {0, 73, 808}) {
    gufo::sampling::SamplerState masked({.temperature = 1.0F, .seed = seed});
    auto rng = masked.rng_state();
    const auto expected =
        gufo::sampling::BuildDistribution(masked_logits, masked.config())
            .Sample(&rng);
    Expect(masked.Sample(masked_logits) == expected,
           "linear CPU sampling must ignore nonfinite candidates");
  }
}

void TestStrategyReplayAgainstReference() {
  const std::array<float, 5> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (const auto seed : {0, 1, 73, 808}) {
      auto config = test.config;
      config.seed = seed;
      std::vector<gufo::sampling::TokenId> history{1, 1, 2, 3};
      gufo::sampling::SamplerState sampler(config, history);
      std::vector<gufo::sampling::TokenId> generated;
      auto replay = sampler;
      for (int step = 0; step < 16; ++step) {
        const auto distribution = gufo::sampling::BuildDistribution(
            logits, config, history, generated);
        std::vector<gufo::sampling::Probability> ordered(
            distribution.entries().begin(), distribution.entries().end());
        if (config.top_k == 0 && config.top_p == 1.0F &&
            !(config.min_p > 0.0F && config.min_keep > 1)) {
          std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
        }
        auto rng = sampler.rng_state();
        double threshold = gufo::sampling::Uniform(&rng);
        auto expected = distribution.best_token();
        if (config.temperature > 0.0F) {
          for (const auto& entry : ordered) {
            if (entry.value > 0.0 && threshold < entry.value) {
              expected = entry.token;
              break;
            }
            threshold -= entry.value;
          }
        }
        const auto token = sampler.Sample(logits);
        Expect(token == expected && token == replay.Sample(logits), test.name);
        generated.push_back(token);
        sampler.Accept(token);
        replay.Accept(token);
      }
    }
  }
}

void TestFilteredDistributionAgainstFullSort() {
  std::vector<float> logits(4099);
  const std::array<gufo::sampling::TokenId, 6> history{8, 8, 8, 11, 17, 29};
  for (bool concentrated : {false, true}) {
    for (std::size_t i = 0; i < logits.size(); ++i)
      logits[i] = concentrated ? -(float)((i * 73) % logits.size()) * 0.2F
                               : std::sin((float)i * 0.7F);
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      gufo::sampling::SamplerState sampler(test.config, history);
      const auto rng = sampler.rng_state();
      const auto expected =
          gufo::sampling::BuildDistribution(logits, test.config, history);
      const auto actual = sampler.Distribution(logits);
      const auto positive = std::ranges::count_if(
          expected.entries(), [](const auto& e) { return e.value > 0; });
      Expect(actual.entries().size() == static_cast<std::size_t>(positive),
             test.name);
      for (const auto& entry : expected.entries())
        Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
               test.name);
      Expect(sampler.rng_state() == rng,
             "materializing probabilities consumes no random draws");
    }
  }
}

void TestDeferredResidualReplay() {
  const std::array<float, 3> logits{100.0F, -100.0F, -100.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 7});
  // A retained residual must win over a fresh draw, even when that fresh
  // draw would almost certainly select a different token.
  sampler.DeferSample(2);
  const auto rng = sampler.rng_state();
  auto copied = sampler;
  gufo::sampling::SamplerState assigned;
  assigned = sampler;
  bool duplicate_rejected = false;
  try {
    sampler.DeferSample(1);
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  Expect(duplicate_rejected, "pending residual cannot be overwritten");
  const std::array<gufo::sampling::TokenId, 1> prompt{1};
  gufo::sampling::SamplerState published({.temperature = 1.0F}, prompt);
  published.CopyDrawStateFrom(sampler);
  Expect(published.history().size() == 1 && published.history()[0] == 1,
         "publishing draw state leaves committed history unchanged");
  gufo::sampling::SamplerState behind({.temperature = 1.0F, .seed = 7});
  auto ahead = behind;
  ahead.Accept(0);
  (void)ahead.Uniform();
  const auto behind_rng = behind.rng_state();
  behind.CopyDrawStateFrom(ahead);
  Expect(behind.rng_state() == behind_rng,
         "draws at a later position never become this position's draws");
  for (auto* state : {&sampler, &copied, &assigned, &published}) {
    Expect(state->Sample(logits) == 2 && state->rng_state() == rng,
           "copied residual is consumed without another RNG draw");
    Expect(state->Sample(logits) == 0 && state->rng_state() != rng,
           "residual is consumed only once");
  }
  sampler.DeferSample(2);
  sampler.ResetHistory({});
  Expect(sampler.Sample(logits) == 0, "new history discards pending residual");
  sampler.DeferSample(2);
  sampler.SetRngState(rng);
  Expect(sampler.Sample(logits) == 0, "RNG reset discards pending residual");
}

void TestCanonicalTargetDistribution() {
  std::vector<float> logits(4099, -1000.0F);
  logits[0] = 0;
  logits[17] = -1;
  logits[4098] = -2;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (float temperature : {0.5F, 0.99999994F, 2.0F}) {
      auto config = test.config;
      config.temperature = temperature;
      config.seed = 19;
      gufo::sampling::SamplerState sampler(config);
      for (unsigned step = 0; step < 16; ++step) {
        const auto distribution = sampler.Distribution(logits);
        auto rng = sampler.rng_state();
        const auto expected = distribution.Sample(&rng);
        Expect(sampler.Sample(logits) == expected && sampler.rng_state() == rng,
               "AR and verification share exact support, CDF and RNG draws");
        sampler.Accept(expected);
      }
    }
  }
}

void TestResponsePenaltyScope() {
  using gufo::sampling::SamplerState;
  using gufo::sampling::TokenId;
  const std::array<float, 3> logits{3, 2, 1};
  SamplerState sampler({.temperature = 1,
                        .seed = 7,
                        .repeat_last_n = 0,
                        .frequency_penalty = .5F,
                        .presence_penalty = .25F},
                       std::vector<TokenId>{0, 0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "frequency/presence must not penalize prompt tokens");
  sampler.Accept(std::vector<TokenId>{0, 0});
  sampler.Accept(std::vector<TokenId>(80, 2));
  const std::array<float, 3> expected_logits{1.75F, 2, -39.25F};
  const auto expected =
      gufo::sampling::BuildDistribution(expected_logits, {.temperature = 1});
  for (const auto token : {0U, 1U, 2U})
    Expect(std::abs(sampler.Distribution(logits).probability(token) -
                    expected.probability(token)) < 1e-12,
           "full response counts must outlive the repetition window");
  auto rejected = sampler;
  rejected.Accept(1);
  Expect(sampler.Distribution(logits).best_token() == 1 &&
             rejected.Distribution(logits).best_token() == 0,
         "tentative penalty counts must not leak through speculative rollback");
  auto replay = sampler;
  Expect(replay.Sample(logits) == sampler.Sample(logits),
         "penalty copies preserve RNG replay");
  sampler.ResetHistory(std::vector<TokenId>{0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "new request resets generated counts");
}

}  // namespace

/// Seeded penalized nucleus sampling, recorded on main before structured
/// outputs: requests without a constraint must keep drawing these tokens.
std::vector<gufo::sampling::TokenId> SeededDraws(
    const std::shared_ptr<gufo::sampling::TokenConstraint>& constraint) {
  std::vector<float> logits(48);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = static_cast<float>((i * 37) % 23) * 0.25F - 2.0F;
  gufo::sampling::SamplingConfig config{.temperature = 0.8F,
                                        .top_p = 0.95F,
                                        .seed = 42,
                                        .repeat_penalty = 1.1F,
                                        .presence_penalty = 0.2F};
  config.constraint = constraint;
  gufo::sampling::SamplerState sampler(config);
  std::vector<gufo::sampling::TokenId> tokens;
  for (int i = 0; i < 32; ++i) {
    tokens.push_back(sampler.Sample(logits));
    sampler.Accept(tokens.back());
  }
  return tokens;
}

/// A constraint that never restricts (free reasoning) and counts masks.
class IdleConstraint final : public gufo::sampling::TokenConstraint {
public:
  [[nodiscard]] bool Restricts() const override { return restricts; }
  [[nodiscard]] bool Ready() const override { return true; }
  void Mask(std::span<float> logits) override {
    ++masks;
    for (std::size_t token = 0; token < logits.size(); ++token)
      if (token != allowed)
        logits[token] = -std::numeric_limits<float>::infinity();
  }
  void Accept(gufo::sampling::TokenId) override {}
  bool restricts{false};
  std::size_t allowed{0};
  int masks{0};
};

void TestUnconstrainedDrawsAreUnchanged() {
  const std::vector<gufo::sampling::TokenId> recorded{
      18, 41, 24, 18, 36, 13, 21, 31, 13, 44, 41, 8,  13, 29, 26, 18,
      11, 13, 37, 3,  13, 14, 3,  41, 26, 16, 24, 39, 41, 3,  21, 16};
  Expect(SeededDraws(nullptr) == recorded,
         "unconstrained seeded sampling is byte-identical to main");
  auto idle = std::make_shared<IdleConstraint>();
  Expect(SeededDraws(idle) == recorded && idle->masks == 0,
         "a constraint that does not restrict leaves draws unchanged");
  Expect(!gufo::sampling::SamplingConfig{.constraint = idle}
                 .can_use_unmodified_argmax() &&
             gufo::sampling::SamplingConfig{.constraint = idle}.host_sampled(),
         "constrained requests sample on the host");

  auto only = std::make_shared<IdleConstraint>();
  only->restricts = true;
  only->allowed = 7;
  const auto masked = SeededDraws(only);
  Expect(only->masks == 32 &&
             std::ranges::all_of(masked, [](auto token) { return token == 7; }),
         "a restricting constraint masks every draw");
  const std::array<float, 4> logits = {0.0F, 9.0F, 1.0F, 2.0F};
  only->allowed = 2;
  gufo::sampling::SamplerState greedy({.top_logprobs = 1, .constraint = only});
  Expect(greedy.Sample(logits) == 2, "greedy decoding honours the mask");
  Expect(greedy.TakeLogprobs()->top.front().first == 1,
         "logprobs still report the raw model distribution");
}

int main() {
  TestUnconstrainedDrawsAreUnchanged();
  bool sub_float_resolution = false;
  for (std::uint64_t seed = 1; seed < 100; ++seed) {
    auto state = seed;
    auto reference = seed;
    const auto bits = gufo::sampling::NextRandom(&reference);
    const double draw = gufo::sampling::Uniform(&state);
    Expect(draw >= 0 && draw < 1 && state == reference,
           "uniform stays half-open and consumes one RNG word");
    Expect(draw == static_cast<double>(bits >> 11) * 0x1.0p-53,
           "uniform retains all 53 random mantissa bits");
    sub_float_resolution |=
        draw != static_cast<double>(static_cast<float>(draw));
  }
  Expect(sub_float_resolution, "uniform must exceed float resolution");
  TestCanonicalTargetDistribution();
  TestResponsePenaltyScope();
  TestGreedySelectsFiniteArgmaxWithoutAdvancingRng();
  TestLogprobProbeAndForcedTokens();
  TestDefaultConfigPreservesGreedyDecoding();
  TestTemperatureSamplingIsDeterministicAndNonGreedy();
  TestTopKFiltersTheCandidateSet();
  TestTopPFiltersByCumulativeProbability();
  TestMinPFiltersRelativeToTheBestToken();
  TestMinKeepProvidesAFilterFloor();
  TestTemperatureBeforeProbabilityFilters();
  TestRepeatPenaltyUsesCommittedHistory();
  TestFrequencyAndPresencePenaltiesUseCounts();
  TestRepeatWindowIsBounded();
  TestFastGreedyMatchesNormalizedPenaltyRoute();
  TestFastTopKSamplingNeverEscapesSelectedSet();
  TestFastMinPSamplingNeverEscapesRelativeThreshold();
  TestSeedAndStateAreRequestLocal();
  TestRngIsKeyedByCommittedPosition();
  TestDistributionIsNormalized();
  TestSamplingFailsClosedOnInvalidInputs();
  TestZeroDrawAndNonFiniteCandidates();
  TestStrategyReplayAgainstReference();
  TestFilteredDistributionAgainstFullSort();
  TestDeferredResidualReplay();
  std::cout << "All logit sampler tests passed.\n";
  return 0;
}
