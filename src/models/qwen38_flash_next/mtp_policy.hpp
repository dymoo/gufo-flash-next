#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>

#include "src/models/qwen38_flash_next/mtp_costs.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;
/// Widest physical decode batch (sessions) the runtime packs.
inline constexpr std::uint32_t kMaxBatchSessions = 512;

struct MtpLengthState {
  std::array<float, kMaxMtpDraftTokens> successes;
  std::array<float, kMaxMtpDraftTokens> failures;
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

// Length depends on committed acceptance history, never request timings.
// Only the first rejected draft is an observed failure; its unverified tail
// and a fully accepted, budget-limited chain are censored observations.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit,
                               std::uint32_t concurrency = 1)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)),
        concurrency_(std::clamp(concurrency, 1U, kMaxBatchSessions)) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t context = 0) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0) {
      return 0;
    }
    // After a bounded AR interval, try one proposal to detect recovery.
    return std::min(
        cap, std::max({1U, BestLength(cap, context), state_.probe_depth}));
  }

  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const noexcept {
    float expected = 1.0F, survival = 1.0F;
    for (unsigned depth = 0; depth < std::min(drafts, limit_); ++depth) {
      survival *= Probability(depth);
      expected += survival;
    }
    return expected;
  }

  [[nodiscard]] std::uint32_t concurrency() const noexcept {
    return concurrency_;
  }

  void Observe(std::size_t accepted, std::size_t drafted,
               std::uint32_t context = 0) noexcept {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      state_.successes[depth] =
          0.75F * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          0.75F * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    // A fully accepted chain supplies no observations for its tail. Probe
    // farther rather than permanently treating those depths as the prior.
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_, context) == 0)
      state_.retry_tokens = 16;
  }

  void ObserveArToken() noexcept {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }

  void Reset() noexcept {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }

  [[nodiscard]] MtpLengthState State() const noexcept { return state_; }

  [[nodiscard]] bool Restore(const MtpLengthState& state) noexcept {
    if (state.retry_tokens > 16 || state.probe_depth > limit_ ||
        state.explored_depth > limit_ || state.probe_delay > 16 ||
        state.failed_depths >= (1U << limit_))
      return false;
    for (std::size_t depth = 0; depth < kMaxMtpDraftTokens; ++depth) {
      const float success = state.successes[depth];
      const float failure = state.failures[depth];
      if (!std::isfinite(success) || !std::isfinite(failure) ||
          success < 0.0F || failure < 0.0F ||
          !std::isfinite(success + failure) || success + failure <= 0.0F)
        return false;
    }
    state_ = state;
    return true;
  }

private:
  [[nodiscard]] float Probability(unsigned depth) const noexcept {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }
  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap,
                                         std::uint32_t context) const noexcept {
    const auto costs = MtpCycleCosts(context, concurrency_);
    float survival = 1.0F;
    float expected_tokens = 1.0F;  // The sampled target anchor.
    float best_score = 1.0F;       // Ordinary target decoding.
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      const auto depth = length - 1;
      // An explored depth with no actual rejection remains optimistic.
      // Otherwise the prior alone shortens perfect chains and prevents
      // their less frequently sampled tails from collecting evidence.
      const float probability = Probability(depth);
      survival *= probability;
      expected_tokens += survival;
      const float cost = costs[length] / costs[0];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  std::uint32_t concurrency_;
  MtpLengthState state_{};
};

/// GUFO_LOOKUP_ONLY (benchmarks): speculate with prompt-lookup drafts only.
/// MTP drafts are never chosen and the draft block never runs (its catch-up is
/// skipped), so the cost of a lookup-only server is the verification rows
/// alone. Needs the MTP sidecar loaded (the rollback state and wide
/// verification path belong to MTP mode); output is unchanged by the draft
/// source (lookup drafts are verified by exact target draws).
[[nodiscard]] inline bool LookupOnly() noexcept {
  static const bool value = std::getenv("GUFO_LOOKUP_ONLY") != nullptr;
  return value;
}

/// Draft source and length for one session's next cycle.
struct DraftPlan {
  std::uint32_t drafts{0};
  bool lookup{false};  ///< prompt-lookup continuation instead of MTP
};

/// Deterministic per-session plan from committed history only (sampled
/// requests: seeded replay must not depend on timing or batch company).
/// Picks the source with the better expected tokens per prior cycle cost.
[[nodiscard]] inline DraftPlan ChooseSessionDraft(
    const MtpLengthController& mtp, const MtpLengthController& lookup,
    std::uint32_t budget, std::uint32_t lookup_size, std::uint32_t context,
    std::uint32_t concurrency) noexcept {
  const auto m = LookupOnly() ? 0 : mtp.Choose(budget, context);
  const auto l = lookup_size != 0
                     ? lookup.Choose(std::min(budget, lookup_size), context)
                     : 0;
  if (l == 0 || m == 0)
    return l == 0 ? DraftPlan{m, false} : DraftPlan{l, true};
  const auto costs = MtpCycleCosts(context, concurrency);
  return lookup.ExpectedTokens(l) / costs[l] >= mtp.ExpectedTokens(m) / costs[m]
             ? DraftPlan{l, true}
             : DraftPlan{m, false};
}

/// Batch draft allocation. Each session gets its own draft length and
/// source: greedily add the draft (or prompt-lookup extension) with the most
/// expected accepted tokens per predicted millisecond while that beats the
/// batch's current tokens per millisecond, i.e. while it raises aggregate
/// throughput. Low-acceptance sessions stay at 0 drafts. The step-time
/// model is fitted online from executed steps (StepCostModel). Greedy output
/// does not depend on the chosen widths; sampled rows arrive `fixed`.
class MtpBatchController {
public:
  struct Row {
    const MtpLengthController* mtp;
    const MtpLengthController* lookup;
    std::uint32_t budget;       ///< maximum drafts
    std::uint32_t lookup_size;  ///< prompt-lookup drafts available
    std::uint32_t context;
    std::optional<DraftPlan> fixed{};
  };

  void Choose(std::span<const Row> rows, std::span<DraftPlan> plans) noexcept {
    const auto n =
        std::min<std::size_t>({rows.size(), plans.size(), kMaxBatchSessions});
    ++cycles_;
    for (std::size_t i = 0; i < n; ++i)
      plans[i] = rows[i].fixed.value_or(DraftPlan{});
    // Periodic plain steps anchor the AR cost that drafts must beat.
    if (cycles_ % 32 == 0)
      return;
    // A sequential MTP round costs the batch once, which a one-draft-at-a-
    // time greedy never pays for: fix the round count, allocate, keep the
    // best rate. Depth 0 still allows prompt-lookup drafts.
    std::uint32_t deepest = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (!rows[i].fixed)
        deepest = std::max(deepest, rows[i].budget);
    std::array<DraftPlan, kMaxBatchSessions> trial{};
    double best = 0;
    for (std::uint32_t depth = 0; depth <= deepest; ++depth) {
      const double rate = Allocate(rows.first(n), trial, depth);
      if (depth == 0 || rate > best * 1.01) {
        best = rate;
        std::copy_n(trial.begin(), n, plans.begin());
      }
    }
    // Exploration: idle sessions draft once so their acceptance estimates
    // recover when the text changes (the online model sees wider shapes).
    if (cycles_ % 32 == 16)
      for (std::size_t i = 0; i < n; ++i)
        if (!rows[i].fixed && plans[i].drafts == 0 && rows[i].budget != 0) {
          if (LookupOnly()) {
            if (rows[i].lookup_size != 0)
              plans[i] = {1, true};
          } else {
            plans[i] = {
                1, rows[i].lookup_size != 0 &&
                       Gain(rows[i], {1, true}) > Gain(rows[i], {1, false})};
          }
        }
  }

  void Observe(const StepShape& shape, float ms) noexcept {
    model_.Observe(shape, ms);
  }
  [[nodiscard]] const StepCostModel& Model() const noexcept { return model_; }

private:
  // Greedy allocation with at most `depth` MTP drafts per session (the
  // round count is fixed, so each extra draft has a convex marginal cost).
  // Returns expected tokens per predicted millisecond.
  double Allocate(std::span<const Row> rows, std::span<DraftPlan> plans,
                  std::uint32_t depth) const noexcept {
    const auto n = rows.size();
    StepShape shape{};
    double tokens = 0;
    for (std::size_t i = 0; i < n; ++i) {
      plans[i] = rows[i].fixed.value_or(DraftPlan{});
      Add(shape, rows[i], plans[i], 1);
      tokens += Gain(rows[i], plans[i]);
    }
    shape.rounds = std::max(shape.rounds, depth);
    double ms = model_.Predict(shape);
    for (;;) {
      std::size_t best = n;
      DraftPlan best_plan{};
      double best_ratio = tokens / ms * 1.01, best_gain = 0, best_ms = ms;
      StepShape best_shape{};
      for (std::size_t i = 0; i < n; ++i) {
        const auto& row = rows[i];
        if (row.fixed || plans[i].drafts >= row.budget)
          continue;
        const auto next = plans[i].drafts + 1;
        const std::array<DraftPlan, 3> candidates{
            DraftPlan{next, false}, DraftPlan{next, true},
            DraftPlan{std::min(row.budget, row.lookup_size), true}};
        for (const auto& candidate : candidates) {
          if ((LookupOnly() && !candidate.lookup) ||
              candidate.drafts <= plans[i].drafts ||
              (candidate.lookup ? candidate.drafts > row.lookup_size
                                : candidate.drafts > depth))
            continue;
          const double gain = Gain(row, candidate) - Gain(row, plans[i]);
          if (gain <= 0)
            continue;
          auto trial = shape;
          Add(trial, row, plans[i], -1);
          Add(trial, row, candidate, 1);
          const double trial_ms = model_.Predict(trial);
          const double ratio = gain / std::max(trial_ms - ms, 1e-3);
          if (ratio > best_ratio) {
            best = i;
            best_plan = candidate;
            best_ratio = ratio;
            best_gain = gain;
            best_ms = trial_ms;
            best_shape = trial;
          }
        }
      }
      if (best == n)
        break;
      plans[best] = best_plan;
      tokens += best_gain;
      ms = best_ms;
      shape = best_shape;
    }
    return tokens / ms;
  }

  static double Gain(const Row& row, const DraftPlan& plan) noexcept {
    if (plan.drafts == 0)
      return 1.0;
    return plan.lookup ? row.lookup->ExpectedTokens(plan.drafts)
                       : row.mtp->ExpectedTokens(plan.drafts);
  }
  // `rounds` keeps the deepest MTP chain seen; a session leaving MTP for a
  // prompt lookup does not shorten it (conservative).
  static void Add(StepShape& shape, const Row& row, const DraftPlan& plan,
                  int sign) noexcept {
    const auto rows = 1 + plan.drafts;
    const auto mtp = plan.lookup ? 0U : plan.drafts;
    if (sign > 0) {
      shape.sessions += 1;
      shape.rows += rows;
      shape.mtp_drafts += mtp;
      shape.rounds = std::max(shape.rounds, mtp);
    } else {
      shape.sessions -= 1;
      shape.rows -= rows;
      shape.mtp_drafts -= mtp;
    }
    shape.row_context += static_cast<float>(sign) * static_cast<float>(rows) *
                         static_cast<float>(row.context) / 65536.0F;
  }

  StepCostModel model_;
  std::uint64_t cycles_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
