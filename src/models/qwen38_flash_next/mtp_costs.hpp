#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

/// One batched decode/verify step: `rows` target rows from `sessions`
/// sessions (one anchor each plus drafts), `mtp_drafts` of which the draft
/// block proposed in `rounds` sequential rounds. `row_context` sums each
/// target row's context in 64K-token units.
struct StepShape {
  std::uint32_t sessions{0};
  std::uint32_t rows{0};
  std::uint32_t mtp_drafts{0};
  std::uint32_t rounds{0};
  float row_context{0};
};

inline constexpr std::size_t kStepFeatures = 7;
using StepFeatures = std::array<double, kStepFeatures>;

/// Expected distinct routed experts per layer for `rows` uniform top-10 of
/// 512 routes: the device-planned MoE streams each touched expert once.
[[nodiscard]] inline double UniqueExperts(std::uint32_t rows) noexcept {
  static const auto table = [] {
    std::array<double, 1025> t{};
    for (std::size_t r = 0; r < t.size(); ++r)
      t[r] = 512.0 * (1.0 - std::pow(502.0 / 512.0, static_cast<double>(r)));
    return t;
  }();
  return table[std::min<std::size_t>(rows, table.size() - 1)];
}

[[nodiscard]] inline StepFeatures Features(const StepShape& s) noexcept {
  return {1.0, UniqueExperts(s.rows), static_cast<double>(s.sessions),
          // Draft rows by source; disjoint so neither is collinear with
          // the other when only one source runs.
          static_cast<double>(
              s.rows - std::min(s.rows, s.sessions) -
              std::min(s.mtp_drafts, s.rows - std::min(s.rows, s.sessions))),
          static_cast<double>(s.mtp_drafts), static_cast<double>(s.rounds),
          static_cast<double>(s.row_context)};
}

/// Milliseconds per feature. Fitted to gfx1151 window-3 AR decode (tq4,
/// depth 0: C1 36.8, C8 54.1, C16 73.1, C32 116.4, C64 190.5 ms/step; per
/// session ~2.4 ms, the expert union barely shows), the 32K-depth C8/C32
/// steps, and MTP window-3 throughput (C32 draft rows cost about as much as
/// sessions). Deliberately pessimistic about drafts: the online fit in
/// StepCostModel earns wider speculation from measured steps.
inline constexpr StepFeatures kStepPrior{
    34.2,  // fixed: dense weights, launches
    0.02,  // per unique routed expert per layer
    2.4,   // per session: recurrent state, attention, sampling
    1.5,   // per prompt-lookup draft (verify row)
    2.3,   // per MTP draft (verify row + predictor row)
    5.0,   // per sequential MTP round (predictor weights + LM head)
    1.5,   // per row x 64K context (indexer scan, sparse attention)
};

[[nodiscard]] inline double PredictStep(const StepFeatures& theta,
                                        const StepShape& shape) noexcept {
  const auto x = Features(shape);
  double ms = theta[0];
  // Collinear online estimates may drift negative; a draft never saves time.
  for (std::size_t i = 1; i < kStepFeatures; ++i)
    ms += std::max(theta[i], 0.0) * x[i];
  return std::max(ms, 1.0);
}

/// Prior cost of a cohort of `concurrency` sessions all proposing w MTP
/// drafts, w = 0..7. Sampled requests use it at the configured capacity so
/// their draft length depends only on committed history (seeded replay).
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto b = std::max(concurrency, 1U);
  std::array<float, 8> costs{};
  for (std::uint32_t w = 0; w < costs.size(); ++w)
    costs[w] = static_cast<float>(PredictStep(
        kStepPrior, {.sessions = b,
                     .rows = b * (1 + w),
                     .mtp_drafts = b * w,
                     .rounds = w,
                     .row_context = static_cast<float>(b * (1 + w)) *
                                    static_cast<float>(context) / 65536.0F}));
  return costs;
}

/// Online step-time model: recursive least squares over Features(), started
/// at kStepPrior (ridge prior) with forgetting so it tracks kernel, context
/// and workload changes. Outliers (first-use GEMM plans, preemption) are
/// clipped before they move the estimate.
class StepCostModel {
public:
  StepCostModel() noexcept { Reset(); }

  void Reset() noexcept {
    theta_ = kStepPrior;
    for (auto& row : p_)
      row.fill(0.0);
    for (std::size_t i = 0; i < kStepFeatures; ++i) {
      // Prior standard deviation ~ the prior itself, in units of the ~3 ms
      // step noise.
      const double sd = std::max(kStepPrior[i], 0.2) / 3.0;
      p0_[i] = sd * sd;
      p_[i][i] = p0_[i];
    }
    observations_ = 0;
  }

  [[nodiscard]] double Predict(const StepShape& shape) const noexcept {
    return PredictStep(theta_, shape);
  }
  [[nodiscard]] const StepFeatures& Coefficients() const noexcept {
    return theta_;
  }
  [[nodiscard]] std::uint64_t Observations() const noexcept {
    return observations_;
  }

  void Observe(const StepShape& shape, double ms) noexcept {
    if (shape.sessions == 0 || shape.rows < shape.sessions ||
        !std::isfinite(ms) || ms <= 0)
      return;
    const auto x = Features(shape);
    StepFeatures px{};
    double denominator = 0;
    for (std::size_t i = 0; i < kStepFeatures; ++i) {
      for (std::size_t j = 0; j < kStepFeatures; ++j)
        px[i] += p_[i][j] * x[j];
      denominator += x[i] * px[i];
    }
    constexpr double lambda = 0.995;  // ~200-step memory
    denominator += lambda;
    double predicted = 0;
    for (std::size_t i = 0; i < kStepFeatures; ++i)
      predicted += theta_[i] * x[i];
    const double limit = std::max(5.0, 0.3 * std::abs(predicted));
    const double error = std::clamp(ms - predicted, -limit, limit);
    for (std::size_t i = 0; i < kStepFeatures; ++i)
      theta_[i] += px[i] / denominator * error;
    for (std::size_t i = 0; i < kStepFeatures; ++i)
      for (std::size_t j = 0; j < kStepFeatures; ++j)
        p_[i][j] = (p_[i][j] - px[i] * px[j] / denominator) / lambda;
    // Forgetting inflates unexcited directions (covariance wind-up): cap
    // each variance at its prior by symmetric scaling, which keeps P PSD.
    StepFeatures scale{};
    for (std::size_t i = 0; i < kStepFeatures; ++i)
      scale[i] = p_[i][i] > p0_[i] ? std::sqrt(p0_[i] / p_[i][i]) : 1.0;
    for (std::size_t i = 0; i < kStepFeatures; ++i)
      for (std::size_t j = 0; j < kStepFeatures; ++j)
        p_[i][j] *= scale[i] * scale[j];
    ++observations_;
  }

private:
  StepFeatures theta_{};
  StepFeatures p0_{};
  std::array<StepFeatures, kStepFeatures> p_{};
  std::uint64_t observations_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
