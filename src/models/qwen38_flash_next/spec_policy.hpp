#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_POLICY_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// Knobs of the combined C1 speculative policy (greedy requests only). mode 0
/// keeps the controllers; 1 is the policy (`--mtp-policy adaptive`).
struct SpecKnobs {
  int mode{0};
  /// Scales every marginal-cost threshold: > 1 drafts less, < 1 more.
  float aggr{0.5F};
  /// A prompt-lookup token whose calibrated acceptance reaches this extends
  /// the chain instead of another MTP round.
  float lk_pref{0.7F};
  /// Assumed acceptance of a draft whose confidence is not yet known.
  float p_next{0.6F};
  bool lookup{true};
  bool mtp{true};
  /// Skip the draft head after AR cycles with a hopeless first draft.
  bool idle{true};
  /// Batched greedy decode (C >= 2): cut each session's planned chain where
  /// its calibrated survival no longer pays its verify row.
  bool bgate{false};
  /// Draft with the compact head (`--mtp-draft-vocab` loaded) or the full head.
  bool vocab{true};
  /// Every explore-th cycle lowers the threshold to 30%: acceptance of deep
  /// or low-confidence drafts is only observed for drafts that were sent, so
  /// without exploring the calibration of the cut region never updates.
  std::uint32_t explore{8};
};

/// The process-wide policy (scheduler thread only), set once at model load by
/// `--mtp-policy`: `adaptive` is the combined greedy C1 policy, `off` keeps
/// the plain MTP controllers. Batched greedy cutting (`bgate`) stays off.
[[nodiscard]] inline SpecKnobs& MutableSpecKnobs() noexcept {
  static SpecKnobs knobs;
  return knobs;
}
inline void SetSpecPolicy(bool adaptive) noexcept {
  SpecKnobs k;
  k.mode = adaptive ? 1 : 0;
  MutableSpecKnobs() = k;
}
[[nodiscard]] inline const SpecKnobs& SpecPolicyKnobs() noexcept {
  return MutableSpecKnobs();
}

/// Source of one drafted token (for calibration).
enum class DraftSource : std::uint8_t { kMtp, kLookup, kAgree };

/// Online acceptance probabilities of the three draft sources, learned from
/// verified outcomes (and from the next anchor when a draft was not sent).
/// Priors come from window-1 forced-width traces (acceptance by top-1
/// confidence) and are [INFERENCE] for lookup and agreement.
class SpecCalibrator {
public:
  static constexpr std::size_t kConfBins = 6;
  static constexpr std::size_t kMlBins = 5;
  static constexpr std::size_t kDepthClasses = 3;

  SpecCalibrator() noexcept { Reset(); }

  void Reset() noexcept {
    constexpr std::array<float, kConfBins> conf{0.25F, 0.35F, 0.45F,
                                                0.60F, 0.75F, 0.965F};
    for (std::size_t i = 0; i < kConfBins; ++i)
      mtp_[i] = Beta(conf[i], kPriorWeight);
    constexpr std::array<float, kMlBins> ml{0.35F, 0.50F, 0.65F, 0.80F, 0.90F};
    constexpr std::array<float, kDepthClasses> depth{1.0F, 0.95F, 0.9F};
    for (std::size_t m = 0; m < kMlBins; ++m) {
      agree_[m] = Beta(std::min(0.97F, ml[m] + 0.2F), kPriorWeight);
      for (std::size_t d = 0; d < kDepthClasses; ++d)
        lookup_[m][d] = Beta(ml[m] * depth[d], kPriorWeight);
    }
  }

  [[nodiscard]] float Mtp(float conf) const noexcept {
    return mtp_[ConfBin(conf)].p();
  }
  [[nodiscard]] float Agree(std::uint32_t ml) const noexcept {
    return agree_[MlBin(ml)].p();
  }
  /// `depth` counts from 1 (the first draft after the anchor).
  [[nodiscard]] float Lookup(std::uint32_t ml,
                             std::uint32_t depth) const noexcept {
    return lookup_[MlBin(ml)][DepthClass(depth)].p();
  }

  void ObserveMtp(float conf, bool ok) noexcept { mtp_[ConfBin(conf)].Add(ok); }
  void ObserveAgree(std::uint32_t ml, bool ok) noexcept {
    agree_[MlBin(ml)].Add(ok);
  }
  void ObserveLookup(std::uint32_t ml, std::uint32_t depth, bool ok) noexcept {
    lookup_[MlBin(ml)][DepthClass(depth)].Add(ok);
  }

private:
  static constexpr float kPriorWeight = 6.0F;
  struct Beta {
    float s{1}, f{1};
    Beta() = default;
    Beta(float p, float w) : s(p * w), f((1.0F - p) * w) {}
    [[nodiscard]] float p() const noexcept { return s / (s + f); }
    void Add(bool ok) noexcept {
      // Forgetting keeps ~40 effective observations, so a change of text
      // (prose to a copied block) is learned within a few cycles.
      constexpr float kDecay = 0.975F;
      s *= kDecay;
      f *= kDecay;
      (ok ? s : f) += 1.0F;
    }
  };
  static std::size_t ConfBin(float c) noexcept {
    return c < 0.5F    ? 0
           : c < 0.7F  ? 1
           : c < 0.8F  ? 2
           : c < 0.9F  ? 3
           : c < 0.99F ? 4
                       : 5;
  }
  static std::size_t MlBin(std::uint32_t ml) noexcept {
    return ml < 4 ? 0 : ml < 6 ? 1 : ml < 10 ? 2 : ml < 20 ? 3 : 4;
  }
  static std::size_t DepthClass(std::uint32_t d) noexcept {
    return d <= 1 ? 0 : d <= 3 ? 1 : 2;
  }
  std::array<Beta, kConfBins> mtp_;
  std::array<Beta, kMlBins> agree_;
  std::array<std::array<Beta, kDepthClasses>, kMlBins> lookup_;
};

/// Online cost of one greedy C1 cycle by verify rows (anchor + drafts),
/// excluding the sequential MTP rounds, which are priced separately. The
/// prior is window-1 (32K draft head, UD, 1.8K context, wall time of
/// PrepareDecode + verify + finish per width); measured cycles scale it.
class SpecCosts {
public:
  static constexpr std::size_t kMaxRows = 8;

  [[nodiscard]] double Cycle(std::uint32_t rows) const noexcept {
    rows = std::clamp<std::uint32_t>(rows, 1, kMaxRows);
    if (count_[rows] >= 3)
      return ema_[rows];
    // Unmeasured width: the line through the two nearest measured widths
    // (their slope tells how much a verify row costs on this model, e.g.
    // fewer routed experts per token), else the scaled prior.
    int a = -1, b = -1;
    for (std::uint32_t r = 1; r <= kMaxRows; ++r) {
      if (count_[r] < 3)
        continue;
      const auto dist = [&](int i) {
        return std::abs(static_cast<int>(rows) - i);
      };
      if (a < 0 || dist(static_cast<int>(r)) < dist(a)) {
        b = a;
        a = static_cast<int>(r);
      } else if (b < 0 || dist(static_cast<int>(r)) < dist(b)) {
        b = static_cast<int>(r);
      }
    }
    if (a >= 0 && b >= 0) {
      const double prior_slope =
          (kPrior[std::max(a, b)] - kPrior[std::min(a, b)]) / std::abs(a - b);
      double slope = (ema_[a] - ema_[b]) / (a - b);
      slope = std::clamp(slope, 0.3 * prior_slope, 1.5 * prior_slope);
      return std::max(ema_[a] + slope * (static_cast<int>(rows) - a), 1.0);
    }
    if (a >= 0) {
      // One measured width: carry its offset from the prior to this width.
      return std::max(ema_[a] + (kPrior[rows] - kPrior[a]) * Scale(), 1.0);
    }
    return kPrior[rows] * Scale();
  }
  [[nodiscard]] double Round() const noexcept { return round_; }

  void ObserveCycle(std::uint32_t rows, double ms) noexcept {
    if (rows < 1 || rows > kMaxRows || !(ms > 0))
      return;
    auto& n = count_[rows];
    ema_[rows] = n == 0 ? ms : 0.9 * ema_[rows] + 0.1 * ms;
    n = std::min<std::uint32_t>(n + 1, 1000);
  }
  void ObserveRound(double ms) noexcept {
    if (ms > 0)
      round_ = 0.9 * round_ + 0.1 * ms;
  }

private:
  // Mean measured over prior, shrunk toward 1 (as three virtual cycles).
  [[nodiscard]] double Scale() const noexcept {
    double num = 3.0, den = 3.0;
    for (std::size_t r = 1; r <= kMaxRows; ++r) {
      if (count_[r] < 3)
        continue;
      num += count_[r] * ema_[r] / kPrior[r];
      den += count_[r];
    }
    return num / den;
  }
  static constexpr std::array<double, kMaxRows + 1> kPrior{
      0, 40.0, 46.4, 51.4, 56.0, 60.3, 64.8, 69.0, 73.3};
  std::array<double, kMaxRows + 1> ema_{};
  std::array<std::uint32_t, kMaxRows + 1> count_{};
  double round_{1.5};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_POLICY_HPP_
