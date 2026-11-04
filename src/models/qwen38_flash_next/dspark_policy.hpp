#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DSPARK_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DSPARK_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>

namespace gufo::models::qwen38_flash_next {

/// Milliseconds of one MiniCPM-V 4.7 (qwen35moe) DSpark cycle on gfx1151,
/// per session of a `concurrency` cohort: the drafter block plus a verify of
/// `rows` target rows. Verify bytes are the Q8_0 dense weights (2.07 GB,
/// shared by the cohort) plus 1.94 MB per distinct routed expert per layer
/// (distinct experts over consecutive tokens measured from router traces),
/// at 220 GB/s, plus the 5.6 ms a decode step spends beyond bandwidth.
/// ponytail: a static table; fit it online (StepCostModel) if workloads or
/// kernels move it.
[[nodiscard]] inline float DsparkCycleMs(std::uint32_t rows,
                                         std::uint32_t concurrency,
                                         bool draft) noexcept {
  static constexpr std::array<float, 9> kExperts{
      0, 8.0F, 12.8F, 16.8F, 20.4F, 23.6F, 26.7F, 29.5F, 32.1F};
  constexpr float kDenseMs = 2.07F / 0.220F + 5.6F;
  constexpr float kExpertMs = 40 * 1.94e-3F / 0.220F;
  constexpr float kDraftMs = 4.5F;
  const auto c = static_cast<float>(std::max(concurrency, 1U));
  return kDenseMs / c +
         kExperts[std::clamp<std::uint32_t>(rows, 1, 8)] * kExpertMs +
         (draft ? kDraftMs : 0.0F);
}

/// Whether a drafting pass pays at `cohort` sessions, given `yield`, the
/// session's running mean of accepted drafts per pass (declined passes
/// count zero). Once a cohort shares the dense weights, a token saves only
/// the session's share of a step while a pass costs the whole drafter and
/// widens the step's verify, which past eight rows leaves the GEMV route
/// for every projection. Measured on gfx1151 (production shape, 32
/// sessions), passes pay only above 1.6x the table, so they must return
/// that. Alone a pass always runs and its confidence head decides.
[[nodiscard]] inline bool DsparkPassPays(float yield,
                                         std::uint32_t cohort) noexcept {
  if (cohort <= 1)
    return true;
  constexpr float kMeasured = 1.6F;
  const float token_ms = DsparkCycleMs(1, cohort, false);
  const float pass_ms =
      DsparkCycleMs(2, cohort, true) - DsparkCycleMs(1, cohort, false);
  return yield * token_ms >= kMeasured * pass_ms;
}

/// Drafts to verify given each proposal's confidence (calibrated as the
/// probability it is accepted after its predecessors were): the length with
/// the most expected tokens per millisecond, zero when plain decoding wins.
/// Depends only on the committed context, so seeded replay is preserved.
[[nodiscard]] inline std::uint32_t ChooseDsparkLength(
    std::span<const float> confidence, std::uint32_t concurrency) noexcept {
  float best = 1.0F / DsparkCycleMs(1, concurrency, false);
  float survival = 1.0F, expected = 1.0F;
  std::uint32_t length = 0;
  for (std::uint32_t l = 1; l <= confidence.size() && l < 8; ++l) {
    const float c = confidence[l - 1];
    if (!std::isfinite(c) || c < 0 || c > 1)
      break;
    survival *= c;
    expected += survival;
    const float rate = expected / DsparkCycleMs(l + 1, concurrency, true);
    if (rate > best) {
      best = rate;
      length = l;
    }
  }
  return length;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DSPARK_POLICY_HPP_
