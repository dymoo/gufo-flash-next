// DSpark length policy: confident blocks verify long, doubtful ones decode
// plainly, and a non-finite confidence ends the block.
#include "src/models/qwen38_flash_next/dspark_policy.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace q = gufo::models::qwen38_flash_next;

int main() {
  const auto check = [](bool ok, const char* what) {
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", what);
      std::exit(1);
    }
  };
  const std::array<float, 7> sure{0.99F, 0.99F, 0.99F, 0.99F,
                                  0.99F, 0.99F, 0.99F};
  const std::array<float, 7> doubtful{0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F};
  const std::array<float, 7> fading{0.95F, 0.9F, 0.6F, 0.3F, 0.2F, 0.1F, 0.1F};
  const std::array<float, 3> broken{0.99F, NAN, 0.99F};
  check(q::ChooseDsparkLength(sure, 1) == 7, "confident block verifies all");
  check(q::ChooseDsparkLength(doubtful, 1) == 0, "doubtful block decodes");
  const auto fade = q::ChooseDsparkLength(fading, 1);
  check(fade >= 1 && fade < 7, "fading block stops early");
  check(q::ChooseDsparkLength(broken, 1) <= 1, "non-finite ends the block");
  check(q::ChooseDsparkLength(fading, 8) <= fade,
        "a shared dense pass makes drafts relatively dearer");
  check(q::DsparkPassPays(0.0F, 1), "a lone session always drafts");
  check(q::DsparkPassPays(3.5F, 8), "productive passes pay in a cohort");
  check(!q::DsparkPassPays(1.0F, 4), "prose-like yields stop in a cohort");
  check(q::DsparkPassPays(1.5F, 2) && !q::DsparkPassPays(1.5F, 32),
        "the bar rises with the cohort");
  std::puts("dspark_policy: ok");
  return 0;
}
