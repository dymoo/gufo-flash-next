// Spike knobs: GUFO_SPIKE_* parsing and the scalar activation
// numerics shared with the emulation kernels, checked on the CPU.
#include "src/models/qwen38_flash_next/kernels/rocm/spike_knobs.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace s = gufo::models::qwen38_flash_next::rocm::spike;

namespace {

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
  }
}

// Parses a "NAME=value NAME=value" string the way the stages files write
// their rung environments.
s::Knobs Parse(const std::string& words) {
  std::map<std::string, std::string> env;
  std::size_t pos = 0;
  while (pos < words.size()) {
    std::size_t end = words.find(' ', pos);
    if (end == std::string::npos)
      end = words.size();
    const std::string w = words.substr(pos, end - pos);
    const std::size_t eq = w.find('=');
    if (eq != std::string::npos)
      env[w.substr(0, eq)] = w.substr(eq + 1);
    pos = end + 1;
  }
  return s::ParseKnobs([&](const char* name) -> const char* {
    const auto it = env.find(name);
    return it == env.end() ? nullptr : it->second.c_str();
  });
}

void TestParse() {
  const s::Knobs none = Parse("");
  Check(!none.enabled && none.dense.grid == s::kActNone &&
            none.targets == 127 && none.min_rows == 0 && none.outliers == 0 &&
            none.warnings.empty(),
        "empty environment is disabled");

  const s::Knobs dense = Parse(
      "GUFO_SPIKE_WGROUP=32 GUFO_SPIKE_ACT=a4 GUFO_SPIKE_AGROUP=32 "
      "GUFO_SPIKE_HAD=128 GUFO_SPIKE_TARGETS=1");
  Check(dense.enabled && dense.dense.grid == s::kActInt4 &&
            dense.dense.group == 32 && dense.routed.grid == s::kActInt4 &&
            dense.routed.group == 32 && dense.had == 128 &&
            dense.wgroup == 32 && dense.targets == 1,
        "dense family recipe; routed inherits it");

  const s::Knobs stack = Parse(
      "GUFO_SPIKE_WGROUP=32 GUFO_SPIKE_ACT=a4 GUFO_SPIKE_AGROUP=32 "
      "GUFO_SPIKE_HAD=128 GUFO_SPIKE_CLIP=0.9 GUFO_SPIKE_ROUTED_ACT=fp4 "
      "GUFO_SPIKE_ROUTED_AGROUP=16 GUFO_SPIKE_ROUTED_CLIP=1 "
      "GUFO_SPIKE_TARGETS=127");
  Check(stack.dense.grid == s::kActInt4 && stack.dense.group == 32 &&
            std::fabs(stack.dense.clip - 0.9F) < 1e-6F &&
            stack.routed.grid == s::kActFp4 && stack.routed.group == 16 &&
            stack.routed.clip == 1.0F && stack.warnings.empty(),
        "dense and routed recipes stack");

  const s::Knobs fp4 = Parse("GUFO_SPIKE_ACT=fp4");
  Check(fp4.dense.group == 16 && fp4.routed.group == 16 && fp4.enabled,
        "fp4 defaults to NVFP4's group of 16");
  const s::Knobs routed_only =
      Parse("GUFO_SPIKE_ROUTED_ACT=fp4 GUFO_SPIKE_TARGETS=96");
  Check(routed_only.enabled && routed_only.dense.grid == s::kActNone &&
            routed_only.routed.grid == s::kActFp4 &&
            routed_only.routed.group == 16,
        "a routed-only recipe enables the spike");
  const s::Knobs shared_group =
      Parse("GUFO_SPIKE_ACT=a8 GUFO_SPIKE_AGROUP=64 GUFO_SPIKE_ROUTED_ACT=a4");
  Check(shared_group.routed.grid == s::kActInt4 &&
            shared_group.routed.group == 64,
        "an explicit AGROUP is the routed default");

  const s::Knobs gate = Parse(
      "GUFO_SPIKE_ACT=a4 GUFO_SPIKE_MIN_ROWS=2048 GUFO_SPIKE_ACLIP_SEARCH=1 "
      "GUFO_SPIKE_OUTLIERS=16 GUFO_SPIKE_OUTLIER_MODE=call "
      "GUFO_SPIKE_A8_LAYERS=0,1,46,47 GUFO_SPIKE_KEEP=3,5");
  Check(gate.min_rows == 2048 && gate.clip_search && gate.outliers == 16 &&
            gate.outlier_mode == s::OutlierMode::kCall &&
            gate.a8_layers == std::set<int>{0, 1, 46, 47} &&
            gate.keep == std::set<int>{3, 5} && gate.warnings.empty(),
        "min rows, clip search, outliers, layer lists");
  Check(Parse("GUFO_SPIKE_ACT=a4 GUFO_SPIKE_OUTLIERS=8").outlier_mode ==
            s::OutlierMode::kRow,
        "outlier mode defaults to row");

  const s::Knobs bad = Parse(
      "GUFO_SPIKE_ACT=int4 GUFO_SPIKE_OUTLIERS=999 GUFO_SPIKE_OUTLIER_MODE=x "
      "GUFO_SPIKE_AGROUP=24");
  Check(bad.dense.grid == s::kActNone && bad.outliers == s::kMaxOutliers &&
            bad.outlier_mode == s::OutlierMode::kRow &&
            bad.warnings.size() == 5,  // ACT, OUTLIERS, MODE, 2x group 24
        "bad values warn and fall back");

  const s::Knobs real = Parse("GUFO_SPIKE_REAL=1 GUFO_SPIKE_HAD=128");
  Check(real.real && real.wgroup == 32 && real.dense.grid == s::kActInt4 &&
            real.targets == 15 && real.warnings.empty(),
        "real mode defaults");
  const s::Knobs routed_real = Parse(
      "GUFO_SPIKE_ROUTED_REAL=1 GUFO_SPIKE_ROUTED_AGROUP=32 "
      "GUFO_SPIKE_ROUTED_DOWN4=1 GUFO_SPIKE_ROUTED_CHECK=4 "
      "GUFO_SPIKE_TARGETS=96");
  Check(routed_real.routed_real && routed_real.routed_down4 &&
            routed_real.routed_check == 4 &&
            routed_real.routed.grid == s::kActInt4 &&
            routed_real.routed.group == 32 && routed_real.enabled &&
            routed_real.warnings.empty(),
        "routed real mode defaults to int4");
  Check(Parse("GUFO_SPIKE_ROUTED_DOWN4=1").enabled,
        "down4 alone enables the spike");
  Check(Parse("GUFO_SPIKE_ROUTED_REAL=1 GUFO_SPIKE_ROUTED_AGROUP=16")
                .warnings.size() == 1,
        "routed real warns on an unsupported group");
  const s::Knobs rot = Parse(
      "GUFO_SPIKE_ROUTED_HAD=128 GUFO_SPIKE_ROUTED_TALL=1 "
      "GUFO_SPIKE_ROUTED_REAL=1 GUFO_SPIKE_ROUTED_AGROUP=64");
  Check(rot.routed_had == 128 && rot.routed_tall && rot.enabled &&
            rot.routed.group == 64 && rot.warnings.empty(),
        "routed rotation and tall blocks parse; group 64 is a real group");
  Check(Parse("GUFO_SPIKE_ROUTED_HAD=128").enabled,
        "routed rotation alone enables the spike");
  const s::Knobs bad_rot = Parse("GUFO_SPIKE_ROUTED_HAD=64");
  Check(bad_rot.routed_had == 0 && bad_rot.warnings.size() == 1,
        "routed rotation other than 128 warns and falls back to 0");
  Check(Parse("GUFO_SPIKE_JITTER=3 GUFO_SPIKE_ACT=a4").jitter == 3 &&
            Parse("GUFO_SPIKE_ACT=a4").jitter == 0 &&
            !Parse("GUFO_SPIKE_JITTER=1").enabled,
        "jitter parses; it alone does not enable the spike");
  const s::Knobs act = Parse(
      "GUFO_SPIKE_WGROUP=32 GUFO_SPIKE_HAD=128 GUFO_SPIKE_ACT=a4 "
      "GUFO_SPIKE_AGROUP=32 GUFO_SPIKE_ROUTED_ACT=a4 "
      "GUFO_SPIKE_ROUTED_AGROUP=16 "
      "GUFO_SPIKE_TARGETS=127 GUFO_SPIKE_MIN_ROWS=2048 "
      "GUFO_SPIKE_MIN_ROWS_MODE=act");
  Check(act.min_rows == 2048 && act.min_rows_act && act.warnings.empty(),
        "MIN_ROWS_MODE=act parses");
  Check(!Parse("GUFO_SPIKE_MIN_ROWS=2048").min_rows_act &&
            !Parse("GUFO_SPIKE_MIN_ROWS_MODE=all").min_rows_act,
        "MIN_ROWS_MODE defaults to all");
  const s::Knobs bad_mode = Parse("GUFO_SPIKE_MIN_ROWS_MODE=weights");
  Check(!bad_mode.min_rows_act && bad_mode.warnings.size() == 1,
        "unknown MIN_ROWS_MODE warns and falls back to all");
  for (int b : {4, 5, 6, 8})
    Check(Parse("GUFO_SPIKE_WBITS=" + std::to_string(b)).wbits == b &&
              Parse("GUFO_SPIKE_WBITS=" + std::to_string(b)).warnings.empty(),
          "WBITS 4|5|6|8 parse");
  const s::Knobs bad_bits = Parse("GUFO_SPIKE_WBITS=7");
  Check(bad_bits.wbits == 4 && bad_bits.warnings.size() == 1,
        "WBITS 7 warns and falls back to 4");
  const s::Knobs wonly = Parse(
      "GUFO_SPIKE_WGROUP=32 GUFO_SPIKE_WBITS=5 GUFO_SPIKE_HAD=128 "
      "GUFO_SPIKE_TARGETS=1 GUFO_SPIKE_MIN_ROWS_MODE=act GUFO_SPIKE_JITTER=1");
  Check(wonly.enabled && wonly.dense.grid == s::kActNone &&
            wonly.routed.grid == s::kActNone && wonly.min_rows_act &&
            wonly.min_rows == 0 && wonly.wbits == 5 && wonly.warnings.empty(),
        "weights-only decode recipe: shadow on, activations never quantized");

  const auto unknown = s::UnknownKnobs(
      {"GUFO_SPIKE_ACT", "GUFO_SPIKE_ROUTED_TARGETS", "GUFO_SPIKE_MIN_ROWS",
       "PATH", "GUFO_SPIKE_ROUTED_CLIP", "GUFO_SPIKE_CLIPS"});
  Check(unknown.size() == 2 && unknown[0] == "GUFO_SPIKE_ROUTED_TARGETS" &&
            unknown[1] == "GUFO_SPIKE_CLIPS",
        "unknown GUFO_SPIKE_* names are reported");
  Check(s::RecipeText(stack.routed) == "fp4/g16/clip1.00", "recipe text");
}

void TestScalars() {
  const float in[] = {0.0F, 0.25F, 0.26F, 0.75F, 1.25F, 1.3F,   1.75F, 2.5F,
                      2.6F, 3.5F,  5.0F,  5.1F,  100.F, -0.75F, -5.0F};
  const float want[] = {0.0F, 0.0F, 0.5F, 1.0F, 1.0F, 1.5F,  2.0F, 2.0F,
                        3.0F, 4.0F, 4.0F, 6.0F, 6.0F, -1.0F, -4.0F};
  for (std::size_t i = 0; i < sizeof(in) / sizeof(in[0]); ++i)
    Check(s::E2M1Round(in[i]) == want[i], "E2M1 round half to even");

  Check(s::E4M3Round(1.0F) == 1.0F, "E4M3 exact");
  Check(s::E4M3Round(1.0625F) == 1.0F, "E4M3 tie to even (down)");
  Check(s::E4M3Round(1.1875F) == 1.25F, "E4M3 tie to even (up)");
  Check(s::E4M3Round(448.0F) == 448.0F && s::E4M3Round(470.0F) == 448.0F &&
            s::E4M3Round(1.0e6F) == 448.0F,
        "E4M3 saturates at 448");
  Check(s::E4M3Round(std::ldexp(1.0F, -10)) == 0.0F, "E4M3 subnormal tie");
  Check(s::E4M3Round(std::ldexp(3.0F, -10)) == std::ldexp(1.0F, -8),
        "E4M3 subnormal rounding");
  Check(s::E4M3Round(std::ldexp(1.0F, -6)) == std::ldexp(1.0F, -6),
        "E4M3 min normal");
  Check(s::E4M3Round(0.3F) == 0.3125F, "E4M3 0.3 -> 0.3125");

  Check(s::ActScale(7.0F, s::kActInt4, 1.0F) == 1.0F &&
            s::ActScale(127.0F, s::kActInt8, 1.0F) == 1.0F,
        "integer scales");
  const float gs = s::Fp4RowScale(6.0F * 448.0F);
  Check(gs == 1.0F && s::ActScale(6.0F * 448.0F, s::kActFp4, gs) == 448.0F,
        "fp4 row max lands on 6 x 448");
  Check(s::ActQuant(9.0F, 1.0F, 1.0F, s::kActInt4) == 7.0F &&
            s::ActQuant(-9.0F, 1.0F, 1.0F, s::kActInt4) == -8.0F &&
            s::ActQuant(2.5F, 1.0F, 1.0F, s::kActInt4) == 2.0F,
        "int4 clamp and round half to even");
}

// The original FqRows integer arithmetic (before the recipe knobs).
std::vector<float> Original(std::vector<float> x, int bits, int group,
                            float clip) {
  const float qmax = bits == 4 ? 7.0F : 127.0F;
  const int k = static_cast<int>(x.size());
  const int g = group == 0 ? k : group;
  for (int g0 = 0; g0 < k; g0 += g) {
    float m = 0.0F;
    for (int i = g0; i < g0 + g; ++i)
      m = std::fmax(m, std::fabs(x[i]));
    const float sc = clip * m / qmax;
    const float inv = sc > 0.0F ? 1.0F / sc : 0.0F;
    for (int i = g0; i < g0 + g; ++i)
      x[i] =
          std::fmin(std::fmax(std::rint(x[i] * inv), -qmax - 1.0F), qmax) * sc;
  }
  return x;
}

double Mse(const std::vector<float>& a, const std::vector<float>& b) {
  double e = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    e +=
        (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
  return e / static_cast<double>(a.size());
}

void TestRows() {
  std::mt19937 rng(1234);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(2560);
  for (float& v : x)
    v = normal(rng);
  // Heavy channels like post-SwiGLU / out-projection inputs.
  for (int c : {7, 300, 301, 1500, 2559})
    x[c] = (c % 2 != 0 ? -1.0F : 1.0F) * (40.0F + static_cast<float>(c % 7));

  for (int bits : {4, 8})
    for (int group : {0, 16, 32, 128})
      for (float clip : {1.0F, 0.9F}) {
        std::vector<float> y = x;
        s::QuantizeRowRef(y, {bits, group, clip}, false, 0);
        Check(y == Original(x, bits, group, clip),
              "integer recipes are bit-identical to the original emulation");
      }

  // Clip search never loses to the plain absmax scale (1.0 is a candidate).
  for (int grid : {s::kActInt4, s::kActFp4})
    for (int group : {0, 16, 32}) {
      std::vector<float> plain = x, searched = x;
      s::QuantizeRowRef(plain, {grid, group, 1.0F}, false, 0);
      s::QuantizeRowRef(searched, {grid, group, 1.0F}, true, 0);
      Check(Mse(searched, x) <= Mse(plain, x), "clip search lowers the MSE");
    }
  {
    std::vector<float> clean(2560);
    for (float& v : clean)
      v = normal(rng);
    std::vector<float> plain = clean, searched = clean;
    s::QuantizeRowRef(plain, {s::kActInt4, 0, 1.0F}, false, 0);
    s::QuantizeRowRef(searched, {s::kActInt4, 0, 1.0F}, true, 0);
    Check(Mse(searched, clean) < 0.9 * Mse(plain, clean),
          "per-row int4 clip search helps on a Gaussian row");
  }

  // Row outliers: the K largest stay exact and stop dominating the scale.
  {
    std::vector<float> y = x;
    s::QuantizeRowRef(y, {s::kActInt4, 0, 1.0F}, false, 5);
    for (int c : {7, 300, 301, 1500, 2559})
      Check(y[c] == x[c], "row outliers pass through");
    std::vector<float> plain = x;
    s::QuantizeRowRef(plain, {s::kActInt4, 0, 1.0F}, false, 0);
    Check(Mse(y, x) < 0.05 * Mse(plain, x),
          "retaining 5 outliers shrinks per-row int4 error");
  }
  // Call-mode keep flags behave the same for the flagged channels.
  {
    std::vector<std::uint8_t> keep(x.size(), 0);
    keep[7] = keep[300] = 1;
    std::vector<float> y = x;
    s::QuantizeRowRef(y, {s::kActInt4, 32, 1.0F}, false, 0, keep.data());
    Check(y[7] == x[7] && y[300] == x[300] && y[302] != x[302],
          "call-mode keep flags pass exactly the flagged channels");
  }

  // FP4: every quantized value is an E2M1 code times a UE4M3 block scale
  // times the row scale.
  {
    std::vector<float> y = x;
    s::QuantizeRowRef(y, {s::kActFp4, 16, 1.0F}, false, 0);
    float row_max = 0.0F;
    for (float v : x)
      row_max = std::fmax(row_max, std::fabs(v));
    const float gs = s::Fp4RowScale(row_max);
    for (int g0 = 0; g0 < 2560; g0 += 16) {
      float amax = 0.0F;
      for (int i = g0; i < g0 + 16; ++i)
        amax = std::fmax(amax, std::fabs(x[i]));
      const float bs = s::E4M3Round(amax / 6.0F / gs);
      Check(s::E4M3Round(bs) == bs, "block scale is UE4M3");
      for (int i = g0; i < g0 + 16; ++i) {
        const float code = y[i] / (bs * gs);
        Check(std::fabs(code - s::E2M1Round(code)) < 1e-4F,
              "fp4 values are E2M1 codes");
      }
    }
    std::vector<float> int4 = x;
    s::QuantizeRowRef(int4, {s::kActInt4, 16, 1.0F}, false, 0);
    Check(Mse(y, x) > 0.0 && Mse(int4, x) > 0.0, "both grids are lossy");
  }
}

// RequantRow's emulation arithmetic for one row (no rotation), sequential:
// per-group clip search, 6-bit group scales x row scale, F16-stored scale,
// codes clamped to the WBITS grid. Mirrors spike.hip.cpp's RequantRow<false>.
std::vector<float> RequantRef(const std::vector<float>& w, int group, int wbits,
                              std::vector<int>* codes) {
  const float qpos = s::WeightQmax(wbits), qneg = -qpos - 1.0F;
  const float clip[8] = {1.0F, 0.95F, 0.9F, 0.85F, 0.8F, 0.75F, 0.7F, 0.65F};
  const std::size_t ng = w.size() / static_cast<std::size_t>(group);
  std::vector<float> gs(ng);
  for (std::size_t g = 0; g < ng; ++g) {
    float amax = 0.0F;
    for (int i = 0; i < group; ++i)
      amax = std::fmax(amax, std::fabs(w[g * group + i]));
    float best = amax / qpos, err_best = 3.0e38F;
    for (float c : clip) {
      const float t = amax * c / qpos, inv = 1.0F / t;
      float err = 0.0F;
      for (int i = 0; i < group; ++i) {
        const float v = w[g * group + i];
        const float e =
            v - std::fmin(std::fmax(std::rint(v * inv), qneg), qpos) * t;
        err += e * e;
      }
      if (err < err_best) {
        err_best = err;
        best = t;
      }
    }
    gs[g] = best;
  }
  float m = 0.0F;
  for (float v : gs)
    m = std::fmax(m, v);
  const float d = m / 63.0F;
  std::vector<float> out(w.size());
  for (std::size_t g = 0; g < ng; ++g) {
    const float sc =
        d * std::fmin(std::fmax(std::rint(gs[g] / d), 1.0F), 63.0F);
    const float inv = 1.0F / sc;
    for (int i = 0; i < group; ++i) {
      const float q =
          std::fmin(std::fmax(std::rint(w[g * group + i] * inv), qneg), qpos);
      codes->push_back(static_cast<int>(q));
      out[g * group + i] = q * sc;
    }
  }
  return out;
}

// GUFO_SPIKE_WBITS 5 and 6: the same group/scale6 arithmetic on a wider
// grid; codes stay in the int8 container, use the new range and cut the error.
void TestWeightBits() {
  Check(s::WeightQmax(4) == 7.0F && s::WeightQmax(8) == 127.0F &&
            s::WeightQmax(5) == 15.0F && s::WeightQmax(6) == 31.0F,
        "weight grids: 4 -> 7, 5 -> 15, 6 -> 31, 8 -> 127");
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 0.02F);
  std::vector<float> w(6144);
  for (float& v : w)
    v = normal(rng);
  double prev = 1e30;
  for (int b : {4, 5, 6, 8}) {
    std::vector<int> codes;
    const std::vector<float> q = RequantRef(w, 32, b, &codes);
    const int qpos = static_cast<int>(s::WeightQmax(b));
    int lo = 0, hi = 0;
    for (int c : codes) {
      lo = std::min(lo, c);
      hi = std::max(hi, c);
    }
    Check(lo >= -qpos - 1 && hi <= qpos && hi >= qpos - 1 && lo <= -qpos + 1,
          "codes use the WBITS range and fit int8");
    const double mse = Mse(q, w);
    std::printf("wbits %d: rel rms %.4f, codes [%d, %d]\n", b,
                std::sqrt(mse) / 0.02, lo, hi);
    Check(mse < prev, "more weight bits, less error");
    prev = mse;
  }
}

// GUFO_SPIKE_JITTER: seed 0 is exactly off; nonzero seeds give bounded,
// deterministic, seed-dependent perturbations that flip some codes only.
void TestJitter() {
  Check(s::JitterKey(0, 32, 5) == 0 && s::JitterEps(0, 123) == 0.0F,
        "seed 0 is no jitter");
  const std::uint32_t k1 = s::JitterKey(1, 32, 5);
  Check(k1 != 0 && k1 == s::JitterKey(1, 32, 5) &&
            k1 != s::JitterKey(2, 32, 5) && k1 != s::JitterKey(1, 64, 5) &&
            k1 != s::JitterKey(1, 32, 6),
        "keys depend on seed, role and layer, deterministically");
  double sum = 0.0;
  float lo = 1.0F, hi = -1.0F;
  for (std::uint32_t i = 0; i < 100000; ++i) {
    const float e = s::JitterEps(k1, i);
    lo = std::fmin(lo, e);
    hi = std::fmax(hi, e);
    sum += e;
  }
  const float bound = std::ldexp(1.0F, -20);
  Check(lo >= -bound && hi < bound && lo < -0.99F * bound && hi > 0.99F * bound,
        "jitter spans [-2^-20, 2^-20)");
  Check(std::fabs(sum / 100000.0) < 0.01 * bound, "jitter is centered");

  std::mt19937 rng(99);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(2560 * 64);
  for (float& v : x)
    v = normal(rng);
  std::vector<float> plain = x, zero = x, j1 = x, j2 = x;
  const s::ActRecipe a4{s::kActInt4, 16, 1.0F};
  s::QuantizeRowRef(plain, a4, false, 0);
  s::QuantizeRowRef(zero, a4, false, 0, nullptr, s::JitterKey(0, 32, 5));
  s::QuantizeRowRef(j1, a4, false, 0, nullptr, k1);
  s::QuantizeRowRef(j2, a4, false, 0, nullptr, s::JitterKey(2, 32, 5));
  Check(zero == plain, "seed 0 is bit-identical");
  std::size_t flips1 = 0, flips12 = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    flips1 += j1[i] != plain[i] ? 1 : 0;
    flips12 += j1[i] != j2[i] ? 1 : 0;
  }
  std::printf(
      "jitter: %zu of %zu values move (seed 1 vs 0), %zu differ "
      "between seeds 1 and 2\n",
      flips1, x.size(), flips12);
  Check(flips1 > 0 && flips12 > 0, "nonzero seeds change the realization");
  // Values move with their group's scale; codes flip only at boundaries.
  std::size_t code_flips = 0;
  for (std::size_t g0 = 0; g0 < x.size(); g0 += 16) {
    float amax = 0.0F;
    for (std::size_t i = g0; i < g0 + 16; ++i)
      amax = std::fmax(amax, std::fabs(x[i]));
    const float sc = amax / 7.0F;
    for (std::size_t i = g0; i < g0 + 16; ++i)
      code_flips += std::rint(plain[i] / sc) != std::rint(j1[i] / sc) ? 1 : 0;
  }
  std::printf("jitter: %zu int4 codes flipped by seed 1\n", code_flips);
  Check(code_flips < x.size() / 1000,
        "seed 1 flips at most a few boundary codes, not the recipe");
}

}  // namespace

int main() {
  TestParse();
  TestScalars();
  TestRows();
  TestJitter();
  TestWeightBits();
  std::puts("spike_knobs: ok");
  return 0;
}
