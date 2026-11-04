#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_KNOBS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_KNOBS_HPP_

// Spike: the GUFO_SPIKE_* knob parser and the scalar quantization
// primitives the emulation kernels in spike.hip.cpp use. Plain C++ with no HIP
// runtime dependency, so the CPU test (spike_knobs_test) checks parsing and
// numerics without touching a GPU. Knob semantics are documented in spike.hpp.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#if defined(__HIPCC__)
#define GUFO_SPIKE_HD __host__ __device__
#else
#define GUFO_SPIKE_HD
#endif

namespace gufo::models::qwen38_flash_next::rocm::spike {

/// Activation grid of a recipe. The values are passed to the kernels as int.
enum ActGrid : int {
  kActNone = 0,  // rotate only (if the weights are rotated)
  kActFp4 = 2,   // E2M1 codes, per-group UE4M3 scales x per-row FP32 scale
  kActInt4 = 4,  // symmetric int4 [-8, 7]
  kActInt8 = 8,  // symmetric int8 [-128, 127]
};

/// One activation recipe: grid, group size (0 = one scale per row) and the
/// clip ratio applied to the group absmax before the scale is derived.
struct ActRecipe {
  int grid{kActNone};
  int group{0};
  float clip{1.0F};
};

enum class OutlierMode : int {
  kRow = 0,   // the K largest |x| channels of every row stay F16
  kCall = 1,  // one set per call: the K channels with the largest column max
};

struct Knobs {
  bool enabled{false};
  ActRecipe dense;   // roles 1, 2, 4, 8, 16
  ActRecipe routed;  // roles 32, 64 (defaults to the dense recipe)
  int had{0};        // block Hadamard size on rotated (dense) tensors
  int wgroup{0};     // dense weight requantization group (0 = keep)
  int wbits{4};      // weight grid of the emulation: 4, 5, 6 or 8 bits
  int scale6{1};     // group scales are d_row x 6-bit integer (kernel format)
  unsigned targets{127};
  std::set<int> keep;        // layers left untouched
  std::set<int> a8_layers;   // layers whose activation grid becomes int8
  std::size_t min_rows{0};   // calls with fewer rows forward unchanged
  bool min_rows_act{false};  // MIN_ROWS_MODE=act: small calls keep W4
  bool clip_search{false};   // per-group MSE-optimal clip among AClip(0..7)
  int outliers{0};           // F16-retained channels per row / per call
  OutlierMode outlier_mode{OutlierMode::kRow};
  bool log{false};
  bool real{false};   // run the integer WMMA kernels instead of the emulation
  bool check{false};  // real mode also keeps the emulated shadow (tests)
  bool routed_real{false};   // routed experts on the iu4 kernels (prefill)
  bool routed_down4{false};  // Q5_1 down experts re-quantized to 4 bits
  int routed_check{0};       // routed real: emulate + log the first N calls
  int routed_had{0};         // routed experts stored rotated: Hadamard 128
  bool routed_tall{false};   // routed real: 128 x 64 blocks on 128-row maps
  std::uint32_t jitter{0};   // scale-jitter seed (0 = off, bit-identical)
  std::vector<std::string> warnings;
};

/// Every knob the binary reads; anything else with the prefix is reported.
inline constexpr const char* kKnownKnobs[] = {
    "GUFO_SPIKE_ACT",           "GUFO_SPIKE_AGROUP",
    "GUFO_SPIKE_CLIP",          "GUFO_SPIKE_ROUTED_ACT",
    "GUFO_SPIKE_ROUTED_AGROUP", "GUFO_SPIKE_ROUTED_CLIP",
    "GUFO_SPIKE_ACLIP_SEARCH",  "GUFO_SPIKE_OUTLIERS",
    "GUFO_SPIKE_OUTLIER_MODE",  "GUFO_SPIKE_MIN_ROWS",
    "GUFO_SPIKE_MIN_ROWS_MODE", "GUFO_SPIKE_A8_LAYERS",
    "GUFO_SPIKE_HAD",           "GUFO_SPIKE_WGROUP",
    "GUFO_SPIKE_WBITS",         "GUFO_SPIKE_SCALE6",
    "GUFO_SPIKE_TARGETS",       "GUFO_SPIKE_KEEP",
    "GUFO_SPIKE_LOG",           "GUFO_SPIKE_REAL",
    "GUFO_SPIKE_REAL_CHECK",    "GUFO_SPIKE_ROUTED_REAL",
    "GUFO_SPIKE_ROUTED_DOWN4",  "GUFO_SPIKE_ROUTED_CHECK",
    "GUFO_SPIKE_ROUTED_HAD",    "GUFO_SPIKE_ROUTED_TALL",
    "GUFO_SPIKE_JITTER",
};

/// Upper bound on GUFO_SPIKE_OUTLIERS (selection is K block-wide argmaxes).
inline constexpr int kMaxOutliers = 256;

/// Clip ratios of the activation clip search (the weight search's list).
inline constexpr int kAClipCount = 8;
GUFO_SPIKE_HD inline float AClip(int c) {
  constexpr float kRatios[kAClipCount] = {1.0F, 0.95F, 0.9F, 0.85F,
                                          0.8F, 0.75F, 0.7F, 0.65F};
  return kRatios[c];
}

/// Largest UE4M3 (FP8 E4M3, finite-only) magnitude and the E2M1 maximum.
inline constexpr float kE4M3Max = 448.0F;
inline constexpr float kE2M1Max = 6.0F;

/// Rounds |v| to the nearest UE4M3 value (round half to even, subnormal step
/// 2^-9, saturating at 448), as the NVFP4 block-scale cast does.
GUFO_SPIKE_HD inline float E4M3Round(float v) {
  v = fabsf(v);
  if (!(v > 0.0F))
    return 0.0F;
  if (v >= kE4M3Max)
    return kE4M3Max;
  int e = 0;
  (void)frexpf(v, &e);                   // v = m * 2^e, m in [0.5, 1)
  const int step = e < -5 ? -9 : e - 4;  // 3 mantissa bits, min normal 2^-6
  const float q = ldexpf(__builtin_rintf(ldexpf(v, -step)), step);
  return fminf(q, kE4M3Max);
}

/// Rounds v to the nearest E2M1 value {0, .5, 1, 1.5, 2, 3, 4, 6} with its
/// sign (round half to even on the mantissa, saturating at 6).
GUFO_SPIKE_HD inline float E2M1Round(float v) {
  const float a = fabsf(v);
  float q;
  if (a <= 0.25F)
    q = 0.0F;
  else if (a < 0.75F)
    q = 0.5F;
  else if (a <= 1.25F)
    q = 1.0F;
  else if (a < 1.75F)
    q = 1.5F;
  else if (a <= 2.5F)
    q = 2.0F;
  else if (a < 3.5F)
    q = 3.0F;
  else if (a <= 5.0F)
    q = 4.0F;
  else
    q = 6.0F;
  return v < 0.0F ? -q : q;
}

/// Group scale for a (clipped) group absmax. Integer grids: amax / qmax.
/// FP4: the UE4M3-rounded block scale amax / 6 / gs times the row scale gs.
GUFO_SPIKE_HD inline float ActScale(float clipped_amax, int grid, float gs) {
  if (grid == kActFp4)
    return gs > 0.0F ? E4M3Round(clipped_amax / kE2M1Max / gs) * gs : 0.0F;
  return clipped_amax / (grid == kActInt8 ? 127.0F : 7.0F);
}

/// Quantize-dequantize one value with scale `sc` (inv = 1 / sc, or 0).
GUFO_SPIKE_HD inline float ActQuant(float v, float inv, float sc, int grid) {
  if (grid == kActFp4)
    return E2M1Round(v * inv) * sc;
  const float qmax = grid == kActInt8 ? 127.0F : 7.0F;
  return fminf(fmaxf(__builtin_rintf(v * inv), -qmax - 1.0F), qmax) * sc;
}

/// Positive code limit of the dense weight grid (GUFO_SPIKE_WBITS): codes
/// span [-qmax - 1, qmax]; all fit the Q8_0 container's int8 codes.
GUFO_SPIKE_HD inline float WeightQmax(int wbits) {
  return wbits == 8 ? 127.0F : wbits == 6 ? 31.0F : wbits == 5 ? 15.0F : 7.0F;
}

/// The per-row FP32 scale of the FP4 grid: the row absmax lands on 6 x 448.
GUFO_SPIKE_HD inline float Fp4RowScale(float row_amax) {
  return row_amax / (kE2M1Max * kE4M3Max);
}

/// Scale jitter (GUFO_SPIKE_JITTER): a deterministic relative perturbation
/// e in [-2^-20, 2^-20) of one group scale, from a 32-bit integer hash of
/// (seed, role, layer) -- the key -- and the group index. Enough to flip
/// codes that sit on a rounding boundary, not enough to change the recipe;
/// seeds give an ensemble of equally valid quantized realizations.
GUFO_SPIKE_HD inline std::uint32_t JitterMix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352DU;
  x ^= x >> 15;
  x *= 0x846CA68BU;
  x ^= x >> 16;
  return x;
}

/// Key of one tensor's jitter stream; 0 (no jitter) exactly when seed is 0.
GUFO_SPIKE_HD inline std::uint32_t JitterKey(std::uint32_t seed, unsigned role,
                                             int layer) {
  if (seed == 0)
    return 0;
  const std::uint32_t l =
      JitterMix(static_cast<std::uint32_t>(layer) + 0x632BE5ABU);
  return JitterMix(seed ^ JitterMix(role * 0x9E3779B9U ^ l)) | 1U;
}

/// The relative perturbation e of group `index` (0 when key is 0). Applied as
/// scale += scale * e, so e keeps its full resolution.
GUFO_SPIKE_HD inline float JitterEps(std::uint32_t key, std::uint32_t index) {
  if (key == 0)
    return 0.0F;
  const std::uint32_t h = JitterMix(key ^ JitterMix(index + 0x85EBCA6BU));
  // 24 uniform bits -> u in [-1, 1) -> e = u * 2^-20.
  const float u = static_cast<float>(h >> 8) * 0x1p-23F - 1.0F;
  return u * 0x1p-20F;
}

namespace knobs_detail {

inline std::set<int> ParseList(std::string_view s) {
  std::set<int> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const std::size_t end = s.find(',', pos);
    const std::string item(
        s.substr(pos, end == std::string_view::npos ? end : end - pos));
    if (!item.empty())
      out.insert(std::atoi(item.c_str()));
    if (end == std::string_view::npos)
      break;
    pos = end + 1;
  }
  return out;
}

inline int ParseGrid(const char* v, std::vector<std::string>& warnings,
                     const char* name) {
  const std::string_view s(v);
  if (s == "a4")
    return kActInt4;
  if (s == "a8")
    return kActInt8;
  if (s == "fp4")
    return kActFp4;
  if (!s.empty() && s != "none")
    warnings.push_back(std::string(name) + "=" + v +
                       " is not none|a4|a8|fp4; using none");
  return kActNone;
}

}  // namespace knobs_detail

/// Parses the knobs from `env` (a getenv-like callable: name -> value or
/// nullptr). Production passes std::getenv; the CPU test passes a map.
template<class Env>
Knobs ParseKnobs(const Env& env) {
  Knobs r;
  const auto str = [&](const char* name) -> const char* {
    const char* v = env(name);
    return v != nullptr && *v != '\0' ? v : nullptr;
  };
  const auto num = [&](const char* name, int fallback) {
    const char* v = str(name);
    return v != nullptr ? std::atoi(v) : fallback;
  };
  const auto flt = [&](const char* name, float fallback) {
    const char* v = str(name);
    return v != nullptr ? static_cast<float>(std::atof(v)) : fallback;
  };
  const auto group_for = [&](const char* name, int grid, int inherited) {
    if (str(name) != nullptr)
      return num(name, 0);
    return inherited >= 0 ? inherited : grid == kActFp4 ? 16 : 0;
  };
  r.real = num("GUFO_SPIKE_REAL", 0) != 0;
  r.check = num("GUFO_SPIKE_REAL_CHECK", 0) != 0;
  r.log = num("GUFO_SPIKE_LOG", 0) != 0;
  if (const char* a = str("GUFO_SPIKE_ACT"))
    r.dense.grid = knobs_detail::ParseGrid(a, r.warnings, "GUFO_SPIKE_ACT");
  if (r.real && r.dense.grid == kActNone)
    r.dense.grid = kActInt4;
  r.dense.group = group_for("GUFO_SPIKE_AGROUP", r.dense.grid, -1);
  r.dense.clip = flt("GUFO_SPIKE_CLIP", 1.0F);
  r.routed = r.dense;
  if (const char* a = str("GUFO_SPIKE_ROUTED_ACT"))
    r.routed.grid =
        knobs_detail::ParseGrid(a, r.warnings, "GUFO_SPIKE_ROUTED_ACT");
  r.routed_real = num("GUFO_SPIKE_ROUTED_REAL", 0) != 0;
  r.routed_down4 = num("GUFO_SPIKE_ROUTED_DOWN4", 0) != 0;
  r.routed_check = num("GUFO_SPIKE_ROUTED_CHECK", 0);
  r.routed_had = num("GUFO_SPIKE_ROUTED_HAD", 0);
  if (r.routed_had != 0 && r.routed_had != 128) {
    r.warnings.push_back("GUFO_SPIKE_ROUTED_HAD is 0 or 128; using 0");
    r.routed_had = 0;
  }
  r.routed_tall = num("GUFO_SPIKE_ROUTED_TALL", 0) != 0;
  if (const char* j = str("GUFO_SPIKE_JITTER"))
    r.jitter = static_cast<std::uint32_t>(std::strtoul(j, nullptr, 10));
  if (r.routed_real && r.routed.grid == kActNone)
    r.routed.grid = kActInt4;
  // ROUTED_AGROUP, else an explicit AGROUP, else the routed grid's default.
  r.routed.group =
      group_for("GUFO_SPIKE_ROUTED_AGROUP", r.routed.grid,
                str("GUFO_SPIKE_AGROUP") != nullptr ? r.dense.group : -1);
  r.routed.clip = flt("GUFO_SPIKE_ROUTED_CLIP", r.dense.clip);
  r.had = num("GUFO_SPIKE_HAD", 0);
  r.wgroup = num("GUFO_SPIKE_WGROUP", 0);
  if (r.real && r.wgroup == 0)
    r.wgroup = 32;
  r.wbits = num("GUFO_SPIKE_WBITS", 4);
  if (r.wbits != 4 && r.wbits != 5 && r.wbits != 6 && r.wbits != 8) {
    r.warnings.push_back("GUFO_SPIKE_WBITS=" + std::to_string(r.wbits) +
                         " is not 4|5|6|8; using 4");
    r.wbits = 4;
  }
  r.scale6 = num("GUFO_SPIKE_SCALE6", 1);
  r.targets =
      static_cast<unsigned>(num("GUFO_SPIKE_TARGETS", r.real ? 15 : 127));
  if (const char* k = str("GUFO_SPIKE_KEEP"))
    r.keep = knobs_detail::ParseList(k);
  if (const char* k = str("GUFO_SPIKE_A8_LAYERS"))
    r.a8_layers = knobs_detail::ParseList(k);
  const int min_rows = num("GUFO_SPIKE_MIN_ROWS", 0);
  r.min_rows = min_rows > 0 ? static_cast<std::size_t>(min_rows) : 0;
  if (const char* m = str("GUFO_SPIKE_MIN_ROWS_MODE")) {
    const std::string_view s(m);
    if (s == "act")
      r.min_rows_act = true;
    else if (s != "all")
      r.warnings.push_back(std::string("GUFO_SPIKE_MIN_ROWS_MODE=") + m +
                           " is not all|act; using all");
  }
  r.clip_search = num("GUFO_SPIKE_ACLIP_SEARCH", 0) != 0;
  r.outliers = num("GUFO_SPIKE_OUTLIERS", 0);
  if (r.outliers < 0 || r.outliers > kMaxOutliers) {
    r.warnings.push_back("GUFO_SPIKE_OUTLIERS clamped to [0, 256]");
    r.outliers = r.outliers < 0 ? 0 : kMaxOutliers;
  }
  if (const char* m = str("GUFO_SPIKE_OUTLIER_MODE")) {
    const std::string_view s(m);
    if (s == "call")
      r.outlier_mode = OutlierMode::kCall;
    else if (s != "row")
      r.warnings.push_back(std::string("GUFO_SPIKE_OUTLIER_MODE=") + m +
                           " is not row|call; using row");
  }
  for (const ActRecipe* a : {&r.dense, &r.routed}) {
    if (a->group < 0 || (a->group != 0 && a->group % 16 != 0))
      r.warnings.push_back("activation group " + std::to_string(a->group) +
                           " is not 0 or a multiple of 16");
  }
  if (r.real && (r.clip_search || r.outliers > 0 || r.dense.grid != kActInt4 ||
                 r.dense.group != 0))
    r.warnings.push_back(
        "GUFO_SPIKE_REAL runs the per-row int4 kernel; ACT/AGROUP/"
        "ACLIP_SEARCH/OUTLIERS apply only to the emulated (unpacked) roles");
  if (r.routed_real && (r.routed.grid != kActInt4 || r.routed.clip != 1.0F ||
                        (r.routed.group != 0 && r.routed.group != 32 &&
                         r.routed.group != 64 && r.routed.group != 128)))
    r.warnings.push_back(
        "GUFO_SPIKE_ROUTED_REAL runs int4 activations, clip 1, groups "
        "0|32|64|128 only; other routed recipes stay on the emulation");
  r.enabled = r.dense.grid != kActNone || r.routed.grid != kActNone ||
              r.wgroup != 0 || r.had != 0 || r.routed_down4 ||
              r.routed_had != 0;
  return r;
}

/// GUFO_SPIKE_* names in `names` that this binary does not read.
inline std::vector<std::string> UnknownKnobs(
    const std::vector<std::string>& names) {
  std::vector<std::string> out;
  for (const std::string& n : names) {
    if (n.rfind("GUFO_SPIKE_", 0) != 0)
      continue;
    bool known = false;
    for (const char* k : kKnownKnobs)
      known = known || n == k;
    if (!known)
      out.push_back(n);
  }
  return out;
}

/// Short text for a recipe, e.g. "a4/g32/clip1.00", for the startup log.
inline std::string RecipeText(const ActRecipe& a) {
  const char* g = a.grid == kActInt4   ? "a4"
                  : a.grid == kActInt8 ? "a8"
                  : a.grid == kActFp4  ? "fp4"
                                       : "none";
  char clip[32];
  std::snprintf(clip, sizeof(clip), "%.2f", static_cast<double>(a.clip));
  return std::string(g) + "/g" + std::to_string(a.group) + "/clip" + clip;
}

/// CPU reference of one activation row (no rotation): the arithmetic of the
/// FqRows kernel, sequential. `keep` (optional) marks channels that stay
/// unquantized and are excluded from every absmax; with `outliers` > 0 and no
/// `keep`, the row's own K largest |x| (lowest index on ties) are kept.
/// `jitter` is the tensor's JitterKey: integer grids perturb the chosen scale
/// of group g (index within the row), FP4 perturbs the row scale.
inline void QuantizeRowRef(std::vector<float>& x, const ActRecipe& a,
                           bool clip_search, int outliers,
                           const std::uint8_t* keep = nullptr,
                           std::uint32_t jitter = 0) {
  const int k = static_cast<int>(x.size());
  std::vector<std::uint8_t> fl(x.size(), 0);
  if (keep != nullptr) {
    for (int i = 0; i < k; ++i)
      fl[i] = keep[i];
  } else {
    for (int r = 0; r < outliers && r < k; ++r) {
      int best = -1;
      for (int i = 0; i < k; ++i)
        if (fl[i] == 0 && (best < 0 || fabsf(x[i]) > fabsf(x[best])))
          best = i;
      if (best < 0)
        break;
      fl[static_cast<std::size_t>(best)] = 1;
    }
  }
  if (a.grid == kActNone)
    return;
  float gs = 1.0F;
  if (a.grid == kActFp4) {
    float m = 0.0F;
    for (int i = 0; i < k; ++i)
      if (fl[i] == 0)
        m = fmaxf(m, fabsf(x[i]));
    gs = Fp4RowScale(m);
    if (jitter != 0)
      gs += gs * JitterEps(jitter, 0xFFFFFFFFU);
  }
  const int group = a.group == 0 || k % a.group != 0 ? k : a.group;
  for (int g0 = 0; g0 < k; g0 += group) {
    float amax = 0.0F;
    for (int i = g0; i < g0 + group; ++i)
      if (fl[i] == 0)
        amax = fmaxf(amax, fabsf(x[i]));
    float sc = ActScale(a.clip * amax, a.grid, gs);
    if (clip_search && amax > 0.0F) {
      float best = 3.0e38F;
      for (int c = 0; c < kAClipCount; ++c) {
        const float t = ActScale(AClip(c) * amax, a.grid, gs);
        const float inv = t > 0.0F ? 1.0F / t : 0.0F;
        float err = 0.0F;
        for (int i = g0; i < g0 + group; ++i) {
          if (fl[i] != 0)
            continue;
          const float e = x[i] - ActQuant(x[i], inv, t, a.grid);
          err += e * e;
        }
        if (err < best) {
          best = err;
          sc = t;
        }
      }
    }
    if (jitter != 0 && a.grid != kActFp4)
      sc += sc * JitterEps(jitter, static_cast<std::uint32_t>(g0 / group));
    const float inv = sc > 0.0F ? 1.0F / sc : 0.0F;
    for (int i = g0; i < g0 + group; ++i)
      if (fl[i] == 0)
        x[i] = ActQuant(x[i], inv, sc, a.grid);
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm::spike

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SPIKE_KNOBS_HPP_
