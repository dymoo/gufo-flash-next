// WMMA issue-rate probe for gfx1151 (Strix Halo, RDNA 3.5): fp16, bf16,
// fp16-accumulate, int8 (iu8) and int4 (iu4), side by side.
//
// Build: nix develop -c tools/bench/build.sh wmma_issue_probe
//        (or: hipcc -O3 --offload-arch=gfx1151 -std=c++20 wmma_issue_probe.hip)
// Run:   /tmp/wmma_issue_probe [quick]
//
// Sections
//   1. peak:   4 independent accumulator chains per wave (the same shape as
//              gfx1151_peak.hip). Every type is measured in interleaved
//              rounds, so a clock, power or thermal drift shows up as a
//              difference between rounds rather than as a difference between
//              types. Resident waves per SIMD are swept for the integer types.
//   2. chains: 1/2/4/8 independent chains per wave: dependent-issue latency
//              versus the saturated issue interval.
//   3. mix:    a TIxTJ grid of output tiles per wave, S WMMAs per scale block, then the
//              int32->fp32 scaled accumulate a W4A4/W8A8 GEMM runs per scale
//              group (v_cvt_f32_i32 + v_fma_f32 per element). Shows how much
//              of that VALU epilogue overlaps the matrix pipe, per type and
//              per scale-group width. The operands change every step so the
//              WMMAs cannot be hoisted out of the loop.
//
// Each row prints TOPS from HIP event time, the effective WMMA instructions
// per CU-clock implied by the amdgpu-reported shader clock sampled right after
// the run, and that clock, package power and edge temperature. The in-kernel
// clock64() is not used: on gfx11 it is a 20-bit SHADER_CYCLES field.
// Integer and fp32-accumulate peak rows are verified against a host value.
// Device memory: a few KiB. Total run time is bounded (default under 45 s).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    const hipError_t err_ = (expr);                                     \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using i32x2 = __attribute__((__vector_size__(8))) int;
using i32x4 = __attribute__((__vector_size__(16))) int;
using i32x8 = __attribute__((__vector_size__(32))) int;
using bf16x16 = __attribute__((__vector_size__(32))) short;
using f16x16 = __attribute__((__vector_size__(32))) _Float16;
using f32x8 = __attribute__((__vector_size__(32))) float;

enum class Kind { kF16, kBf16, kF16Acc, kIu8, kIu4 };

constexpr const char* Name(Kind k) {
  switch (k) {
    case Kind::kF16: return "f16 (f32 acc)";
    case Kind::kBf16: return "bf16 (f32 acc)";
    case Kind::kF16Acc: return "f16 (f16 acc)";
    case Kind::kIu8: return "iu8 (i32 acc)";
    case Kind::kIu4: return "iu4 (i32 acc)";
  }
  return "?";
}
constexpr bool IsInt(Kind k) { return k == Kind::kIu8 || k == Kind::kIu4; }

// Operand and accumulator bundles. All A/B elements are +1 (or +1 / 0.5), so
// an fp32 or int32 accumulator advances by an exact constant per WMMA.
// Perturb(x, v) changes one operand word with a loop-varying value.
template<Kind K>
struct Ops;
template<>
struct Ops<Kind::kF16> {
  using Acc = f32x8;
  using Frag = f16x16;
  static constexpr double kStep = 8.0;  // 16 * (1.0 * 0.5)
  __device__ static Frag A() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 1.0F; return v; }
  __device__ static Frag B() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 0.5F; return v; }
  __device__ static Frag Perturb(Frag v, int x) { v[0] = static_cast<_Float16>(x & 3); return v; }
  __device__ static Acc Zero() { return Acc{}; }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  }
  __device__ static float ToF(Acc c, int j) { return c[j]; }
};
template<>
struct Ops<Kind::kBf16> {
  using Acc = f32x8;
  using Frag = bf16x16;
  static constexpr double kStep = 8.0;
  __device__ static Frag A() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 0x3f80; return v; }
  __device__ static Frag B() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 0x3f00; return v; }
  __device__ static Frag Perturb(Frag v, int x) { v[0] = static_cast<short>(0x3f80 + (x & 3)); return v; }
  __device__ static Acc Zero() { return Acc{}; }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  }
  __device__ static float ToF(Acc c, int j) { return c[j]; }
};
template<>
struct Ops<Kind::kF16Acc> {
  // 16 half accumulators per lane; opsel selects the low/high half packing.
  using Acc = f16x16;
  using Frag = f16x16;
  static constexpr double kStep = 0.0;  // not exactly checkable (f16 rounding)
  __device__ static Frag A() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 1.0F; return v; }
  __device__ static Frag B() { Frag v; for (int i = 0; i < 16; ++i) v[i] = 0.0009765625F; return v; }
  __device__ static Frag Perturb(Frag v, int x) { v[0] = static_cast<_Float16>(x & 3); return v; }
  __device__ static Acc Zero() { return Acc{}; }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, false);
  }
  __device__ static float ToF(Acc c, int j) { return static_cast<float>(c[j]); }
};
template<>
struct Ops<Kind::kIu8> {
  using Acc = i32x8;
  using Frag = i32x4;
  static constexpr double kStep = 16.0;  // 16 * (1 * 1)
  __device__ static Frag A() { return Frag{0x01010101, 0x01010101, 0x01010101, 0x01010101}; }
  __device__ static Frag B() { return A(); }
  __device__ static Frag Perturb(Frag v, int x) { v[0] ^= (x & 7); return v; }
  __device__ static Acc Zero() { return Acc{}; }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
  }
  __device__ static float ToF(Acc c, int j) { return static_cast<float>(c[j]); }
};
template<>
struct Ops<Kind::kIu4> {
  using Acc = i32x8;
  using Frag = i32x2;
  static constexpr double kStep = 16.0;  // 16 * (1 * 1), every nibble is +1
  __device__ static Frag A() { return Frag{0x11111111, 0x11111111}; }
  __device__ static Frag B() { return A(); }
  __device__ static Frag Perturb(Frag v, int x) { v[0] ^= (x & 7); return v; }
  __device__ static Acc Zero() { return Acc{}; }
  __device__ static Acc Mma(Frag a, Frag b, Acc c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, true);
  }
  __device__ static float ToF(Acc c, int j) { return static_cast<float>(c[j]); }
};

constexpr std::uint32_t kSeed = 2166136261U;
__host__ __device__ inline std::uint32_t Fold(std::uint32_t h, std::uint32_t v) {
  return (h ^ v) * 16777619U;
}
template<typename T>
__host__ __device__ inline std::uint32_t Word(T v) {
  static_assert(sizeof(T) == 4);
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return u;
}

// 1+2. Saturation: CH independent accumulator chains, 16 WMMA per chain per
// rep (the loop is fully unrolled over both).
template<Kind K, int CH>
__global__ __launch_bounds__(256) void Peak(std::uint32_t* sink, int reps) {
  using O = Ops<K>;
  const auto a = O::A();
  const auto b = O::B();
  typename O::Acc c[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    c[i] = O::Zero();
    if constexpr (K != Kind::kF16Acc) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (IsInt(K))
          c[i][j] = i * 8 + j;
        else
          c[i][j] = static_cast<float>(i * 8 + j);
      }
    }
  }
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int s = 0; s < 16; ++s) {
#pragma unroll
      for (int i = 0; i < CH; ++i)
        c[i] = O::Mma(a, b, c[i]);
    }
  }
  std::uint32_t h = kSeed;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      h = Fold(h, Word(O::ToF(c[i], j)));
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = h;
}

// 3. Scale-group epilogue mix. A wave computes a TI x TJ grid of 16x16 output
// tiles from TI A fragments and TJ B fragments (as a real GEMM wave tile does,
// so no two tiles share operands and the compiler cannot merge them). Each tile
// runs S dependent WMMAs into a zeroed accumulator, then folds the result into
// an fp32 accumulator with one scale: facc += scale * float(acc). For int types
// that is one v_cvt_f32_i32 and one v_fma_f32 per element. Operands change with
// (rep, s), so nothing is loop-invariant. Direct = accumulate across all steps
// in the matrix accumulator with no epilogue (the fp16 GEMM shape).
template<Kind K, int TI, int TJ, int S, bool Direct>
__global__ __launch_bounds__(256) void Mix(std::uint32_t* sink, int reps,
                                           float scale) {
  using O = Ops<K>;
  constexpr int T = TI * TJ;
  const auto a0 = O::A();
  const auto b0 = O::B();
  f32x8 facc[T];
  typename O::Acc dacc[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    facc[t] = f32x8{};
    dacc[t] = O::Zero();
  }
  for (int r = 0; r < reps; ++r) {
    typename O::Acc acc[T];
    if constexpr (!Direct) {
#pragma unroll
      for (int t = 0; t < T; ++t)
        acc[t] = O::Zero();
    }
#pragma unroll
    for (int s = 0; s < S; ++s) {
      typename O::Frag a[TI], b[TJ];
#pragma unroll
      for (int i = 0; i < TI; ++i)
        a[i] = O::Perturb(a0, r + s + i);
#pragma unroll
      for (int j = 0; j < TJ; ++j)
        b[j] = O::Perturb(b0, 3 * r + s + j + 5);
#pragma unroll
      for (int i = 0; i < TI; ++i) {
#pragma unroll
        for (int j = 0; j < TJ; ++j) {
          if constexpr (Direct)
            dacc[i * TJ + j] = O::Mma(a[i], b[j], dacc[i * TJ + j]);
          else
            acc[i * TJ + j] = O::Mma(a[i], b[j], acc[i * TJ + j]);
        }
      }
    }
    if constexpr (!Direct) {
#pragma unroll
      for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          facc[t][j] = __builtin_fmaf(scale, O::ToF(acc[t], j), facc[t][j]);
      }
    }
  }
  std::uint32_t h = kSeed;
#pragma unroll
  for (int t = 0; t < T; ++t) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      h = Fold(Fold(h, Word(facc[t][j])), Word(O::ToF(dacc[t], j)));
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = h;
}

// ---------------------------------------------------------------- host side

// amdgpu sysfs sampling (clock, power, temperature) for the context columns.
struct Sys {
  std::string freq, power, temp;
  static std::string Find() {
    for (int i = 0; i < 16; ++i) {
      char p[96];
      std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/name", i);
      if (FILE* f = std::fopen(p, "r")) {
        char n[32] = {};
        if (std::fgets(n, sizeof n, f) && std::strncmp(n, "amdgpu", 6) == 0) {
          std::fclose(f);
          std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/", i);
          return p;
        }
        std::fclose(f);
      }
    }
    return {};
  }
  static double Read(const std::string& path) {
    if (path.empty())
      return -1;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
      return -1;
    double v = -1;
    if (std::fscanf(f, "%lf", &v) != 1)
      v = -1;
    std::fclose(f);
    return v;
  }
  void Init() {
    const auto d = Find();
    if (d.empty())
      return;
    freq = d + "freq1_input";
    power = d + "power1_average";
    temp = d + "temp1_input";
  }
  double MHz() const { return Read(freq) / 1e6; }
  double Watts() const { return Read(power) / 1e6; }
  double Celsius() const { return Read(temp) / 1e3; }
};

struct Device {
  int cus = 0;
  int wgps = 0;
  std::uint32_t* sink = nullptr;
  std::size_t sink_cap = 0;
  hipEvent_t ev0{}, ev1{};
  Sys sys;
};

static double g_deadline_s = 45.0;
static std::chrono::steady_clock::time_point g_t0;
static bool OutOfTime() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0)
             .count() > g_deadline_s;
}

// Calibrate reps so one launch is about target_ms.
template<typename Launch>
int Calibrate(Launch&& launch, hipEvent_t a, hipEvent_t b, double target_ms) {
  int reps = 64;
  for (int it = 0; it < 5; ++it) {
    launch(reps);
    HIP_OK(hipEventRecord(a));
    launch(reps);
    HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, a, b));
    if (ms > 0.8 * target_ms && ms < 1.25 * target_ms)
      break;
    reps = std::max(4, static_cast<int>(reps * target_ms / std::max(0.02F, ms)));
  }
  return reps;
}

struct Row {
  double tops;
  double mhz, watts, celsius;
  bool checked, ok;
};

// Expected sink hash for Peak<K, CH>: each lane folds 8*CH values in order.
template<Kind K, int CH>
std::uint32_t PeakExpect(int reps) {
  std::uint32_t h = kSeed;
  const double n = 16.0 * reps;
  for (int i = 0; i < CH; ++i) {
    for (int j = 0; j < 8; ++j) {
      const double v = i * 8 + j + n * Ops<K>::kStep;
      if constexpr (IsInt(K))
        h = Fold(h, Word(static_cast<float>(static_cast<std::int32_t>(v))));
      else
        h = Fold(h, Word(static_cast<float>(v)));
    }
  }
  return h;
}

// Times `launch` (3 launches back to back), samples sysfs right after.
template<typename Launch>
Row Time(Device& d, int blocks, double wmma_per_wave_rep, int reps,
         Launch&& launch) {
  HIP_OK(hipEventRecord(d.ev0));
  for (int i = 0; i < 3; ++i)
    launch(reps);
  HIP_OK(hipEventRecord(d.ev1));
  HIP_OK(hipEventSynchronize(d.ev1));
  float ms = 0;
  HIP_OK(hipEventElapsedTime(&ms, d.ev0, d.ev1));
  ms /= 3;
  Row r{};
  r.mhz = d.sys.MHz();
  r.watts = d.sys.Watts();
  r.celsius = d.sys.Celsius();
  const double wmma = static_cast<double>(blocks) * 8 * wmma_per_wave_rep * reps;
  r.tops = wmma * 8192.0 / (ms * 1e-3) / 1e12;
  return r;
}

template<Kind K, int CH>
Row PeakRun(Device& d, int blocks_per_wgp, double target_ms) {
  const int blocks = d.wgps * blocks_per_wgp;
  const auto launch = [&](int reps) { Peak<K, CH><<<blocks, 256>>>(d.sink, reps); };
  const int reps = Calibrate(launch, d.ev0, d.ev1, target_ms);
  HIP_OK(hipMemset(d.sink, 0, d.sink_cap * 4));
  launch(reps);
  HIP_OK(hipDeviceSynchronize());
  Row r{};
  r.checked = Ops<K>::kStep != 0.0;
  r.ok = true;
  if (r.checked) {
    std::vector<std::uint32_t> got(static_cast<std::size_t>(blocks) * 256);
    HIP_OK(hipMemcpy(got.data(), d.sink, got.size() * 4, hipMemcpyDeviceToHost));
    const auto want = PeakExpect<K, CH>(reps);
    for (const auto v : got)
      r.ok = r.ok && v == want;
  }
  const auto t = Time(d, blocks, 16.0 * CH, reps, launch);
  r.tops = t.tops;
  r.mhz = t.mhz;
  r.watts = t.watts;
  r.celsius = t.celsius;
  return r;
}

static void Print(const char* label, const char* unit, const Row& r,
                  const Device& d, double peak_ops_per_clk_cu) {
  const double eff =
      r.mhz > 0 ? r.tops * 1e12 / (d.cus * r.mhz * 1e6) : 0;  // ops/clk/CU
  std::printf("  %-40s %8.2f T%s  %6.0f ops/clk/CU (%5.1f%% of %4.0f)  %4.0f MHz %5.1f W %3.0f C %s\n",
              label, r.tops, unit, eff, 100.0 * eff / peak_ops_per_clk_cu,
              peak_ops_per_clk_cu, r.mhz, r.watts, r.celsius,
              r.checked ? (r.ok ? "[verified]" : "[MISMATCH]") : "");
}

template<Kind K, int CH>
void PeakPrint(Device& d, int bpw, double target_ms, const char* extra = "") {
  if (OutOfTime())
    return;
  const auto r = PeakRun<K, CH>(d, bpw, target_ms);
  char label[96];
  std::snprintf(label, sizeof label, "%s ch=%d w/SIMD=%d%s", Name(K), CH, bpw * 8 / 4, extra);
  Print(label, IsInt(K) ? "OPS" : "FLOPS", r, d, K == Kind::kIu4 ? 1024 : 512);
}

template<Kind K, int TI, int TJ, int S, bool Direct>
Row MixRun(Device& d, int blocks_per_wgp, double target_ms) {
  const int blocks = d.wgps * blocks_per_wgp;
  const auto launch = [&](int reps) {
    Mix<K, TI, TJ, S, Direct><<<blocks, 256>>>(d.sink, reps, 0.0078125F);
  };
  const int reps = Calibrate(launch, d.ev0, d.ev1, target_ms);
  return Time(d, blocks, static_cast<double>(S) * TI * TJ, reps, launch);
}

template<Kind K, int TI, int TJ, int S, bool Direct>
void MixPrint(Device& d, double target_ms) {
  if (OutOfTime())
    return;
  const auto r = MixRun<K, TI, TJ, S, Direct>(d, 4, target_ms);
  char label[96];
  std::snprintf(label, sizeof label, "%s %dx%d tiles K=%d/group %s", Name(K), TI,
                TJ, 16 * S, Direct ? "direct   " : "i32->f32 ");
  Print(label, IsInt(K) ? "OPS" : "FLOPS", r, d, K == Kind::kIu4 ? 1024 : 512);
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "quick";
  g_t0 = std::chrono::steady_clock::now();
  if (quick)
    g_deadline_s = 20.0;
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  Device d;
  d.wgps = prop.multiProcessorCount;  // HIP counts WGPs on RDNA
  d.cus = d.wgps * 2;
  d.sys.Init();
  std::printf("device: %s arch=%s WGPs=%d CUs=%d max clock %.2f GHz; sysfs clock %s\n",
              prop.name, prop.gcnArchName, d.wgps, d.cus, prop.clockRate / 1e6,
              d.sys.freq.empty() ? "unavailable" : d.sys.freq.c_str());
  d.sink_cap = static_cast<std::size_t>(d.wgps) * 8 * 256;
  HIP_OK(hipMalloc(&d.sink, d.sink_cap * 4));
  HIP_OK(hipEventCreate(&d.ev0));
  HIP_OK(hipEventCreate(&d.ev1));
  const double target_ms = quick ? 25.0 : 60.0;

  std::printf("\n== 1. saturated issue: 4 independent chains, 8 waves/SIMD, interleaved rounds ==\n"
              "   ops/clk/CU: 512 = one 16x16x16 WMMA per 32 clk per SIMD; 1024 = per 16 clk\n");
  const int rounds = quick ? 1 : 3;
  for (int round = 0; round < rounds; ++round) {
    std::printf(" round %d\n", round + 1);
    PeakPrint<Kind::kF16, 4>(d, 4, target_ms);
    PeakPrint<Kind::kIu4, 4>(d, 4, target_ms);
    PeakPrint<Kind::kIu8, 4>(d, 4, target_ms);
    PeakPrint<Kind::kBf16, 4>(d, 4, target_ms);
    PeakPrint<Kind::kF16Acc, 4>(d, 4, target_ms);
  }
  if (!quick) {
    std::printf("\n== 1b. resident waves per SIMD (integer types and f16) ==\n");
    for (const int bpw : {1, 2, 8}) {
      PeakPrint<Kind::kF16, 4>(d, bpw, target_ms);
      PeakPrint<Kind::kIu8, 4>(d, bpw, target_ms);
      PeakPrint<Kind::kIu4, 4>(d, bpw, target_ms);
    }
    std::printf("\n== 2. independent chains per wave, 8 waves/SIMD ==\n");
    PeakPrint<Kind::kF16, 1>(d, 4, target_ms);
    PeakPrint<Kind::kF16, 2>(d, 4, target_ms);
    PeakPrint<Kind::kF16, 8>(d, 4, target_ms);
    PeakPrint<Kind::kIu8, 1>(d, 4, target_ms);
    PeakPrint<Kind::kIu8, 2>(d, 4, target_ms);
    PeakPrint<Kind::kIu8, 8>(d, 4, target_ms);
    PeakPrint<Kind::kIu4, 1>(d, 4, target_ms);
    PeakPrint<Kind::kIu4, 2>(d, 4, target_ms);
    PeakPrint<Kind::kIu4, 8>(d, 4, target_ms);
  }

  std::printf("\n== 3. scale-group epilogue mix (8 waves/SIMD): S WMMAs per tile, then facc += scale * float(acc) ==\n");
  MixPrint<Kind::kF16, 4, 2, 2, true>(d, target_ms);
  MixPrint<Kind::kIu8, 4, 2, 2, true>(d, target_ms);
  MixPrint<Kind::kIu4, 4, 2, 2, true>(d, target_ms);
  MixPrint<Kind::kIu8, 4, 2, 1, false>(d, target_ms);
  MixPrint<Kind::kIu8, 4, 2, 2, false>(d, target_ms);
  MixPrint<Kind::kIu8, 4, 2, 4, false>(d, target_ms);
  MixPrint<Kind::kIu8, 4, 2, 8, false>(d, target_ms);
  MixPrint<Kind::kIu4, 4, 2, 1, false>(d, target_ms);
  MixPrint<Kind::kIu4, 4, 2, 2, false>(d, target_ms);
  MixPrint<Kind::kIu4, 4, 2, 4, false>(d, target_ms);
  MixPrint<Kind::kIu4, 4, 2, 8, false>(d, target_ms);
  if (!quick) {
    // 2x2 tiles: fewer live registers (no spills), shorter dependent runs.
    MixPrint<Kind::kIu4, 2, 2, 2, false>(d, target_ms);
    MixPrint<Kind::kIu4, 2, 2, 4, false>(d, target_ms);
    MixPrint<Kind::kF16, 4, 2, 2, false>(d, target_ms);
  }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
  std::printf("\ntotal %.1f s%s\n", secs,
              OutOfTime() ? " (deadline reached; later rows skipped)" : "");
  HIP_OK(hipFree(d.sink));
  return 0;
}
