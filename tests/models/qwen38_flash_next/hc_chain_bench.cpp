// One hyper-connection sub-layer of wide prefill, as the executor runs it:
// combine (plain block output or the fused MoE epilogue) with the next
// mixer's norm, the W8A8 HC down projection, and the F16 up projection with
// the mix epilogue and inject partials. Times each kernel and the chain, and
// checks variants of the chain bit for bit against it. Random Q8_0 weights.
//   qwen38_flash_next_hc_chain_bench [--tokens 512,2048,8192] [--reps 20]
//       [--slices 256,512,1024]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kHidden = 2560, kStreams = 4, kRank = 320;
constexpr std::uint32_t kRow = kHidden * kStreams;
constexpr std::uint32_t kUsed = 10, kExperts = 512;
constexpr float kEps = 1e-6F;

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
struct Buf {
  explicit Buf(std::size_t n) : n(n) {
    Hip(hipMalloc(&p, n * sizeof(T) + 4096), "hipMalloc");
    Hip(hipMemset(p, 0, n * sizeof(T) + 4096), "hipMemset");
  }
  ~Buf() { (void)hipFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void Up(const std::vector<T>& h) {
    Hip(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  }
  std::vector<std::uint8_t> Bytes(std::size_t count) const {
    std::vector<std::uint8_t> h(count * sizeof(T));
    Hip(hipMemcpy(h.data(), p, h.size(), hipMemcpyDeviceToHost), "download");
    return h;
  }
  T* p{nullptr};
  std::size_t n;
};

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}
std::vector<float> Random(std::size_t n, std::uint32_t seed, float scale,
                          float offset = 0.0F) {
  std::vector<float> v(n);
  for (float& x : v)
    x = offset +
        scale * (static_cast<float>(Next(&seed) & 0xFFFF) / 32768.0F - 1.0F);
  return v;
}
std::vector<std::uint8_t> RandomQ8(std::size_t rows, std::size_t cols,
                                   std::uint32_t seed) {
  std::vector<std::uint8_t> w(rows * cols / 32 * 34);
  for (std::size_t b = 0; b < rows * cols / 32; ++b) {
    const __half d = __float2half(0.002F + 0.001F * (Next(&seed) % 8));
    std::memcpy(w.data() + b * 34, &d, 2);
    for (int i = 0; i < 32; ++i)
      w[b * 34 + 2 + i] = static_cast<std::uint8_t>(Next(&seed));
  }
  return w;
}

double TimeMs(const std::function<void()>& f, int reps) {
  hipEvent_t a, b;
  Hip(hipEventCreate(&a), "event");
  Hip(hipEventCreate(&b), "event");
  f();
  Hip(hipDeviceSynchronize(), "warm");
  std::vector<float> ms;
  for (int i = 0; i < reps; ++i) {
    Hip(hipEventRecord(a, nullptr), "record");
    f();
    Hip(hipEventRecord(b, nullptr), "record");
    Hip(hipEventSynchronize(b), "sync");
    float t = 0.0F;
    Hip(hipEventElapsedTime(&t, a, b), "elapsed");
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return ms[ms.size() / 2];
}

std::vector<std::uint32_t> ParseList(const std::string& s) {
  std::vector<std::uint32_t> v;
  std::size_t p = 0;
  while (p < s.size()) {
    const std::size_t e = s.find(',', p);
    v.push_back(static_cast<std::uint32_t>(std::stoul(s.substr(p, e - p))));
    if (e == std::string::npos)
      break;
    p = e + 1;
  }
  return v;
}

struct Chain {
  explicit Chain(std::uint32_t n)
      : n(n),
        parts(q::HcInjectPartsVec4(kHidden)),
        res0(std::size_t(n) * kRow),
        res(std::size_t(n) * kRow),
        block(std::size_t(n) * kHidden),
        inject_in(std::size_t(n) * kStreams * parts),
        inject_out(std::size_t(n) * kStreams * parts),
        gamma(kRow),
        inject_w(std::size_t(kStreams) * kRow),
        weights(std::size_t(n) * kUsed),
        shared(std::size_t(n) * kHidden),
        router(std::size_t(n) * (kExperts + 1)),
        mixed(std::size_t(n) * kHidden),
        experts(std::size_t(n) * kUsed * kHidden),
        xn(std::size_t(n) * kRow),
        low(std::size_t(n) * kRank),
        mixed_half(std::size_t(n) * kHidden),
        xn_q8(q::Q8TiledBytes(n, kRow)),
        mixed_q8(q::Q8TiledBytes(n, kHidden)),
        down(std::size_t(kRank) * kRow / 32 * 34),
        up(std::size_t(kRow) * kRank / 32 * 34),
        res_half(std::size_t(n) * kRow),
        xn_scale(std::size_t(n) * kStreams) {
    res0.Up(Random(res0.n, 1, 1.0F));
    block.Up(Random(block.n, 2, 1.0F));
    inject_in.Up(Random(inject_in.n, 3, 0.5F));
    gamma.Up(Random(gamma.n, 4, 0.5F, 1.0F));
    inject_w.Up(Random(inject_w.n, 5, 0.01F));
    weights.Up(Random(weights.n, 6, 0.1F, 0.1F));
    shared.Up(Random(shared.n, 7, 1.0F));
    router.Up(Random(router.n, 8, 2.0F));
    std::vector<__half> e(experts.n);
    auto ev = Random(experts.n, 9, 1.0F);
    for (std::size_t i = 0; i < e.size(); ++i)
      e[i] = __float2half(ev[i]);
    experts.Up(e);
    down.Up(RandomQ8(kRank, kRow, 10));
    up.Up(RandomQ8(kRow, kRank, 11));
  }
  void Reset() {
    Hip(hipMemcpy(res.p, res0.p, res.n * sizeof(float),
                  hipMemcpyDeviceToDevice),
        "reset");
  }
  /// Combine over tokens [t0, t0 + count) into xn/xn_q8 at the slice base
  /// (`local` keeps slice-local scratch), then down and mix.
  void Run(std::uint32_t t0, std::uint32_t count, bool moe, bool local,
           bool fuse = false) {
    const std::size_t x0 = local ? 0 : t0;
    __half* xs = xn.p + x0 * kRow;
    auto* xq = xn_q8.p + (x0 / 16) * (kRow / 32) * 576;
    float* r = res.p + std::size_t(t0) * kRow;
    const float* inj = inject_in.p + std::size_t(t0) * kStreams * parts;
    float* inj_out = inject_out.p + std::size_t(t0) * kStreams * parts;
    if (moe) {
      if (!q::HcCombineMoeF16(
              r, experts.p + std::size_t(t0) * kUsed * kHidden,
              weights.p + std::size_t(t0) * kUsed,
              shared.p + std::size_t(t0) * kHidden,
              router.p + std::size_t(t0) * (kExperts + 1) + kExperts,
              kExperts + 1, kUsed, inj, parts, gamma.p, xs, xq, count, kHidden,
              kStreams, kEps, nullptr, nullptr, fuse ? inject_w.p : nullptr,
              fuse ? inj_out : nullptr))
        throw std::runtime_error("moe combine refused");
    } else {
      q::HcCombineF16(r, block.p + std::size_t(t0) * kHidden, inj, parts,
                      gamma.p, xs, xq, count, kHidden, kStreams, kEps, nullptr,
                      fuse ? inject_w.p : nullptr, fuse ? inj_out : nullptr);
    }
    if (!q::HcDownF16Gemm(down.p, xq, low.p + std::size_t(t0) * kRank, count,
                          nullptr))
      throw std::runtime_error("hc down refused");
    auto* mq = mixed_q8.p + (t0 / 16) * (kHidden / 32) * 576;
    if (!q::HcMixF16Gemm(
            up.p, low.p + std::size_t(t0) * kRank, xs,
            fuse ? nullptr : inject_w.p, mixed.p + std::size_t(t0) * kHidden,
            mixed_half.p + std::size_t(t0) * kHidden, mq,
            fuse ? nullptr : inj_out, count, kHidden, kRank, nullptr))
      throw std::runtime_error("hc mix refused");
  }
  /// The whole sub-layer with fused inject partials. `mode` 0 is the F32
  /// residual; 1 and 2 keep it in `res_half` (`nx` also drops the F16 norm).
  void RunMode(int mode, bool nx, bool moe) {
    q::HcResArgs ra;
    ra.mode = mode;
    ra.res_half = res_half.p;
    ra.xn_scale = nx ? xn_scale.p : nullptr;
    __half* xs = nx ? nullptr : xn.p;
    if (moe) {
      if (!q::HcCombineMoeF16(res.p, experts.p, weights.p, shared.p,
                              router.p + kExperts, kExperts + 1, kUsed,
                              inject_in.p, parts, gamma.p, xs, xn_q8.p, n,
                              kHidden, kStreams, kEps, nullptr, nullptr,
                              inject_w.p, inject_out.p, ra))
        throw std::runtime_error("moe combine refused");
    } else {
      q::HcCombineF16(res.p, block.p, inject_in.p, parts, gamma.p, xs, xn_q8.p,
                      n, kHidden, kStreams, kEps, nullptr, inject_w.p,
                      inject_out.p, ra);
    }
    if (!q::HcDownF16Gemm(down.p, xn_q8.p, low.p, n, nullptr))
      throw std::runtime_error("hc down refused");
    if (!q::HcMixF16Gemm(up.p, low.p, nx ? res_half.p : xn.p, nullptr, mixed.p,
                         mixed_half.p, mixed_q8.p, nullptr, n, kHidden, kRank,
                         nullptr, nx ? xn_scale.p : nullptr,
                         nx ? gamma.p : nullptr))
      throw std::runtime_error("hc mix refused");
  }
  std::vector<std::uint8_t> Snapshot() const {
    std::vector<std::uint8_t> all;
    const auto add = [&](std::vector<std::uint8_t> v) {
      all.insert(all.end(), v.begin(), v.end());
    };
    add(res.Bytes(res.n));
    add(mixed.Bytes(mixed.n));
    add(mixed_half.Bytes(mixed_half.n));
    add(mixed_q8.Bytes(q::Q8TiledBytes(n, kHidden) / 576 * 576));
    add(inject_out.Bytes(inject_out.n));
    add(low.Bytes(low.n));
    return all;
  }
  std::uint32_t n, parts;
  Buf<float> res0, res, block, inject_in, inject_out, gamma, inject_w, weights,
      shared, router, mixed;
  Buf<__half> experts, xn, low, mixed_half;
  Buf<std::uint8_t> xn_q8, mixed_q8, down, up;
  Buf<__half> res_half;
  Buf<float> xn_scale;
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::uint32_t> sizes{512, 2048, 8192};
  std::vector<std::uint32_t> slices{256, 512, 1024};
  int reps = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const std::string next = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--tokens") {
      sizes = ParseList(next);
      ++i;
    } else if (arg == "--slices") {
      slices = ParseList(next);
      ++i;
    } else if (arg == "--reps") {
      reps = std::stoi(next);
      ++i;
    }
  }
  int failures = 0;
  try {
    for (const std::uint32_t n : sizes) {
      Chain c(n);
      for (const bool moe : {false, true}) {
        const char* kind = moe ? "moe" : "plain";
        c.Reset();
        c.Run(0, n, moe, false);
        Hip(hipDeviceSynchronize(), "run");
        const auto golden = c.Snapshot();
        std::cout << std::fixed << std::setprecision(3);
        // Per-kernel timings over the whole batch.
        const double t_comb = TimeMs(
            [&] {
              if (moe)
                (void)q::HcCombineMoeF16(c.res.p, c.experts.p, c.weights.p,
                                         c.shared.p, c.router.p + kExperts,
                                         kExperts + 1, kUsed, c.inject_in.p,
                                         c.parts, c.gamma.p, c.xn.p, c.xn_q8.p,
                                         n, kHidden, kStreams, kEps, nullptr);
              else
                q::HcCombineF16(c.res.p, c.block.p, c.inject_in.p, c.parts,
                                c.gamma.p, c.xn.p, c.xn_q8.p, n, kHidden,
                                kStreams, kEps, nullptr);
            },
            reps);
        const double t_down = TimeMs(
            [&] {
              (void)q::HcDownF16Gemm(c.down.p, c.xn_q8.p, c.low.p, n, nullptr);
            },
            reps);
        const double t_mix = TimeMs(
            [&] {
              (void)q::HcMixF16Gemm(c.up.p, c.low.p, c.xn.p, nullptr, c.mixed.p,
                                    c.mixed_half.p, c.mixed_q8.p, nullptr, n,
                                    kHidden, kRank, nullptr);
            },
            reps);
        const double t_mixinj = TimeMs(
            [&] {
              (void)q::HcMixF16Gemm(c.up.p, c.low.p, c.xn.p, c.inject_w.p,
                                    c.mixed.p, c.mixed_half.p, c.mixed_q8.p,
                                    c.inject_out.p, n, kHidden, kRank, nullptr);
            },
            reps);
        const double t_chain = TimeMs([&] { c.Run(0, n, moe, false); }, reps);
        // Inject partials fused into the combine.
        c.Reset();
        Hip(hipMemset(c.inject_out.p, 0, c.inject_out.n * 4), "clear");
        c.Run(0, n, moe, false, true);
        Hip(hipDeviceSynchronize(), "fused");
        const bool fused_same = c.Snapshot() == golden;
        // Which output differs, and by how much (inject partials are summed
        // per stream before the sigmoid weighting of the next combine).
        {
          const auto inj = c.inject_out.Bytes(c.inject_out.n);
          Chain& g = c;
          (void)g;
          std::vector<float> fused(c.inject_out.n), ref(c.inject_out.n);
          std::memcpy(fused.data(), inj.data(), inj.size());
          c.Reset();
          c.Run(0, n, moe, false, false);
          Hip(hipDeviceSynchronize(), "ref");
          const auto rb = c.inject_out.Bytes(c.inject_out.n);
          std::memcpy(ref.data(), rb.data(), rb.size());
          double worst = 0.0, scale = 0.0;
          std::size_t diff = 0;
          for (std::size_t i = 0; i < ref.size(); ++i) {
            scale = std::max(scale, double(std::abs(ref[i])));
            if (std::memcmp(&ref[i], &fused[i], 4) != 0) {
              ++diff;
              worst = std::max(worst, double(std::abs(ref[i] - fused[i])));
            }
          }
          std::cout << "  inject partials differing " << diff << "/"
                    << ref.size() << " worst abs " << worst
                    << " (max |partial| " << scale << ")\n";
        }
        if (!fused_same)
          ++failures;
        const double t_fused =
            TimeMs([&] { c.Run(0, n, moe, false, true); }, reps);
        const double t_comb_inj = TimeMs(
            [&] {
              if (moe)
                (void)q::HcCombineMoeF16(
                    c.res.p, c.experts.p, c.weights.p, c.shared.p,
                    c.router.p + kExperts, kExperts + 1, kUsed, c.inject_in.p,
                    c.parts, c.gamma.p, c.xn.p, c.xn_q8.p, n, kHidden, kStreams,
                    kEps, nullptr, nullptr, c.inject_w.p, c.inject_out.p);
              else
                q::HcCombineF16(c.res.p, c.block.p, c.inject_in.p, c.parts,
                                c.gamma.p, c.xn.p, c.xn_q8.p, n, kHidden,
                                kStreams, kEps, nullptr, c.inject_w.p,
                                c.inject_out.p);
            },
            reps);
        const double comb_bytes =
            n *
            (kRow * 8.0 + kRow * 3.125 +
             (moe ? (kUsed * kHidden * 2.0 + kHidden * 4.0) : kHidden * 4.0));
        std::cout << "n=" << n << ' ' << kind << " | combine " << t_comb << " ("
                  << std::setprecision(0) << comb_bytes / (t_comb * 1e6)
                  << " GB/s) " << std::setprecision(3) << "| down " << t_down
                  << " | mix " << t_mix << " | mix+inject " << t_mixinj
                  << " | chain " << t_chain << '\n'
                  << "  inject fused into combine: combine " << t_comb_inj
                  << " | chain " << t_fused << " | "
                  << (fused_same ? "bit-identical" : "MISMATCH") << '\n';
        for (const std::uint32_t s : slices) {
          if (s >= n || s < 128 || s % 128 != 0)
            continue;
          const auto sliced = [&](bool local) {
            for (std::uint32_t t0 = 0; t0 < n;) {
              std::uint32_t count = std::min(s, n - t0);
              if (n - t0 - count < 128)
                count = n - t0;  // fold a short tail into this slice
              c.Run(t0, count, moe, local);
              t0 += count;
            }
          };
          for (const bool local : {false, true}) {
            c.Reset();
            sliced(local);
            Hip(hipDeviceSynchronize(), "sliced");
            const bool same = c.Snapshot() == golden;
            if (!same)
              ++failures;
            const double t = TimeMs([&] { sliced(local); }, reps);
            std::cout << "  slices " << s
                      << (local ? " local scratch" : " in place") << " | chain "
                      << t << " | " << (same ? "bit-identical" : "MISMATCH")
                      << '\n';
          }
        }
      }
    }
    // F16 residual arms (GUFO_HC_STREAMS): the two-combine sequence the
    // executor runs (mode 1 reads F32, mode 2 steady state), against the F32
    // route, with the same inject fusion.
    for (const std::uint32_t n : sizes) {
      Chain c(n);
      for (const bool moe : {false, true}) {
        std::vector<float> ref;
        for (const int arm : {0, 1, 2}) {
          const int mode1 = arm == 0 ? 0 : 1;
          const int mode2 = arm == 0 ? 0 : 2;
          const bool nx = arm == 2;
          c.Reset();
          Hip(hipMemset(c.res_half.p, 0, c.res_half.n * 2), "clear");
          c.RunMode(mode1, nx, moe);
          c.RunMode(mode2, nx, moe);
          Hip(hipDeviceSynchronize(), "arm");
          const auto bytes = c.mixed.Bytes(c.mixed.n);
          std::vector<float> out(c.mixed.n);
          std::memcpy(out.data(), bytes.data(), bytes.size());
          double err = 0.0, norm = 0.0, worst = 0.0;
          if (arm == 0) {
            ref = out;
          } else {
            for (std::size_t i = 0; i < out.size(); ++i) {
              const double d = out[i] - ref[i];
              err += d * d;
              norm += double(ref[i]) * ref[i];
              worst = std::max(worst, std::abs(d));
            }
          }
          Hip(hipMemset(c.res_half.p, 0, c.res_half.n * 2), "clear");
          const double t1 = TimeMs([&] { c.RunMode(mode1, nx, moe); }, reps);
          const double t2 = TimeMs([&] { c.RunMode(mode2, nx, moe); }, reps);
          const char* name = arm == 0   ? "f32   "
                             : arm == 1 ? "f16   "
                                        : "f16nx ";
          std::cout << "n=" << n << ' ' << (moe ? "moe  " : "plain") << " arm "
                    << name << "| sub-layer (combine+down+mix) first " << t1
                    << " ms, steady " << t2 << " ms";
          if (arm != 0)
            std::cout << " | mixed rel RMS err " << std::scientific
                      << std::sqrt(err / norm) << " worst abs " << worst
                      << std::fixed << std::setprecision(3);
          std::cout << '\n';
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
