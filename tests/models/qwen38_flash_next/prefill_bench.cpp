// Prefill-at-depth timings of the per-layer non-GEMM stages: indexer
// pooling, block scoring/selection and sparse WMMA attention (attention
// layers), and the chunked GDN recurrence with its convolution/history
// setup (linear-attention layers). Shapes are the model's; caches are sized
// for a 131072-token context so they exceed the 32 MiB MALL. Selections
// come from the real selector over random keys; `--overlap A` mixes a
// shared query into every row (0: independent rows, the attention worst
// case; real text lies in between). Run under rocprofv3 --stats for the
// per-kernel split.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
using gufo::models::qwen38_flash_next::KvCacheFormat;
namespace {

constexpr std::uint32_t kContext = 131072 + 2048;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kIdxHeads = 4, kIdxDim = 128, kBudget = 512;
constexpr std::uint32_t kHeads = 24, kKvHeads = 2, kDim = 256;
constexpr std::uint32_t kQWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;
constexpr std::uint32_t kKHeads = 16, kVHeads = 48, kGdn = 128;
constexpr std::uint32_t kChannels = (2 * kKHeads + kVHeads) * kGdn;
constexpr std::uint32_t kQkvz = kChannels + kVHeads * kGdn;
std::uint32_t kSelectChunk = 256;  // --select-chunk; as the executor

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}

template<typename T>
struct Buf {
  explicit Buf(std::size_t n) : n(n) {
    CheckHip(hipMalloc(&p, n * sizeof(T) + 4096), "hipMalloc");
    CheckHip(hipMemset(p, 0, n * sizeof(T) + 4096), "hipMemset");
  }
  ~Buf() { (void)hipFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  T* get() const { return p; }
  void Upload(const std::vector<T>& h) {
    CheckHip(
        hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
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
float Uniform(std::uint32_t* s) {
  return static_cast<float>(static_cast<int>(Next(s) & 0xFFFFU) - 32768) /
         32768.0F;
}
std::vector<float> Random(std::size_t n, std::uint32_t seed, float scale) {
  std::vector<float> v(n);
  for (float& x : v)
    x = scale * Uniform(&seed);
  return v;
}

/// FNV-1a over device bytes: identical hashes across builds mean
/// bit-identical outputs.
std::uint64_t Hash(const void* device, std::size_t bytes) {
  std::vector<unsigned char> h(bytes);
  CheckHip(hipMemcpy(h.data(), device, bytes, hipMemcpyDeviceToHost), "hash");
  std::uint64_t x = 1469598103934665603ULL;
  for (const unsigned char c : h)
    x = (x ^ c) * 1099511628211ULL;
  return x;
}

template<typename Run>
double Ms(Run&& run, int reps) {
  run();
  CheckHip(hipDeviceSynchronize(), "warm-up");
  hipEvent_t a, b;
  CheckHip(hipEventCreate(&a), "event");
  CheckHip(hipEventCreate(&b), "event");
  CheckHip(hipEventRecord(a, nullptr), "event");
  for (int i = 0; i < reps; ++i)
    run();
  CheckHip(hipEventRecord(b, nullptr), "event");
  CheckHip(hipEventSynchronize(b), "event");
  float ms = 0.0F;
  CheckHip(hipEventElapsedTime(&ms, a, b), "event");
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return ms / static_cast<float>(reps);
}

void BenchAttention(const std::vector<std::uint32_t>& depths,
                    const std::vector<std::uint32_t>& chunks, float overlap,
                    int reps) {
  const std::uint32_t max_blocks = (kContext / kRatio + 31) / 32 * 32;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::uint32_t max_n = *std::max_element(chunks.begin(), chunks.end());
  const std::uint32_t capacity = std::bit_ceil(2048U + max_n);
  Buf<__half> k(std::size_t(kContext) * kKvWidth),
      v(std::size_t(kContext) * kKvWidth),
      blocks(std::size_t(max_blocks) * kIdxDim);
  Buf<std::uint8_t> tk(std::size_t(kContext) * kKvHeads *
                       q::KvCodeBytes(KvCacheFormat::kTq4)),
      tv(std::size_t(kContext) * kKvHeads *
         q::KvCodeBytes(KvCacheFormat::kTq4));
  Buf<__half> tks(std::size_t(kContext) * kKvHeads),
      tvs(std::size_t(kContext) * kKvHeads);
  {
    // Random rows: timing does not depend on cache content.
    constexpr std::uint32_t kStep = 8192;
    Buf<float> rows(std::size_t(kStep) * kKvWidth);
    rows.Upload(Random(std::size_t(kStep) * kKvWidth, 3, 1.0F));
    Buf<std::uint32_t> pos(1);
    for (std::uint32_t p = 0; p < kContext; p += kStep) {
      pos.Upload({p});
      const auto n = std::min(kStep, kContext - p);
      q::StoreKv(rows.get(), k.get(), n, kKvWidth, pos.get(), nullptr);
      q::StoreKv(rows.get(), v.get(), n, kKvWidth, pos.get(), nullptr);
      q::QuantizeKv(rows.get(), rows.get(),
                    q::KvView{KvCacheFormat::kTq4, tk.get(), tv.get(),
                              tks.get(), tvs.get()},
                    n, kKvHeads, pos.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "caches");
    }
    std::vector<float> keys =
        Random(std::size_t(max_blocks) * kIdxDim, 7, 1.0F);
    std::vector<__half> hk(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
      hk[i] = __float2half(keys[i]);
    blocks.Upload(hk);
  }
  Buf<float> iq(std::size_t(max_n) * kIdxHeads * kIdxDim),
      raw(std::size_t(capacity) * kIdxDim), gamma(kIdxDim),
      scores(std::size_t(kSelectChunk) * max_blocks),
      qa(std::size_t(max_n) * kQWidth), gate(std::size_t(max_n) * kQWidth),
      ctx(std::size_t(max_n) * kQWidth);
  Buf<std::uint32_t> mask(std::size_t(max_n) * mask_words), pos(1), first(1);
  {
    std::uint32_t s = 11;
    std::vector<float> shared(kIdxHeads * kIdxDim), rows(iq.n);
    for (float& x : shared)
      x = Uniform(&s);
    for (std::size_t i = 0; i < rows.size(); ++i)
      rows[i] =
          overlap * shared[i % shared.size()] + (1 - overlap) * Uniform(&s);
    iq.Upload(rows);
    qa.Upload(Random(qa.n, 5, 1.0F));
    gamma.Upload(std::vector<float>(kIdxDim, 1.0F));
  }
  const q::KvView f16{KvCacheFormat::kF16, k.get(), v.get()};
  const q::KvView tq4{KvCacheFormat::kTq4, tk.get(), tv.get(), tks.get(),
                      tvs.get()};
  std::cout << "attention layer (overlap " << overlap
            << "), ms per call: depth n | pool select wmma_f16 wmma_tq4 | "
               "union/4q\n";
  for (const std::uint32_t depth : depths) {
    for (const std::uint32_t n : chunks) {
      pos.Upload({depth});
      first.Upload({depth / kRatio});
      const bool sparse = depth + n > 2048;
      const double pool = Ms(
          [&] {
            q::PoolIndexerBlocks(raw.get(), gamma.get(), blocks.get(),
                                 first.get(), pos.get(), n, n / kRatio + 1,
                                 kRatio, kIdxDim, 64, 1e7F, 1e-6F, capacity,
                                 nullptr);
          },
          reps);
      const auto select = [&] {
        for (std::uint32_t t0 = 0; t0 < n; t0 += kSelectChunk)
          q::SelectBlocks(
              iq.get() + std::size_t(t0) * kIdxHeads * kIdxDim, blocks.get(),
              mask.get() + std::size_t(t0) * mask_words, scores.get(),
              std::min(kSelectChunk, n - t0), pos.get(), t0, kIdxHeads, kIdxDim,
              kRatio, kBudget, mask_words, max_blocks, nullptr);
      };
      const double sel = sparse ? Ms(select, reps) : 0.0;
      const std::uint32_t* m = sparse ? mask.get() : nullptr;
      const double wf = Ms(
          [&] {
            (void)q::WmmaCausalAttention(
                qa.get(), gate.get(), f16, m, mask_words, ctx.get(), n, depth,
                kHeads, kKvHeads, kDim, kRatio, nullptr);
          },
          reps);
      const std::uint64_t hf = Hash(ctx.get(), std::size_t(n) * kQWidth * 4);
      const double wq = Ms(
          [&] {
            q::RotateHeads(qa.get(), n * kHeads, nullptr);
            (void)q::WmmaCausalAttention(qa.get(), nullptr, tq4, m, mask_words,
                                         ctx.get(), n, depth, kHeads, kKvHeads,
                                         kDim, kRatio, nullptr);
            q::UnrotateHeads(ctx.get(), gate.get(), n * kHeads, nullptr);
          },
          reps);
      const std::uint64_t hq = Hash(ctx.get(), std::size_t(n) * kQWidth * 4);
      const std::uint64_t hm =
          sparse ? Hash(mask.get(), std::size_t(n) * mask_words * 4) : 0;
      double uni = 0;
      if (sparse) {
        std::vector<std::uint32_t> hm(std::size_t(n) * mask_words);
        CheckHip(hipMemcpy(hm.data(), mask.get(), hm.size() * 4,
                           hipMemcpyDeviceToHost),
                 "mask");
        for (std::uint32_t g = 0; g < n; g += 4) {
          for (std::uint32_t w = 0; w < mask_words; ++w) {
            std::uint32_t bits = 0;
            for (std::uint32_t r = g; r < std::min(g + 4, n); ++r)
              bits |= hm[std::size_t(r) * mask_words + w];
            uni += std::popcount(bits);
          }
        }
        uni /= (n + 3) / 4;
      }
      std::cout << std::fixed << std::setprecision(3) << "attn d=" << depth
                << " n=" << n << " | " << pool << ' ' << sel << ' ' << wf << ' '
                << wq << " | " << std::setprecision(0) << uni << std::hex
                << " | mask " << hm << " f16 " << hf << " tq4 " << hq
                << std::dec << '\n';
    }
  }
}

void BenchGdn(const std::vector<std::uint32_t>& chunks, int reps) {
  const std::uint32_t max_n = *std::max_element(chunks.begin(), chunks.end());
  Buf<float> qkvz(std::size_t(max_n) * kQkvz),
      ab(std::size_t(max_n) * 2 * kVHeads), conv_w(std::size_t(kChannels) * 4),
      a(kVHeads), dt(kVHeads), norm(kGdn), conv_state(3 * kChannels),
      scratch(std::size_t(max_n + 4) * kChannels),
      qn(std::size_t(max_n) * kKHeads * kGdn),
      kn(std::size_t(max_n) * kVHeads * kGdn),
      rawo(std::size_t(max_n) * kVHeads * kGdn),
      state(std::size_t(kVHeads) * kGdn * kGdn),
      out(std::size_t(max_n) * kVHeads * kGdn);
  Buf<__half> out_half(std::size_t(max_n) * kVHeads * kGdn);
  qkvz.Upload(Random(qkvz.n, 21, 1.0F));
  ab.Upload(Random(ab.n, 22, 1.0F));
  conv_w.Upload(Random(conv_w.n, 23, 0.5F));
  a.Upload(Random(a.n, 24, 1.0F));
  dt.Upload(Random(dt.n, 25, 1.0F));
  norm.Upload(std::vector<float>(kGdn, 1.0F));
  std::cout << "gdn layer, ms per call: n | conv+recurrence (unconvolved) | "
               "recurrence only (convolved)\n";
  for (const std::uint32_t n : chunks) {
    const auto run = [&](bool convolved) {
      q::GatedDeltaNet(
          qkvz.get(), kQkvz, qkvz.get() + kChannels, kQkvz, ab.get(),
          conv_w.get(), a.get(), dt.get(), norm.get(), conv_state.get(),
          scratch.get(), qn.get(), kn.get(), rawo.get(), state.get(), out.get(),
          nullptr, q::RollbackRows{}, q::RollbackRows{}, n, kKHeads, kVHeads,
          kGdn, 4, true, convolved, 1e-6F, nullptr, out_half.get());
    };
    const double full = Ms([&] { run(false); }, reps);
    const double rec = Ms([&] { run(true); }, reps);
    std::cout << std::fixed << std::setprecision(3) << "gdn n=" << n << " | "
              << full << " | " << rec << std::hex << " | out "
              << Hash(out_half.get(), std::size_t(n) * kVHeads * kGdn * 2)
              << " state " << Hash(state.get(), state.n * 4) << std::dec
              << '\n';
  }
}

/// The two hyper-connection combines of a layer (block output; MoE
/// epilogue fused) and the vec4 mixer epilogue, with their DRAM bytes.
void BenchHc(const std::vector<std::uint32_t>& chunks, int reps) {
  constexpr std::uint32_t kHidden = 2560, kStreams = 4, kUsed = 8;
  constexpr std::uint32_t kExperts = 512;
  const std::uint32_t max_n = *std::max_element(chunks.begin(), chunks.end());
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t row = std::size_t(kStreams) * kHidden;
  Buf<float> res(max_n * row), block(std::size_t(max_n) * kHidden),
      inject(std::size_t(max_n) * kStreams * parts), gamma(row),
      weights(std::size_t(max_n) * kUsed), shared(std::size_t(max_n) * kHidden),
      gate(std::size_t(max_n) * (kExperts + 1)),
      mixed(std::size_t(max_n) * kHidden), hc_gate(max_n * row), inject_w(row);
  Buf<__half> xn(max_n * row), experts(std::size_t(max_n) * kUsed * kHidden),
      mixed_half(std::size_t(max_n) * kHidden);
  Buf<std::uint8_t> xn_q8(q::Q8TiledBytes(max_n, row)),
      mixed_q8(q::Q8TiledBytes(max_n, kHidden));
  res.Upload(Random(res.n, 31, 1.0F));
  block.Upload(Random(block.n, 32, 1.0F));
  gamma.Upload(std::vector<float>(row, 1.0F));
  weights.Upload(Random(weights.n, 33, 0.1F));
  inject_w.Upload(Random(inject_w.n, 34, 0.01F));
  std::cout << "hc, ms per call (GB/s): n | combine | combine+moe | mix "
               "epilogue\n";
  for (const std::uint32_t n : chunks) {
    const double c = Ms(
        [&] {
          q::HcCombineF16(res.get(), block.get(), inject.get(), parts,
                          gamma.get(), xn.get(), xn_q8.get(), n, kHidden,
                          kStreams, 1e-6F, nullptr);
        },
        reps);
    const double m = Ms(
        [&] {
          (void)q::HcCombineMoeF16(res.get(), experts.get(), weights.get(),
                                   shared.get(), gate.get() + kExperts,
                                   kExperts + 1, kUsed, inject.get(), parts,
                                   gamma.get(), xn.get(), xn_q8.get(), n,
                                   kHidden, kStreams, 1e-6F, nullptr);
        },
        reps);
    const double e = Ms(
        [&] {
          q::HcMixEpilogueVec4F16(xn.get(), hc_gate.get(), inject_w.get(),
                                  mixed.get(), mixed_half.get(), mixed_q8.get(),
                                  inject.get(), n, kHidden, nullptr);
        },
        reps);
    // res read+write F32, norm F16 + Q8 out, block F32 or experts F16 in.
    const double base = n * (row * 4.0 * 2 + row * 2.0 + row * 1.125);
    const double gb_c = (base + n * kHidden * 4.0) / (c * 1e6);
    const double gb_m = (base + n * (kUsed + 2.0) * kHidden * 2) / (m * 1e6);
    // xn F16 + gate F32 in; mixed F32 + F16 + Q8 out.
    const double gb_e =
        n * (row * 2.0 + row * 4.0 + kHidden * 7.125) / (e * 1e6);
    std::cout << std::fixed << std::setprecision(3) << "hc n=" << n << " | "
              << c << " (" << std::setprecision(0) << gb_c << ") | "
              << std::setprecision(3) << m << " (" << std::setprecision(0)
              << gb_m << ") | " << std::setprecision(3) << e << " ("
              << std::setprecision(0) << gb_e << ")\n";
  }
}

std::vector<std::uint32_t> List(const char* s) {
  std::vector<std::uint32_t> v;
  for (std::string t(s); !t.empty();) {
    const auto comma = t.find(',');
    v.push_back(static_cast<std::uint32_t>(std::stoul(t.substr(0, comma))));
    t = comma == std::string::npos ? "" : t.substr(comma + 1);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::uint32_t> depths{0, 32768, 65536, 100000, 131072};
    std::vector<std::uint32_t> chunks{256, 512, 1024, 2048};
    float overlap = 0.0F;
    int reps = 5;
    bool attn = true, gdn = true, hc = true;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const char* next = i + 1 < argc ? argv[i + 1] : "";
      if (a == "--depths") {
        depths = List(next);
        ++i;
      } else if (a == "--chunks") {
        chunks = List(next);
        ++i;
      } else if (a == "--overlap") {
        overlap = std::stof(next);
        ++i;
      } else if (a == "--reps") {
        reps = std::stoi(next);
        ++i;
      } else if (a == "--select-chunk") {
        kSelectChunk = static_cast<std::uint32_t>(std::stoul(next));
        ++i;
      } else if (a == "--attention-only") {
        gdn = hc = false;
      } else if (a == "--gdn-only") {
        attn = hc = false;
      } else if (a == "--hc-only") {
        attn = gdn = false;
      } else {
        throw std::runtime_error("unknown option " + a);
      }
    }
    if (attn)
      BenchAttention(depths, chunks, overlap, reps);
    if (gdn)
      BenchGdn(chunks, reps);
    if (hc)
      BenchHc(chunks, reps);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
