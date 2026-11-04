// Validates the lossy-prefill emulation (kernels/rocm/spike.*) on real
// Flash-Next dense weights, outside the engine. For several dense tensors it
// registers the weight with the GUFO_SPIKE_* environment, feeds synthetic
// activations (unit normal plus a few outlier channels, like post-norm
// hidden states), and reports the relative RMS error of the emulated GEMM
// against the original Q8_0 GEMM. A rotation-mismatch control (rotated shadow
// weights, unrotated activations) must be ~100% wrong; it proves the rotation
// pairing is wired both ways.
//   GUFO_SPIKE_ACT=a4 GUFO_SPIKE_HAD=128 GUFO_SPIKE_WGROUP=64 \
//     qwen38_flash_next_spike_bench <first-target-shard.gguf> [rows]
// Device memory: the tensors, a shadow of each, and a few activation buffers
// (about 0.6 GiB at 256 rows).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/spike.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace spike = gufo::models::qwen38_flash_next::rocm::spike;
using gufo::models::qwen38_flash_next::TensorRef;

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

// Concatenates stacked row blocks (ssm qkv|gate, attention q|k|v).
q::DeviceTensor Upload(std::initializer_list<const TensorRef*> parts,
                       std::uint32_t rows, std::uint32_t cols) {
  q::DeviceTensor d;
  std::size_t bytes = 0;
  for (const auto* t : parts)
    bytes += t->SizeBytes();
  Check(hipMalloc(&d.data, bytes), "hipMalloc");
  std::size_t off = 0;
  for (const auto* t : parts) {
    Check(hipMemcpy(static_cast<std::uint8_t*>(d.data) + off, t->data,
                    t->SizeBytes(), hipMemcpyHostToDevice),
          "upload");
    off += t->SizeBytes();
  }
  d.type = gufo::core::GgmlType::kQ8_0;
  d.rows = rows;
  d.cols = cols;
  return d;
}

double TimeUs(hipStream_t stream, const std::function<void()>& fn) {
  hipEvent_t a, b;
  Check(hipEventCreate(&a), "event");
  Check(hipEventCreate(&b), "event");
  fn();
  fn();
  Check(hipStreamSynchronize(stream), "warm");
  std::vector<double> us;
  for (int rep = 0; rep < 5; ++rep) {
    Check(hipEventRecord(a, stream), "record");
    for (int i = 0; i < 8; ++i)
      fn();
    Check(hipEventRecord(b, stream), "record");
    Check(hipEventSynchronize(b), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, a, b), "elapsed");
    us.push_back(1000.0 * ms / 8);
  }
  std::sort(us.begin(), us.end());
  Check(hipEventDestroy(a), "event");
  Check(hipEventDestroy(b), "event");
  return us[2];
}

double RelRms(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - ref[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  return std::sqrt(num / (den + 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <first-target-shard.gguf> [rows]\n",
                 argv[0]);
    return 2;
  }
  const std::uint32_t rows = argc > 2 ? std::atoi(argv[2]) : 256;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto weights =
      gufo::models::qwen38_flash_next::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto& L = weights->layers;
  std::uint32_t linear = 0, attention = 0;
  for (std::uint32_t i = 0; i < L.size(); ++i) {
    if (L[i].linear() && linear == 0)
      linear = i;
    if (!L[i].linear() && attention == 0)
      attention = i;
  }
  hipStream_t stream;
  Check(hipStreamCreate(&stream), "stream");
  std::printf("spike enabled: %d\n", spike::Enabled() ? 1 : 0);

  struct Case {
    const char* name;
    q::DeviceTensor w;
    unsigned role;
  };
  std::vector<Case> cases;
  cases.push_back(
      {"ssm_in", Upload({&L[linear].ssm_qkv, &L[linear].ssm_gate}, 16384, 2560),
       spike::kSsmIn});
  cases.push_back({"attn_qkv",
                   Upload({&L[attention].attn_q, &L[attention].attn_k,
                           &L[attention].attn_v},
                          13312, 2560),
                   spike::kAttnQkv});
  cases.push_back(
      {"ssm_out", Upload({&L[linear].ssm_out}, 2560, 6144), spike::kSsmOut});
  cases.push_back({"attn_out", Upload({&L[attention].attn_out}, 2560, 6144),
                   spike::kAttnOut});
  cases.push_back({"shexp_gate", Upload({&L[linear].shexp_gate}, 640, 2560),
                   spike::kShared});
  cases.push_back({"shexp_down", Upload({&L[linear].shexp_down}, 2560, 640),
                   spike::kShared});
  for (auto& c : cases)
    spike::RegisterTensor(c.w, c.role, 0, true, stream);

  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  const bool real = std::getenv("GUFO_SPIKE_REAL") != nullptr;
  std::printf(
      "%-11s %6s %6s | relRMS vs original: spike emulated | integer | int vs "
      "emu | us: original  emulated  integer  speedup\n",
      "tensor", "m", "k");
  for (auto& c : cases) {
    const std::uint32_t m = c.w.rows, k = c.w.cols;
    std::vector<__half> hx(static_cast<std::size_t>(rows) * k);
    std::vector<int> outlier;
    for (int i = 0; i < 8; ++i)
      outlier.push_back(static_cast<int>(rng() % k));
    for (std::uint32_t r = 0; r < rows; ++r)
      for (std::uint32_t j = 0; j < k; ++j) {
        float v = normal(rng);
        for (int o : outlier)
          if (static_cast<int>(j) == o)
            v *= 20.0F;
        hx[static_cast<std::size_t>(r) * k + j] = __float2half(v);
      }
    __half* dx;
    float *y0, *y1, *y2;
    Check(hipMalloc(&dx, hx.size() * 2), "x");
    Check(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice), "x");
    const std::size_t ybytes = static_cast<std::size_t>(rows) * m * 4;
    Check(hipMalloc(&y0, ybytes), "y0");
    Check(hipMalloc(&y1, ybytes), "y1");
    Check(hipMalloc(&y2, ybytes), "y2");
    const auto original = [&] {
      (void)q::DenseF16Gemm(c.w.data, dx, y0, rows, m, k, stream);
    };
    const auto integer = [&] {
      (void)spike::DenseF16Gemm(true, c.w.data, dx, y1, rows, m, k, stream);
    };
    const auto emulated = [&] {
      (void)spike::DenseF16GemmEmulated(c.w.data, dx, y2, rows, m, k, stream);
    };
    original();
    integer();
    emulated();
    Check(hipStreamSynchronize(stream), "sync");
    std::vector<float> r0(static_cast<std::size_t>(rows) * m), r1(r0.size()),
        r2(r0.size());
    Check(hipMemcpy(r0.data(), y0, ybytes, hipMemcpyDeviceToHost), "r0");
    Check(hipMemcpy(r1.data(), y1, ybytes, hipMemcpyDeviceToHost), "r1");
    Check(hipMemcpy(r2.data(), y2, ybytes, hipMemcpyDeviceToHost), "r2");
    const double t0 = TimeUs(stream, original);
    const double t1 = real ? TimeUs(stream, integer) : 0.0;
    const double t2 = TimeUs(stream, emulated);
    std::printf(
        "%-11s %6u %6u | %8.4f %8.4f %8.4f | %9.0f %9.0f %9.0f  %5.2fx\n",
        c.name, m, k, RelRms(r2, r0), RelRms(r1, r0), RelRms(r1, r2), t0, t2,
        t1, real && t1 > 0 ? t0 / t1 : 0.0);
    (void)hipFree(dx);
    (void)hipFree(y0);
    (void)hipFree(y1);
    (void)hipFree(y2);
  }
  return 0;
}
