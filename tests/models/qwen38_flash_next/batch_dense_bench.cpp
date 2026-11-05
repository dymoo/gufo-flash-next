// Batched-decode dense projection crossover on real Flash-Next weights.
// For R token rows it times the decode GEMV chunks that DenseBatch used
// (Q8_1 activations, weights re-read per chunk) against the tiled W8A8 and
// F16 WMMA GEMMs (weights read once), and F16 SmallGemm against hipBLASLt.
// Every route is checked against an FP64 dot of the dequantized weights.
//   qwen38_flash_next_batch_dense_bench <first-target-shard.gguf> [prefill]
// `prefill` times the tiled routes alone (activations staged once) at
// 256..4096 rows, with TFLOPS and an FNV hash of the whole output.
// DENSE_BENCH_{SHAPE,ROWS,ROUTE,REPS,LAUNCHES,FLUSH} select one case for
// counter passes and interleaved A/B runs (details at the prefill loop).
// Uses about 1.8 GiB of device memory; four weight copies per shape rotate
// beyond the 32 MiB MALL.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
using gufo::models::qwen38_flash_next::TensorRef;

namespace {

constexpr int kCopies = 4;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

struct Shape {
  const char* name;
  std::uint32_t m, k;
  std::vector<void*> copies;       // device Q8_0 [m][k]
  std::vector<std::uint8_t> host;  // first copy, for the reference
};

// Concatenates stacked row blocks (ssm qkv|gate, attention q|k|v).
void* Upload(std::initializer_list<const TensorRef*> parts,
             std::vector<std::uint8_t>* host) {
  std::size_t bytes = 0;
  for (const auto* t : parts)
    bytes += t->SizeBytes();
  void* d = nullptr;
  Check(hipMalloc(&d, bytes), "hipMalloc weights");
  std::size_t off = 0;
  for (const auto* t : parts) {
    Check(hipMemcpy(static_cast<std::uint8_t*>(d) + off, t->data,
                    t->SizeBytes(), hipMemcpyHostToDevice),
          "upload");
    if (host != nullptr && host->size() < bytes) {
      host->resize(off);
      const auto* p = static_cast<const std::uint8_t*>(t->data);
      host->insert(host->end(), p, p + t->SizeBytes());
    }
    off += t->SizeBytes();
  }
  return d;
}

double Q8Dot(const std::uint8_t* row, const float* x, std::uint32_t k) {
  double sum = 0;
  for (std::uint32_t b = 0; b < k / 32; ++b) {
    const std::uint8_t* blk = row + b * 34;
    __half d;
    std::memcpy(&d, blk, 2);
    const double scale = __half2float(d);
    for (int i = 0; i < 32; ++i)
      sum += scale * static_cast<std::int8_t>(blk[2 + i]) * x[b * 32 + i];
  }
  return sum;
}

}  // namespace

int main(int argc, char** argv) {
  const bool prefill = argc == 3 && std::string(argv[2]) == "prefill";
  if (argc != 2 && !prefill) {
    std::fprintf(stderr, "usage: %s <first-target-shard.gguf> [prefill]\n",
                 argv[0]);
    return 2;
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);
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
  std::vector<std::uint32_t> linear, attention;
  for (std::uint32_t i = 0; i < L.size(); ++i)
    (L[i].linear() ? linear : attention).push_back(i);
  std::vector<Shape> shapes;
  const auto add =
      [&](const char* name, std::uint32_t m, std::uint32_t k,
          const std::function<void*(int, std::vector<std::uint8_t>*)>& make,
          int copies = kCopies) {
        Shape s{name, m, k, {}, {}};
        for (int c = 0; c < copies; ++c)
          s.copies.push_back(make(c, c == 0 ? &s.host : nullptr));
        shapes.push_back(std::move(s));
      };
  add("hc_down", 320, 10240,
      [&](int c, auto* h) { return Upload({&L[c].hc_attn.down}, h); });
  add("hc_up", 10240, 320,
      [&](int c, auto* h) { return Upload({&L[c].hc_attn.up}, h); });
  add("ssm_in", 16384, 2560, [&](int c, auto* h) {
    return Upload({&L[linear[c]].ssm_qkv, &L[linear[c]].ssm_gate}, h);
  });
  add("attn_qkv", 13312, 2560, [&](int c, auto* h) {
    const auto& l = L[attention[c]];
    return Upload({&l.attn_q, &l.attn_k, &l.attn_v}, h);
  });
  add("ssm_out", 2560, 6144,
      [&](int c, auto* h) { return Upload({&L[linear[c]].ssm_out}, h); });
  add("attn_out", 2560, 6144,
      [&](int c, auto* h) { return Upload({&L[attention[c]].attn_out}, h); });
  add("shexp_gate", 640, 2560,
      [&](int c, auto* h) { return Upload({&L[c].shexp_gate}, h); });
  add("shexp_down", 2560, 640,
      [&](int c, auto* h) { return Upload({&L[c].shexp_down}, h); });
  if (!prefill)
    add(
        "output", weights->config.vocab_size, 2560,
        [&](int, auto* h) { return Upload({&weights->output}, h); }, 1);

  hipStream_t stream;
  Check(hipStreamCreate(&stream), "stream");
  if (qfn_mmq_init(0) != 0)
    return 1;
  auto blaslt = q::BlasLt::Create(stream, &error);
  const std::uint32_t kMaxRows = prefill ? 4096 : 512;
  std::vector<float> x(std::size_t{kMaxRows} * 10240);
  std::uint32_t rng = 12345;
  for (auto& v : x) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    // Mostly unit-scale values with rare large channels, like HC activations.
    const float u = static_cast<float>(rng & 0xFFFF) / 32768.0F - 1.0F;
    v = (rng >> 28) == 0 ? 8.0F * u : u;
  }
  float *dx, *dout;
  void *xq8, *xq8t, *xh;
  Check(hipMalloc(&dx, x.size() * 4), "x");
  Check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  Check(
      hipMalloc(&dout, std::size_t{kMaxRows} *
                           (prefill ? 16384 : weights->config.vocab_size) * 4),
      "out");
  Check(hipMalloc(&xq8, qfn_mmq_q8_1_bytes(kMaxRows, 10240)), "xq8");
  Check(hipMalloc(&xq8t, q::Q8TiledBytes(kMaxRows, 10240)), "xq8t");
  Check(hipMalloc(&xh, std::size_t{kMaxRows} * 10240 * 2), "xh");

  // DenseBatch's previous GEMV policy (see batch.cpp history).
  const auto vec = [&](const void* w, std::uint32_t m, std::uint32_t k,
                       std::uint32_t rows) {
    const auto padded = (rows + 7) / 8 * 8;
    if (padded != rows)
      (void)hipMemsetAsync(
          static_cast<std::uint8_t*>(xq8) + qfn_mmq_q8_1_bytes(rows, k), 0,
          qfn_mmq_q8_1_bytes(padded - rows, k), stream);
    qfn_mmq_quantize_q8_1(dx, xq8, rows, k, stream);
    const auto project = [&](std::uint32_t first, std::uint32_t count) {
      qfn_mmq_q8_0_dense_vec_preq(
          w, nullptr,
          static_cast<const std::uint8_t*>(xq8) + qfn_mmq_q8_1_bytes(first, k),
          dout + std::size_t{first} * m, m, count, k, stream);
    };
    if (rows <= 8) {
      project(0, rows);
      return;
    }
    if (m == 320 && k == 10240) {
      const auto full = rows / 8 * 8;
      project(0, full);
      if (full != rows)
        project(full, rows - full);
      return;
    }
    const auto chunk =
        k == 2560 && m >= 8192 ? (rows > 32 && rows <= 48 ? 48U : 32U) : 8U;
    for (std::uint32_t r = 0; r < rows; r += chunk)
      project(r, std::min(chunk, rows - r));
  };
  const auto w8a8 = [&](const void* w, std::uint32_t m, std::uint32_t k,
                        std::uint32_t rows) {
    q::QuantizeQ8Tiled(dx, xq8t, rows, k, stream);
    q::W8A8Gemm(w, xq8t, dout, rows, m, k, stream);
  };
  const auto f16 = [&](const void* w, std::uint32_t m, std::uint32_t k,
                       std::uint32_t rows) {
    q::NarrowActivations(dx, xh, false, std::size_t{rows} * k, stream);
    q::DenseF16Gemm(w, static_cast<const __half*>(xh), dout, rows, m, k,
                    stream);
  };
  hipEvent_t start, end;
  Check(hipEventCreate(&start), "event");
  Check(hipEventCreate(&end), "event");
  if (prefill) {
    // Diagnostics, all optional. DENSE_BENCH_SHAPE=ssm_in selects one shape,
    // DENSE_BENCH_ROWS=2048 one row count (any value up to 4096, ragged ones
    // too), DENSE_BENCH_ROUTE=w8a8|f16|hcdown one route. DENSE_BENCH_REPS and
    // DENSE_BENCH_LAUNCHES set the timed repetitions (default 5) and launches
    // per repetition (default 8); DENSE_BENCH_FLUSH=1 writes 64 MiB before
    // every repetition so the MALL starts cold. A counter pass uses REPS=1
    // LAUNCHES=4 so each dispatch is counted once.
    const char* want_shape = std::getenv("DENSE_BENCH_SHAPE");
    const char* want_route = std::getenv("DENSE_BENCH_ROUTE");
    const auto env_uint = [](const char* name, unsigned fallback) {
      const char* v = std::getenv(name);
      return v != nullptr && *v != '\0'
                 ? static_cast<unsigned>(std::strtoul(v, nullptr, 10))
                 : fallback;
    };
    const unsigned want_rows = env_uint("DENSE_BENCH_ROWS", 0);
    std::vector<std::uint32_t> row_list{256U, 512U, 1024U, 2048U, 4096U};
    if (want_rows != 0)
      row_list = {std::min(want_rows, kMaxRows)};
    const unsigned reps = std::max(1U, env_uint("DENSE_BENCH_REPS", 5));
    const unsigned launches = std::max(1U, env_uint("DENSE_BENCH_LAUNCHES", 8));
    void* flush = nullptr;
    if (env_uint("DENSE_BENCH_FLUSH", 0) != 0)
      Check(hipMalloc(&flush, std::size_t{64} << 20), "flush");
    std::puts("shape,m,k,rows,route,us,TFLOPS,hash");
    for (auto& s : shapes) {
      if (want_shape != nullptr && std::string(want_shape) != s.name)
        continue;
      for (const std::uint32_t rows : row_list) {
        q::QuantizeQ8Tiled(dx, xq8t, rows, s.k, stream);
        q::NarrowActivations(dx, xh, false, std::size_t{rows} * s.k, stream);
        // hc_down also runs its production fused kernel (F16 SiLU output).
        const int routes = s.m == 320 && s.k == 10240 ? 3 : 2;
        for (int route = 0; route < routes; ++route) {
          const char* route_name = route == 0   ? "w8a8"
                                   : route == 1 ? "f16"
                                                : "hcdown";
          if (want_route != nullptr && std::string(want_route) != route_name)
            continue;
          const auto run = [&](int i) {
            const void* w = s.copies[i % s.copies.size()];
            if (route == 2)
              q::HcDownF16Gemm(w, xq8t, reinterpret_cast<__half*>(dout), rows,
                               stream);
            else if (route == 0)
              q::W8A8Gemm(w, xq8t, dout, rows, s.m, s.k, stream);
            else
              q::DenseF16Gemm(w, static_cast<const __half*>(xh), dout, rows,
                              s.m, s.k, stream);
          };
          run(0);
          std::vector<float> y(std::size_t{rows} * s.m / (route == 2 ? 2 : 1));
          Check(hipMemcpy(y.data(), dout, y.size() * 4, hipMemcpyDeviceToHost),
                "out");
          std::uint64_t hash = 1469598103934665603ull;
          for (const float v : y) {
            std::uint32_t u;
            std::memcpy(&u, &v, 4);
            hash = (hash ^ u) * 1099511628211ull;
          }
          for (int i = 1; i < 3; ++i)
            run(i);
          std::vector<float> us;
          for (unsigned rep = 0; rep < reps; ++rep) {
            if (flush != nullptr)
              Check(hipMemsetAsync(flush, static_cast<int>(rep), 64 << 20,
                                   stream),
                    "flush");
            Check(hipEventRecord(start, stream), "record");
            for (unsigned i = 0; i < launches; ++i)
              run(static_cast<int>(i));
            Check(hipEventRecord(end, stream), "record");
            Check(hipEventSynchronize(end), "sync");
            float ms = 0;
            Check(hipEventElapsedTime(&ms, start, end), "elapsed");
            us.push_back(1000.0F * ms / static_cast<float>(launches));
          }
          std::sort(us.begin(), us.end());
          const float median = us[us.size() / 2];
          std::printf("%s,%u,%u,%u,%s,%.1f,%.1f,%016llx\n", s.name, s.m, s.k,
                      rows, route_name, median,
                      2.0 * s.m * s.k * rows / median / 1e6,
                      static_cast<unsigned long long>(hash));
        }
      }
    }
    return 0;
  }
  std::puts("shape,m,k,rows,route,us,weight_GBps,max_err_over_rms");
  const std::uint32_t row_list[] = {8,  16,  24,  32,  40,  48, 64,
                                    96, 128, 192, 256, 384, 512};
  std::vector<float> out;
  for (auto& s : shapes) {
    const double bytes = double(s.m) * (s.k / 32) * 34;
    for (const auto rows : row_list) {
      struct Route {
        const char* name;
        std::function<void(const void*, std::uint32_t, std::uint32_t,
                           std::uint32_t)>
            run;
      };
      const Route routes[] = {{"vec", vec}, {"w8a8", w8a8}, {"f16", f16}};
      for (const auto& route : routes) {
        route.run(s.copies[0], s.m, s.k, rows);
        Check(hipStreamSynchronize(stream), route.name);
        // Accuracy against FP64 dots of the dequantized weights.
        out.resize(std::size_t{rows} * s.m);
        Check(
            hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost),
            "out");
        double max_err = 0, sum_sq = 0;
        const std::size_t row_bytes = std::size_t{s.k / 32} * 34;
        for (int i = 0; i < 128; ++i) {
          const std::uint32_t r = (i * 37) % rows;
          const std::uint32_t mm = (i * 7919U + 13) % s.m;
          const double ref = Q8Dot(s.host.data() + mm * row_bytes,
                                   x.data() + std::size_t{r} * s.k, s.k);
          sum_sq += ref * ref;
          max_err =
              std::max(max_err, std::abs(ref - out[std::size_t{r} * s.m + mm]));
        }
        const double rms = std::sqrt(sum_sq / 128);
        const int iters = s.copies.size() == 1 ? 3 : 12;
        for (int i = 0; i < 2; ++i)
          route.run(s.copies[i % s.copies.size()], s.m, s.k, rows);
        Check(hipEventRecord(start, stream), "record");
        for (int i = 0; i < iters; ++i)
          route.run(s.copies[i % s.copies.size()], s.m, s.k, rows);
        Check(hipEventRecord(end, stream), "record");
        Check(hipEventSynchronize(end), "sync");
        float ms = 0;
        Check(hipEventElapsedTime(&ms, start, end), "elapsed");
        const double us = 1000.0 * ms / iters;
        std::printf("%s,%u,%u,%u,%s,%.1f,%.1f,%.2e\n", s.name, s.m, s.k, rows,
                    route.name, us, bytes / us / 1000.0, max_err / rms);
      }
    }
  }
  // F16 small projections: grouped SmallGemm rows versus hipBLASLt.
  for (const auto [name, m] :
       {std::pair{"router", 513U}, std::pair{"alpha_beta", 96U},
        std::pair{"indexer_q", 512U}}) {
    std::vector<__half> hw(std::size_t{m} * 2560);
    for (auto& v : hw) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      v = __float2half(static_cast<float>(rng & 0xFFFF) / 65536.0F - 0.5F);
    }
    void* w[kCopies];
    for (auto& p : w) {
      Check(hipMalloc(&p, hw.size() * 2), "w16");
      Check(hipMemcpy(p, hw.data(), hw.size() * 2, hipMemcpyHostToDevice),
            "w16");
    }
    for (const auto rows : row_list) {
      for (int route = 0; route < 2; ++route) {
        const auto run = [&](int i) {
          if (route == 0) {
            q::SmallGemm(w[i % kCopies], q::WeightType::kF16, dx, dout, rows, m,
                         2560, stream);
          } else {
            q::NarrowActivations(dx, xh, false, std::size_t{rows} * 2560,
                                 stream);
            (void)blaslt->Gemm(w[i % kCopies], xh, dout, HIP_R_16F, m, rows,
                               2560, &error);
          }
        };
        for (int i = 0; i < 2; ++i)
          run(i);
        Check(hipEventRecord(start, stream), "record");
        for (int i = 0; i < 12; ++i)
          run(i);
        Check(hipEventRecord(end, stream), "record");
        Check(hipEventSynchronize(end), "sync");
        float ms = 0;
        Check(hipEventElapsedTime(&ms, start, end), "elapsed");
        std::printf("%s,%u,2560,%u,%s,%.1f,,\n", name, m, rows,
                    route == 0 ? "small" : "blaslt", 1000.0 * ms / 12);
      }
    }
  }
  return 0;
}
