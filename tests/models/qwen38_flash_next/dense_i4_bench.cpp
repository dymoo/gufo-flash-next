// W4A4 dense prefill route (kernels/rocm/dense_i4.*) on real Flash-Next
// weights against the production routes, per projection and prompt rows.
//   qwen38_flash_next_dense_i4_bench <first-target-shard.gguf> [rows...]
// For every dense prefill shape it packs four layers' Q8_0 weights (rotating
// beyond the 32 MiB MALL) with both rotations and reports, as CSV:
//   route  f16 (DenseF16Gemm), w8a8 (W8A8Gemm), i4-block / i4-full (int4
//          GEMM), and the activation staging of each (narrow, q8tiled,
//          quant-block, quant-full: F32 rows in);
//   us     median of five 8-call rounds; TOPS = 2 m k rows / time;
//   err    relative RMS error of 256 sampled outputs against an FP64 dot
//          of the Q8_0 weights and the F32 activations (the production
//          contract's reference);
//   oracle int4 GEMM outputs that differ from a CPU recomputation of the
//          same codes and scales (exact integer sum, then the kernel's two
//          float products); must be 0.
// The fused SSM (conv) and attention epilogues are checked against the
// generic int4 GEMM: Z rows and attention outputs must be bit-identical,
// convolved rows match a CPU convolution of the GEMM rows.
// Synthetic activations: uniform values with eight 20x outlier channels,
// like post-norm hidden states. Device memory about 2.5 GiB.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dense_i4.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace i4 = gufo::models::qwen38_flash_next::rocm::dense_i4;
using gufo::models::qwen38_flash_next::TensorRef;

namespace {

constexpr int kCopies = 4;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

void Require(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "failed: %s\n", what);
    std::exit(1);
  }
}

struct Shape {
  const char* name;
  std::uint32_t m, k;
  std::vector<void*> copies;       // device Q8_0 [m][k]
  std::vector<std::uint8_t> host;  // first copy, for the reference
  std::vector<void*> packed[2];    // int4 copies per rotation
  std::vector<void*> packed_g;     // slab-grouped int4 copies (block rotation)
};

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
    if (host != nullptr) {
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
    const std::uint8_t* blk = row + (b * 34);
    __half d;
    std::memcpy(&d, blk, 2);
    const double scale = __half2float(d);
    for (int i = 0; i < 32; ++i)
      sum += scale * static_cast<std::int8_t>(blk[2 + i]) * x[(b * 32) + i];
  }
  return sum;
}

int Nibble(const std::vector<std::uint8_t>& codes, std::uint32_t row,
           std::uint32_t e, std::uint32_t k) {
  const std::size_t off =
      ((std::size_t{row / 16} * (k / 32) + (e / 32)) * 256) +
      ((row % 16) * 16) + ((e % 32) / 2);
  const int v = (e % 2 == 0) ? (codes[off] & 0xF) : (codes[off] >> 4);
  return v >= 8 ? v - 16 : v;
}

struct Timer {
  hipStream_t stream;
  hipEvent_t start, end;
  double Median(const std::function<void(int)>& run) {
    for (int i = 0; i < 3; ++i)
      run(i);
    std::vector<double> us;
    for (int rep = 0; rep < 5; ++rep) {
      Check(hipEventRecord(start, stream), "record");
      for (int i = 0; i < 8; ++i)
        run(i);
      Check(hipEventRecord(end, stream), "record");
      Check(hipEventSynchronize(end), "sync");
      float ms = 0;
      Check(hipEventElapsedTime(&ms, start, end), "elapsed");
      us.push_back(1000.0 * ms / 8);
    }
    std::sort(us.begin(), us.end());
    return us[2];
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <first-target-shard.gguf> [rows...]\n",
                 argv[0]);
    return 2;
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<std::uint32_t> row_list;
  for (int a = 2; a < argc; ++a)
    row_list.push_back(static_cast<std::uint32_t>(std::atoi(argv[a])));
  if (row_list.empty())
    row_list = {512, 1024, 2048, 4096, 8192};
  const std::uint32_t max_rows =
      *std::max_element(row_list.begin(), row_list.end());
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  Require(reader != nullptr, error.c_str());
  auto weights =
      gufo::models::qwen38_flash_next::ModelWeights::Bind(*reader, &error);
  Require(weights.has_value(), error.c_str());
  const auto& L = weights->layers;
  std::vector<std::uint32_t> linear, attention;
  for (std::uint32_t i = 0; i < L.size(); ++i)
    (L[i].linear() ? linear : attention).push_back(i);

  std::vector<Shape> shapes;
  const auto add =
      [&](const char* name, std::uint32_t m, std::uint32_t k,
          const std::function<void*(int, std::vector<std::uint8_t>*)>& make) {
        Shape s{name, m, k, {}, {}, {}};
        for (int c = 0; c < kCopies; ++c)
          s.copies.push_back(make(c, c == 0 ? &s.host : nullptr));
        shapes.push_back(std::move(s));
      };
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

  hipStream_t stream;
  Check(hipStreamCreate(&stream), "stream");
  constexpr i4::Rotation kRot[2] = {i4::Rotation::kBlock128,
                                    i4::Rotation::kFull};
  const char* kRotName[2] = {"block", "full"};
  for (auto& s : shapes) {
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < kCopies; ++c) {
        void* p = nullptr;
        Check(hipMalloc(&p, i4::WeightBytes(s.m, s.k)), "packed");
        Require(i4::PackQ8_0(s.copies[c], s.m, s.k, kRot[r], p, stream),
                "pack");
        s.packed[r].push_back(p);
      }
    for (int c = 0; c < kCopies; ++c) {
      void* p = nullptr;
      Check(hipMalloc(&p, i4::WeightBytes(s.m, s.k, true)), "packed groups");
      Require(i4::PackQ8_0(s.copies[c], s.m, s.k, kRot[0], p, stream, 1, true),
              "pack groups");
      s.packed_g.push_back(p);
    }
  }
  Check(hipStreamSynchronize(stream), "pack");

  std::vector<float> x(std::size_t{max_rows} * 6144);
  std::uint32_t rng = 12345;
  for (std::size_t i = 0; i < x.size(); ++i) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    const float u = static_cast<float>(rng & 0xFFFF) / 32768.0F - 1.0F;
    const std::size_t channel = i % 6144;
    x[i] = (channel * 2654435761U) % 640 < 2 ? 20.0F * u : u;
  }
  float *dx, *dout, *dout2, *dconv;
  void *xh, *xq8t, *xi4;
  Check(hipMalloc(&dx, x.size() * 4), "x");
  Check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  Check(hipMalloc(&dout, std::size_t{max_rows} * 16384 * 4), "out");
  Check(hipMalloc(&dout2, std::size_t{max_rows} * 16384 * 4), "out2");
  Check(hipMalloc(&dconv, std::size_t{max_rows} * 10240 * 4), "conv");
  Check(hipMalloc(&xh, std::size_t{max_rows} * 6144 * 2), "xh");
  Check(hipMalloc(&xq8t, q::Q8TiledBytes(max_rows, 6144)), "xq8t");
  Check(hipMalloc(&xi4, i4::ActivationBytes(max_rows, 6144)), "xi4");
  Timer timer{stream, {}, {}};
  Check(hipEventCreate(&timer.start), "event");
  Check(hipEventCreate(&timer.end), "event");

  std::puts("shape,m,k,rows,route,us,TOPS,err,oracle");
  for (auto& s : shapes) {
    const std::size_t row_bytes = std::size_t{s.k / 32} * 34;
    for (const std::uint32_t rows : row_list) {
      const double ops = 2.0 * s.m * s.k * rows;
      const auto report = [&](const char* route, double us, double err,
                              long oracle) {
        std::printf("%s,%u,%u,%u,%s,%.1f,%.1f,%.3e,%ld\n", s.name, s.m, s.k,
                    rows, route, us, ops / us / 1e6, err, oracle);
      };
      // Relative RMS error of 256 sampled outputs against FP64.
      const auto accuracy = [&]() {
        std::vector<float> y(std::size_t{rows} * s.m);
        Check(hipMemcpy(y.data(), dout, y.size() * 4, hipMemcpyDeviceToHost),
              "out");
        double se = 0, sr = 0;
        for (int i = 0; i < 256; ++i) {
          const std::uint32_t t = (i * 37U) % rows;
          const std::uint32_t r = (i * 7919U + 13) % s.m;
          const double ref = Q8Dot(s.host.data() + (r * row_bytes),
                                   x.data() + std::size_t{t} * s.k, s.k);
          const double d = y[std::size_t{t} * s.m + r] - ref;
          se += d * d;
          sr += ref * ref;
        }
        return std::sqrt(se / sr);
      };
      // Production routes and their staging.
      const double narrow = timer.Median([&](int) {
        q::NarrowActivations(dx, xh, false, std::size_t{rows} * s.k, stream);
      });
      report("narrow", narrow, 0, 0);
      const double f16 = timer.Median([&](int i) {
        q::DenseF16Gemm(s.copies[i % kCopies], static_cast<const __half*>(xh),
                        dout, rows, s.m, s.k, stream);
      });
      q::DenseF16Gemm(s.copies[0], static_cast<const __half*>(xh), dout, rows,
                      s.m, s.k, stream);
      report("f16", f16, accuracy(), 0);
      const double q8t = timer.Median(
          [&](int) { q::QuantizeQ8Tiled(dx, xq8t, rows, s.k, stream); });
      report("q8tiled", q8t, 0, 0);
      const double w8a8 = timer.Median([&](int i) {
        q::W8A8Gemm(s.copies[i % kCopies], xq8t, dout, rows, s.m, s.k, stream);
      });
      q::W8A8Gemm(s.copies[0], xq8t, dout, rows, s.m, s.k, stream);
      report("w8a8", w8a8, accuracy(), 0);
      for (int r = 0; r < 2; ++r) {
        const std::string qname = std::string("quant-") + kRotName[r];
        const double quant = timer.Median(
            [&](int) { i4::Quantize(dx, rows, s.k, kRot[r], xi4, stream); });
        report(qname.c_str(), quant, 0, 0);
        q::NarrowActivations(dx, xh, false, std::size_t{rows} * s.k, stream);
        const double quant_half = timer.Median([&](int) {
          i4::Quantize(static_cast<const __half*>(xh), rows, s.k, kRot[r], xi4,
                       stream);
        });
        report((qname + "-f16").c_str(), quant_half, 0, 0);
        i4::Quantize(dx, rows, s.k, kRot[r], xi4, stream);
        const i4::Matrix act = i4::ActivationView(xi4, rows, s.k);
        const i4::Matrix w = i4::WeightView(s.packed[r][0], s.m, s.k);
        // Oracle: the same codes and scales on the CPU.
        std::vector<std::uint8_t> wc(std::size_t{i4::Padded(s.m, i4::kRowPad)} *
                                     (s.k / 2));
        std::vector<std::uint8_t> xc(
            std::size_t{i4::Padded(rows, i4::kTokenPad)} * (s.k / 2));
        std::vector<float> ws(i4::Padded(s.m, i4::kRowPad)), xs(rows);
        Check(hipMemcpy(wc.data(), w.codes, wc.size(), hipMemcpyDeviceToHost),
              "wc");
        Check(hipMemcpy(xc.data(), act.codes, xc.size(), hipMemcpyDeviceToHost),
              "xc");
        Check(
            hipMemcpy(ws.data(), w.scale, ws.size() * 4, hipMemcpyDeviceToHost),
            "ws");
        Check(hipMemcpy(xs.data(), act.scale, xs.size() * 4,
                        hipMemcpyDeviceToHost),
              "xs");
        std::vector<float> expect(512);
        for (int i = 0; i < 512; ++i) {
          const std::uint32_t t = (i * 41U) % rows;
          const std::uint32_t row = (i * 7907U + 5) % s.m;
          long long acc = 0;
          for (std::uint32_t e = 0; e < s.k; ++e)
            acc += Nibble(wc, row, e, s.k) * Nibble(xc, t, e, s.k);
          const std::uint32_t tile = row / 16, rr = row % 16;
          const float wscale = ws[(((tile * 2) + (rr % 2)) * 8) + (rr / 2)];
          expect[i] = static_cast<float>(acc) * (wscale * xs[t]);
        }
        std::vector<float> y(std::size_t{rows} * s.m);
        // Variants 1..4 (dense_i4::Gemm) for the block rotation, the default
        // for the full rotation; the default runs last so dout holds it.
        for (const int v :
             r == 0 ? std::vector<int>{1, 2, 3, 4, 0} : std::vector<int>{0}) {
          if (!i4::Gemm(w, act, dout, stream, v))
            continue;
          const double gemm = timer.Median([&](int i) {
            i4::Gemm(i4::WeightView(s.packed[r][i % kCopies], s.m, s.k), act,
                     dout, stream, v);
          });
          Require(i4::Gemm(w, act, dout, stream, v), "i4 gemm");
          Check(hipMemcpy(y.data(), dout, y.size() * 4, hipMemcpyDeviceToHost),
                "y");
          long mismatches = 0;
          for (int i = 0; i < 512; ++i) {
            const std::uint32_t t = (i * 41U) % rows;
            const std::uint32_t row = (i * 7907U + 5) % s.m;
            if (std::memcmp(&expect[i], &y[std::size_t{t} * s.m + row], 4) != 0)
              ++mismatches;
          }
          report((std::string("i4-") + kRotName[r] + "-v" + std::to_string(v))
                     .c_str(),
                 gemm, accuracy(), mismatches);
        }
        // Fused attention projection (QKV + head norms + RoPE + KV store):
        // F16 route and int4 route, F16 and TurboQuant-4 caches. The int4
        // gate heads (no norm) must equal the generic int4 GEMM `yref` bit
        // for bit.
        const auto attention_case = [&](const i4::Matrix& av,
                                        const std::vector<float>& yref,
                                        const std::function<i4::Matrix(int)>&
                                            wview,
                                        const char* tag, bool f16_route) {
          if (s.m != 13312 || rows < 1024)
            return;
          const auto& l = L[attention[0]];
          float *qg, *kg, *query, *gate;
          void *keys, *values;
          __half *kscale, *vscale;
          std::uint32_t* position;
          const std::size_t width = 6144;
          Check(hipMalloc(&qg, 256 * 4), "qg");
          Check(hipMalloc(&kg, 256 * 4), "kg");
          Check(
              hipMemcpy(qg, l.attn_q_norm.data, 256 * 4, hipMemcpyHostToDevice),
              "qg");
          Check(
              hipMemcpy(kg, l.attn_k_norm.data, 256 * 4, hipMemcpyHostToDevice),
              "kg");
          Check(hipMalloc(&query, std::size_t{rows} * width * 4), "query");
          Check(hipMalloc(&gate, std::size_t{rows} * width * 4), "gate");
          Check(hipMalloc(&keys, std::size_t{rows} * 512 * 2), "keys");
          Check(hipMalloc(&values, std::size_t{rows} * 512 * 2), "values");
          Check(hipMalloc(&kscale, std::size_t{rows} * 2 * 2), "kscale");
          Check(hipMalloc(&vscale, std::size_t{rows} * 2 * 2), "vscale");
          Check(hipMalloc(&position, 4), "position");
          Check(hipMemset(position, 0, 4), "position");
          for (const auto format :
               {gufo::models::qwen38_flash_next::KvCacheFormat::kF16,
                gufo::models::qwen38_flash_next::KvCacheFormat::kTq4}) {
            const q::KvView kv{format, keys, values, kscale, vscale};
            const bool tq =
                format != gufo::models::qwen38_flash_next::KvCacheFormat::kF16;
            if (f16_route) {
              const double f16a = timer.Median([&](int i) {
                q::AttentionF16Gemm(
                    s.copies[i % kCopies], static_cast<const __half*>(xh), qg,
                    kg, query, gate, kv, rows, position, 1e7F, 1e-6F, stream);
              });
              report(tq ? "attn-f16-tq4" : "attn-f16-f16kv", f16a, 0, 0);
              // The cache the F16 route writes, for exactness checks.
              q::AttentionF16Gemm(s.copies[0], static_cast<const __half*>(xh),
                                  qg, kg, query, gate, kv, rows, position, 1e7F,
                                  1e-6F, stream);
              Check(hipStreamSynchronize(stream), "attention f16");
              // Everything the fused kernel wrote: the KV cache (codes
              // and scales) and the query and gate rows, hashed so two
              // builds of the F16 route can be compared bit for bit.
              const std::size_t kv_bytes =
                  tq ? std::size_t{rows} * 2 * 128 : std::size_t{rows} * 1024;
              std::vector<std::uint8_t> kb(kv_bytes), vb(kv_bytes),
                  ks(std::size_t{rows} * 4), vs(std::size_t{rows} * 4),
                  qb(std::size_t{rows} * 6144 * 4),
                  gb(std::size_t{rows} * 6144 * 4);
              Check(
                  hipMemcpy(kb.data(), keys, kb.size(), hipMemcpyDeviceToHost),
                  "keys");
              Check(hipMemcpy(vb.data(), values, vb.size(),
                              hipMemcpyDeviceToHost),
                    "values");
              if (tq) {
                Check(hipMemcpy(ks.data(), kscale, ks.size(),
                                hipMemcpyDeviceToHost),
                      "kscale");
                Check(hipMemcpy(vs.data(), vscale, vs.size(),
                                hipMemcpyDeviceToHost),
                      "vscale");
              }
              Check(
                  hipMemcpy(qb.data(), query, qb.size(), hipMemcpyDeviceToHost),
                  "query");
              Check(
                  hipMemcpy(gb.data(), gate, gb.size(), hipMemcpyDeviceToHost),
                  "gate");
              std::uint64_t h = 1469598103934665603ULL;
              for (const auto* v : {&kb, &vb, &ks, &vs, &qb, &gb})
                for (const std::uint8_t b : *v)
                  h = (h ^ b) * 1099511628211ULL;
              std::printf("# attn-f16 %s rows=%u hash %016llx\n",
                          tq ? "tq4" : "f16kv", rows,
                          static_cast<unsigned long long>(h));
            }
            const double i4a = timer.Median([&](int i) {
              i4::AttentionGemm(wview(i % kCopies), av, qg, kg, query, gate, kv,
                                position, 1e7F, 1e-6F, stream);
            });
            Require(i4::AttentionGemm(wview(0), av, qg, kg, query, gate, kv,
                                      position, 1e7F, 1e-6F, stream),
                    "i4 attention");
            std::vector<float> g(std::size_t{rows} * width);
            Check(
                hipMemcpy(g.data(), gate, g.size() * 4, hipMemcpyDeviceToHost),
                "gate");
            long bad = 0;
            for (std::uint32_t t = 0; t < rows; t += 5)
              for (std::uint32_t head = 0; head < 24; head += 5)
                for (std::uint32_t d = 0; d < 256; d += 37)
                  bad +=
                      std::memcmp(&g[std::size_t{t} * width + head * 256 + d],
                                  &yref[std::size_t{t} * s.m +
                                        (2 * head + 1) * 256 + d],
                                  4) != 0;
            report((std::string("attn-i4-") + tag + (tq ? "-tq4" : "-f16kv"))
                       .c_str(),
                   i4a, 0, bad);
          }
          for (void* p :
               {static_cast<void*>(qg), static_cast<void*>(kg),
                static_cast<void*>(query), static_cast<void*>(gate), keys,
                values, static_cast<void*>(kscale), static_cast<void*>(vscale),
                static_cast<void*>(position)})
            Check(hipFree(p), "free");
        };
        // Grouped activations and weights (one integer scale per 128-slab).
        if (r == 0) {
          const double quant_g = timer.Median([&](int) {
            i4::Quantize(static_cast<const __half*>(xh), rows, s.k, kRot[r],
                         xi4, stream, true);
          });
          report("quant-groups-f16", quant_g, 0, 0);
          i4::Quantize(dx, rows, s.k, kRot[r], xi4, stream, true);
          const i4::Matrix gact = i4::ActivationView(xi4, rows, s.k, true);
          const i4::Matrix wg = i4::WeightView(s.packed_g[0], s.m, s.k, true);
          const double gemm_g = timer.Median([&](int i) {
            i4::Gemm(i4::WeightView(s.packed_g[i % kCopies], s.m, s.k, true),
                     gact, dout, stream);
          });
          Require(i4::Gemm(wg, gact, dout, stream), "i4 grouped gemm");
          const std::size_t slabs = s.k / 128;
          const std::size_t wrows = i4::Padded(s.m, i4::kRowPad);
          std::vector<std::uint8_t> wgc(wrows * (s.k / 2));
          std::vector<std::uint8_t> wgg(wrows * slabs);
          std::vector<float> wgs(wrows);
          std::vector<std::uint8_t> gc(
              std::size_t{i4::Padded(rows, i4::kTokenPad)} * (s.k / 2));
          std::vector<std::uint8_t> gg(
              std::size_t{i4::Padded(rows, i4::kTokenPad)} * slabs);
          std::vector<float> gs(rows);
          Check(hipMemcpy(wgc.data(), wg.codes, wgc.size(),
                          hipMemcpyDeviceToHost),
                "wgc");
          Check(hipMemcpy(wgg.data(), wg.group, wgg.size(),
                          hipMemcpyDeviceToHost),
                "wgg");
          Check(hipMemcpy(wgs.data(), wg.scale, wgs.size() * 4,
                          hipMemcpyDeviceToHost),
                "wgs");
          Check(hipMemcpy(gc.data(), gact.codes, gc.size(),
                          hipMemcpyDeviceToHost),
                "gc");
          Check(hipMemcpy(gg.data(), gact.group, gg.size(),
                          hipMemcpyDeviceToHost),
                "gg");
          Check(hipMemcpy(gs.data(), gact.scale, gs.size() * 4,
                          hipMemcpyDeviceToHost),
                "gs");
          Check(hipMemcpy(y.data(), dout, y.size() * 4, hipMemcpyDeviceToHost),
                "y");
          long gbad = 0;
          double se = 0, sr = 0;
          for (int i = 0; i < 512; ++i) {
            const std::uint32_t t = (i * 41U) % rows;
            const std::uint32_t row = (i * 7907U + 5) % s.m;
            long long tot = 0;
            const std::uint32_t tile = row / 16, rr = row % 16;
            for (std::uint32_t slab = 0; slab < slabs; ++slab) {
              long long part = 0;
              for (std::uint32_t e = slab * 128; e < slab * 128 + 128; ++e)
                part += Nibble(wgc, row, e, s.k) * Nibble(gc, t, e, s.k);
              tot += part * gg[((t / 16) * slabs + slab) * 16 + t % 16] *
                     wgg[((tile * slabs + slab) * 2 + (rr % 2)) * 8 + rr / 2];
            }
            const float wscale = wgs[(((tile * 2) + (rr % 2)) * 8) + (rr / 2)];
            const float expect = static_cast<float>(tot) * (wscale * gs[t]);
            gbad +=
                std::memcmp(&expect, &y[std::size_t{t} * s.m + row], 4) != 0;
          }
          report("i4g-block", gemm_g, accuracy(), gbad);
          attention_case(
              gact, y,
              [&](int c) {
                return i4::WeightView(s.packed_g[c], s.m, s.k, true);
              },
              "g", false);
          (void)se;
          (void)sr;
          if (s.m == 16384 && rows >= 1024) {
            float* conv_w = nullptr;
            float* history = nullptr;
            const auto& l = L[linear[0]];
            Check(hipMalloc(&conv_w, l.ssm_conv1d.SizeBytes()), "conv_w");
            Check(hipMemcpy(conv_w, l.ssm_conv1d.data, l.ssm_conv1d.SizeBytes(),
                            hipMemcpyHostToDevice),
                  "conv_w");
            Check(hipMalloc(&history, 3 * 10240 * 4), "history");
            Check(hipMemset(history, 0, 3 * 10240 * 4), "history");
            const double ssm_g = timer.Median([&](int i) {
              i4::SsmGemm(
                  i4::WeightView(s.packed_g[i % kCopies], s.m, s.k, true), gact,
                  conv_w, history, dout2, dconv, 10240, 4, stream);
            });
            Require(i4::SsmGemm(wg, gact, conv_w, history, dout2, dconv, 10240,
                                4, stream),
                    "i4 grouped ssm");
            Check(hipStreamSynchronize(stream), "ssm");
            std::vector<float> z(std::size_t{rows} * s.m);
            Check(
                hipMemcpy(z.data(), dout2, z.size() * 4, hipMemcpyDeviceToHost),
                "z");
            long zbad = 0;
            long first_t = -1, first_c = -1;
            for (std::uint32_t t = 0; t < rows; t += 7)
              for (std::uint32_t c = 10240; c < s.m; c += 61)
                if (std::memcmp(&z[std::size_t{t} * s.m + c],
                                &y[std::size_t{t} * s.m + c], 4) != 0) {
                  if (zbad++ == 0) {
                    first_t = t;
                    first_c = c;
                  }
                }
            if (zbad != 0)
              std::printf("# i4g-ssm first Z mismatch t=%ld c=%ld z=%g y=%g\n",
                          first_t, first_c,
                          z[std::size_t(first_t) * s.m + first_c],
                          y[std::size_t(first_t) * s.m + first_c]);
            // Convolved channels against the FP64 SiLU(conv) of the generic
            // grouped GEMM rows (history zero); tile edges included.
            std::vector<float> cv(std::size_t{rows} * 10240);
            Check(hipMemcpy(cv.data(), dconv, cv.size() * 4,
                            hipMemcpyDeviceToHost),
                  "cv");
            const auto* taps = static_cast<const float*>(l.ssm_conv1d.data);
            double cse = 0, csr = 0;
            for (std::uint32_t t = 0; t < rows; ++t) {
              if (t % 7 != 0 && t % 32 > 3 && t % 32 < 29 && t + 3 < rows)
                continue;
              for (std::uint32_t c = 0; c < 10240; c += 61) {
                double ref = 0;
                for (int j = 0; j < 4; ++j) {
                  const int src = static_cast<int>(t) - 3 + j;
                  if (src >= 0)
                    ref += taps[c * 4 + j] * y[std::size_t(src) * s.m + c];
                }
                ref = ref / (1.0 + std::exp(-ref));
                const double d = cv[std::size_t{t} * 10240 + c] - ref;
                cse += d * d;
                csr += ref * ref;
              }
            }
            report("i4g-ssm-block", ssm_g, std::sqrt(cse / csr), zbad);
            Check(hipFree(conv_w), "free");
            Check(hipFree(history), "free");
          }
          // Leave the per-token quantization for the checks below.
          i4::Quantize(dx, rows, s.k, kRot[r], xi4, stream);
          Require(i4::Gemm(w, act, dout, stream), "i4 gemm");
          Check(hipMemcpy(y.data(), dout, y.size() * 4, hipMemcpyDeviceToHost),
                "y");
        }
        attention_case(
            act, y,
            [&](int c) { return i4::WeightView(s.packed[r][c], s.m, s.k); },
            kRotName[r], r == 0);
        // Fused epilogues reuse the F16 route's code: compare with the
        // generic int4 GEMM in dout.
        if (s.m == 16384 && rows >= 1024) {
          const auto& l = L[linear[0]];
          float* conv_w = nullptr;
          float* history = nullptr;
          Check(hipMalloc(&conv_w, l.ssm_conv1d.SizeBytes()), "conv_w");
          Check(hipMemcpy(conv_w, l.ssm_conv1d.data, l.ssm_conv1d.SizeBytes(),
                          hipMemcpyHostToDevice),
                "conv_w");
          Check(hipMalloc(&history, 3 * 10240 * 4), "history");
          Check(hipMemset(history, 0, 3 * 10240 * 4), "history");
          const double ssm = timer.Median([&](int i) {
            i4::SsmGemm(i4::WeightView(s.packed[r][i % kCopies], s.m, s.k), act,
                        conv_w, history, dout2, dconv, 10240, 4, stream);
          });
          const double ssm_v1 = timer.Median([&](int i) {
            i4::SsmGemm(i4::WeightView(s.packed[r][i % kCopies], s.m, s.k), act,
                        conv_w, history, dout2, dconv, 10240, 4, stream, 2);
          });
          report((std::string("i4-ssm-") + kRotName[r] + "-v2").c_str(), ssm_v1,
                 0, 0);
          Require(i4::SsmGemm(w, act, conv_w, history, dout2, dconv, 10240, 4,
                              stream),
                  "i4 ssm");
          std::vector<float> z(std::size_t{rows} * s.m),
              cv(std::size_t{rows} * 10240);
          Check(hipMemcpy(z.data(), dout2, z.size() * 4, hipMemcpyDeviceToHost),
                "z");
          Check(
              hipMemcpy(cv.data(), dconv, cv.size() * 4, hipMemcpyDeviceToHost),
              "cv");
          const auto* taps = static_cast<const float*>(l.ssm_conv1d.data);
          long bad = 0;
          double se = 0, sr = 0;
          for (std::uint32_t t = 0; t < rows; t += 7)
            for (std::uint32_t c = 0; c < s.m; c += 61) {
              const std::size_t at = std::size_t{t} * s.m + c;
              if (c >= 10240) {
                bad += std::memcmp(&z[at], &y[at], 4) != 0;
                continue;
              }
              double ref = 0;
              for (int j = 0; j < 4; ++j) {
                const int src = static_cast<int>(t) - 3 + j;
                if (src >= 0)
                  ref += taps[c * 4 + j] * y[std::size_t(src) * s.m + c];
              }
              ref = ref / (1.0 + std::exp(-ref));  // SiLU
              const double d = cv[std::size_t{t} * 10240 + c] - ref;
              se += d * d;
              sr += ref * ref;
            }
          report((std::string("i4-ssm-") + kRotName[r]).c_str(), ssm,
                 std::sqrt(se / sr), bad);
          Check(hipFree(conv_w), "free");
          Check(hipFree(history), "free");
        }
      }
    }
  }
  return 0;
}
