// Routed-expert microbenchmark on one layer's real GGUF expert tensors.
//
// moe_bench --model FIRST_SHARD.gguf [--layer L] [--rows 1,8,64]
//           [--skew S] [--session N] [--iters N] [--paths new,wmma]
//
// Rows get top-10 distinct experts from a log-normal popularity (sigma S;
// 0 is uniform); rows of one session (N rows) prefer the previous row's
// experts. Each timed call follows a 64 MiB read so the 32 MiB MALL never
// holds the previous call's experts. GB/s counts the union of routed
// experts' gate/up/down bytes once per call. Paths: new (RoutedExperts,
// the decode route), wmma (prefill route), stream and
// streamrows (read-only bounds). --check 1 compares the new route with an
// FP64 reference and checks that rows are independent of their batch.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/routed_prefill.hpp"
#include "tests/models/qwen38_flash_next/moe_reference.hpp"

// Traced copy of the routed-expert kernels (--trace 1): every work item
// records its start/end time (100 MHz) and hardware slot.
__device__ unsigned long long* g_moe_trace;
#define MOE_TRACE_BEGIN(kind) \
  const unsigned long long trace_t0 = wall_clock64();
#define MOE_TRACE_END(kind)                                                  \
  if (threadIdx.x == 0) {                                                    \
    const unsigned slot =                                                    \
        atomicAdd(reinterpret_cast<unsigned*>(g_moe_trace), 1u);             \
    unsigned long long* rec = g_moe_trace + 2 + 3 * slot;                    \
    rec[0] = trace_t0;                                                       \
    rec[1] = wall_clock64();                                                 \
    rec[2] =                                                                 \
        (unsigned long long)(kind) << 32 | __builtin_amdgcn_s_getreg(63511); \
  }
#define RoutedExperts RoutedExpertsTraced
#define RoutedExpertsScratchBytes RoutedExpertsScratchBytesTraced
#define RoutedPacked RoutedPackedTraced
#define PackRoutedExperts PackRoutedExpertsTraced
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hip.cpp"
#undef RoutedExperts
#undef RoutedExpertsScratchBytes
#undef RoutedPacked
#undef PackRoutedExperts

namespace {

namespace r = gufo::models::qwen38_flash_next::rocm;
namespace rp = gufo::models::qwen38_flash_next::rocm::routed_prefill;

/// --epilogue 0 drops the MoE combine epilogue from the prefill paths (the
/// engine folds it into the HC combine), so their times are the GEMMs.
bool g_epilogue = true;
using gufo::core::GgmlType;

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kHidden = 2560;
constexpr int kFf = 640;

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
T* DeviceAlloc(std::size_t count) {
  void* p = nullptr;
  Hip(hipMalloc(&p, count * sizeof(T) + 4096), "hipMalloc");
  Hip(hipMemset(p, 0, count * sizeof(T) + 4096), "hipMemset");
  return static_cast<T*>(p);
}

struct Tensor {
  void* data{nullptr};
  GgmlType type{GgmlType::kF32};
  std::size_t bytes{0};
  const void* host{nullptr};
};

Tensor Upload(const gufo::core::GgufReader& reader, const std::string& name) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr)
    throw std::runtime_error("missing tensor " + name);
  Tensor t;
  t.type = info->type;
  t.bytes = info->size_bytes;
  t.host = info->data;
  t.data = DeviceAlloc<std::uint8_t>(t.bytes);
  Hip(hipMemcpy(t.data, info->data, t.bytes, hipMemcpyHostToDevice), "upload");
  return t;
}

struct Layer {
  Tensor gate, up, down;
  std::size_t expert_bytes() const {
    return (gate.bytes + up.bytes + down.bytes) / kExperts;
  }
};

struct Routing {
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::size_t unique{0};
};

Routing MakeRouting(int rows, double sigma, int session, double bonus,
                    std::mt19937_64& rng, int window = kExperts) {
  std::normal_distribution<double> normal;
  std::vector<double> log_p(kExperts);
  for (auto& v : log_p)
    v = sigma * normal(rng);
  std::extreme_value_distribution<double> gumbel;
  Routing out;
  out.ids.resize(static_cast<std::size_t>(rows) * kUsed);
  out.weights.resize(out.ids.size());
  std::set<int> unique;
  for (int t = 0; t < rows; ++t) {
    std::vector<std::pair<double, int>> score(kExperts);
    for (int e = 0; e < kExperts; ++e)
      score[e] = {e < window ? log_p[e] + gumbel(rng) : -1e30, e};
    if (session > 1 && t % session != 0) {
      for (int s = 0; s < kUsed; ++s)
        score[out.ids[(t - 1) * kUsed + s]].first += bonus;
    }
    std::partial_sort(score.begin(), score.begin() + kUsed, score.end(),
                      std::greater<>());
    double sum = 0;
    std::vector<double> w(kUsed);
    for (int s = 0; s < kUsed; ++s) {
      out.ids[t * kUsed + s] = score[s].second;
      unique.insert(score[s].second);
      w[s] =
          std::exp(-0.3 * s) * (0.5 + std::uniform_real_distribution<>()(rng));
      sum += w[s];
    }
    for (int s = 0; s < kUsed; ++s)
      out.weights[t * kUsed + s] = static_cast<float>(w[s] / sum);
  }
  out.unique = unique.size();
  return out;
}

struct Buffers {
  float* x;
  std::int32_t* ids;
  float* weights;
  float* shared;
  float* router;  // [rows][513], last column gates the shared expert
  float* gate_e;
  float* up_e;
  float* down_e;
  float* out;
  // WMMA route
  std::uint32_t* counts;
  std::int32_t* bounds;
  std::int32_t* cursors;
  std::int32_t* rows_token;
  std::int32_t* rows_slot;
  std::int32_t* tiles;
  __half* x_half;
  std::uint8_t* a4;  // QuantizeA4 rows (largest: group 64)
  std::uint8_t* h4;  // int4 SwiGLU rows (group 128) for DownIu4
  std::uint32_t* counts_host;
  std::int32_t* tiles_host;
};

Buffers AllocBuffers(int max_rows) {
  const std::size_t slots = static_cast<std::size_t>(max_rows) * kUsed;
  Buffers b{};
  b.x = DeviceAlloc<float>(static_cast<std::size_t>(max_rows) * kHidden);
  b.ids = DeviceAlloc<std::int32_t>(slots);
  b.weights = DeviceAlloc<float>(slots);
  b.shared = DeviceAlloc<float>(static_cast<std::size_t>(max_rows) * kHidden);
  b.router = DeviceAlloc<float>(static_cast<std::size_t>(max_rows) * 513);
  b.gate_e = DeviceAlloc<float>(slots * kFf);
  b.up_e = DeviceAlloc<float>(slots * kFf);
  b.down_e = DeviceAlloc<float>(slots * kHidden);
  b.out = DeviceAlloc<float>(static_cast<std::size_t>(max_rows) * kHidden);
  b.counts = DeviceAlloc<std::uint32_t>(kExperts);
  const std::size_t compact = r::RoutedCompactRows(slots, kExperts);
  b.bounds = DeviceAlloc<std::int32_t>(kExperts + 1);
  b.cursors = DeviceAlloc<std::int32_t>(kExperts);
  b.rows_token = DeviceAlloc<std::int32_t>(compact);
  b.rows_slot = DeviceAlloc<std::int32_t>(compact);
  const std::size_t tiles = 3 * ((slots + kExperts * 15) / 16 + kExperts + 1);
  b.tiles = DeviceAlloc<std::int32_t>(tiles);
  b.x_half = DeviceAlloc<__half>(static_cast<std::size_t>(max_rows) * kHidden);
  b.a4 = DeviceAlloc<std::uint8_t>(rp::A4Bytes(max_rows, kHidden, 64));
  b.h4 = DeviceAlloc<std::uint8_t>(rp::A4Bytes(slots, kFf, 128));
  Hip(hipHostMalloc(&b.counts_host, kExperts * sizeof(std::uint32_t)), "host");
  Hip(hipHostMalloc(&b.tiles_host, tiles * sizeof(std::int32_t)), "host");
  return b;
}

/// The prefill WMMA route (MoeExperts with ExpertMatrixRows), including its
/// host round trip for the tile map, with 16-row tiles.
void OldWmma(const Layer& l, const Buffers& b, int rows, hipStream_t s) {
  r::ExpertCounts(b.ids, b.counts, rows, kExperts, kUsed, s);
  Hip(hipMemcpyAsync(b.counts_host, b.counts, kExperts * 4,
                     hipMemcpyDeviceToHost, s),
      "counts");
  Hip(hipStreamSynchronize(s), "sync");
  int n_tiles = 0;
  for (int e = 0; e < kExperts; ++e) {
    const int padded = (b.counts_host[e] + 15) / 16 * 16;
    for (int j = 0; j < padded / 16; ++j)
      b.tiles_host[n_tiles++] = e | (j << 16);
  }
  Hip(hipMemcpyAsync(b.tiles, b.tiles_host, n_tiles * 4, hipMemcpyHostToDevice,
                     s),
      "tiles");
  r::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                   b.rows_slot, rows, kUsed, kExperts, s);
  r::NarrowActivations(b.x, b.x_half, false,
                       static_cast<std::size_t>(rows) * kHidden, s);
  const auto gt = l.gate.type == GgmlType::kQ5_K ? r::WeightType::kQ5_K
                                                 : r::WeightType::kQ4_K;
  const auto dt = l.down.type == GgmlType::kQ8_0 ? r::WeightType::kQ8_0
                                                 : r::WeightType::kQ5_1;
  auto* up_half = reinterpret_cast<__half*>(b.up_e);
  if (!r::RoutedF16Gemm(l.gate.data, gt, b.x_half, b.tiles, n_tiles, 16,
                        b.bounds, b.rows_token, b.rows_slot, nullptr, b.gate_e,
                        nullptr, kFf, kHidden, s) ||
      !r::RoutedF16Gemm(l.up.data, gt, b.x_half, b.tiles, n_tiles, 16, b.bounds,
                        b.rows_token, b.rows_slot, b.gate_e, nullptr, up_half,
                        kFf, kHidden, s) ||
      !r::RoutedF16Gemm(l.down.data, dt, up_half, b.tiles, n_tiles, 16,
                        b.bounds, b.rows_slot, b.rows_slot, nullptr, nullptr,
                        reinterpret_cast<__half*>(b.down_e), kHidden, kFf, s))
    throw std::runtime_error("routed F16 GEMM failed");
  r::MoeEpilogueVec4F16(reinterpret_cast<const __half*>(b.down_e), b.weights,
                        b.shared, b.router + kExperts, 513, b.out, rows, kUsed,
                        kHidden, s);
}

// Streaming bounds: read every byte of the routed experts once, either as
// contiguous 64 KiB chunks or with one lane per weight row (lane-strided).
struct Range {
  const std::uint8_t* base;
  std::uint32_t bytes;
  std::uint32_t row_bytes;
};

using U4 = __attribute__((ext_vector_type(4))) unsigned;

// One wave per 32-row tile of one expert matrix (contiguous bytes). `Rows`
// gives each lane its own weight row; otherwise the lanes read consecutive
// 16-byte pieces. Every lane keeps kDepth loads in flight.
template<bool Rows>
__global__ void StreamTiles(const Range* ranges, const int* tile_range,
                            const int* tile_index, std::uint32_t* sink) {
  constexpr int kDepth = 8;
  const Range rg = ranges[tile_range[blockIdx.x]];
  const int lane = threadIdx.x;
  const auto* tile =
      rg.base +
      static_cast<std::size_t>(tile_index[blockIdx.x]) * 32 * rg.row_bytes;
  const int pieces = static_cast<int>(rg.row_bytes / 16);  // per row
  U4 acc{0, 0, 0, 0};
  for (int p0 = 0; p0 < pieces; p0 += kDepth) {
    U4 v[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      const int p = min(p0 + d, pieces - 1);
      const std::size_t off =
          Rows ? static_cast<std::size_t>(lane) * rg.row_bytes + p * 16
               : (static_cast<std::size_t>(p) * 32 + lane) * 16;
      v[d] = *reinterpret_cast<const U4*>(tile + off);
    }
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
      acc ^= v[d];
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u)
    sink[0] = acc.x;
}

// Clean MALL eviction: read (never write) a buffer twice the cache size.
__global__ void FlushRead(const U4* p, std::size_t n, std::uint32_t* sink) {
  U4 acc{0, 0, 0, 0};
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    acc ^= p[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u)
    sink[0] = acc.x;
}

struct StreamPlan {
  Range* ranges{nullptr};
  int* chunk_range{nullptr};
  int* chunk_index{nullptr};
  int chunks{0};
  int tiles{0};
  int* tile_range{nullptr};
  int* tile_index{nullptr};
};

StreamPlan MakeStreamPlan(const Layer& l, const Routing& routing) {
  std::set<int> experts(routing.ids.begin(), routing.ids.end());
  std::vector<Range> ranges;
  std::vector<int> cr, ci, tr, ti;
  for (const int e : experts) {
    for (const Tensor* t : {&l.gate, &l.up, &l.down}) {
      const std::uint32_t per = static_cast<std::uint32_t>(t->bytes / kExperts);
      const std::uint32_t rows = t == &l.down ? kHidden : kFf;
      ranges.push_back({static_cast<const std::uint8_t*>(t->data) +
                            static_cast<std::size_t>(e) * per,
                        per, per / rows});
      for (std::uint32_t c = 0; c * 65536u < per; ++c) {
        cr.push_back(static_cast<int>(ranges.size() - 1));
        ci.push_back(static_cast<int>(c));
      }
      for (std::uint32_t c = 0; c < rows / 32; ++c) {
        tr.push_back(static_cast<int>(ranges.size() - 1));
        ti.push_back(static_cast<int>(c));
      }
    }
  }
  StreamPlan p;
  p.ranges = DeviceAlloc<Range>(ranges.size());
  p.chunk_range = DeviceAlloc<int>(cr.size());
  p.chunk_index = DeviceAlloc<int>(ci.size());
  p.tile_range = DeviceAlloc<int>(tr.size());
  p.tile_index = DeviceAlloc<int>(ti.size());
  Hip(hipMemcpy(p.ranges, ranges.data(), ranges.size() * sizeof(Range),
                hipMemcpyHostToDevice),
      "plan");
  Hip(hipMemcpy(p.chunk_range, cr.data(), cr.size() * 4, hipMemcpyHostToDevice),
      "plan");
  Hip(hipMemcpy(p.chunk_index, ci.data(), ci.size() * 4, hipMemcpyHostToDevice),
      "plan");
  Hip(hipMemcpy(p.tile_range, tr.data(), tr.size() * 4, hipMemcpyHostToDevice),
      "plan");
  Hip(hipMemcpy(p.tile_index, ti.data(), ti.size() * 4, hipMemcpyHostToDevice),
      "plan");
  p.chunks = static_cast<int>(cr.size());
  p.tiles = static_cast<int>(tr.size());
  return p;
}

void FreeStreamPlan(StreamPlan& p) {
  for (void* q :
       {static_cast<void*>(p.ranges), static_cast<void*>(p.chunk_range),
        static_cast<void*>(p.chunk_index), static_cast<void*>(p.tile_range),
        static_cast<void*>(p.tile_index)})
    (void)hipFree(q);
}

/// WeightType values are the GGUF type ids.
r::WeightType Type(GgmlType t) {
  return static_cast<r::WeightType>(t);
}

void New(const Layer& l, const Buffers& b, void* scratch, int rows,
         hipStream_t s, float* y = nullptr,
         const hipEvent_t* phases = nullptr) {
  if (!r::RoutedExperts(l.gate.data, l.up.data, Type(l.gate.type), l.down.data,
                        Type(l.down.type), kExperts, b.x, b.ids,
                        y ? y : b.down_e, rows, kUsed, scratch, s, phases))
    throw std::runtime_error("RoutedExperts failed");
  if (y == nullptr)
    r::MoeEpilogueVec4(b.down_e, b.weights, b.shared, b.router + kExperts, 513,
                       b.out, rows, kUsed, kHidden, s);
}

/// The production prefill route at its tile choices (Executor::RouteHints
/// and MoeExperts): 48-row tiles once slots >= 16 per expert, paired
/// gate/up on 64/128-row tiles from 1024 rows, down on 64-row tiles when
/// they pad less; otherwise 16-row separate GEMMs.
void Prefill(const Layer& l, const Buffers& b, int rows, hipStream_t s,
             const hipEvent_t* phases = nullptr) {
  r::ExpertCounts(b.ids, b.counts, rows, kExperts, kUsed, s);
  Hip(hipMemcpyAsync(b.counts_host, b.counts, kExperts * 4,
                     hipMemcpyDeviceToHost, s),
      "counts");
  Hip(hipStreamSynchronize(s), "sync");
  const int tile_rows = rows * kUsed >= 16 * kExperts ? 48 : 16;
  const auto build = [&](int& n, int rows_per_tile, bool padded) {
    const int first = n;
    for (int e = 0; e < kExperts; ++e) {
      const int c = padded ? (b.counts_host[e] + 15) / 16 * 16
                           : static_cast<int>(b.counts_host[e]);
      for (int j = 0; j < (c + rows_per_tile - 1) / rows_per_tile; ++j)
        b.tiles_host[n++] = e | (j << 16);
    }
    return n - first;
  };
  int n = 0;
  const int n_main = build(n, tile_rows, true);
  int n64 = 0, pair_off = 0, pair_n = 0, pair_rows = 64;
  // Every prompt batch pairs gate/up (RouteHints); MOE_PAIR_MIN (A/B) only
  // pairs from that many rows.
  static const int pair_min = [] {
    const char* v = std::getenv("MOE_PAIR_MIN");
    return v ? std::atoi(v) : 0;
  }();
  const bool pair = rows >= pair_min;
  if (pair) {
    n64 = build(n, 64, true);
    pair_off = n_main;
    pair_n = n64;
    int n128 = 0;
    for (int e = 0; e < kExperts; ++e)
      n128 += (b.counts_host[e] + 127) / 128;
    // MOE_PAIR_ROWS=64|128 forces the paired tile height (A/B).
    static const int pair_rows_env = [] {
      const char* v = std::getenv("MOE_PAIR_ROWS");
      return v ? std::atoi(v) : 0;
    }();
    if (pair_rows_env == 128 || (pair_rows_env == 0 && n128 * 4 <= n64 * 3)) {
      pair_rows = 128;
      pair_off = n;
      pair_n = build(n, 128, false);
    }
  }
  Hip(hipMemcpyAsync(b.tiles, b.tiles_host, n * 4, hipMemcpyHostToDevice, s),
      "tiles");
  r::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                   b.rows_slot, rows, kUsed, kExperts, s);
  r::NarrowActivations(b.x, b.x_half, false,
                       static_cast<std::size_t>(rows) * kHidden, s);
  const auto gt = Type(l.gate.type);
  const auto dt = Type(l.down.type);
  auto* up_half = reinterpret_cast<__half*>(b.up_e);
  if (phases)
    Hip(hipEventRecord(phases[0], s), "record");
  const bool ok =
      pair
          ? r::RoutedGatedF16Gemm(l.gate.data, l.up.data, gt, b.x_half,
                                  b.tiles + pair_off, pair_n, pair_rows,
                                  b.bounds, b.rows_token, b.rows_slot, up_half,
                                  kFf, kHidden, s)
          : r::RoutedF16Gemm(l.gate.data, gt, b.x_half, b.tiles, n_main,
                             tile_rows, b.bounds, b.rows_token, b.rows_slot,
                             nullptr, b.gate_e, nullptr, kFf, kHidden, s) &&
                r::RoutedF16Gemm(l.up.data, gt, b.x_half, b.tiles, n_main,
                                 tile_rows, b.bounds, b.rows_token, b.rows_slot,
                                 b.gate_e, nullptr, up_half, kFf, kHidden, s);
  if (phases)
    Hip(hipEventRecord(phases[1], s), "record");
  const bool wide = tile_rows == 48 && n64 != 0 && n64 * 4 <= n_main * 3;
  if (!ok ||
      !r::RoutedF16Gemm(l.down.data, dt, up_half, b.tiles + (wide ? n_main : 0),
                        wide ? n64 : n_main, wide ? 64 : tile_rows, b.bounds,
                        b.rows_slot, b.rows_slot, nullptr, nullptr,
                        reinterpret_cast<__half*>(b.down_e), kHidden, kFf, s))
    throw std::runtime_error("prefill routed GEMM failed");
  if (g_epilogue)
    r::MoeEpilogueVec4F16(reinterpret_cast<const __half*>(b.down_e), b.weights,
                          b.shared, b.router + kExperts, 513, b.out, rows,
                          kUsed, kHidden, s);
}

/// The integer routed GEMMs (routed_prefill.hpp) on the 64-row tile map of
/// the 16-padded buckets. `variant`: "base" (the production paired F16 gate/up
/// and down), "iu4" / "iu4g128" / "iu4g64" (Q4_K as shipped, activation group
/// 256 / 128 / 64; F16 down), "iu4full" (group 256 gate/up writing int4
/// SwiGLU rows + DownIu4, needs --down4 1), "sym128" / "sym64" (symmetric
/// weights, --sym G: gate/up writing int4 rows + DownSym) and "symup128" /
/// "symup64" (symmetric gate/up, F16 down).
void RoutedInt(const Layer& l, const Buffers& b, int rows, hipStream_t s,
               const std::string& variant, const hipEvent_t* phases = nullptr) {
  r::ExpertCounts(b.ids, b.counts, rows, kExperts, kUsed, s);
  Hip(hipMemcpyAsync(b.counts_host, b.counts, kExperts * 4,
                     hipMemcpyDeviceToHost, s),
      "counts");
  Hip(hipStreamSynchronize(s), "sync");
  int n = 0;
  const auto build = [&](int height) {
    const int first = n;
    for (int e = 0; e < kExperts; ++e) {
      const int c = (b.counts_host[e] + 15) / 16 * 16;
      for (int j = 0; j < (c + height - 1) / height; ++j)
        b.tiles_host[n++] = e | (j << 16);
    }
    return n - first;
  };
  const int n_64 = build(64);
  Hip(hipMemcpyAsync(b.tiles, b.tiles_host, n * 4, hipMemcpyHostToDevice, s),
      "tiles");
  r::RoutedCompact(b.ids, b.counts, b.bounds, b.cursors, b.rows_token,
                   b.rows_slot, rows, kUsed, kExperts, s);
  r::NarrowActivations(b.x, b.x_half, false,
                       static_cast<std::size_t>(rows) * kHidden, s);
  const auto gt = Type(l.gate.type);
  const auto dt = Type(l.down.type);
  auto* up_half = reinterpret_cast<__half*>(b.up_e);
  const bool sym = variant.rfind("sym", 0) == 0;
  const bool full = variant == "iu4full" || (sym && variant[3] != 'u');
  const int group =
      sym ? (variant.back() == '4' && variant[variant.size() - 2] == '6' ? 64
                                                                         : 128)
      : variant == "iu4g64"  ? 64
      : variant == "iu4g128" ? 128
                             : 256;
  if (phases)
    Hip(hipEventRecord(phases[0], s), "record");
  bool ok = false;
  const auto h4 =
      rp::CarveA4(b.h4, static_cast<std::size_t>(rows) * kUsed, kFf, 128);
  if (variant == "base") {
    ok = r::RoutedGatedF16Gemm(l.gate.data, l.up.data, gt, b.x_half, b.tiles,
                               n_64, 64, b.bounds, b.rows_token, b.rows_slot,
                               up_half, kFf, kHidden, s);
  } else {
    auto q = rp::CarveA4(b.a4, rows, kHidden, group);
    if (sym)
      q.sums = nullptr;
    ok = rp::QuantizeA4(b.x_half, rows, q, s) &&
         rp::GateUpIu4(l.gate.data, l.up.data, gt, q, b.tiles, n_64, 64,
                       b.bounds, b.rows_token, b.rows_slot,
                       full ? nullptr : up_half, full ? &h4 : nullptr, kFf, sym,
                       s);
  }
  if (phases)
    Hip(hipEventRecord(phases[1], s), "record");
  if (!ok)
    throw std::runtime_error("routed gate/up failed (" + variant + ")");
  if (sym && full)
    ok = rp::DownSym(l.down.data, dt, h4, b.tiles, n_64, 64, b.bounds,
                     b.rows_slot, b.rows_slot,
                     reinterpret_cast<__half*>(b.down_e), kHidden, s);
  else if (full)
    ok = rp::DownIu4(l.down.data, dt, h4, b.tiles, n_64, 64, b.bounds,
                     b.rows_slot, b.rows_slot,
                     reinterpret_cast<__half*>(b.down_e), kHidden, s);
  else
    ok = r::RoutedF16Gemm(l.down.data, dt, up_half, b.tiles, n_64, 64, b.bounds,
                          b.rows_slot, b.rows_slot, nullptr, nullptr,
                          reinterpret_cast<__half*>(b.down_e), kHidden, kFf, s);
  if (!ok)
    throw std::runtime_error("routed down failed (" + variant + ")");
  if (g_epilogue)
    r::MoeEpilogueVec4F16(reinterpret_cast<const __half*>(b.down_e), b.weights,
                          b.shared, b.router + kExperts, 513, b.out, rows,
                          kUsed, kHidden, s);
}

/// FP64 reference of expert e for token row x (see moe_reference.hpp).
std::vector<double> Reference(const Layer& l, const float* x, int e,
                              bool quantized) {
  return moe_reference::Expert(l.gate.type, l.gate.host, l.up.host, l.down.type,
                               l.down.host, e, x, quantized);
}

std::vector<float> Download(const float* p, std::size_t n);

/// Numerics of an integer variant (down outputs, F16) on `rows` rows: against
/// the exact FP64 expert on the shipped weights; against the production F16
/// route run on the same (possibly re-quantized) device weights with the
/// dequantized int4 token rows (isolates the kernel from the activation
/// grid); and, for non-symmetric variants, FP64 on the int4 rows.
void CheckVariant(const Layer& l, const Buffers& b, int rows, hipStream_t s,
                  const std::string& variant, std::mt19937_64& rng) {
  Routing rt = MakeRouting(rows, 0.5, 1, 0.0, rng);
  Hip(hipMemcpy(b.ids, rt.ids.data(), rt.ids.size() * 4, hipMemcpyHostToDevice),
      "ids");
  RoutedInt(l, b, rows, s, variant);
  Hip(hipStreamSynchronize(s), "sync");
  const std::vector<float> x = Download(b.x, std::size_t(rows) * kHidden);
  std::vector<std::uint16_t> y(std::size_t(rows) * kUsed * kHidden);
  Hip(hipMemcpy(y.data(), b.down_e, y.size() * 2, hipMemcpyDeviceToHost),
      "download");
  const bool sym = variant.rfind("sym", 0) == 0;
  const int group =
      sym ? (variant.size() > 2 && variant[variant.size() - 2] == '6' ? 64
                                                                      : 128)
      : variant == "iu4g64"  ? 64
      : variant == "iu4g128" ? 128
                             : 256;
  const auto q = rp::CarveA4(b.a4, rows, kHidden, group);
  std::vector<std::uint8_t> codes(std::size_t(rows) * kHidden / 2);
  std::vector<float> scales(std::size_t(rows) * kHidden / group);
  Hip(hipMemcpy(codes.data(), q.codes, codes.size(), hipMemcpyDeviceToHost),
      "codes");
  Hip(hipMemcpy(scales.data(), q.scale, scales.size() * 4,
                hipMemcpyDeviceToHost),
      "scales");
  // The int4 token rows, dequantized. WMMA nibble order: group j, slice h,
  // dword d, nibble p = 2 q (+1): element 32 j + 16 h + 8 d + q (+4).
  std::vector<float> xh(std::size_t(rows) * kHidden);
  for (int t = 0; t < rows; ++t)
    for (int j = 0; j < kHidden / 32; ++j)
      for (int h = 0; h < 2; ++h)
        for (int d = 0; d < 2; ++d) {
          std::uint32_t w;
          std::memcpy(
              &w, &codes[std::size_t(t) * kHidden / 2 + 16 * j + 8 * h + 4 * d],
              4);
          for (int p = 0; p < 8; ++p) {
            int c = static_cast<int>((w >> (4 * p)) & 0xF);
            c = c >= 8 ? c - 16 : c;
            const int el = 32 * j + 16 * h + 8 * d + (p >> 1) + 4 * (p & 1);
            xh[std::size_t(t) * kHidden + el] =
                static_cast<float>(c) *
                scales[std::size_t(t) * (kHidden / group) + el / group];
          }
        }
  // Production route on the dequantized rows (same weights, same ids).
  Hip(hipMemcpy(b.x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice), "x");
  RoutedInt(l, b, rows, s, "base");
  Hip(hipStreamSynchronize(s), "sync");
  std::vector<std::uint16_t> y2(y.size());
  Hip(hipMemcpy(y2.data(), b.down_e, y2.size() * 2, hipMemcpyDeviceToHost),
      "download");
  Hip(hipMemcpy(b.x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  double e_exact = 0, e_emul = 0, w_exact = 0, w_emul = 0, e_f64 = 0;
  int n = 0;
  const auto half_at = [](const std::vector<std::uint16_t>& v, std::size_t i) {
    __half hv;
    std::memcpy(&hv, &v[i], 2);
    return static_cast<double>(__half2float(hv));
  };
  for (int t : {0, rows / 3, rows - 1}) {
    for (int slot = 0; slot < kUsed; ++slot) {
      const int e = rt.ids[t * kUsed + slot];
      const auto ref =
          Reference(l, x.data() + std::size_t(t) * kHidden, e, false);
      std::vector<double> emu64;
      if (!sym)
        emu64 = Reference(l, xh.data() + std::size_t(t) * kHidden, e, false);
      double r2 = 0, d2 = 0, m2 = 0, d3 = 0, d4 = 0, m4 = 0;
      for (int i = 0; i < kHidden; ++i) {
        const std::size_t at = (std::size_t(t) * kUsed + slot) * kHidden + i;
        const double g = half_at(y, at);
        const double g2 = half_at(y2, at);
        r2 += ref[i] * ref[i];
        d2 += (g - ref[i]) * (g - ref[i]);
        m2 += g2 * g2;
        d3 += (g - g2) * (g - g2);
        if (!sym) {
          m4 += emu64[i] * emu64[i];
          d4 += (g - emu64[i]) * (g - emu64[i]);
        }
      }
      const double re = std::sqrt(d2 / std::max(r2, 1e-300));
      e_exact += re;
      w_exact = std::max(w_exact, re);
      const double rm = std::sqrt(d3 / std::max(m2, 1e-300));
      e_emul += rm;
      w_emul = std::max(w_emul, rm);
      if (!sym)
        e_f64 += std::sqrt(d4 / std::max(m4, 1e-300));
      ++n;
    }
  }
  std::printf(
      "check %s R=%d (%d pairs): rel RMS vs exact FP64 on shipped weights "
      "mean %.4g max %.4g; vs production F16 on the same weights with "
      "int4 rows mean %.4g max %.4g",
      variant.c_str(), rows, n, e_exact / n, w_exact, e_emul / n, w_emul);
  if (!sym)
    std::printf("; vs FP64 on the int4 rows mean %.4g", e_f64 / n);
  std::printf("\n");
}

std::vector<float> Download(const float* p, std::size_t n) {
  std::vector<float> v(n);
  Hip(hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost), "download");
  return v;
}

/// FP64 comparison on sampled (row, slot) pairs plus bitwise checks that a
/// row's output is independent of its batch, position and grouping.
bool Check(const Layer& l, const Buffers& b, void* scratch, hipStream_t s,
           std::mt19937_64& rng, double sigma) {
  bool ok = true;
  const int rows = 64;
  Routing rt = MakeRouting(rows, sigma, 1, 0.0, rng);
  // A hot expert with 11 rows (two groups), an unrouted slot, a repeat.
  for (int t = 0; t < 11; ++t) {
    bool present = false;
    for (int k = 0; k < kUsed; ++k)
      present |= rt.ids[t * kUsed + k] == 7;
    if (!present)
      rt.ids[t * kUsed + 9] = 7;
  }
  rt.ids[5 * kUsed + 3] = -1;
  std::vector<float> x = Download(b.x, std::size_t(rows) * kHidden);
  std::copy_n(x.begin(), kHidden, x.begin() + 40 * kHidden);
  std::copy_n(rt.ids.begin(), kUsed, rt.ids.begin() + 40 * kUsed);
  Hip(hipMemcpy(b.x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  Hip(hipMemcpy(b.ids, rt.ids.data(), rt.ids.size() * 4, hipMemcpyHostToDevice),
      "ids");
  New(l, b, scratch, rows, s, b.down_e);
  Hip(hipStreamSynchronize(s), "sync");
  const auto y = Download(b.down_e, std::size_t(rows) * kUsed * kHidden);
  // Max and mean relative RMS error against the exact and Q8 references.
  double err[2] = {};
  double rms[2] = {};
  int pairs = 0;
  for (int t : {0, 5, 10, 33, 63}) {
    for (int slot = 0; slot < kUsed; ++slot) {
      const float* got = y.data() + (std::size_t(t) * kUsed + slot) * kHidden;
      const int e = rt.ids[t * kUsed + slot];
      if (e < 0) {
        for (int i = 0; i < kHidden; ++i)
          ok &= got[i] == 0.0f;
        continue;
      }
      for (int q = 0; q < 2; ++q) {
        const auto ref =
            Reference(l, x.data() + std::size_t(t) * kHidden, e, q == 1);
        double den = 0, num = 0, r2 = 0, e2 = 0;
        for (int i = 0; i < kHidden; ++i) {
          den = std::max(den, std::fabs(ref[i]));
          num = std::max(num, std::fabs(got[i] - ref[i]));
          r2 += ref[i] * ref[i];
          e2 += (got[i] - ref[i]) * (got[i] - ref[i]);
        }
        rms[q] += std::sqrt(e2 / std::max(r2, 1e-300));
        err[q] = std::max(err[q], num / std::max(den, 1e-30));
      }
      ++pairs;
    }
  }
  std::printf(
      "check (%d pairs) max|err|/max|ref|, mean relative RMS: exact FP64 "
      "%.3g / %.3g, Q8 FP64 %.3g / %.3g\n",
      pairs, err[0], rms[0] / pairs, err[1], rms[1] / pairs);
  {
    // The prefill route (F16 WMMA over the packed experts) against the
    // exact operator, per (row, slot) down output.
    Prefill(l, b, rows, s);
    Hip(hipStreamSynchronize(s), "sync");
    std::vector<__half> m(std::size_t(rows) * kUsed * kHidden);
    Hip(hipMemcpy(m.data(), b.down_e, m.size() * 2, hipMemcpyDeviceToHost),
        "download");
    double worst = 0, mean = 0;
    int n = 0;
    for (int t : {0, 10, 63}) {
      for (int slot = 0; slot < kUsed; ++slot) {
        const int e = rt.ids[t * kUsed + slot];
        if (e < 0)
          continue;
        const __half* got =
            m.data() + (std::size_t(t) * kUsed + slot) * kHidden;
        const auto ref =
            Reference(l, x.data() + std::size_t(t) * kHidden, e, false);
        double r2 = 0, e2 = 0;
        for (int i = 0; i < kHidden; ++i) {
          const double gv = __half2float(got[i]);
          r2 += ref[i] * ref[i];
          e2 += (gv - ref[i]) * (gv - ref[i]);
        }
        const double rel = std::sqrt(e2 / std::max(r2, 1e-300));
        worst = std::max(worst, rel);
        mean += rel;
        ++n;
      }
    }
    std::printf(
        "check prefill (WMMA) relative RMS vs exact FP64: mean %.3g "
        "max %.3g\n",
        mean / n, worst);
    ok &= worst < 0.05;
  }
  ok &= err[1] < 1e-3;
  const bool repeat =
      std::memcmp(y.data(), y.data() + std::size_t(40) * kUsed * kHidden,
                  std::size_t(kUsed) * kHidden * 4) == 0;
  if (!repeat)
    std::printf("check: repeated row differs\n");
  ok &= repeat;
  for (int t : {0, 5, 10, 63}) {
    Hip(hipMemcpy(b.x, x.data() + std::size_t(t) * kHidden, kHidden * 4,
                  hipMemcpyHostToDevice),
        "x");
    Hip(hipMemcpy(b.ids, rt.ids.data() + t * kUsed, kUsed * 4,
                  hipMemcpyHostToDevice),
        "ids");
    New(l, b, scratch, 1, s, b.down_e);
    Hip(hipStreamSynchronize(s), "sync");
    const auto one = Download(b.down_e, std::size_t(kUsed) * kHidden);
    const bool same =
        std::memcmp(one.data(), y.data() + std::size_t(t) * kUsed * kHidden,
                    one.size() * 4) == 0;
    if (!same)
      std::printf("check: row %d alone differs from its batch\n", t);
    ok &= same;
  }
  Hip(hipMemcpy(b.x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  std::printf("check: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

/// Runs the traced kernels once and prints per-phase work-item durations
/// and wave concurrency.
void Trace(const Layer& l, const Buffers& b, void* scratch, int rows,
           hipStream_t s) {
  constexpr std::size_t kRecords = 1 << 20;
  unsigned long long* buf = DeviceAlloc<unsigned long long>(2 + 3 * kRecords);
  Hip(hipMemcpyToSymbol(HIP_SYMBOL(g_moe_trace), &buf, sizeof(buf)), "symbol");
  Hip(hipMemset(buf, 0, 16), "trace");
  if (!r::RoutedExpertsTraced(l.gate.data, l.up.data, Type(l.gate.type),
                              l.down.data, Type(l.down.type), kExperts, b.x,
                              b.ids, b.down_e, rows, kUsed, scratch, s, nullptr,
                              kHidden, kFf, 0))
    throw std::runtime_error("traced RoutedExperts failed");
  Hip(hipStreamSynchronize(s), "sync");
  unsigned n = 0;
  Hip(hipMemcpy(&n, buf, 4, hipMemcpyDeviceToHost), "trace");
  std::vector<unsigned long long> rec(3 * std::size_t(n));
  Hip(hipMemcpy(rec.data(), buf + 2, rec.size() * 8, hipMemcpyDeviceToHost),
      "trace");
  (void)hipFree(buf);
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<double> dur;
    std::vector<std::pair<unsigned long long, int>> ev;
    unsigned long long t0 = ~0ull, t1 = 0;
    std::map<unsigned, int> slots;
    for (unsigned i = 0; i < n; ++i) {
      if (int(rec[3 * i + 2] >> 32) != kind)
        continue;
      dur.push_back((rec[3 * i + 1] - rec[3 * i]) * 0.01);
      ev.push_back({rec[3 * i], 1});
      ev.push_back({rec[3 * i + 1], -1});
      t0 = std::min(t0, rec[3 * i]);
      t1 = std::max(t1, rec[3 * i + 1]);
      ++slots[(rec[3 * i + 2] >> 10) & 0x7FF];  // WGP / SA / SE
    }
    if (dur.empty())
      continue;
    std::sort(dur.begin(), dur.end());
    std::sort(ev.begin(), ev.end());
    int live = 0, peak = 0;
    double area = 0;
    for (std::size_t i = 0; i + 1 < ev.size(); ++i) {
      live += ev[i].second;
      peak = std::max(peak, live);
      area += live * double(ev[i + 1].first - ev[i].first);
    }
    const double span = (t1 - t0) * 0.01;
    std::printf(
        "  trace %s: %zu items, span %.1f us, item us p10 %.1f p50 %.1f "
        "p90 %.1f max %.1f, waves live avg %.0f peak %d, WGPs %zu\n",
        kind ? "down   " : "gate/up", dur.size(), span, dur[dur.size() / 10],
        dur[dur.size() / 2], dur[dur.size() * 9 / 10], dur.back(),
        area / (t1 - t0), peak, slots.size());
  }
}

std::vector<int> ParseList(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    v.push_back(std::stoi(item));
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  int layer = 0;
  std::vector<int> row_list{1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  double sigma = 0.5;
  int session = 1;
  double bonus = 1.0;
  int iters = 20;
  std::string paths = "new";
  int window = kExperts;
  bool check = false;
  bool trace = false;
  std::string checkv;  // comma list of integer variants for CheckVariant
  bool down4 = false;  // re-quantize the down experts to 4-bit codes
  int sym = 0;         // re-quantize gate/up/down to symmetric int4 (64|128)
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const std::string v = argv[i + 1];
    if (a == "--model")
      model = v;
    else if (a == "--layer")
      layer = std::stoi(v);
    else if (a == "--rows")
      row_list = ParseList(v);
    else if (a == "--skew")
      sigma = std::stod(v);
    else if (a == "--session")
      session = std::stoi(v);
    else if (a == "--bonus")
      bonus = std::stod(v);
    else if (a == "--iters")
      iters = std::stoi(v);
    else if (a == "--paths")
      paths = v;
    else if (a == "--window")
      window = std::stoi(v);
    else if (a == "--check")
      check = v == "1";
    else if (a == "--trace")
      trace = v == "1";
    else if (a == "--sym")
      sym = std::stoi(v);
    else if (a == "--down4")
      down4 = v == "1";
    else if (a == "--checkv")
      checkv = v;
    else if (a == "--epilogue")
      g_epilogue = v != "0";
    else
      throw std::runtime_error("unknown option " + a);
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader)
    throw std::runtime_error("open failed: " + error);
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init failed");
  const std::string p = "blk." + std::to_string(layer) + ".";
  Layer l{Upload(*reader, p + "ffn_gate_exps.weight"),
          Upload(*reader, p + "ffn_up_exps.weight"),
          Upload(*reader, p + "ffn_down_exps.weight")};
  // Kernels read the packed layout; Tensor::host keeps the GGUF bytes.
  if (r::RoutedPacked(Type(l.gate.type), Type(l.down.type))) {
    Hip(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (const Tensor* t : {&l.gate, &l.up, &l.down}) {
      const bool down = t == &l.down;
      if (!r::PackRoutedExperts(t->data, Type(t->type),
                                std::size_t{kExperts} * (down ? kHidden : kFf),
                                down ? kFf : kHidden, nullptr))
        throw std::runtime_error("PackRoutedExperts failed");
    }
    Hip(hipDeviceSynchronize(), "sync");
    std::printf("packed %.1f MB in %.2f ms\n",
                (l.gate.bytes + l.up.bytes + l.down.bytes) / 1e6,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
  }
  if (down4) {
    if (!rp::RequantDown4(l.down.data, Type(l.down.type),
                          std::size_t{kExperts} * kHidden, kFf, nullptr))
      throw std::runtime_error("--down4 needs Q5_1 down experts");
    Hip(hipDeviceSynchronize(), "sync");
    std::printf("down experts re-quantized to 4-bit codes\n");
  }
  if (sym != 0) {
    Hip(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    if (!rp::RequantQ4KSym(l.gate.data, Type(l.gate.type),
                           std::size_t{kExperts} * kFf, kHidden, sym,
                           nullptr) ||
        !rp::RequantQ4KSym(l.up.data, Type(l.up.type),
                           std::size_t{kExperts} * kFf, kHidden, sym,
                           nullptr) ||
        !rp::RequantDownSym(l.down.data, Type(l.down.type),
                            std::size_t{kExperts} * kHidden, kFf, nullptr))
      throw std::runtime_error(
          "--sym needs Q4_K gate/up and Q5_1 down experts");
    Hip(hipDeviceSynchronize(), "sync");
    std::printf("experts re-quantized to symmetric int4 g%d in %.1f ms\n", sym,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
  }
  std::printf("layer %d gate/up %s down %s expert %.3f MB\n", layer,
              std::string(gufo::core::ToString(l.gate.type)).c_str(),
              std::string(gufo::core::ToString(l.down.type)).c_str(),
              l.expert_bytes() / 1e6);
  const int max_rows = *std::max_element(row_list.begin(), row_list.end());
  Buffers b = AllocBuffers(std::max(max_rows, 64));
  void* scratch = DeviceAlloc<std::uint8_t>(
      r::RoutedExpertsScratchBytes(std::max(max_rows, 64), kUsed));
  std::mt19937_64 rng(1234);
  {
    std::vector<float> x(static_cast<std::size_t>(max_rows) * kHidden);
    std::normal_distribution<float> normal;
    for (auto& v : x)
      v = normal(rng);
    Hip(hipMemcpy(b.x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  }
  auto* sink = DeviceAlloc<std::uint32_t>(1);
  void* flush = nullptr;
  constexpr std::size_t kFlush = 64 << 20;
  Hip(hipMalloc(&flush, kFlush), "flush");
  Hip(hipMemset(flush, 1, kFlush), "flush");
  hipStream_t s = nullptr;
  Hip(hipStreamCreate(&s), "stream");
  std::vector<hipEvent_t> ev(2 * iters);
  std::vector<hipEvent_t> phase_ev(2 * iters);
  for (auto& e : ev)
    Hip(hipEventCreate(&e), "event");
  for (auto& e : phase_ev)
    Hip(hipEventCreate(&e), "event");

  if (check && !Check(l, b, scratch, s, rng, sigma))
    return 1;
  if (!checkv.empty()) {
    std::stringstream cs(checkv);
    for (std::string v; std::getline(cs, v, ',');)
      CheckVariant(l, b, std::max(max_rows, 64) >= 512 ? 512 : 64, s, v, rng);
  }
  for (const int rows : row_list) {
    std::vector<Routing> sets;
    std::vector<StreamPlan> plans;
    for (int k = 0; k < 8; ++k) {
      sets.push_back(MakeRouting(rows, sigma, session, bonus, rng, window));
      if (paths.find("stream") != std::string::npos)
        plans.push_back(MakeStreamPlan(l, sets.back()));
    }
    double unique = 0;
    for (const auto& set : sets)
      unique += set.unique;
    unique /= sets.size();
    if (trace) {
      Hip(hipMemcpy(b.ids, sets[0].ids.data(), sets[0].ids.size() * 4,
                    hipMemcpyHostToDevice),
          "ids");
      Trace(l, b, scratch, rows, s);
    }
    std::stringstream ps(paths);
    for (std::string path; std::getline(ps, path, ',');) {
      const auto run = [&](int k) {
        if (path == "wmma")
          OldWmma(l, b, rows, s);
        else if (path == "prefill")
          Prefill(l, b, rows, s,
                  k >= 2 ? phase_ev.data() + 2 * (k - 2) : nullptr);
        else if (path == "base" || path.rfind("iu4", 0) == 0 ||
                 path.rfind("sym", 0) == 0)
          RoutedInt(l, b, rows, s, path,
                    k >= 2 ? phase_ev.data() + 2 * (k - 2) : nullptr);
        else if (path == "new")
          New(l, b, scratch, rows, s, nullptr,
              k >= 2 ? phase_ev.data() + 2 * (k - 2) : nullptr);
        else if (path == "stream")
          StreamTiles<false><<<plans[k % 8].tiles, 32, 0, s>>>(
              plans[k % 8].ranges, plans[k % 8].tile_range,
              plans[k % 8].tile_index, sink);
        else if (path == "streamrows")
          StreamTiles<true><<<plans[k % 8].tiles, 32, 0, s>>>(
              plans[k % 8].ranges, plans[k % 8].tile_range,
              plans[k % 8].tile_index, sink);
        else
          throw std::runtime_error("unknown path " + path);
      };
      std::vector<float> ms;
      for (int it = -2; it < iters; ++it) {
        const auto& set = sets[(it + 2) % sets.size()];
        Hip(hipMemcpyAsync(b.ids, set.ids.data(), set.ids.size() * 4,
                           hipMemcpyHostToDevice, s),
            "ids");
        Hip(hipMemcpyAsync(b.weights, set.weights.data(),
                           set.weights.size() * 4, hipMemcpyHostToDevice, s),
            "weights");
        FlushRead<<<1024, 256, 0, s>>>(static_cast<const U4*>(flush),
                                       kFlush / 16, sink);
        if (it >= 0)
          Hip(hipEventRecord(ev[2 * it], s), "record");
        run(it + 2);
        if (it >= 0)
          Hip(hipEventRecord(ev[2 * it + 1], s), "record");
        Hip(hipStreamSynchronize(s), "sync");
      }
      for (int it = 0; it < iters; ++it) {
        float t = 0;
        Hip(hipEventElapsedTime(&t, ev[2 * it], ev[2 * it + 1]), "elapsed");
        ms.push_back(t);
      }
      if (path == "new" || path == "prefill" || path == "base" ||
          path.rfind("iu4", 0) == 0 || path.rfind("sym", 0) == 0) {
        // Median phase split: plan, gate/up, down + epilogue.
        std::vector<float> p0, p1, p2;
        for (int it = 0; it < iters; ++it) {
          float a = 0, b2 = 0, c = 0;
          Hip(hipEventElapsedTime(&a, ev[2 * it], phase_ev[2 * it]), "e");
          Hip(hipEventElapsedTime(&b2, phase_ev[2 * it], phase_ev[2 * it + 1]),
              "e");
          Hip(hipEventElapsedTime(&c, phase_ev[2 * it + 1], ev[2 * it + 1]),
              "e");
          p0.push_back(a);
          p1.push_back(b2);
          p2.push_back(c);
        }
        for (auto* v : {&p0, &p1, &p2})
          std::sort(v->begin(), v->end());
        std::printf(
            "       phases: plan %.1f us, gate/up %.1f us, down+epilogue %.1f "
            "us\n",
            p0[iters / 2] * 1e3, p1[iters / 2] * 1e3, p2[iters / 2] * 1e3);
      }
      std::uint64_t hash = 1469598103934665603ull;
      if (path == "new" || path == "prefill" || path == "wmma" ||
          path == "base" || path.rfind("iu4", 0) == 0 ||
          path.rfind("sym", 0) == 0) {
        const auto out =
            Download(b.out, static_cast<std::size_t>(rows) * kHidden);
        for (const float v : out) {
          std::uint32_t u;
          std::memcpy(&u, &v, 4);
          hash = (hash ^ u) * 1099511628211ull;
        }
      }
      std::sort(ms.begin(), ms.end());
      const double med = ms[ms.size() / 2];
      const double gb = unique * l.expert_bytes() / 1e9;
      std::printf(
          "R=%4d %-5s unique=%6.1f  %8.1f us  %6.1f GB/s  (min %.1f) hash "
          "%016llx\n",
          rows, path.c_str(), unique, med * 1e3, gb / (med * 1e-3),
          ms.front() * 1e3, static_cast<unsigned long long>(hash));
    }
    for (auto& plan : plans)
      FreeStreamPlan(plan);
  }
  return 0;
}
