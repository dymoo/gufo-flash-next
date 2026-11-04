#include "src/models/mimo_v26_flash/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/mimo_v26_flash/kernels/rocm/kernels.hpp"
#include "src/models/mimo_v26_flash/kernels/rocm/mmq/mimo_mmq.h"

namespace gufo::models::mimo_v26_flash::rocm {
namespace {

using core::GgmlType;

/// Quantized GEMM tiles may read past a row; every weight carries this tail.
constexpr std::size_t kTailMargin = 4096;
constexpr int kMaxItems = 256;
constexpr int kFullLayers = 9;
constexpr int kSwaLayers = 39;
/// Halves of one layer in a page or ring (see kernels.hip.cpp).
constexpr std::size_t kLayerHalves =
    4 * 256 * (Config::kHeadDimQk + Config::kHeadDimV);
constexpr std::size_t kPageBytes = kFullLayers * kLayerHalves * 2;
constexpr std::size_t kRingBytes = kSwaLayers * kLayerHalves * 2;
constexpr int kMaxSplits = 32;
/// Steps up to this many rows use the decode GEMVs; wider ones the MMQ tiles.
constexpr int kGemvRows = 64;

struct DTensor {
  void* data{nullptr};
  GgmlType type{GgmlType::kF32};
  std::uint32_t cols{0}, rows{0};
  [[nodiscard]] const float* f32() const {
    return static_cast<const float*>(data);
  }
};

struct DLayer {
  DTensor attn_norm, attn_qkv, attn_out, attn_sinks, ffn_norm;
  DTensor ffn_gate, ffn_up, ffn_down;
  DTensor router, router_bias, gate_exps, up_exps, down_exps;
  bool swa{false}, dense{false};
  int full_index{-1}, swa_index{-1};
  int kv_heads{0};
  float rope_base{0};
};

/// GUFO_MIMO_PROFILE=1: GPU time per op group from events after each group,
/// logged to stderr every 16 steps (split by decode vs wider steps).
struct StepProfile {
  static constexpr const char* kNames[] = {
      "embed",  "norm",     "qkv",         "rope",   "attention", "kv_commit",
      "o_proj", "residual", "dense_ffn",   "router", "moe_prep",  "moe_gate_up",
      "swiglu", "moe_down", "moe_combine", "head"};
  static constexpr int kCats = sizeof(kNames) / sizeof(kNames[0]);
  bool on = std::getenv("GUFO_MIMO_PROFILE") != nullptr;
  std::vector<hipEvent_t> events;
  std::vector<int> cats;
  double total[2][kCats] = {};
  int steps[2] = {};
  hipStream_t stream{};

  void Begin(hipStream_t s) {
    if (!on)
      return;
    stream = s;
    if (events.empty()) {
      events.resize(4096);
      for (auto& e : events)
        (void)hipEventCreate(&e);
    }
    cats.clear();
    Mark(-1);
  }
  void Mark(int cat) {
    if (!on || cats.size() >= events.size())
      return;
    (void)hipEventRecord(events[cats.size()], stream);
    cats.push_back(cat);
  }
  /// After the stream synchronised.
  void End(bool decode) {
    if (!on || cats.empty())
      return;
    const int k = decode ? 0 : 1;
    for (std::size_t i = 1; i < cats.size(); ++i) {
      float ms = 0;
      (void)hipEventElapsedTime(&ms, events[i - 1], events[i]);
      total[k][cats[i]] += ms;
    }
    if (++steps[k] % 16 != 0)
      return;
    double sum = 0;
    for (double t : total[k])
      sum += t;
    std::fprintf(stderr, "[mimo-profile] %s steps=%d gpu_ms/step=%.2f",
                 decode ? "decode" : "wide", steps[k], sum / steps[k]);
    for (int c = 0; c < kCats; ++c)
      if (total[k][c] > 0)
        std::fprintf(stderr, " %s=%.2f", kNames[c], total[k][c] / steps[k]);
    std::fprintf(stderr, "\n");
    std::fill(std::begin(total[k]), std::end(total[k]), 0.0);
    steps[k] = 0;
  }
};

bool Check(hipError_t e, const char* what, std::string* error) {
  if (e == hipSuccess)
    return true;
  if (error != nullptr)
    *error = std::string(what) + ": " + hipGetErrorString(e);
  return false;
}

}  // namespace

struct Executor::Impl {
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::vector<void*> allocations;

  DTensor token_embd, output_norm, output;
  std::vector<DLayer> layers;

  // Step scratch, sized for kMaxRows.
  std::int32_t *tokens{}, *positions{}, *ids{}, *logit_rows{};
  float *hidden{}, *norm{}, *qkv{}, *q{}, *attn{}, *proj{}, *attn_scratch{};
  float *router_logits{}, *weights{}, *gate{}, *up{}, *mid{}, *y{};
  float *final_rows{}, *logits{};
  std::uint16_t *k{}, *v{};
  void* yq{};  ///< q8_1 activations for the GEMV path
  std::int32_t *ids_src1{}, *ids_dst{}, *bounds{};
  AttnItem* items{};
  SessionKv* sessions{};
  std::size_t attn_scratch_floats{0};

  // Full-attention page pool.
  std::uint16_t* pool{nullptr};
  std::size_t page_count{0};
  std::vector<std::int32_t> free_pages;
  std::uint32_t max_pages{0};
  std::mutex pool_mutex;

  std::vector<float> logits_host;
  StepProfile profile;

  template<class T>
  bool Alloc(T** ptr, std::size_t count, std::string* error) {
    void* p = nullptr;
    if (!Check(hipMalloc(&p, count * sizeof(T) + 256), "hipMalloc scratch",
               error))
      return false;
    allocations.push_back(p);
    *ptr = static_cast<T*>(p);
    return true;
  }

  ~Impl() {
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    for (void* p : allocations)
      (void)hipFree(p);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
};

KvState::~KvState() {
  Release();
  (void)hipFree(device_pages_);
  (void)hipFree(ring_);
}

void KvState::Release() noexcept {
  if (owner_ != nullptr)
    owner_->ReturnPages(pages_);
  pages_.clear();
  device_page_count_ = 0;
}

std::size_t KvState::committed_bytes() const noexcept {
  return pages_.size() * kPageBytes + kRingBytes;
}

Executor::Executor(Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}
Executor::~Executor() = default;

std::size_t Executor::page_bytes() const noexcept {
  return kPageBytes;
}
std::size_t Executor::ring_bytes() const noexcept {
  return kRingBytes;
}
std::size_t Executor::pool_bytes() const noexcept {
  return impl_->page_count * kPageBytes;
}
std::size_t Executor::pool_used_bytes() const noexcept {
  std::lock_guard lock(impl_->pool_mutex);
  return (impl_->page_count - impl_->free_pages.size()) * kPageBytes;
}

std::unique_ptr<Executor> Executor::Create(const ModelWeights& weights,
                                           const core::GgufReader& reader,
                                           std::uint32_t max_context,
                                           std::size_t kv_pool_bytes,
                                           std::string* error) {
  const Config& c = weights.config;
  if (c.FullLayerCount() != kFullLayers) {
    if (error)
      *error = "MiMo kernels need 9 full-attention layers";
    return nullptr;
  }
  std::unique_ptr<Executor> ex(new Executor(c));
  auto& im = *ex->impl_;
  if (!Check(hipStreamCreateWithFlags(&im.stream, hipStreamNonBlocking),
             "hipStreamCreate", error) ||
      mimo_mmq_init(0) != 0 ||
      hipblasCreate(&im.blas) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(im.blas, im.stream) != HIPBLAS_STATUS_SUCCESS) {
    if (error && error->empty())
      *error = "HIP runtime initialisation failed";
    return nullptr;
  }

  auto stager = hip::WeightUpload::Create(reader.GetMappedRegions(), error);
  if (stager == nullptr)
    return nullptr;
  bool ok = true;
  const auto upload = [&](const TensorRef& t) {
    DTensor d;
    if (!ok || t.empty())
      return d;
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (!Check(hipMalloc(&ptr, size + kTailMargin), "hipMalloc weight",
               error)) {
      ok = false;
      return d;
    }
    im.allocations.push_back(ptr);
    ex->weight_bytes_ += size + kTailMargin;
    if (!stager->Copy(t.shard, t.file_offset, size, ptr, error)) {
      ok = false;
      return d;
    }
    (void)hipMemset(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    return d;
  };

  im.token_embd = upload(weights.token_embd);
  im.output_norm = upload(weights.output_norm);
  im.output = upload(weights.output);
  int full = 0, swa = 0;
  for (std::uint32_t l = 0; l < Config::kLayers && ok; ++l) {
    const auto& L = weights.layers[l];
    DLayer d;
    d.attn_norm = upload(L.attn_norm);
    d.attn_qkv = upload(L.attn_qkv);
    d.attn_out = upload(L.attn_out);
    d.attn_sinks = upload(L.attn_sinks);
    d.ffn_norm = upload(L.ffn_norm);
    d.ffn_gate = upload(L.ffn_gate);
    d.ffn_up = upload(L.ffn_up);
    d.ffn_down = upload(L.ffn_down);
    d.router = upload(L.router);
    d.router_bias = upload(L.router_bias);
    d.gate_exps = upload(L.gate_exps);
    d.up_exps = upload(L.up_exps);
    d.down_exps = upload(L.down_exps);
    d.swa = c.swa[l];
    d.dense = c.dense_ffn[l];
    d.full_index = d.swa ? -1 : full++;
    d.swa_index = d.swa ? swa++ : -1;
    d.kv_heads = static_cast<int>(c.KvHeads(l));
    d.rope_base = d.swa ? c.rope_base_swa : c.rope_base_full;
    im.layers.push_back(d);
  }
  if (!ok || !stager->Finish(error))
    return nullptr;
  stager.reset();

  constexpr std::size_t R = kMaxRows;
  constexpr std::size_t H = Config::kHidden;
  constexpr std::size_t E = Config::kExpertsUsed;
  im.attn_scratch_floats = AttentionScratchFloats(static_cast<int>(R), 1);
  if (!im.Alloc(&im.tokens, R, error) || !im.Alloc(&im.positions, R, error) ||
      !im.Alloc(&im.hidden, R * H, error) ||
      !im.Alloc(&im.norm, R * H, error) ||
      !im.Alloc(&im.qkv,
                R * (Config::kQDim + Config::kSwaKvHeads * (Config::kHeadDimQk +
                                                            Config::kHeadDimV)),
                error) ||
      !im.Alloc(&im.q, R * Config::kQDim, error) ||
      !im.Alloc(&im.k, R * 8 * Config::kHeadDimQk, error) ||
      !im.Alloc(&im.v, R * 8 * Config::kHeadDimV, error) ||
      !im.Alloc(&im.attn, R * Config::kAttnOutDim, error) ||
      !im.Alloc(&im.proj, R * H, error) ||
      !im.Alloc(&im.attn_scratch, im.attn_scratch_floats, error) ||
      !im.Alloc(&im.router_logits, R * Config::kExperts, error) ||
      !im.Alloc(&im.ids, R * E, error) ||
      !im.Alloc(&im.weights, R * E, error) ||
      !im.Alloc(&im.gate, R * E * Config::kExpertFf, error) ||
      !im.Alloc(&im.up, R * E * Config::kExpertFf, error) ||
      !im.Alloc(&im.mid, R * E * Config::kExpertFf, error) ||
      !im.Alloc(&im.y, R * E * H, error) ||
      !im.Alloc(&im.logit_rows, kMaxItems, error) ||
      !im.Alloc(&im.final_rows, kMaxItems * H, error) ||
      !im.Alloc(&im.logits, static_cast<std::size_t>(kMaxItems) * c.vocab,
                error) ||
      !im.Alloc(&im.items, kMaxItems, error) ||
      !im.Alloc(&im.sessions, kMaxItems, error) ||
      !im.Alloc(
          reinterpret_cast<char**>(&im.yq),
          mimo_q8_1_bytes(std::max(kGemvRows * static_cast<int>(E), kMaxItems),
                          Config::kDenseFf),
          error) ||
      !im.Alloc(&im.ids_src1, kGemvRows * E, error) ||
      !im.Alloc(&im.ids_dst, kGemvRows * E, error) ||
      !im.Alloc(&im.bounds, Config::kExperts + 1, error))
    return nullptr;
  // The dense layer-0 FFN reuses the expert gate/up/mid buffers.
  static_assert(E * Config::kExpertFf == Config::kDenseFf);

  im.max_pages = (max_context + kPageTokens - 1) / kPageTokens;
  if (kv_pool_bytes == 0) {
    // Free device memory less headroom for the GEMM pools and HIP runtime.
    constexpr std::size_t kHeadroom = std::size_t{3} << 30;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (!Check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo",
               error))
      return nullptr;
    kv_pool_bytes = free_bytes > kHeadroom ? free_bytes - kHeadroom : 0;
  }
  im.page_count = kv_pool_bytes / kPageBytes;
  if (im.page_count < im.max_pages) {
    if (error)
      *error = "KV pool of " + std::to_string(kv_pool_bytes >> 20) +
               " MiB cannot hold one " + std::to_string(max_context) +
               "-token session";
    return nullptr;
  }
  void* pool = nullptr;
  if (!Check(hipMalloc(&pool, im.page_count * kPageBytes), "hipMalloc KV pool",
             error))
    return nullptr;
  im.allocations.push_back(pool);
  im.pool = static_cast<std::uint16_t*>(pool);
  im.free_pages.resize(im.page_count);
  for (std::size_t i = 0; i < im.page_count; ++i)
    im.free_pages[i] = static_cast<std::int32_t>(im.page_count - 1 - i);
  return ex;
}

std::unique_ptr<KvState> Executor::CreateKv(std::string* error) {
  std::unique_ptr<KvState> kv(new KvState(this));
  void* ring = nullptr;
  void* pages = nullptr;
  if (!Check(hipMalloc(&ring, kRingBytes), "hipMalloc ring", error) ||
      !Check(hipMalloc(&pages, impl_->max_pages * sizeof(std::int32_t)),
             "hipMalloc page table", error)) {
    (void)hipFree(ring);
    return nullptr;
  }
  kv->ring_ = static_cast<std::uint16_t*>(ring);
  kv->device_pages_ = static_cast<std::int32_t*>(pages);
  return kv;
}

void Executor::ReturnPages(std::vector<std::int32_t>& pages) noexcept {
  std::lock_guard lock(impl_->pool_mutex);
  impl_->free_pages.insert(impl_->free_pages.end(), pages.begin(), pages.end());
}

bool Executor::Reserve(KvState& kv, std::uint32_t positions,
                       std::string* error) {
  const std::size_t need = (positions + kPageTokens - 1) / kPageTokens;
  if (need > impl_->max_pages) {
    if (error)
      *error = "context exceeds the configured maximum";
    return false;
  }
  if (need > kv.pages_.size()) {
    std::lock_guard lock(impl_->pool_mutex);
    if (impl_->free_pages.size() < need - kv.pages_.size()) {
      if (error)
        *error = "KV pool exhausted";
      return false;
    }
    while (kv.pages_.size() < need) {
      kv.pages_.push_back(impl_->free_pages.back());
      impl_->free_pages.pop_back();
    }
  }
  if (kv.device_page_count_ < kv.pages_.size()) {
    const std::size_t first = kv.device_page_count_;
    if (!Check(
            hipMemcpyAsync(kv.device_pages_ + first, kv.pages_.data() + first,
                           (kv.pages_.size() - first) * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, impl_->stream),
            "page table upload", error))
      return false;
    kv.device_page_count_ = static_cast<std::uint32_t>(kv.pages_.size());
  }
  return true;
}

bool Executor::Forward(std::span<const StepItem> items, std::string* error) {
  auto& im = *impl_;
  const auto s = im.stream;
  if (items.empty() || items.size() > kMaxItems) {
    if (error)
      *error = "invalid step width";
    return false;
  }
  std::vector<std::int32_t> tokens, positions, logit_rows;
  std::vector<AttnItem> attn_items;
  std::vector<SessionKv> sessions;
  int max_count = 0;
  std::uint32_t max_keys = 0;
  for (const auto& it : items) {
    if (it.tokens.empty()) {
      if (error)
        *error = "empty step item";
      return false;
    }
    const auto count = static_cast<std::uint32_t>(it.tokens.size());
    if (!Reserve(*it.kv, it.start + count, error))
      return false;
    attn_items.push_back({static_cast<std::int32_t>(tokens.size()),
                          static_cast<std::int32_t>(count),
                          static_cast<std::int32_t>(it.start),
                          static_cast<std::int32_t>(sessions.size())});
    sessions.push_back({it.kv->device_pages_, it.kv->ring_});
    for (std::uint32_t i = 0; i < count; ++i) {
      tokens.push_back(it.tokens[i]);
      positions.push_back(static_cast<std::int32_t>(it.start + i));
    }
    if (!it.logits.empty())
      logit_rows.push_back(static_cast<std::int32_t>(tokens.size() - 1));
    max_count = std::max<int>(max_count, static_cast<int>(count));
    max_keys = std::max(max_keys, it.start + count);
  }
  const int n = static_cast<int>(tokens.size());
  if (n > static_cast<int>(kMaxRows)) {
    if (error)
      *error = "step exceeds " + std::to_string(kMaxRows) + " rows";
    return false;
  }
  const auto h2d = [&](void* dst, const void* src, std::size_t bytes) {
    return Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, s),
                 "step upload", error);
  };
  if (!h2d(im.tokens, tokens.data(), n * 4) ||
      !h2d(im.positions, positions.data(), n * 4) ||
      !h2d(im.items, attn_items.data(), attn_items.size() * sizeof(AttnItem)) ||
      !h2d(im.sessions, sessions.data(), sessions.size() * sizeof(SessionKv)) ||
      (!logit_rows.empty() &&
       !h2d(im.logit_rows, logit_rows.data(), logit_rows.size() * 4)))
    return false;

  // Decode rows split long full-attention key ranges across workgroups.
  int full_splits = 1;
  if (max_count == 1) {
    full_splits = std::clamp<int>(static_cast<int>((max_keys + 511) / 512), 1,
                                  kMaxSplits);
    while (AttentionScratchFloats(n, full_splits) > im.attn_scratch_floats)
      --full_splits;
  }

  constexpr int H = Config::kHidden;
  constexpr int E = Config::kExpertsUsed;
  const int item_count = static_cast<int>(attn_items.size());
  const float one = 1.0f, zero = 0.0f;
  int rc = 0;
  const bool gemv = n <= kGemvRows;
  // out[rows x M] = X[rows x K] . W^T through the GEMV (decode) or MMQ tiles.
  const auto project = [&](const DTensor& w, const float* x, float* out, int M,
                           int K, int rows) {
    if (rows <= kGemvRows) {
      rc |= mimo_q8_1_quantize(x, im.yq, rows, K, s);
      rc |= mimo_gemv_dense(static_cast<int>(w.type), w.data, im.yq, out, M,
                            rows, K, s);
    } else {
      rc |= mimo_mmq_dense(static_cast<int>(w.type), w.data, x, out, M, rows, K,
                           s);
    }
  };
  auto& prof = im.profile;
  enum {
    kEmbed,
    kNorm,
    kQkv,
    kRope,
    kAttn,
    kCommit,
    kOproj,
    kResid,
    kDense,
    kRouter,
    kMoePrep,
    kGateUp,
    kSwiglu,
    kDown,
    kCombine,
    kHead
  };
  prof.Begin(s);
  EmbedQ8_0(im.token_embd.data, im.tokens, im.hidden, n, s);
  prof.Mark(kEmbed);
  for (const DLayer& L : im.layers) {
    RmsNorm(im.hidden, L.attn_norm.f32(), im.norm, n, config_.rms_eps, s);
    prof.Mark(kNorm);
    project(L.attn_qkv, im.norm, im.qkv, static_cast<int>(L.attn_qkv.rows), H,
            n);
    prof.Mark(kQkv);
    QkvRope(im.qkv, im.positions, im.q, im.k, im.v, n, L.kv_heads, L.rope_base,
            s);
    prof.Mark(kRope);
    Attention(im.q, im.k, im.v, im.items, item_count, max_count,
              static_cast<int>(max_keys), im.sessions, im.pool, L.full_index,
              L.swa_index, L.swa ? L.attn_sinks.f32() : nullptr, im.attn,
              im.attn_scratch, L.swa ? 1 : full_splits, n, s);
    prof.Mark(kAttn);
    KvCommit(im.k, im.v, im.items, item_count, max_count, im.sessions, im.pool,
             L.full_index, L.swa_index, L.kv_heads, s);
    prof.Mark(kCommit);
    project(L.attn_out, im.attn, im.proj, H, Config::kAttnOutDim, n);
    prof.Mark(kOproj);
    AddScaled(im.hidden, im.proj, config_.value_scale,
              static_cast<std::int64_t>(n) * H, s);

    RmsNorm(im.hidden, L.ffn_norm.f32(), im.norm, n, config_.rms_eps, s);
    prof.Mark(kResid);
    if (L.dense) {
      constexpr int F = Config::kDenseFf;
      project(L.ffn_gate, im.norm, im.gate, F, H, n);
      project(L.ffn_up, im.norm, im.up, F, H, n);
      SwiGlu(im.gate, im.up, im.mid, static_cast<std::int64_t>(n) * F, s);
      project(L.ffn_down, im.mid, im.proj, H, F, n);
      AddScaled(im.hidden, im.proj, 1.0f, static_cast<std::int64_t>(n) * H, s);
      prof.Mark(kDense);
      continue;
    }
    if (gemv)
      RouterLogits(im.norm, L.router.f32(), im.router_logits, n, s);
    else if (hipblasSgemm(im.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, Config::kExperts,
                          n, H, &one, L.router.f32(), H, im.norm, H, &zero,
                          im.router_logits,
                          Config::kExperts) != HIPBLAS_STATUS_SUCCESS)
      rc |= 1;
    RouterTopK(im.router_logits, L.router_bias.f32(), im.ids, im.weights, n, s);
    prof.Mark(kRouter);
    constexpr int F = Config::kExpertFf;
    constexpr int X = Config::kExperts;
    if (gemv) {
      // Each active expert's weights are read once for all its tokens. The
      // down projection's input rows are the (token, slot) rows = ids_dst.
      rc |= mimo_q8_1_quantize(im.norm, im.yq, n, H, s);
      rc |= mimo_mmq_build_ids_maps(im.ids, im.ids_src1, im.ids_dst, im.bounds,
                                    X, n, E, 1, E, 1, s);
      prof.Mark(kMoePrep);
      rc |= mimo_gemv_moe(static_cast<int>(L.gate_exps.type), L.gate_exps.data,
                          im.yq, im.ids_src1, im.ids_dst, im.bounds, im.gate, F,
                          H, X, n * E, n, s);
      rc |= mimo_gemv_moe(static_cast<int>(L.up_exps.type), L.up_exps.data,
                          im.yq, im.ids_src1, im.ids_dst, im.bounds, im.up, F,
                          H, X, n * E, n, s);
      prof.Mark(kGateUp);
      SwiGlu(im.gate, im.up, im.mid, static_cast<std::int64_t>(n) * E * F, s);
      rc |= mimo_q8_1_quantize(im.mid, im.yq, n * E, F, s);
      prof.Mark(kSwiglu);
      rc |= mimo_gemv_moe(static_cast<int>(L.down_exps.type), L.down_exps.data,
                          im.yq, im.ids_dst, im.ids_dst, im.bounds, im.y, H, F,
                          X, n * E, n, s);
    } else {
      rc |= mimo_mmq_moe(static_cast<int>(L.gate_exps.type), L.gate_exps.data,
                         im.norm, im.ids, im.gate, F, H, n, X, E, 1, s);
      rc |= mimo_mmq_moe(static_cast<int>(L.up_exps.type), L.up_exps.data,
                         im.norm, im.ids, im.up, F, H, n, X, E, 1, s);
      SwiGlu(im.gate, im.up, im.mid, static_cast<std::int64_t>(n) * E * F, s);
      rc |= mimo_mmq_moe(static_cast<int>(L.down_exps.type), L.down_exps.data,
                         im.mid, im.ids, im.y, H, F, n, X, E, E, s);
    }
    prof.Mark(kDown);
    MoeCombine(im.hidden, im.y, im.weights, n, s);
    prof.Mark(kCombine);
  }
  if (rc != 0) {
    if (error)
      *error = "quantized GEMM dispatch failed";
    return false;
  }

  const int m = static_cast<int>(logit_rows.size());
  if (m > 0) {
    GatherRows(im.hidden, im.logit_rows, im.final_rows, m, s);
    RmsNorm(im.final_rows, im.output_norm.f32(), im.norm, m, config_.rms_eps,
            s);
    project(im.output, im.norm, im.logits, static_cast<int>(config_.vocab), H,
            m);
    if (rc != 0) {
      if (error)
        *error = "LM head dispatch failed";
      return false;
    }
    im.logits_host.resize(static_cast<std::size_t>(m) * config_.vocab);
    if (!Check(hipMemcpyAsync(im.logits_host.data(), im.logits,
                              im.logits_host.size() * sizeof(float),
                              hipMemcpyDeviceToHost, s),
               "logits download", error))
      return false;
  }
  prof.Mark(kHead);
  if (!Check(hipStreamSynchronize(s), "forward", error))
    return false;
  prof.End(max_count == 1);
  int row = 0;
  for (const auto& it : items) {
    if (it.logits.empty())
      continue;
    std::memcpy(
        it.logits.data(),
        im.logits_host.data() + static_cast<std::size_t>(row++) * config_.vocab,
        std::min<std::size_t>(it.logits.size(), config_.vocab) * sizeof(float));
  }
  return true;
}

std::size_t Executor::KvSnapshotBytes(std::uint32_t positions) const {
  return kRingBytes + (positions + kPageTokens - 1) / kPageTokens * kPageBytes;
}

bool Executor::SaveKv(const KvState& kv, std::uint32_t positions,
                      std::uint8_t* dst, std::string* error) {
  const std::size_t pages = (positions + kPageTokens - 1) / kPageTokens;
  if (pages > kv.pages_.size()) {
    if (error)
      *error = "snapshot past committed context";
    return false;
  }
  if (!Check(hipStreamSynchronize(impl_->stream), "snapshot sync", error) ||
      !Check(hipMemcpy(dst, kv.ring_, kRingBytes, hipMemcpyDeviceToHost),
             "ring download", error))
    return false;
  dst += kRingBytes;
  for (std::size_t p = 0; p < pages; ++p, dst += kPageBytes)
    if (!Check(
            hipMemcpy(dst,
                      reinterpret_cast<std::uint8_t*>(impl_->pool) +
                          static_cast<std::size_t>(kv.pages_[p]) * kPageBytes,
                      kPageBytes, hipMemcpyDeviceToHost),
            "page download", error))
      return false;
  return true;
}

bool Executor::RestoreKv(KvState& kv, std::uint32_t positions,
                         const std::uint8_t* src, std::string* error) {
  kv.Release();
  if (positions > 0 && !Reserve(kv, positions, error))
    return false;
  if (!Check(hipStreamSynchronize(impl_->stream), "restore sync", error) ||
      !Check(hipMemcpy(kv.ring_, src, kRingBytes, hipMemcpyHostToDevice),
             "ring upload", error))
    return false;
  src += kRingBytes;
  for (std::size_t p = 0; p < kv.pages_.size(); ++p, src += kPageBytes)
    if (!Check(
            hipMemcpy(reinterpret_cast<std::uint8_t*>(impl_->pool) +
                          static_cast<std::size_t>(kv.pages_[p]) * kPageBytes,
                      src, kPageBytes, hipMemcpyHostToDevice),
            "page upload", error))
      return false;
  return true;
}

}  // namespace gufo::models::mimo_v26_flash::rocm
