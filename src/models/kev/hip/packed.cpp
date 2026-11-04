#include "src/models/kev/hip/packed.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/kev/hip/kernels.hpp"
#include "src/models/kev/model.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::models::kev {
namespace {

namespace gh = ::gufo::hip;
namespace kh = ::gufo::models::kev::hip;

template<typename T>
T* DeviceAlloc(std::size_t count, std::size_t* total) {
  void* p = nullptr;
  HIP_CHECK(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)));
  *total += count * sizeof(T);
  return static_cast<T*>(p);
}

template<typename T>
void Upload(T* device, const std::vector<T>& host, hipStream_t stream) {
  if (!host.empty())
    HIP_CHECK(hipMemcpyAsync(device, host.data(), host.size() * sizeof(T),
                             hipMemcpyHostToDevice, stream));
}

/// First-fit allocator of contiguous KV slot ranges.
class RangeAllocator {
public:
  explicit RangeAllocator(std::uint32_t size) { free_.push_back({0, size}); }
  bool Allocate(std::uint32_t length, std::uint32_t* first) {
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second < length)
        continue;
      *first = it->first;
      it->first += length;
      it->second -= length;
      if (it->second == 0)
        free_.erase(it);
      return true;
    }
    return false;
  }
  void Release(std::uint32_t first, std::uint32_t length) {
    auto it = std::lower_bound(free_.begin(), free_.end(), first,
                               [](const auto& range, std::uint32_t value) {
                                 return range.first < value;
                               });
    it = free_.insert(it, {first, length});
    if (it + 1 != free_.end() && it->first + it->second == (it + 1)->first) {
      it->second += (it + 1)->second;
      free_.erase(it + 1);
    }
    if (it != free_.begin() &&
        (it - 1)->first + (it - 1)->second == it->first) {
      (it - 1)->second += it->second;
      free_.erase(it);
    }
  }

private:
  std::vector<std::pair<std::uint32_t, std::uint32_t>> free_;
};

}  // namespace

/// KEV_PROFILE=1: per-stage GPU time from event pairs, printed per pass.
class StageTimer {
public:
  explicit StageTimer(hipStream_t stream)
      : stream_(stream), enabled_(std::getenv("KEV_PROFILE") != nullptr) {}
  ~StageTimer() {
    for (auto& [name, start, stop] : marks_) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
    }
  }
  template<typename F>
  void operator()(const char* name, F&& launch) {
    if (!enabled_) {
      launch();
      return;
    }
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream_));
    launch();
    HIP_CHECK(hipEventRecord(stop, stream_));
    marks_.emplace_back(name, start, stop);
  }
  void Report(std::size_t tokens) {
    if (!enabled_)
      return;
    std::vector<std::pair<std::string, float>> totals;
    float all = 0.0F;
    for (auto& [name, start, stop] : marks_) {
      float ms = 0.0F;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      all += ms;
      auto it = std::find_if(totals.begin(), totals.end(),
                             [&](const auto& t) { return t.first == name; });
      if (it == totals.end())
        totals.emplace_back(name, ms);
      else
        it->second += ms;
    }
    std::sort(totals.begin(), totals.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::fprintf(stderr, "kev pass: %zu tokens, %.2f ms in stages\n", tokens,
                 all);
    for (const auto& [name, ms] : totals)
      std::fprintf(stderr, "  %-16s %8.2f ms %5.1f%%\n", name.c_str(), ms,
                   100.0F * ms / all);
  }

private:
  hipStream_t stream_;
  bool enabled_;
  std::vector<std::tuple<std::string, hipEvent_t, hipEvent_t>> marks_;
};

struct PackedExecutor::Impl {
  std::shared_ptr<const gh::QwenGpuModel> model;
  const PointerHead* head{nullptr};
  PackedOptions options;
  hipStream_t stream{nullptr};
  std::size_t reserved{0};
  std::function<void(std::uint32_t)> slice_hook;
  std::uint32_t slice_layers{4};

  // Shapes.
  std::uint32_t hidden{0}, ffn{0}, layers{0}, ssm_layers{0}, attn_layers{0};
  std::uint32_t k_heads{0}, v_heads{0}, channels{0}, inner{0};
  std::size_t pool_slots{0};

  // Pass buffers.
  std::uint32_t *ids{}, *pos{}, *kv_slot{}, *token_seq{}, *list_a{}, *list_b{},
      *readout_index{};
  std::uint8_t* phase{};
  kh::Sequence* seqs{};
  kh::AttentionTile* tiles{};
  float *h{}, *attn_out{}, *scratch{}, *qn{}, *kn{}, *raw{}, *alpha{}, *beta{};
  void* q8{};

  // Pointer head: final norm and [q; k] projection rows on the device.
  float *head_norm{}, *head_qk{}, *head_rows{}, *head_proj{};

  // Bank.
  float *conv_pool{}, *state_pool{};
  void* kv_pool{};

  struct Entry {
    std::vector<TokenId> tokens;
    std::int32_t rec_slot;
    std::uint32_t kv_first;
    std::uint64_t last_used;
    bool pinned;  ///< In use by the batch being scored.
    bool ready;   ///< Its bank slot holds the computed state.
  };
  std::vector<Entry> cache;
  std::vector<std::int32_t> free_rec;
  std::unique_ptr<RangeAllocator> kv;
  std::uint64_t clock{0};

  kh::RecurrentPools Pools() const {
    return {conv_pool, state_pool, ssm_layers, channels, v_heads};
  }

  void Evict(std::size_t index) {
    free_rec.push_back(cache[index].rec_slot);
    kv->Release(cache[index].kv_first,
                static_cast<std::uint32_t>(cache[index].tokens.size()));
    cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(index));
  }

  /// Finds or allocates the bank entry for a state. Returns the entry index
  /// and whether it must be computed, or -1 when nothing can be evicted.
  std::ptrdiff_t Acquire(const std::vector<TokenId>& state, bool* fresh,
                         PackedStats* stats) {
    for (std::size_t i = 0; i < cache.size(); ++i) {
      if (cache[i].tokens == state) {
        cache[i].pinned = true;
        cache[i].last_used = ++clock;
        *fresh = false;
        ++stats->cache_hits;
        return static_cast<std::ptrdiff_t>(i);
      }
    }
    const auto length = static_cast<std::uint32_t>(state.size());
    std::uint32_t first = 0;
    while (free_rec.empty() || !kv->Allocate(length, &first)) {
      std::ptrdiff_t lru = -1;
      for (std::size_t i = 0; i < cache.size(); ++i)
        if (!cache[i].pinned &&
            (lru < 0 || cache[i].last_used < cache[lru].last_used))
          lru = static_cast<std::ptrdiff_t>(i);
      if (lru < 0)
        return -1;
      Evict(static_cast<std::size_t>(lru));
    }
    const std::int32_t rec = free_rec.back();
    free_rec.pop_back();
    cache.push_back({state, rec, first, ++clock, true, false});
    *fresh = true;
    ++stats->cache_misses;
    return static_cast<std::ptrdiff_t>(cache.size() - 1);
  }

  void Gemm(const models::QwenTensorRef& w, float* y, std::size_t tokens,
            std::size_t m, std::size_t k, const void* act) const {
    gh::LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, act, y, tokens, m, k,
                                           stream);
  }

  struct Readout {
    std::size_t request;
    std::size_t question;
    std::uint32_t first;  ///< Index of its decide row in the gathered rows.
    std::uint32_t options;
  };

  struct Pass {
    std::vector<std::uint32_t> ids, pos, kv_slot, token_seq, list_a, list_b,
        readout_index;
    std::vector<std::uint8_t> phase;
    std::vector<kh::Sequence> seqs;
    std::vector<kh::AttentionTile> tiles;
    std::vector<Readout> readouts;
    std::vector<std::size_t> chunked_entries;

    [[nodiscard]] std::size_t tokens() const { return ids.size(); }
  };

  void AddTiles(Pass* pass, std::uint32_t offset, std::uint32_t length,
                std::uint32_t prefix_slot, std::uint32_t prefix_len,
                std::uint32_t own_slot) const {
    for (std::uint32_t s = 0; s < length; s += kh::kAttentionTileQueries)
      pass->tiles.push_back({offset + s,
                             std::min(kh::kAttentionTileQueries, length - s),
                             prefix_slot, prefix_len, own_slot, s});
  }

  void AddSequence(Pass* pass, std::span<const TokenId> tokens,
                   std::uint32_t first_position, std::int32_t init,
                   std::int32_t final_slot, std::uint32_t first_kv_slot,
                   bool is_state, std::uint32_t prefix_slot,
                   std::uint32_t prefix_len) const {
    const auto offset = static_cast<std::uint32_t>(pass->tokens());
    const auto length = static_cast<std::uint32_t>(tokens.size());
    const auto index = static_cast<std::uint32_t>(pass->seqs.size());
    pass->seqs.push_back({offset, length, init, final_slot});
    pass->phase.push_back(is_state ? 0 : 1);
    (is_state ? pass->list_a : pass->list_b).push_back(index);
    for (std::uint32_t i = 0; i < length; ++i) {
      pass->ids.push_back(tokens[i]);
      pass->pos.push_back(first_position + i);
      pass->kv_slot.push_back(first_kv_slot + i);
      pass->token_seq.push_back(index);
    }
    AddTiles(pass, offset, length, prefix_slot, prefix_len, first_kv_slot);
  }

  void Run(Pass& pass, std::vector<std::vector<std::vector<double>>>* results,
           PackedStats* stats);
};

void PackedExecutor::Impl::Run(
    Pass& pass, std::vector<std::vector<std::vector<double>>>* results,
    PackedStats* stats) {
  const auto started = std::chrono::steady_clock::now();
  const auto& weights = model->GetWeights();
  const auto& config = weights.config;
  const std::size_t tokens = pass.tokens();
  const float eps = head->eps;
  StageTimer timer(stream);
  Upload(ids, pass.ids, stream);
  Upload(pos, pass.pos, stream);
  Upload(kv_slot, pass.kv_slot, stream);
  Upload(token_seq, pass.token_seq, stream);
  Upload(phase, pass.phase, stream);
  Upload(seqs, pass.seqs, stream);
  Upload(list_a, pass.list_a, stream);
  Upload(list_b, pass.list_b, stream);
  Upload(tiles, pass.tiles, stream);
  Upload(readout_index, pass.readout_index, stream);

  timer("embed", [&] {
    gh::LaunchBatchedEmbeddingLookup(weights.token_embd.data,
                                     weights.token_embd.type, ids, h, tokens,
                                     hidden, stream);
  });
  const std::size_t wide = static_cast<std::size_t>(tokens) * 8192;
  for (std::uint32_t l = 0; l < layers; ++l) {
    if (slice_hook && l % slice_layers == 0) {
      // A slice boundary: let the owner pace Kev before queueing more work.
      if (l != 0)
        HIP_CHECK(hipStreamSynchronize(stream));
      slice_hook(l);
    }
    const auto& layer = weights.layers[l];
    timer("norm", [&] {
      gh::LaunchBatchedFusedRMSNormQuantizeQ8_1(
          h, nullptr, static_cast<const float*>(layer.attn_norm.data), nullptr,
          q8, tokens, hidden, eps, stream);
    });
    if (layer.is_full_attention) {
      const std::uint32_t attn_layer = l / config.full_attention_interval;
      float* qg = scratch;
      float* q = scratch + wide;
      float* ctx = q + tokens * 4096;
      float* gate = scratch + 2 * wide;
      float* k = raw;
      float* v = raw + tokens * kh::kKvWidth;
      timer("attn_proj",
            [&] { Gemm(layer.attn_q, qg, tokens, 2 * 4096, hidden, q8); });
      timer("attn_proj",
            [&] { Gemm(layer.attn_k, k, tokens, kh::kKvWidth, hidden, q8); });
      timer("attn_proj",
            [&] { Gemm(layer.attn_v, v, tokens, kh::kKvWidth, hidden, q8); });
      timer("attn_misc", [&] {
        gh::LaunchBatchedUnpackQG(qg, q, gate, tokens, kh::kQueryHeads,
                                  kh::kHeadDim, stream);
      });
      timer("qknorm_rope", [&] {
        kh::LaunchQkNormRopeKv(
            q, k, v, static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), pos, kv_slot,
            kv_pool, pool_slots, attn_layer, tokens, config.rotary_dim,
            config.rope_theta, eps, stream);
      });
      timer("attention", [&] {
        kh::LaunchAttention(q, gate, kv_pool, pool_slots, attn_layer, tiles,
                            static_cast<std::uint32_t>(pass.tiles.size()), ctx,
                            stream);
      });
      timer("attn_misc", [&] {
        gh::LaunchQuantizeActivationQ8_1FromFp32(ctx, q8, tokens, 4096, stream);
      });
      timer("attn_out", [&] {
        Gemm(layer.attn_output, attn_out, tokens, hidden, 4096, q8);
      });
    } else {
      const std::uint32_t ssm_layer = config.SsmLayerIndex(l);
      float* qkv = scratch;
      float* conv = scratch + wide;
      float* z = scratch + 2 * wide;
      timer("ssm_proj",
            [&] { Gemm(layer.attn_qkv, qkv, tokens, channels, hidden, q8); });
      timer("ssm_proj",
            [&] { Gemm(layer.attn_gate, z, tokens, inner, hidden, q8); });
      timer("ssm_ab",
            [&] { Gemm(layer.ssm_alpha, alpha, tokens, v_heads, hidden, q8); });
      timer("ssm_ab",
            [&] { Gemm(layer.ssm_beta, beta, tokens, v_heads, hidden, q8); });
      const auto* conv_w = static_cast<const float*>(layer.ssm_conv1d.data);
      const auto pools = Pools();
      timer("conv", [&] {
        kh::LaunchConv(qkv, conv_w, token_seq, seqs, phase, 0, pools, ssm_layer,
                       tokens, conv, stream);
      });
      timer("conv", [&] {
        kh::LaunchConvStore(qkv, seqs, list_a,
                            static_cast<std::uint32_t>(pass.list_a.size()),
                            pools, ssm_layer, stream);
      });
      timer("conv", [&] {
        kh::LaunchConv(qkv, conv_w, token_seq, seqs, phase, 1, pools, ssm_layer,
                       tokens, conv, stream);
      });
      timer("delta_prep", [&] {
        kh::LaunchDeltaPrep(conv, qn, kn, tokens, k_heads, v_heads, stream);
      });
      const auto* a = static_cast<const float*>(layer.ssm_a.data);
      const auto* dt = static_cast<const float*>(layer.ssm_dt.data);
      timer("delta_rule", [&] {
        kh::LaunchDeltaRule(conv, qn, kn, alpha, beta, a, dt, seqs, list_a,
                            static_cast<std::uint32_t>(pass.list_a.size()),
                            pools, ssm_layer, k_heads, raw, stream);
      });
      timer("delta_rule", [&] {
        kh::LaunchDeltaRule(conv, qn, kn, alpha, beta, a, dt, seqs, list_b,
                            static_cast<std::uint32_t>(pass.list_b.size()),
                            pools, ssm_layer, k_heads, raw, stream);
      });
      timer("ssm_epilogue", [&] {
        gh::LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
            raw, static_cast<const float*>(layer.ssm_norm.data), z, q8, tokens,
            inner, v_heads, kh::kSsmDim, stream);
      });
      timer("ssm_out",
            [&] { Gemm(layer.ssm_out, attn_out, tokens, hidden, inner, q8); });
    }
    timer("norm", [&] {
      gh::LaunchBatchedFusedRMSNormQuantizeQ8_1(
          h, attn_out, static_cast<const float*>(layer.ffn_norm.data), h, q8,
          tokens, hidden, eps, stream);
    });
    float* gate = scratch;
    float* up = scratch + tokens * ffn;
    const void* down_act = q8;
    if (gh::IsFusedSwiGluGemmEpilogueSupported(layer.ffn_gate.type, tokens, ffn,
                                               tokens * ffn * sizeof(float))) {
      // The up projection's epilogue writes the Q8_1 SwiGLU into `up`.
      timer("ffn_gate_up", [&] {
        gh::LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data, q8,
            gate, up, tokens, ffn, hidden, stream);
      });
      down_act = up;
    } else {
      timer("ffn_gate_up", [&] {
        gh::LaunchBatchedDualQuantGEMMPreQuantized(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data, q8,
            gate, up, tokens, ffn, hidden, stream);
      });
      timer("ffn_gate_up", [&] {
        gh::LaunchBatchedFusedSwiGLUQuantizeQ8_1(gate, up, q8, tokens, ffn,
                                                 stream);
      });
    }
    timer("ffn_down", [&] {
      Gemm(layer.ffn_down, attn_out, tokens, hidden, ffn, down_act);
    });
    timer("residual", [&] {
      gh::LaunchBatchedResidualAdd(h, attn_out, h, tokens, hidden, stream);
    });
  }

  const auto rows = static_cast<std::uint32_t>(pass.readout_index.size());
  const std::size_t dim = head->dim;
  std::vector<float> proj(static_cast<std::size_t>(rows) * 2 * dim);
  timer("head", [&] {
    kh::LaunchGatherRows(h, readout_index, rows, hidden, head_rows, stream);
    gh::LaunchBatchedRMSNorm(head_rows, head_norm, attn_out, nullptr, rows,
                             hidden, eps, stream);
    gh::LaunchBatchedGEMM(head_qk, false, attn_out, head_proj, rows, 2 * dim,
                          hidden, stream);
  });
  if (rows != 0)
    HIP_CHECK(hipMemcpyAsync(proj.data(), head_proj,
                             proj.size() * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  timer.Report(tokens);
  const auto gpu_done = std::chrono::steady_clock::now();
  const double scale =
      1.0 / std::sqrt(static_cast<double>(dim)) / head->temperature;
  for (const Readout& r : pass.readouts) {
    const float* q = proj.data() + static_cast<std::size_t>(r.first) * 2 * dim;
    std::vector<double> z(r.options);
    for (std::uint32_t o = 0; o < r.options; ++o) {
      const float* k = proj.data() +
                       static_cast<std::size_t>(r.first + 1 + o) * 2 * dim +
                       dim;
      double dot = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        dot += (static_cast<double>(q[d]) + head->q_bias[d]) *
               (static_cast<double>(k[d]) + head->k_bias[d]);
      z[o] = dot * scale;
    }
    const double top = *std::max_element(z.begin(), z.end());
    double total = 0.0;
    for (double& v : z) {
      v = std::exp(v - top);
      total += v;
    }
    for (double& v : z)
      v /= total;
    (*results)[r.request][r.question] = std::move(z);
  }
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  stats->gpu_ms += ms(started, gpu_done);
  stats->head_ms += ms(gpu_done, std::chrono::steady_clock::now());
}

PackedExecutor::PackedExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)), options_(impl_->options) {}

PackedExecutor::~PackedExecutor() {
  if (impl_ == nullptr)
    return;
  for (void* p : {static_cast<void*>(impl_->ids),
                  static_cast<void*>(impl_->pos),
                  static_cast<void*>(impl_->kv_slot),
                  static_cast<void*>(impl_->token_seq),
                  static_cast<void*>(impl_->list_a),
                  static_cast<void*>(impl_->list_b),
                  static_cast<void*>(impl_->readout_index),
                  static_cast<void*>(impl_->phase),
                  static_cast<void*>(impl_->seqs),
                  static_cast<void*>(impl_->tiles),
                  static_cast<void*>(impl_->h),
                  static_cast<void*>(impl_->attn_out),
                  static_cast<void*>(impl_->scratch),
                  static_cast<void*>(impl_->qn),
                  static_cast<void*>(impl_->kn),
                  static_cast<void*>(impl_->raw),
                  static_cast<void*>(impl_->alpha),
                  static_cast<void*>(impl_->beta),
                  impl_->q8,
                  static_cast<void*>(impl_->conv_pool),
                  static_cast<void*>(impl_->state_pool),
                  impl_->kv_pool})
    if (p != nullptr)
      (void)hipFree(p);
  if (impl_->stream != nullptr)
    (void)hipStreamDestroy(impl_->stream);
}

std::size_t PackedExecutor::ReservedBytes() const noexcept {
  return impl_->reserved;
}

void PackedExecutor::SetSliceHook(std::function<void(std::uint32_t)> hook,
                                  std::uint32_t layers) {
  impl_->slice_hook = std::move(hook);
  impl_->slice_layers = std::max<std::uint32_t>(1, layers);
}

std::unique_ptr<PackedExecutor> PackedExecutor::Create(
    std::shared_ptr<const ::gufo::hip::QwenGpuModel> model,
    const PointerHead& head, const PackedOptions& options, std::string* error) {
  const auto fail =
      [&](std::string message) -> std::unique_ptr<PackedExecutor> {
    if (error != nullptr)
      *error = std::move(message);
    return nullptr;
  };
  const auto& weights = model->GetWeights();
  const auto& c = weights.config;
  if (c.num_attention_heads != kh::kQueryHeads ||
      c.num_key_value_heads != kh::kKvHeads || c.head_dim != kh::kHeadDim ||
      c.ssm_state_size != kh::kSsmDim || c.ssm_conv_kernel != kh::kConvTaps ||
      c.AttentionSize() != 4096 || c.SsmQkvSize() != 8192)
    return fail("the packed Kev pass is built for the Qwen3.5-4B shapes");
  for (const auto& layer : weights.layers) {
    for (const auto* w :
         {&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down, &layer.attn_q,
          &layer.attn_k, &layer.attn_v, &layer.attn_output, &layer.attn_qkv,
          &layer.attn_gate, &layer.ssm_alpha, &layer.ssm_beta, &layer.ssm_out})
      if (!w->empty() && w->type != core::GgmlType::kQ8_0)
        return fail("the packed Kev pass needs Q8_0 projections");
  }
  if (!gh::IsFusedRMSNormQuantizeQ8_1Supported(c.hidden_size) ||
      !gh::IsFusedSSMEpilogueQuantizeQ8_1Supported(kh::kSsmDim,
                                                   c.ssm_inner_size))
    return fail("the fused Q8_1 epilogues do not support these shapes");

  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.model = std::move(model);
  m.head = &head;
  m.options = options;
  m.hidden = c.hidden_size;
  m.ffn = c.intermediate_size;
  m.layers = c.num_layers;
  m.ssm_layers = c.SsmLayerCount();
  m.attn_layers = c.FullAttentionLayerCount();
  m.k_heads = c.ssm_group_count;
  m.v_heads = c.ssm_time_step_rank;
  m.channels = c.SsmQkvSize();
  m.inner = c.ssm_inner_size;
  m.pool_slots =
      static_cast<std::size_t>(options.kv_slots) + options.max_tokens;
  HIP_CHECK(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking));

  const std::size_t t = options.max_tokens;
  std::size_t& total = m.reserved;
  m.ids = DeviceAlloc<std::uint32_t>(t, &total);
  m.pos = DeviceAlloc<std::uint32_t>(t, &total);
  m.kv_slot = DeviceAlloc<std::uint32_t>(t, &total);
  m.token_seq = DeviceAlloc<std::uint32_t>(t, &total);
  m.list_a = DeviceAlloc<std::uint32_t>(t, &total);
  m.list_b = DeviceAlloc<std::uint32_t>(t, &total);
  m.readout_index = DeviceAlloc<std::uint32_t>(t, &total);
  m.phase = DeviceAlloc<std::uint8_t>(t, &total);
  m.seqs = DeviceAlloc<kh::Sequence>(t, &total);
  m.tiles = DeviceAlloc<kh::AttentionTile>(t, &total);
  m.h = DeviceAlloc<float>(t * m.hidden, &total);
  m.attn_out = DeviceAlloc<float>(t * m.hidden, &total);
  // Scratch holds, per phase: qkv | conv | z, or qg | q ctx | gate, or
  // ffn gate | ffn up (2 * 9216 <= 2 * 8192 + 4096).
  m.scratch = DeviceAlloc<float>(
      t * std::max<std::size_t>(2 * 8192 + 4096, 2 * m.ffn), &total);
  m.qn = DeviceAlloc<float>(t * m.k_heads * kh::kSsmDim, &total);
  m.kn = DeviceAlloc<float>(t * m.k_heads * kh::kSsmDim, &total);
  m.raw = DeviceAlloc<float>(t * m.inner, &total);
  m.alpha = DeviceAlloc<float>(t * m.v_heads, &total);
  m.beta = DeviceAlloc<float>(t * m.v_heads, &total);
  m.q8 = DeviceAlloc<std::uint8_t>(
      gh::QuantizedActivationBytes(t, std::max<std::size_t>(m.ffn, 8192)),
      &total);
  m.conv_pool =
      DeviceAlloc<float>(static_cast<std::size_t>(options.state_slots) *
                             m.ssm_layers * m.channels * kh::kConvTaps,
                         &total);
  m.state_pool = DeviceAlloc<float>(
      static_cast<std::size_t>(options.state_slots) * m.ssm_layers * m.v_heads *
          kh::kSsmDim * kh::kSsmDim,
      &total);
  m.kv_pool = DeviceAlloc<std::uint16_t>(
      static_cast<std::size_t>(m.attn_layers) * m.pool_slots * kh::kSlotHalves,
      &total);
  m.head_norm = DeviceAlloc<float>(m.hidden, &total);
  m.head_qk = DeviceAlloc<float>(2 * head.dim * m.hidden, &total);
  m.head_rows = DeviceAlloc<float>(t * m.hidden, &total);
  m.head_proj = DeviceAlloc<float>(t * 2 * head.dim, &total);
  HIP_CHECK(hipMemcpy(m.head_norm, head.final_norm.data(),
                      m.hidden * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(m.head_qk, head.q_weight.data(),
                      head.dim * m.hidden * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(m.head_qk + head.dim * m.hidden, head.k_weight.data(),
                      head.dim * m.hidden * sizeof(float),
                      hipMemcpyHostToDevice));
  for (std::int32_t s = static_cast<std::int32_t>(options.state_slots) - 1;
       s >= 0; --s)
    m.free_rec.push_back(s);
  m.kv = std::make_unique<RangeAllocator>(options.kv_slots);
  return std::unique_ptr<PackedExecutor>(new PackedExecutor(std::move(impl)));
}

std::vector<std::vector<std::vector<double>>> PackedExecutor::Score(
    std::span<const Encoding* const> batch) {
  Impl& m = *impl_;
  std::vector<std::vector<std::vector<double>>> results(batch.size());
  for (std::size_t r = 0; r < batch.size(); ++r)
    results[r].resize(batch[r]->rows.size());
  const std::uint32_t budget = options_.max_tokens;
  const auto row_base = options_.kv_slots;

  // A failed pass leaves its states uncomputed: drop them so no later batch
  // reads an empty slot as a cache hit, and release every pin.
  const auto recover = [&] {
    for (std::size_t i = m.cache.size(); i-- > 0;)
      if (!m.cache[i].ready)
        m.Evict(i);
    for (auto& entry : m.cache)
      entry.pinned = false;
  };
  try {
    std::size_t next = 0;
    while (next < batch.size()) {
      // Resolve a wave of requests whose states fit the bank at once.
      struct Work {
        std::size_t request;
        std::ptrdiff_t entry;
        bool fresh;
      };
      std::vector<Work> wave;
      while (next < batch.size()) {
        const Encoding& e = *batch[next];
        if (e.state.size() > options_.kv_slots)
          throw std::length_error("Kev state exceeds the bank");
        for (const Row& row : e.rows)
          if (row.ids.size() > budget)
            throw std::length_error("Kev question row exceeds the pass");
        bool fresh = false;
        const std::ptrdiff_t entry = m.Acquire(e.state, &fresh, &stats_);
        if (entry < 0)
          break;
        wave.push_back({next, entry, fresh});
        ++next;
      }
      if (wave.empty())
        throw std::runtime_error("Kev bank cannot hold one state");

      Impl::Pass pass;
      std::vector<std::size_t> pass_chunks;
      const auto flush = [&] {
        if (pass.tokens() == 0)
          return;
        ++stats_.passes;
        stats_.pass_tokens += pass.tokens();
        m.Run(pass, &results, &stats_);
        pass = Impl::Pass{};
        pass_chunks.clear();
      };
      for (const Work& w : wave) {
        const Encoding& e = *batch[w.request];
        // Entries shift when a later Acquire evicts, so resolve by tokens.
        const auto entry_it = std::find_if(
            m.cache.begin(), m.cache.end(),
            [&](const Impl::Entry& x) { return x.tokens == e.state; });
        const auto entry = static_cast<std::size_t>(entry_it - m.cache.begin());
        const Impl::Entry& bank = m.cache[entry];
        const auto state_len = static_cast<std::uint32_t>(e.state.size());
        if (w.fresh) {
          for (std::uint32_t b = 0; b < state_len; b += budget) {
            const std::uint32_t len = std::min(budget, state_len - b);
            const bool repeat =
                std::find(pass_chunks.begin(), pass_chunks.end(), entry) !=
                pass_chunks.end();
            if (pass.tokens() + len > budget || repeat)
              flush();
            m.AddSequence(&pass, std::span(e.state).subspan(b, len), b,
                          b == 0 ? kh::kNoSlot : bank.rec_slot, bank.rec_slot,
                          bank.kv_first + b, true, bank.kv_first, b);
            pass_chunks.push_back(entry);
          }
        }
        for (std::size_t q = 0; q < e.rows.size(); ++q) {
          const Row& row = e.rows[q];
          if (pass.tokens() + row.ids.size() > budget)
            flush();
          const auto offset = static_cast<std::uint32_t>(pass.tokens());
          m.AddSequence(&pass, row.ids, state_len, bank.rec_slot, kh::kNoSlot,
                        row_base + offset, false, bank.kv_first, state_len);
          pass.readouts.push_back(
              {w.request, q,
               static_cast<std::uint32_t>(pass.readout_index.size()),
               static_cast<std::uint32_t>(row.option_ends.size())});
          pass.readout_index.push_back(offset + row.decide);
          for (const std::uint32_t end : row.option_ends)
            pass.readout_index.push_back(offset + end);
        }
      }
      flush();
      for (auto& entry : m.cache) {
        entry.ready = true;
        entry.pinned = false;
      }
    }
  } catch (...) {
    recover();
    throw;
  }
  stats_.cached_states = static_cast<std::uint32_t>(m.cache.size());
  return results;
}

}  // namespace gufo::models::kev

#else

namespace gufo::models::kev {
struct PackedExecutor::Impl {};
PackedExecutor::PackedExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PackedExecutor::~PackedExecutor() = default;
std::size_t PackedExecutor::ReservedBytes() const noexcept {
  return 0;
}
void PackedExecutor::SetSliceHook(std::function<void(std::uint32_t)>,
                                  std::uint32_t) {}
std::unique_ptr<PackedExecutor> PackedExecutor::Create(
    std::shared_ptr<const ::gufo::hip::QwenGpuModel>, const PointerHead&,
    const PackedOptions&, std::string* error) {
  if (error != nullptr)
    *error = "Kev needs a HIP build";
  return nullptr;
}
std::vector<std::vector<std::vector<double>>> PackedExecutor::Score(
    std::span<const Encoding* const>) {
  throw std::logic_error("Kev needs a HIP build");
}
}  // namespace gufo::models::kev

#endif
