#include <algorithm>
#include <array>
#include <stdexcept>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/lfm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/spike.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kBatchSessions = Executor::kMaxBatchSessions;
constexpr std::uint32_t kDecodeRows = 8;
constexpr std::uint32_t kBatchRows = kBatchSessions * kDecodeRows;

// Batched rows from which one tiled WMMA GEMM (each weight read once) beats
// the decode GEMVs, which re-read the weights per row chunk. gfx1151, real
// Q8_0 weights, qwen38_flash_next_batch_dense_bench (2026-09-26), us at 64
// rows GEMV -> tiled: ssm_in 16384x2560 583 -> 296, ssm_out 2560x6144
// 350 -> 107, HC up 10240x320 180 -> 39, vocabulary head 9256 -> 3617. The
// long-K 320x10240 HC down GEMV only loses from about 160 rows.
std::uint32_t TiledFrom(const DeviceTensor& w) {
  // ponytail: K-quant (MMQ tiles vs. decode GEMV) reuses the Q8_0
  // crossovers; measure qwen38_flash_next_batch_dense_bench if they matter.
  if (w.type != core::GgmlType::kQ8_0 && !KQuant(w.type))
    return w.rows >= 256 ? 32 : 64;  // hipBLASLt versus grouped SmallGemm
  if (w.rows <= 512 && w.cols >= 8192)
    return 160;
  if (w.rows < 2048)
    return 48;
  return w.cols == 2560 && w.rows >= 8192 ? 24 : 16;
}

// F16 activations win for short K and, up to 64 rows, for K = 6144; the
// int8 tiles elsewhere (for example 96-384 rows of the vocabulary head).
bool F16Tiles(const DeviceTensor& w, std::uint32_t rows) {
  return w.rows >= 2048 && (w.cols <= 640 || (w.cols >= 6144 && rows <= 64));
}

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool Check(hipError_t status, std::string* error) {
  return status == hipSuccess || Fail(error, std::string("batched forward: ") +
                                                 hipGetErrorString(status));
}

// A cancelled row stops touching its private state. Shared projections retain
// their row layout so cancellation cannot change a peer's arithmetic/order.
template<class Item>
bool AnyActive(std::span<const Item> items) {
  bool active = false;
  for (const auto& item : items)
    active |= item.session->CheckCancellation(nullptr);
  return active;
}

WeightType EmbeddingType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next embedding format");
  }
}

}  // namespace

Executor::Scratch Executor::RowScratch(const Scratch& b,
                                       std::uint32_t offset) const {
  const Config& c = config();
  const std::size_t r = offset;
  auto s = b;
  s.tokens += r;
  s.res += r * c.HcDim();
  s.xn += r * c.HcDim();
  s.xn_half += r * c.HcDim();
  s.lo += r * c.hc_low_rank;
  s.hc_gate += r * c.HcDim();
  s.mixed += r * c.hidden_size;
  s.inject += r * c.hc_count * HcInjectParts(c.hidden_size);
  s.block_out += r * c.hidden_size;
  if (s.steer != nullptr)
    s.steer += r;
  if (s.route_keep != nullptr)
    s.route_keep += r;
  s.qkv += r * c.SsmConvChannels();
  s.z += r * c.SsmValueDim();
  s.qkvz += r * (c.SsmConvChannels() + c.SsmValueDim());
  s.alpha_beta += r * 2 * c.ssm_num_v_heads;
  s.qn += r * c.SsmKeyDim();
  s.kn += r * c.SsmKeyDim();
  s.gdn_raw += r * c.SsmValueDim();
  s.gdn_out += r * c.SsmValueDim();
  s.qg += r * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim());
  s.q += r * c.AttentionQDim();
  s.attn_gate += r * c.AttentionQDim();
  s.k += r * c.AttentionKvDim();
  s.v += r * c.AttentionKvDim();
  s.iq += r * c.indexer_heads * c.indexer_head_dim;
  s.ik += r * c.indexer_head_dim;
  s.mask += r * mask_words_;
  s.ctx += r * c.AttentionQDim();
  if (c.ple_layer >= 0) {
    s.ple_emb += r * c.PleEmbeddingDim();
    s.ple_key += r * c.HcDim();
    s.ple_value += r * c.hidden_size;
    s.ple_query += r * c.HcDim();
    s.ple_gated += r * c.HcDim();
    s.ple_norm += r * c.HcDim();
    s.ple_conv += r * c.HcDim();
  }
  s.router += r * (c.num_experts + 1);
  s.ids += r * c.num_experts_used;
  s.weights += r * c.num_experts_used;
  s.gate_e += r * c.num_experts_used * c.expert_ff;
  s.up_e += r * c.num_experts_used * c.expert_ff;
  s.down_e += r * c.num_experts_used * c.hidden_size;
  s.shexp_gate += r * c.shared_expert_ff;
  s.shexp_up += r * c.shared_expert_ff;
  s.shexp_out += r * c.hidden_size;
  s.shexp_half += r * c.shared_expert_ff;
  if (has_mtp()) {
    s.mtp_h += r * c.HcDim();
    s.mtp_embd += r * c.hidden_size;
    // This allocation aliases qkvz, but has a different row stride.
    s.mtp_eproj += r * c.hidden_size;
    s.mtp_res += r * c.HcDim();
  }
  // Activation staging, attention partials and convolution staging have
  // one consumer at a time on stream_. They remain at the allocation base.
  return s;
}

void Executor::UseScratch(const Scratch& scratch) const {
  s_ = scratch;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  xn_half_ = false;
  moe_pending_ = false;
}

bool Executor::AllocateBatch(std::string* error) const {
  // Pinned staging for the widest batch; logit rows are sized per batch.
  if ((batch_controls_ == nullptr &&
       !Check(hipHostMalloc(&batch_controls_,
                            kBatchSessions * sizeof(Session::Control)),
              error)) ||
      (batch_argmax_host_ == nullptr &&
       !Check(hipHostMalloc(&batch_argmax_host_,
                            kBatchRows * sizeof(ArgmaxCandidate)),
              error)))
    return false;
  if (batch_candidates_host_ == nullptr) {
    if (!Check(hipHostMalloc(&batch_candidates_host_,
                             kBatchSessions * sizeof(MtpCandidateLogits)),
               error))
      return false;
    for (std::uint32_t i = 0; i < kBatchSessions; ++i)
      std::construct_at(batch_candidates_host_ + i);
  }
  return true;
}

bool Executor::HeadLogits(std::uint32_t rows, bool keep_moe_scratch,
                          std::string* error) const {
  // The MoE down-projection scratch is idle from the final combine until the
  // next forward and holds about 211 logit rows at the 2048-token prefill
  // chunk. Wider verification batches grow a private buffer.
  const std::size_t vocab = config().vocab_size;
  if (!keep_moe_scratch && std::size_t{rows} * vocab <= logit_alias_floats_) {
    batch_logits_ = s_.down_e;
    return true;
  }
  if (rows > batch_logits_owned_rows_) {
    (void)hipFree(batch_logits_owned_);
    batch_logits_owned_ = nullptr;
    batch_logits_owned_rows_ = 0;
    if (!Check(hipMalloc(&batch_logits_owned_, rows * vocab * sizeof(float)),
               error))
      return false;
    batch_logits_owned_rows_ = rows;
  }
  batch_logits_ = batch_logits_owned_;
  return true;
}

bool Executor::BatchHead(const float* mixed, std::uint32_t rows, bool argmax,
                         std::string* error, bool keep_moe_scratch,
                         const DeviceTensor* head) const {
  if (!HeadLogits(rows, keep_moe_scratch, error) ||
      !DenseBatch(head != nullptr ? *head : model_->output(), mixed,
                  batch_logits_, rows, error))
    return false;
  return !argmax || BatchArgmax(rows, error);
}

bool Executor::BatchArgmax(std::uint32_t rows, std::string* error) const {
  const std::size_t vocab = config().vocab_size;
  // Greedy verification reduces every row now, with GreedyMtpPredictions'
  // kernels; requests then read their predictions without another sync.
  auto* ids = reinterpret_cast<std::uint32_t*>(s_.up_e);
  auto* best = reinterpret_cast<ArgmaxCandidate*>(s_.gate_e);
  gufo::hip::LaunchBatchedGPUArgmax(batch_logits_, ids, rows, vocab,
                                    {s_.qkvz, std::size_t{rows} * 256},
                                    stream_);
  GatherArgmaxCandidates(batch_logits_, ids, best, rows, config().vocab_size,
                         stream_);
  if (!Check(hipMemcpyAsync(batch_argmax_host_, best,
                            rows * sizeof(ArgmaxCandidate),
                            hipMemcpyDeviceToHost, stream_),
             error))
    return false;
  batch_argmax_rows_ = rows;
  return true;
}

bool Executor::HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                          float* mixed, float* inject, std::uint32_t rows,
                          std::string* error) const {
  const auto& c = config();
  xn_half_ = false;
  if (!c.HyperConnections())
    return PlainNorm(m, res, mixed, rows);
  if (!normed)
    RmsNormRows(res, m.norm.f32(), s_.xn, rows, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  if (!DenseBatch(m.down, s_.xn, s_.lo, rows, error))
    return false;
  SiluScale(s_.lo, 1.0F / static_cast<float>(c.hc_count),
            std::size_t{rows} * c.hc_low_rank, stream_);
  if (!DenseBatch(m.up, s_.lo, s_.hc_gate, rows, error))
    return false;
  const bool fused_inject = inject != nullptr && !m.inject.empty() &&
                            m.inject.type == core::GgmlType::kF32;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  // Every request contains at most eight decode rows. Keep its scalar
  // epilogue/reduction even when the concatenated batch exceeds eight.
  HcMixEpilogue(s_.xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                mixed, inject, rows, c.hidden_size, c.hc_count, stream_);
  inject_parts_ = fused_inject ? HcInjectParts(c.hidden_size) : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    if (!DenseBatch(m.inject, s_.xn, inject, rows, error))
      return false;
  }
  return true;
}

void Executor::CombineBatch(float* res, const float* gamma, std::uint32_t rows,
                            std::int32_t moe_layer) const {
  const auto& c = config();
  Steer(rows, moe_layer);
  if (!c.HyperConnections()) {
    MtpAddEmbedding(s_.block_out, res, rows, c.hidden_size, 1, stream_);
    return;
  }
  HcCombine(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn, rows,
            c.hidden_size, c.hc_count, c.rms_eps, stream_, true);
}

bool Executor::MtpForwardBatch(std::span<const MtpBatchItem> items,
                               std::string* error) const {
  const Config& c = config();
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "invalid MTP body batch");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::array<std::uint32_t, kBatchSessions> final_rows{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto n = item.tokens.size();
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || n == 0 || n > kDecodeRows ||
        (item.hidden_row < 0 && n != 1) ||
        (item.hidden_row >= 0 &&
         static_cast<std::uint32_t>(item.hidden_row) + n >
             options_.max_speculative) ||
        item.session->mtp_.position + n > item.session->max_context_) {
      return Fail(error, "invalid MTP body request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session)
        return Fail(error, "MTP body requests must be independent");
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size)
        return Fail(error, "MTP token out of range");
    }
    offsets[i] = rows;
    rows += n;
    final_rows[i] = rows - 1;
  }
  if (rows > options_.max_batch)
    return Fail(error, "MTP body batch exceeds executor capacity");
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    const auto& item = items.front();
    return MtpForward(*item.session, item.tokens, item.hidden_row, {}, error);
  }
  for (const auto& item : items) {
    if (!RequireResident(*item.session, error) ||
        !Reserve(*item.session,
                 item.session->mtp_.position +
                     static_cast<std::uint32_t>(item.tokens.size()),
                 error))
      return false;
  }
  if (!AllocateBatch(error))
    return false;
  // Rows of adapters that project the embedding or the draft block's
  // outputs (SteersMtp) are steered; every other row keeps 0.
  bool steer = false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& session = *item.session;
    batch_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, item.hidden_row,
                          session.mtp_.blocks};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
    steer |= SteersMtp(session.adapter_);
  }
  for (std::size_t i = 0; steer && i < items.size(); ++i) {
    const auto adapter = items[i].session->adapter_;
    std::fill_n(steer_host_ + offsets[i], items[i].tokens.size(),
                SteersMtp(adapter) ? adapter : std::uint8_t{0});
  }
  const Scratch base = s_;
  const auto& l = model_->mtp();
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    const SteerScope steer_scope(*this, steer, steer_writer::kMtp);
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error) ||
        (steer && !Check(hipMemcpyAsync(base.steer, steer_host_, rows,
                                        hipMemcpyHostToDevice, stream_),
                         error)))
      return false;
    EmbedTokens(model_->token_embd().data,
                static_cast<WeightType>(model_->token_embd().type), base.tokens,
                base.mtp_embd, rows, c.hidden_size, 1, stream_);
    SteerWriter(base.mtp_embd, rows, 1, steer_writer::kEmbedding);
    RmsNormRows(base.mtp_embd, l.nextn_enorm.f32(), base.mtp_embd, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      if (!Check(hipMemcpyAsync(session.control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error))
        return false;
      MtpHidden(session.mtp_.target_hidden, session.mtp_.h,
                &session.control_->hidden_row, s_.mtp_h, n, c.HcDim(), stream_);
    }
    RmsNormRows(base.mtp_h, l.nextn_hnorm.f32(), base.mtp_h, rows, c.HcDim(), 1,
                c.rms_eps, stream_);
    // Keep each session's vector/matrix arithmetic while sharing weight
    // reads among adjacent vector-sized inputs. Project the embedding once.
    // Wide batches project every row with one tiled GEMM instead.
    const bool tiled = rows >= TiledFrom(l.nextn_fc_embedding);
    if (tiled) {
      UseScratch(base);
      if (!DenseBatch(l.nextn_fc_embedding, base.mtp_embd, base.mtp_eproj, rows,
                      error) ||
          !DenseBatch(l.nextn_fc_hidden, base.mtp_h, base.mtp_res,
                      rows * c.hc_count, error))
        return false;
      MtpAddEmbedding(base.mtp_eproj, base.mtp_res, rows, c.hidden_size,
                      c.hc_count, stream_);
    }
    for (std::size_t i = 0; !tiled && i < items.size();) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      auto end = i + 1;
      if (n * c.hc_count <= kDecodeRows) {
        while (end < items.size() &&
               items[end].tokens.size() * c.hc_count <= kDecodeRows)
          ++end;
      }
      const auto end_row = end == items.size() ? rows : offsets[end];
      const auto count = end_row - offsets[i];
      UseScratch(RowScratch(base, offsets[i]));
      if (n * c.hc_count <= kDecodeRows) {
        if (!DenseBatch(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, count,
                        error) ||
            !DenseBatch(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res,
                        count * c.hc_count, error))
          return false;
      } else if (!Dense(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n,
                        error) ||
                 !Dense(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
                        error)) {
        return false;
      }
      MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, count, c.hidden_size,
                      c.hc_count, stream_);
      i = end;
    }
    UseScratch(base);
    if (!HcMixBatch(l.hc_attn, base.mtp_res, false, base.mixed, base.inject,
                    rows, error))
      return false;
    if (!l.attn_qkv.empty()) {
      if (!DenseBatch(l.attn_qkv, base.mixed, base.qg, rows, error))
        return false;
    } else if (!DenseBatch(l.attn_q, base.mixed, base.qg, rows, error) ||
               !DenseBatch(l.attn_k, base.mixed, base.k, rows, error) ||
               !DenseBatch(l.attn_v, base.mixed, base.v, rows, error)) {
      return false;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      auto view = RowScratch(base, offsets[i]);
      if (l.attn_qkv.empty())
        view.qg = base.qg + std::size_t{offsets[i]} * 2 * c.AttentionQDim();
      UseScratch(view);
      if (!session.CheckCancellation(nullptr)) {
        if (!Check(hipMemsetAsync(
                       s_.ctx, 0,
                       std::size_t{n} * c.AttentionQDim() * sizeof(float),
                       stream_),
                   error))
          return false;
        continue;
      }
      ++session.mutation_epoch_;
      Session::AttentionState attention;
      attention.rope = session.vision_input_.rope();
      attention.kv = session.mtp_.kv;
      attention.index_k = session.mtp_.index_k;
      attention.block_k = session.mtp_.block_k;
      const bool sparse = session.mtp_.position + n > c.indexer_top_k;
      const auto complete = (session.mtp_.position + n) / c.compress_ratio;
      const auto pool = sparse ? complete - session.mtp_.blocks : 0;
      if (!Attention(l, attention, s_.mixed, s_.block_out, n,
                     &session.control_->mtp_position,
                     &session.control_->mtp_blocks, session.mtp_.position, pool,
                     session.max_context_, sparse, error, false, true, false))
        return false;
    }
    // Attention retains every catch-up KV row. Only each final residual
    // contributes to the next proposal, so compact those rows into idle
    // trunk scratch before the output projection and FFN. A one-row tail
    // in the original eight-row projection has different scalar rounding.
    const bool compact_tail = rows > kDecodeRows && rows % kDecodeRows != 1;
    Scratch tail = base;
    if (compact_tail) {
      // Rows move down in place (final_rows[i] >= i, ascending), never onto
      // a later source. The inject stays in its own buffer: mtp_h aliases
      // xn, which the next combine writes while other blocks read inject.
      tail.mtp_res = base.res;
      tail.ctx = base.qg;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto r = final_rows[i];
        const auto copy = [&](float* dst, const float* src, std::size_t width) {
          return (dst == src && r == i) ||
                 Check(hipMemcpyAsync(dst + i * width, src + r * width,
                                      width * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream_),
                       error);
        };
        if (!copy(tail.mtp_res, base.mtp_res, c.HcDim()) ||
            !copy(tail.ctx, base.ctx, c.AttentionQDim()) ||
            !copy(tail.inject, base.inject, c.hc_count * inject_parts_) ||
            (steer && r != i &&
             !Check(hipMemcpyAsync(base.steer + i, base.steer + r, 1,
                                   hipMemcpyDeviceToDevice, stream_),
                    error)))
          return false;
      }
    }
    const auto tail_rows =
        compact_tail ? static_cast<std::uint32_t>(items.size()) : rows;
    UseScratch(tail);
    if (!DenseBatch(l.attn_out, tail.ctx, tail.block_out, tail_rows, error))
      return false;
    CombineBatch(tail.mtp_res, l.hc_ffn.norm.f32(), tail_rows);
    if (!HcMixBatch(l.hc_ffn, tail.mtp_res, true, tail.mixed, tail.inject,
                    tail_rows, error))
      return false;
    if (!AnyActive(items))
      return true;
    if (!MoeBatch(l, tail.mixed, tail.block_out, tail_rows, error))
      return false;
    CombineBatch(tail.mtp_res, nullptr, tail_rows);
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (!Check(
              hipMemcpyAsync(
                  items[i].session->mtp_.h,
                  tail.mtp_res + (compact_tail ? i : final_rows[i]) * c.HcDim(),
                  c.HcDim() * sizeof(float), hipMemcpyDeviceToDevice, stream_),
              error))
        return false;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (const auto& item : items) {
    if (item.session->Cancelled())
      continue;
    item.session->mtp_.position += item.tokens.size();
    if (item.session->mtp_.position > c.indexer_top_k)
      item.session->mtp_.blocks =
          item.session->mtp_.position / c.compress_ratio;
  }
  return true;
}

bool Executor::MtpHeads(std::span<const MtpHeadItem> items,
                        std::string* error) const {
  selected_logits_ = nullptr;
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions ||
      items.size() > options_.max_batch) {
    return Fail(error, "invalid MTP head batch");
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || item.session->mtp_.position == 0 ||
        item.output.trace != nullptr ||
        (item.output.token == nullptr && item.output.candidates == nullptr)) {
      return Fail(error, "invalid MTP head request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "MTP head requests must be independent");
      }
    }
  }
  const auto& output = DraftHead();
  const auto& head = model_->mtp().nextn_head;
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    for (const auto& item : items) {
      if (!item.session->CheckCancellation(nullptr))
        continue;
      if (!MtpHead(head, item.session->mtp_.h, item.output.token != nullptr,
                   item.output.candidates != nullptr, error) ||
          !Check(hipStreamSynchronize(stream_), error)) {
        return false;
      }
      if (item.output.token != nullptr)
        *item.output.token = DraftToken(*mtp_token_host_);
      if (item.output.candidates != nullptr) {
        *item.output.candidates = *mtp_candidates_host_;
        MapDraftCandidates(*item.output.candidates);
      }
    }
    return true;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  batch_argmax_rows_ = 0;
  const auto n = static_cast<std::uint32_t>(items.size());
  const auto count = std::min<std::size_t>(output.rows, kMtpCandidates);
  const Scratch base = s_;
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    for (std::uint32_t i = 0; i < n; ++i) {
      RmsNormRows(items[i].session->mtp_.h, head.norm.f32(),
                  base.xn + std::size_t{i} * config().HcDim(), 1,
                  config().HcDim(), config().hc_count, config().rms_eps,
                  stream_);
    }
    UseScratch(base);
    if (!HcMixBatch(head, nullptr, true, base.mixed, nullptr, n, error))
      return false;
    if (!AnyActive(items))
      return true;
    if (!BatchHead(base.mixed, n, false, error, false, &output))
      return false;
    // One argmax launch covers every greedy draft row.
    auto* tokens = reinterpret_cast<std::int32_t*>(base.up_e);
    if (std::any_of(items.begin(), items.end(), [](const auto& item) {
          return item.output.candidates == nullptr;
        }))
      Argmax(batch_logits_, reinterpret_cast<ArgmaxCandidate*>(base.gate_e),
             tokens, n, output.rows, stream_);
    for (std::uint32_t i = 0; i < n; ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (items[i].output.candidates == nullptr) {
        if (!Check(hipMemcpyAsync(batch_candidates_host_[i].ids.data(),
                                  tokens + i, sizeof(std::int32_t),
                                  hipMemcpyDeviceToHost, stream_),
                   error))
          return false;
        continue;
      }
      MtpTopCandidates(batch_logits_ + std::size_t(i) * output.rows, s_.mtp_ids,
                       s_.mtp_scratch_ids, s_.mtp_scores, output.rows, stream_);
      if (!Check(hipMemcpyAsync(batch_candidates_host_ + i, s_.mtp_ids,
                                offsetof(MtpCandidateLogits, logits) +
                                    count * sizeof(float),
                                hipMemcpyDeviceToHost, stream_),
                 error)) {
        return false;
      }
      batch_candidates_host_[i].size = count;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    if (items[i].output.token != nullptr)
      *items[i].output.token = DraftToken(
          static_cast<std::int32_t>(batch_candidates_host_[i].ids[0]));
    if (items[i].output.candidates != nullptr) {
      *items[i].output.candidates = batch_candidates_host_[i];
      MapDraftCandidates(*items[i].output.candidates);
    }
  }
  return true;
}

bool Executor::QuantizeBatch(const float* x, std::uint32_t rows,
                             std::uint32_t cols, std::string* error) const {
  // Quantize once for all independent rows. The prefill staging allocation
  // is idle during batch decoding; borrow it without adding another buffer.
  const auto padded = (rows + kDecodeRows - 1) / kDecodeRows * kDecodeRows;
  const auto bytes = qfn_mmq_q8_1_bytes(padded, cols);
  if (bytes > Q8TiledBytes(options_.max_batch, model_->max_q8_cols()))
    return Fail(error, "batched projection exceeds activation scratch");
  void* quantized = s_.x_q8t;
  q8t_src_ = nullptr;
  if (padded != rows &&
      !Check(
          hipMemsetAsync(static_cast<std::uint8_t*>(quantized) +
                             qfn_mmq_q8_1_bytes(rows, cols),
                         0, qfn_mmq_q8_1_bytes(padded - rows, cols), stream_),
          error))
    return false;
  return qfn_mmq_quantize_q8_1(x, quantized, rows, cols, stream_) == 0 ||
         Fail(error, "batched activation quantization failed");
}

bool Executor::TiledDense(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
  // One pass over the weights. Q8 rows keep the decode activation
  // granularity (per-32 absmax int8) or use F16 activations; neither is
  // bit-identical to the GEMV reduction order.
  if (w.type != core::GgmlType::kQ8_0 || rows > options_.max_batch)
    return Dense(w, x, out, rows, error);
  if (F16Tiles(w, rows) &&
      std::size_t{rows} * w.cols <=
          std::size_t{options_.max_batch} * model_->max_half_cols()) {
    PrepareHalfInput(x, rows, w.cols);
    return spike::DenseF16Gemm(prefill_phase, w.data,
                               static_cast<const __half*>(s_.x_half), out, rows,
                               w.rows, w.cols, stream_) ||
           Fail(error, "batched F16 projection failed");
  }
  // Spike: registered prefill weights off the F16 tiles take the
  // spike's F16 path so their activations reach the emulation.
  if (spike::Covers(prefill_phase, w.data, rows, stream_))
    return spike::DenseF32Gemm(prefill_phase, w.data, x, out, rows, w.rows,
                               w.cols, stream_) ||
           Fail(error, "batched spike projection failed");
  if (!(q8t_src_ == x && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
    QuantizeQ8Tiled(x, s_.x_q8t, rows, w.cols, stream_);
    q8t_src_ = x;
    q8t_rows_ = rows;
    q8t_cols_ = w.cols;
  }
  return W8A8Gemm(w.data, s_.x_q8t, out, rows, w.rows, w.cols, stream_) ||
         Fail(error, "batched W8A8 projection failed");
}

bool Executor::DenseBatch(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
  if (rows <= kDecodeRows)
    return Dense(w, x, out, rows, error);
  if (rows >= TiledFrom(w))
    return TiledDense(w, x, out, rows, error);
  // `--gguf-decode-override`: the calibrated twin serves the narrow batches.
  if (w.alt != nullptr && rows <= options_.decode_twin_rows)
    return DenseBatch(*w.alt, x, out, rows, error);
  if (w.type != core::GgmlType::kQ8_0 && !KQuant(w.type)) {
    SmallGemm(w.data, EmbeddingType(w.type), x, out, rows, w.rows, w.cols,
              stream_);
    return true;
  }
  if (!QuantizeBatch(x, rows, w.cols, error))
    return false;
  const auto project = [&](std::uint32_t first, std::uint32_t count) {
    const auto* input = static_cast<const std::uint8_t*>(s_.x_q8t) +
                        qfn_mmq_q8_1_bytes(first, w.cols);
    return VecPreq(w, nullptr, input,
                   out + static_cast<std::size_t>(first) * w.rows,
                   count) == 0 ||
           Fail(error, "batched quantized projection failed");
  };
  if (w.rows == 320 && w.cols == 10240) {
    const auto full = rows / kDecodeRows * kDecodeRows;
    return project(0, full) && (full == rows || project(full, rows - full));
  }
  // Three matrix tiles help 33–48 rows; four tiles cost more than two
  // 32-row launches. Small outputs with long K sweeps stay at eight.
  const auto chunk = w.cols == 2560 && w.rows >= 8192
                         ? (rows > 32 && rows <= 48 ? 48U : 32U)
                         : kDecodeRows;
  for (std::uint32_t r = 0; r < rows; r += chunk) {
    if (!project(r, std::min(chunk, rows - r)))
      return false;
  }
  return true;
}

bool Executor::GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                               const float* x, float* out, std::uint32_t rows,
                               std::string* error) const {
  // Below the tiled crossover the gate/up GEMV reads the calibrated twins.
  if (up.alt != nullptr && gate.alt != nullptr && rows < 64 &&
      rows <= options_.decode_twin_rows)
    return GatedDenseBatch(*up.alt, *gate.alt, x, out, rows, error);
  const Scratch base = s_;
  const bool q8 = (up.type == core::GgmlType::kQ8_0 || KQuant(up.type)) &&
                  gate.type == up.type && up.rows == gate.rows &&
                  up.cols == gate.cols;
  // The paired GEMV reads both weights per row chunk; two tiled GEMMs win
  // from 64 rows (the single-projection crossover is about 48).
  if (q8 && rows >= 64) {
    if (!TiledDense(gate, x, out, rows, error) ||
        !TiledDense(up, x, base.shexp_gate, rows, error))
      return false;
    Swiglu(out, base.shexp_gate, std::size_t{rows} * up.rows, stream_);
    return true;
  }
  if (q8 && !QuantizeBatch(x, rows, up.cols, error))
    return false;
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    const auto n = std::min(kDecodeRows, rows - r);
    if (q8) {
      const auto* input = static_cast<const std::uint8_t*>(base.x_q8t) +
                          qfn_mmq_q8_1_bytes(r, up.cols);
      if (VecPreq(up, gate.data, input, out + std::size_t{r} * up.rows, n) != 0)
        return Fail(error, "batched gated quantized projection failed");
    } else {
      UseScratch(RowScratch(base, r));
      const bool ok =
          GatedDense(up, gate, x + std::size_t{r} * up.cols,
                     out + std::size_t{r} * up.rows, n, nullptr, error);
      UseScratch(base);
      if (!ok)
        return false;
    }
  }
  return true;
}

bool Executor::PleBatch(const DeviceLayer& l, std::span<const BatchItem> items,
                        std::span<const std::uint32_t> offsets,
                        std::uint32_t rows, std::string* error) const {
  // Shared projections, norms and gate cover every row; only the dilated
  // convolution reads and advances a request's history (and rollback rows).
  const Config& c = config();
  const Scratch base = s_;
  const std::uint32_t hc_dim = c.HcDim();
  if (!DenseBatch(l.ple_key, base.ple_emb, base.ple_key, rows, error) ||
      !DenseBatch(l.ple_value, base.ple_emb, base.ple_value, rows, error))
    return false;
  SteerWriter(base.ple_value, rows, 1, steer_writer::kPle);
  RmsNormRows(base.ple_key, l.ple_norm_key.f32(), base.ple_key, rows, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(base.res, l.ple_norm_query.f32(), base.ple_query, rows, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  PleGate(base.ple_key, base.ple_query, base.ple_value, base.ple_gated, rows,
          c.hidden_size, c.hc_count, stream_);
  RmsNormRows(base.ple_gated, l.ple_norm_conv.f32(), base.ple_norm, rows,
              hc_dim, c.hc_count, c.rms_eps, stream_);
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& session = *items[i].session;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const auto view = RowScratch(base, offsets[i]);
    if (session.Cancelled()) {
      // Finite rows for the shared layers; this request's state is untouched.
      if (!Check(
              hipMemsetAsync(view.ple_conv, 0,
                             std::size_t{n} * hc_dim * sizeof(float), stream_),
              error))
        return false;
      continue;
    }
    PleConv(view.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
            base.ple_history_scratch, view.ple_conv,
            items[i].speculative ? session.ple_snapshots_ : RollbackRows{}, n,
            hc_dim, c.ple_conv_kernel, c.ple_ngram_size, stream_);
  }
  PleInject(base.res, base.ple_gated, base.ple_conv, std::size_t{rows} * hc_dim,
            stream_);
  return true;
}

bool Executor::MoeBatchShared(const DeviceLayer& l, const float* x,
                              std::uint32_t rows, std::string* error) const {
  // Narrow batches keep Moe's single-request router and shared expert.
  if (rows <= kDecodeRows)
    return MoeShared(l, x, rows, false, error);
  const Scratch base = s_;
  if (!DenseBatch(l.router, x, base.router, rows, error))
    return false;
  Route(l, rows);
  return GatedDenseBatch(l.shexp_up, l.shexp_gate, x, base.shexp_up, rows,
                         error) &&
         DenseBatch(l.shexp_down, base.shexp_up, base.shexp_out, rows, error);
}

bool Executor::MoeBatch(const DeviceLayer& l, const float* x, float* out,
                        std::uint32_t rows, std::string* error) const {
  if (rows <= kDecodeRows)
    return Moe(l, x, out, rows, error);
  const auto& c = config();
  const Scratch base = s_;
  if (!MoeBatchShared(l, x, rows, error))
    return false;
  if (DeviceRouted(l))
    return DeviceExperts(l, x, out, rows, error);
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    UseScratch(RowScratch(base, r));
    const bool ok = MoeExperts(l, x + std::size_t{r} * c.hidden_size,
                               out + std::size_t{r} * c.hidden_size,
                               std::min(kDecodeRows, rows - r), error);
    UseScratch(base);
    if (!ok)
      return false;
  }
  return true;
}

bool Executor::LfmFfnBatch(const DeviceLayer& l, const float* x, float* out,
                           std::uint32_t rows, std::string* error) const {
  if (rows <= kDecodeRows)
    return LfmFfn(l, x, out, rows, error);
  const auto& c = config();
  const Scratch base = s_;
  if (l.ffn == FfnKind::kDense)
    return GatedDenseBatch(l.shexp_up, l.shexp_gate, x, base.shexp_up, rows,
                           error) &&
           DenseBatch(l.shexp_down, base.shexp_up, out, rows, error);
  LfmRouter(x, l.router.f32(), l.expert_bias.f32(), base.ids, base.weights,
            rows, c.num_experts_used, stream_);
  if (DeviceRouted(l))
    return DeviceExperts(l, x, out, rows, error);
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    UseScratch(RowScratch(base, r));
    const bool ok = MoeExperts(l, x + std::size_t{r} * c.hidden_size,
                               out + std::size_t{r} * c.hidden_size,
                               std::min(kDecodeRows, rows - r), error);
    UseScratch(base);
    if (!ok)
      return false;
  }
  return true;
}

bool Executor::LfmBatchLayer(const DeviceLayer& l, std::uint32_t il,
                             std::span<const BatchItem> items,
                             std::span<const std::uint32_t> offsets,
                             std::uint32_t decode_items, std::uint32_t rows,
                             std::string* error) const {
  const auto& c = config();
  const Scratch base = s_;
  if (items.size() > kMaxBatchSessions)
    return Fail(error, "lfm2 batch exceeds the session table");
  if (!PlainNorm(l.hc_attn, base.res, base.mixed, rows))
    return false;
  if (l.linear()) {
    if (lfm_conv_host_ == nullptr) {
      const std::size_t bytes =
          std::size_t{c.num_layers} * kMaxBatchSessions * sizeof(LfmConvItem);
      if (!Check(hipHostMalloc(&lfm_conv_host_, bytes, hipHostMallocMapped),
                 error) ||
          !Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&lfm_conv_),
                                         lfm_conv_host_, 0),
                 error))
        return false;
    }
    if (!DenseBatch(l.ssm_in, base.mixed, base.qkvz, rows, error))
      return false;
    // One table slice per layer: the GPU may still read the previous one.
    LfmConvItem* table = lfm_conv_host_ + std::size_t{il} * kMaxBatchSessions;
    std::uint32_t max_rows = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      Session& session = *items[i].session;
      table[i] = {session.linear_[il].conv_state, offsets[i],
                  session.CheckCancellation(nullptr)
                      ? static_cast<std::uint32_t>(items[i].tokens.size())
                      : 0U};
      max_rows = std::max(max_rows, table[i].rows);
    }
    LfmShortConvBatch(base.qkvz, l.ssm_in.rows, l.ssm_conv1d.f32(),
                      lfm_conv_ + std::size_t{il} * kMaxBatchSessions,
                      static_cast<std::uint32_t>(items.size()), max_rows,
                      base.ctx, c.hidden_size, stream_);
    if (!DenseBatch(l.ssm_out, base.ctx, base.block_out, rows, error))
      return false;
  } else {
    if (l.attn_qkv.empty())
      return Fail(error, "lfm2 attention needs stacked Q8_0 q/k/v");
    if (batch_attn_host_ == nullptr) {
      const std::size_t bytes = std::size_t{c.num_layers} * kMaxBatchSessions *
                                sizeof(AttentionBatchItem);
      if (!Check(hipHostMalloc(&batch_attn_host_, bytes, hipHostMallocMapped),
                 error) ||
          !Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&batch_attn_),
                                         batch_attn_host_, 0),
                 error))
        return false;
    }
    if (std::size_t{rows} > batch_attn_partial_rows_) {
      if (batch_attn_partials_ != nullptr &&
          !Check(hipStreamSynchronize(stream_), error))
        return false;
      (void)hipFree(batch_attn_partials_);
      batch_attn_partials_ = nullptr;
      batch_attn_partial_rows_ = 0;
      if (!Check(
              hipMalloc(&batch_attn_partials_,
                        std::size_t{rows} * c.num_heads * kMaxAttentionSplits *
                            (c.head_dim + 2) * sizeof(float)),
              error))
        return false;
      batch_attn_partial_rows_ = rows;
    }
    if (!DenseBatch(l.attn_qkv, base.mixed, base.qg, rows, error))
      return false;
    AttentionBatchItem* table =
        batch_attn_host_ + std::size_t{il} * kMaxBatchSessions;
    const AttentionBatchItem* device_table =
        batch_attn_ + std::size_t{il} * kMaxBatchSessions;
    std::uint32_t max_rows = 0, max_decode = 0, partial = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
      Session& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      const bool live = session.CheckCancellation(nullptr);
      table[i] = AttentionBatchItem{
          .kv = session.attention_[il].kv,
          .position = &session.control_->position,
          .row = offsets[i],
          .rows = live ? n : 0,
          .splits = i < decode_items ? AttentionSplits(n) : 1U,
          .partial = partial,
      };
      if (!live)
        continue;
      partial += n * c.num_heads * table[i].splits;
      max_rows = std::max(max_rows, n);
      if (i < decode_items)
        max_decode = std::max(max_decode, n);
    }
    LfmPrepareAttentionBatch(base.qg, l.attn_q_norm.f32(), l.attn_k_norm.f32(),
                             base.q, options_.kv_format, device_table,
                             static_cast<std::uint32_t>(items.size()), max_rows,
                             c.rope_theta, c.rms_eps, stream_);
    const std::uint32_t decode_rows =
        decode_items < items.size() ? offsets[decode_items] : rows;
    LfmAttentionBatch(base.q, options_.kv_format, device_table, decode_items,
                      max_decode, decode_rows, base.ctx, batch_attn_partials_,
                      stream_);
    for (auto i = decode_items; i < items.size(); ++i) {
      if (table[i].rows == 0)
        continue;
      const std::size_t at = std::size_t{offsets[i]} * c.AttentionQDim();
      LfmAttention(base.q + at, table[i].kv, base.ctx + at, nullptr,
                   table[i].rows, &items[i].session->control_->position,
                   stream_);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (table[i].rows == 0 &&
          !Check(hipMemsetAsync(
                     base.ctx + std::size_t{offsets[i]} * c.AttentionQDim(), 0,
                     items[i].tokens.size() * c.AttentionQDim() * sizeof(float),
                     stream_),
                 error))
        return false;
    }
    if (!DenseBatch(l.attn_out, base.ctx, base.block_out, rows, error))
      return false;
  }
  CombineBatch(base.res, nullptr, rows);
  if (!PlainNorm(l.hc_ffn, base.res, base.mixed, rows))
    return false;
  // Decode rows keep the batch-invariant routed kernels; prompt chunks take
  // the prefill route (WMMA routed experts over their concatenated rows).
  const std::uint32_t decode_rows =
      decode_items < items.size() ? offsets[decode_items] : rows;
  if (decode_rows != 0 &&
      !LfmFfnBatch(l, base.mixed, base.block_out, decode_rows, error))
    return false;
  if (decode_rows < rows) {
    PrefillPhase phase(true);
    UseScratch(RowScratch(base, decode_rows));
    const bool ok =
        LfmFfn(l, s_.mixed, s_.block_out, rows - decode_rows, error);
    UseScratch(base);
    if (!ok)
      return false;
  }
  UseScratch(base);
  CombineBatch(base.res, nullptr, rows);
  return true;
}

bool Executor::ForwardBatch(std::span<const BatchItem> items,
                            std::string* error) const {
  selected_logits_ = nullptr;
  batch_argmax_rows_ = 0;
  const Config& c = config();
  if (items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "decode batch must contain 1..64 sessions");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::array<bool, kBatchSessions> sparse{};
  std::array<std::uint32_t, kBatchSessions> complete{};
  std::uint32_t rows = 0;
  std::uint32_t max_tokens = 0;
  bool argmax = false;
  // Decode items come first, prompt chunks after them.
  std::uint32_t decode_items = 0;
  bool prompt_draft = false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        item.tokens.empty())
      return Fail(error, "invalid session or chain in decode batch");
    const auto& session = *item.session;
    const auto n = static_cast<std::uint32_t>(item.tokens.size());
    // A prompt chunk continues after the draft block consumed its frontier.
    const bool bad =
        item.prefill ? item.speculative || item.argmax ||
                           (session.mtp_enabled_ &&
                            session.mtp_.position != session.position_)
                     : decode_items != i || n > kDecodeRows ||
                           n > options_.max_logit_rows ||
                           (item.speculative && n > options_.max_speculative);
    if (bad || session.position_ + n > session.max_context_)
      return Fail(error, "invalid session or chain in decode batch");
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "decode batch contains a duplicate session");
      }
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        return Fail(error, "token out of range");
      }
    }
    argmax |= item.argmax;
    prompt_draft |= item.prefill && session.mtp_enabled_ && n > 1;
    if (!item.prefill) {
      ++decode_items;
      max_tokens = std::max(max_tokens, n);
    }
    offsets[i] = rows;
    rows += n;
    const auto end = session.position_ + n;
    sparse[i] = c.HasIndexer() && end > c.indexer_top_k;
    complete[i] = c.compress_ratio > 0 ? end / c.compress_ratio : 0;
  }
  if (rows > options_.max_batch) {
    return Fail(error, "decode batch exceeds executor capacity");
  }
  const std::uint32_t decode_rows =
      decode_items < items.size() ? offsets[decode_items] : rows;
  const std::uint32_t prompt_rows = rows - decode_rows;
  // A lone prompt chunk runs ForwardBody's own stages, including the fused
  // wide projections; several share concatenated prefill stages.
  const bool single_prompt = items.size() - decode_items == 1;
  // Every decode row, then the last row of each prompt chunk.
  const auto head_rows =
      decode_rows + static_cast<std::uint32_t>(items.size() - decode_items);
  const auto decode_span = items.first(decode_items);
  const auto decode_offsets = std::span(offsets).first(decode_items);
  if (!AnyActive(items))
    return true;
  for (const auto& item : items) {
    if (!RequireResident(*item.session, error) ||
        (item.speculative &&
         !EnsureRollback(*item.session, item.tokens.size() - 1, error)) ||
        !Reserve(*item.session,
                 item.session->position_ +
                     static_cast<std::uint32_t>(item.tokens.size()),
                 error))
      return false;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  // The recurrent batch kernel serves decode rows; a lone decode item keeps
  // the per-request path, as in a decode-only batch.
  const bool batch_gdn =
      decode_items > 1 && c.ssm_head_dim == 128 && c.ssm_conv_kernel == 4;
  if (batch_gdn) {
    const std::size_t bytes =
        std::size_t{c.num_layers} * kBatchSessions * sizeof(GdnBatchItem);
    if ((batch_gdn_host_ == nullptr &&
         !Check(hipHostMalloc(&batch_gdn_host_, bytes, hipHostMallocMapped),
                error)) ||
        (batch_gdn_ == nullptr &&
         !Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&batch_gdn_),
                                        batch_gdn_host_, 0),
                error)))
      return false;
  }
  if (ple_pending_ && !WaitPle(error)) {
    return false;
  }
  batch_rows_ = 0;
  bool steer = false;
  bool route_keep = false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& session = *item.session;
    if (session.adapter_ != 0 || steer) {
      // Normal and adapter rows share the batch; each row names its own.
      if (!steer)
        std::fill_n(steer_host_, offsets[i], std::uint8_t{0});
      std::fill_n(steer_host_ + offsets[i], item.tokens.size(),
                  session.adapter_);
      steer = true;
    }
    if (session.route_keep_ != 0 || route_keep) {
      // Fast-model and full-routing rows share the batch the same way; Route
      // truncates only the decode stage, so prompt rows stay exact.
      if (!route_keep)
        std::fill_n(route_keep_host_, offsets[i], std::uint8_t{0});
      std::fill_n(route_keep_host_ + offsets[i], item.tokens.size(),
                  session.route_keep_);
      route_keep = true;
    }
    ++session.mutation_epoch_;
    session.spec_base_ = session.position_;
    session.spec_tokens_ = item.speculative ? item.tokens.size() : 0;
    batch_controls_[i] = {session.position_, session.blocks_,
                          session.mtp_.position, -1, session.mtp_.blocks};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
    if (c.ple_layer >= 0) {
      for (std::size_t t = 0; t < item.tokens.size(); ++t) {
        HashNgramRows(
            c, session.ngram_, item.tokens.subspan(t, 1),
            std::span(host_rows_)
                .subspan((offsets[i] + t) * c.ple_heads, c.ple_heads));
        if (item.speculative && t + 1 < item.tokens.size()) {
          session.ngram_snapshots_[t] = session.ngram_;
        }
      }
    }
  }
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr ||
        !ngram_->StartRead(std::span(host_rows_).first(rows * c.ple_heads),
                           std::span(host_emb_, static_cast<std::size_t>(rows) *
                                                    c.PleEmbeddingDim()))) {
      return Fail(error, "batched n-gram read could not start");
    }
    ple_pending_ = true;
  }

  const Scratch base = s_;
  const Scratch prompt_view = RowScratch(base, decode_rows);
  // The decode rows run the decode-batch stages and the prompt rows the
  // prefill stages. Each part keeps the mixer flags its next stage reads.
  struct PartFlags {
    bool xn_half{false};
    std::uint32_t inject_parts{1};
  };
  PartFlags decode_flags;
  PartFlags prompt_flags;
  const auto enter = [&](bool prompt) {
    UseScratch(prompt ? prompt_view : base);
    const auto& f = prompt ? prompt_flags : decode_flags;
    xn_half_ = f.xn_half;
    inject_parts_ = f.inject_parts;
  };
  const auto leave = [&](bool prompt) {
    auto& f = prompt ? prompt_flags : decode_flags;
    f.xn_half = xn_half_;
    f.inject_parts = inject_parts_;
  };
  // Every exit restores the ordinary single-session scratch view. Enqueued
  // work is drained before a caller can reuse its pinned staging buffers.
  const auto body = [&]() -> bool {
    if (!AnyActive(items))
      return true;
    const SteerScope steer_scope(*this, steer);
    const RouteKeepScope route_keep_scope(*this, route_keep);
    if ((steer && !Check(hipMemcpyAsync(base.steer, steer_host_, rows,
                                        hipMemcpyHostToDevice, stream_),
                         error)) ||
        (route_keep &&
         !Check(hipMemcpyAsync(base.route_keep, route_keep_host_, rows,
                               hipMemcpyHostToDevice, stream_),
                error)))
      return false;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!Check(hipMemcpyAsync(items[i].session->control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error)) {
        return false;
      }
    }
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data,
                static_cast<WeightType>(model_->token_embd().type), base.tokens,
                base.res, rows, c.hidden_size, c.hc_count, stream_);
    SteerWriter(base.res, rows, c.hc_count, steer_writer::kEmbedding);
    for (std::size_t i = 0; i < items.size(); ++i) {
      items[i].session->vision_input_.Inject(
          base.res + std::size_t{offsets[i]} * c.HcDim(),
          items[i].session->position_, items[i].tokens.size(), c.hidden_size,
          c.hc_count, stream_);
    }
    const auto& layers = model_->layers();
    const bool dspark = std::any_of(items.begin(), items.end(), [](auto& i) {
      return i.session->dspark_enabled_;
    });
    bool normed = false;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!AnyActive(items))
        return true;
      const auto& l = layers[il];
      if (dspark) {
        // DSpark reads each target layer's input residual (every row).
        const auto& targets = model_->dspark()->target_layers;
        const auto hit = std::find(targets.begin(), targets.end(), il);
        if (hit != targets.end() &&
            !Check(hipMemcpy2DAsync(
                       ds_.features + (hit - targets.begin()) * c.hidden_size,
                       targets.size() * c.hidden_size * sizeof(float), base.res,
                       c.HcDim() * sizeof(float), c.hidden_size * sizeof(float),
                       rows, hipMemcpyDeviceToDevice, stream_),
                   error))
          return false;
      }
      // Short-convolution and plain-attention layers (lfm2moe) batch
      // through their own layer.
      if (l.mixer == MixerKind::kShortConv ||
          l.mixer == MixerKind::kAttention) {
        UseScratch(base);
        if (!LfmBatchLayer(l, il, items, offsets, decode_items, rows, error))
          return false;
        continue;
      }
      const float* next_norm =
          il + 1 < c.num_layers
              ? (c.IsPleLayer(il + 1) ? nullptr
                                      : layers[il + 1].hc_attn.norm.f32())
              : model_->hc_head().norm.f32();
      // One routed-expert pass covers both parts on the batch-invariant
      // device-planned route; otherwise each part runs its own experts.
      const bool shared_moe =
          decode_rows != 0 && prompt_rows != 0 && DeviceRouted(l);
      UseScratch(base);
      if (c.IsPleLayer(il) &&
          (!WaitPle(error) ||
           !Check(hipMemcpyAsync(base.ple_emb, host_emb_,
                                 static_cast<std::size_t>(rows) *
                                     c.PleEmbeddingDim() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  error)))
        return false;
      if (decode_rows != 0) {
        enter(false);
        if (c.IsPleLayer(il)) {
          if (!PleBatch(l, decode_span, decode_offsets, decode_rows, error))
            return false;
          UseScratch(base);
        }
        if (!HcMixBatch(l.hc_attn, base.res, normed, base.mixed, base.inject,
                        decode_rows, error))
          return false;
        if (l.linear()) {
          if ((!l.ssm_in.empty() ? !DenseBatch(l.ssm_in, base.mixed, base.qkvz,
                                               decode_rows, error)
                                 : (!DenseBatch(l.ssm_qkv, base.mixed, base.qkv,
                                                decode_rows, error) ||
                                    !DenseBatch(l.ssm_gate, base.mixed, base.z,
                                                decode_rows, error))) ||
              !DenseBatch(l.ssm_alpha_beta, base.mixed, base.alpha_beta,
                          decode_rows, error)) {
            return false;
          }
        } else if (!l.attn_qkv.empty()) {
          if (!DenseBatch(l.attn_qkv, base.mixed, base.qg, decode_rows,
                          error)) {
            return false;
          }
        } else {
          // Separate Q/gate has a shorter row stride than the stacked
          // buffer. This model normally stacks QKV; keep the separate shape
          // explicit.
          if (!DenseBatch(l.attn_q, base.mixed, base.qg, decode_rows, error) ||
              !DenseBatch(l.attn_k, base.mixed, base.k, decode_rows, error) ||
              !DenseBatch(l.attn_v, base.mixed, base.v, decode_rows, error)) {
            return false;
          }
        }
        // Recurrent layers and attention run every active request in one
        // launch per stage, each reading only its request's rows and caches.
        const bool batched_attention = !l.linear() && !l.attn_qkv.empty();
        if (batched_attention &&
            !AttentionBatchLayer(l, il, decode_span, decode_offsets,
                                 decode_rows, error))
          return false;
        std::uint32_t gdn_count = 0;
        for (std::size_t i = 0; i < decode_items && !batched_attention; ++i) {
          const auto& item = items[i];
          auto& session = *item.session;
          const auto n = static_cast<std::uint32_t>(item.tokens.size());
          auto view = RowScratch(base, offsets[i]);
          if (!l.linear() && l.attn_qkv.empty()) {
            view.qg = base.qg + static_cast<std::size_t>(offsets[i]) * 2 *
                                    c.AttentionQDim();
          }
          UseScratch(view);
          if (!session.CheckCancellation(nullptr)) {
            auto* output = l.linear() ? s_.gdn_out : s_.ctx;
            const auto width = l.linear() ? c.SsmValueDim() : c.AttentionQDim();
            if (!Check(hipMemsetAsync(output, 0,
                                      std::size_t{n} * width * sizeof(float),
                                      stream_),
                       error))
              return false;
            continue;
          }
          if (l.linear()) {
            if (batch_gdn) {
              const auto& state = session.linear_[il];
              batch_gdn_host_[il * kBatchSessions + gdn_count++] = {
                  l.ssm_in.empty() ? view.qkv : view.qkvz,
                  l.ssm_in.empty() ? view.z : view.qkvz + c.SsmConvChannels(),
                  view.alpha_beta, state.conv_state,
                  // Short convolution saves history in registers, so each
                  // request needs only its own token rows of staging.
                  base.conv_scratch +
                      std::size_t{offsets[i]} * c.SsmConvChannels(),
                  view.qn, view.kn, view.gdn_raw, state.state, view.gdn_out,
                  item.speculative ? state.state_snapshots : RollbackRows{},
                  item.speculative ? state.conv_snapshots : RollbackRows{}, n};
            } else if (!LinearAttention(l, session.linear_[il], s_.mixed,
                                        s_.block_out, n, item.speculative,
                                        error, true, false)) {
              return false;
            }
          } else {
            const auto pool = sparse[i] && complete[i] > session.blocks_
                                  ? complete[i] - session.blocks_
                                  : 0;
            if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                           &session.control_->position,
                           &session.control_->blocks, session.position_, pool,
                           session.max_context_, sparse[i], error, false, true,
                           false)) {
              return false;
            }
          }
        }
        if (l.linear() && batch_gdn && gdn_count != 0) {
          const auto offset = il * kBatchSessions;
          if (!GatedDeltaNetBatch(
                  batch_gdn_ + offset, gdn_count, max_tokens, gdn_count,
                  l.ssm_in.empty() ? c.SsmConvChannels() : l.ssm_in.rows,
                  l.ssm_in.empty() ? c.SsmValueDim() : l.ssm_in.rows,
                  l.ssm_conv1d.f32(), l.ssm_a.f32(), l.ssm_dt.f32(),
                  l.ssm_norm.f32(), c.ssm_num_k_heads, c.ssm_num_v_heads,
                  c.rms_eps, stream_, c.Traits().gdn_gate == GdnGate::kSilu,
                  options_.gdn_state))
            return Fail(error, "batched GatedDeltaNet launch failed");
        }
        UseScratch(base);
        if (!DenseBatch(l.linear() ? l.ssm_out : l.attn_out,
                        l.linear() ? base.gdn_out : base.ctx, base.block_out,
                        decode_rows, error)) {
          return false;
        }
        CombineBatch(base.res, l.hc_ffn.norm.f32(), decode_rows);
        if (!HcMixBatch(l.hc_ffn, base.res, true, base.mixed, base.inject,
                        decode_rows, error))
          return false;
        if (!shared_moe) {
          if (!MoeBatch(l, base.mixed, base.block_out, decode_rows, error))
            return false;
          UseScratch(base);
          CombineBatch(base.res, next_norm, decode_rows,
                       static_cast<std::int32_t>(il));
        }
        leave(false);
      }
      if (prompt_rows != 0) {
        // The prompt rows run ForwardBody's prefill stages over every chunk
        // at once; only recurrent, attention and PLE state stay per item.
        PrefillPhase phase(true);
        const Scratch& v = prompt_view;
        enter(true);
        if (c.IsPleLayer(il)) {
          for (auto i = decode_items; i < items.size(); ++i) {
            auto& session = *items[i].session;
            if (!session.CheckCancellation(nullptr))
              continue;
            UseScratch(RowScratch(base, offsets[i]));
            if (!Ple(l, session,
                     static_cast<std::uint32_t>(items[i].tokens.size()), s_.res,
                     false, error, true))
              return false;
          }
          enter(true);
        }
        if (!HcMix(l.hc_attn, v.res, normed, v.mixed, v.inject, prompt_rows,
                   error))
          return false;
        if (single_prompt) {
          auto& session = *items.back().session;
          const auto i = items.size() - 1;
          const auto pool = sparse[i] && complete[i] > session.blocks_
                                ? complete[i] - session.blocks_
                                : 0;
          if (!session.CheckCancellation(nullptr)) {
            if (!Check(hipMemsetAsync(v.block_out, 0,
                                      std::size_t{prompt_rows} * c.hidden_size *
                                          sizeof(float),
                                      stream_),
                       error))
              return false;
          } else if (l.linear()
                         ? !LinearAttention(l, session.linear_[il], v.mixed,
                                            v.block_out, prompt_rows, false,
                                            error)
                         : !Attention(
                               l, session.attention_[il], v.mixed, v.block_out,
                               prompt_rows, &session.control_->position,
                               &session.control_->blocks, session.position_,
                               pool, session.max_context_, sparse[i], error)) {
            return false;
          }
        } else {
          // Separate Q/K/V rows keep the unstacked stride (see above).
          float* const prompt_qg =
              !l.linear() && l.attn_qkv.empty()
                  ? base.qg + std::size_t{decode_rows} * 2 * c.AttentionQDim()
                  : v.qg;
          if (l.linear()) {
            if (!l.ssm_in.empty()) {
              if (!Dense(l.ssm_in, v.mixed, v.qkvz, prompt_rows, error))
                return false;
            } else {
              Q8Input xq;
              if (!Quantize(v.mixed, prompt_rows, c.hidden_size, &xq, error) ||
                  !Dense(l.ssm_qkv, xq, v.qkv, error) ||
                  !Dense(l.ssm_gate, xq, v.z, error))
                return false;
            }
            if (!Dense(l.ssm_alpha_beta, v.mixed, v.alpha_beta, prompt_rows,
                       error))
              return false;
          } else if (!l.attn_qkv.empty()) {
            if (!Dense(l.attn_qkv, v.mixed, v.qg, prompt_rows, error))
              return false;
          } else {
            Q8Input xq;
            if (!Quantize(v.mixed, prompt_rows, c.hidden_size, &xq, error) ||
                !Dense(l.attn_q, xq, prompt_qg, error) ||
                !Dense(l.attn_k, xq, v.k, error) ||
                !Dense(l.attn_v, xq, v.v, error))
              return false;
          }
          const bool half_out = l.linear() && HalfSsmOutput(l, prompt_rows);
          auto* const half_rows =
              half_out ? reinterpret_cast<__half*>(v.gdn_out) : nullptr;
          for (auto i = decode_items; i < items.size(); ++i) {
            auto& session = *items[i].session;
            const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
            auto view = RowScratch(base, offsets[i]);
            if (!l.linear() && l.attn_qkv.empty())
              view.qg =
                  base.qg + std::size_t{offsets[i]} * 2 * c.AttentionQDim();
            UseScratch(view);
            auto* const half =
                half_rows == nullptr
                    ? nullptr
                    : half_rows + std::size_t{offsets[i] - decode_rows} *
                                      c.SsmValueDim();
            if (!session.CheckCancellation(nullptr)) {
              // Finite rows for the shared stages; this request stays intact.
              void* output = half != nullptr ? static_cast<void*>(half)
                             : l.linear()    ? static_cast<void*>(s_.gdn_out)
                                             : static_cast<void*>(s_.ctx);
              const std::size_t bytes =
                  half != nullptr ? std::size_t{n} * c.SsmValueDim() * 2
                  : l.linear()    ? std::size_t{n} * c.SsmValueDim() * 4
                                  : std::size_t{n} * c.AttentionQDim() * 4;
              if (!Check(hipMemsetAsync(output, 0, bytes, stream_), error))
                return false;
              continue;
            }
            if (l.linear()) {
              if (!LinearAttention(l, session.linear_[il], s_.mixed,
                                   s_.block_out, n, false, error, true, false,
                                   half))
                return false;
            } else {
              const auto pool = sparse[i] && complete[i] > session.blocks_
                                    ? complete[i] - session.blocks_
                                    : 0;
              if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out,
                             n, &session.control_->position,
                             &session.control_->blocks, session.position_, pool,
                             session.max_context_, sparse[i], error, false,
                             true, false))
                return false;
            }
          }
          // UseScratch keeps inject_parts_; the combine below sets xn_half_.
          UseScratch(v);
          if (half_out) {
            if (!spike::DenseF16Gemm(prefill_phase, l.ssm_out.data, half_rows,
                                     v.block_out, prompt_rows, l.ssm_out.rows,
                                     l.ssm_out.cols, stream_))
              return Fail(error, "prompt SSM output F16 GEMM failed");
          } else if (!Dense(l.linear() ? l.ssm_out : l.attn_out,
                            l.linear() ? v.gdn_out : v.ctx, v.block_out,
                            prompt_rows, error)) {
            return false;
          }
        }
        Combine(v.res, l.hc_ffn.norm.f32(), prompt_rows);
        if (!HcMix(l.hc_ffn, v.res, true, v.mixed, v.inject, prompt_rows,
                   error))
          return false;
        if (!shared_moe) {
          if (!Moe(l, v.mixed, v.block_out, prompt_rows, error))
            return false;
          Combine(v.res, next_norm, prompt_rows, static_cast<std::int32_t>(il));
        }
        leave(true);
      }
      if (shared_moe) {
        // Each part keeps its router and shared-expert arithmetic.
        enter(false);
        if (!MoeBatchShared(l, base.mixed, decode_rows, error))
          return false;
        {
          PrefillPhase phase(true);
          UseScratch(prompt_view);
          if (!MoeShared(l, prompt_view.mixed, prompt_rows, false, error))
            return false;
        }
        UseScratch(base);
        if (!DeviceExperts(l, base.mixed, base.block_out, rows, error))
          return false;
        enter(false);
        CombineBatch(base.res, next_norm, decode_rows,
                     static_cast<std::int32_t>(il));
        leave(false);
        PrefillPhase phase(true);
        enter(true);
        Combine(prompt_view.res, next_norm, prompt_rows,
                static_cast<std::int32_t>(il));
        leave(true);
      }
      normed = next_norm != nullptr;
    }
    UseScratch(base);
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      const auto kept = std::min(n, options_.max_speculative);
      if (!items[i].session->Cancelled() && items[i].session->mtp_enabled_ &&
          !Check(hipMemcpyAsync(
                     items[i].session->mtp_.target_hidden,
                     base.res + std::size_t{offsets[i] + n - kept} * c.HcDim(),
                     static_cast<std::size_t>(kept) * c.HcDim() * sizeof(float),
                     hipMemcpyDeviceToDevice, stream_),
                 error)) {
        return false;
      }
    }
    // The draft catch-ups below reuse the MoE scratch the alias occupies.
    if (!HeadLogits(head_rows, prompt_draft, error))
      return false;
    if (decode_rows != 0 &&
        (!HcMixBatch(model_->hc_head(), base.res, false, base.mixed, nullptr,
                     decode_rows, error) ||
         !DenseBatch(model_->output(), base.mixed, batch_logits_, decode_rows,
                     error) ||
         (argmax && !BatchArgmax(decode_rows, error))))
      return false;
    // Each prompt chunk takes Forward's head for its last row.
    for (auto i = decode_items; i < items.size(); ++i) {
      const auto last = offsets[i] + items[i].tokens.size() - 1;
      UseScratch(prompt_view);
      if (!HcMix(model_->hc_head(), base.res + last * c.HcDim(), false,
                 s_.mixed, nullptr, 1, error) ||
          !Dense(model_->output(), s_.mixed,
                 batch_logits_ +
                     std::size_t{decode_rows + i - decode_items} * c.vocab_size,
                 1, error))
        return false;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& session = *items[i].session;
    if (session.Cancelled())
      continue;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    session.position_ += n;
    if (sparse[i]) {
      session.blocks_ = complete[i];
    }
    // Verified rows join the drafter's context in Rollback; the others
    // join together below.
    session.dspark_.feature_row = offsets[i];
    if (session.dspark_enabled_ && !items[i].speculative)
      session.dspark_.Pend(offsets[i], n, session.position_ - n);
  }
  if (model_->dspark() != nullptr) {
    std::vector<Session*> injected;
    for (const auto& item : items)
      injected.push_back(item.session);
    if (!DsparkFlush(injected, error))
      return false;
  }
  batch_rows_ = head_rows;
  // Every successor but the last is known: catch each draft block up from
  // its chunk's trunk rows, as a kPrefill Forward does. A catch-up
  // overwrites residual rows [0, n - 1), which precede every later chunk's
  // rows, so ascending order reads each chunk intact.
  for (auto i = decode_items; i < items.size(); ++i) {
    auto& session = *items[i].session;
    if (!session.mtp_enabled_ || items[i].tokens.size() < 2 ||
        session.Cancelled())
      continue;
    if (!MtpForward(session, items[i].tokens.subspan(1), 0, {}, error,
                    base.res + std::size_t{offsets[i]} * c.HcDim()))
      return false;
  }
  return true;
}

bool Executor::SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                 float* logits, std::string* error) const {
  if (rows == 0 || rows > options_.max_logit_rows || offset > batch_rows_ ||
      rows > batch_rows_ - offset) {
    return Fail(error, "logit rows outside the completed decode batch");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * config().vocab_size;
  selected_offset_ = offset;
  selected_logits_ =
      batch_logits_ + static_cast<std::size_t>(offset) * config().vocab_size;
  if (logits != nullptr) {
    if (!Check(hipMemcpyAsync(logits_host_, selected_logits_,
                              count * sizeof(float), hipMemcpyDeviceToHost,
                              stream_),
               error) ||
        !Check(hipStreamSynchronize(stream_), error)) {
      return false;
    }
    std::copy_n(logits_host_, count, logits);
  }
  return true;
}

const float* Executor::HostBatchLogits(std::uint32_t rows,
                                       std::string* error) const {
  selected_logits_ = nullptr;
  if (rows == 0 || rows > batch_rows_) {
    Fail(error, "logit rows outside the completed decode batch");
    return nullptr;
  }
  const std::size_t count = std::size_t{rows} * config().vocab_size;
  if (rows > batch_logits_host_rows_) {
    (void)hipHostFree(batch_logits_host_);
    batch_logits_host_ = nullptr;
    batch_logits_host_rows_ = 0;
    if (!Check(hipHostMalloc(&batch_logits_host_, count * sizeof(float)),
               error))
      return nullptr;
    batch_logits_host_rows_ = rows;
  }
  if (!Check(
          hipMemcpyAsync(batch_logits_host_, batch_logits_,
                         count * sizeof(float), hipMemcpyDeviceToHost, stream_),
          error) ||
      !Check(hipStreamSynchronize(stream_), error))
    return nullptr;
  return batch_logits_host_;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
