#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <initializer_list>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

bool Stackable(std::initializer_list<const TensorRef*> parts) {
  for (const TensorRef* t : parts) {
    if (t->empty() ||
        (t->type != core::GgmlType::kQ8_0 && t->type != core::GgmlType::kQ4_0 &&
         !KQuant(t->type)) ||
        t->type != (*parts.begin())->type ||
        t->cols != (*parts.begin())->cols) {
      return false;
    }
  }
  return true;
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if ((t.type == core::GgmlType::kQ8_0 || KQuant(t.type)) && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A quantized stack merges projections of one input
  /// into a single decode GEMV.
  /// `gap_rows` zero rows follow the first part (F32 stacks only): a pruned
  /// router keeps the shared-expert gate row at index num_experts.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts,
                     std::size_t gap_rows = 0) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0 &&
           type != core::GgmlType::kQ4_0 && !KQuant(type))) {
        Fail(
            "stacked upload needs F32, Q8_0, Q4_0 or K-quant tensors of one "
            "shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    if (gap_rows != 0 && type != core::GgmlType::kF32) {
      Fail("zero-padded stacks need F32 tensors");
      return d;
    }
    const std::size_t gap_bytes =
        gap_rows * (*parts.begin())->cols * sizeof(float);
    rows += gap_rows;
    size += gap_bytes;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
      if (t == *parts.begin() && gap_bytes != 0) {
        if (hipMemset(static_cast<std::uint8_t*>(ptr) + offset, 0, gap_bytes) !=
            hipSuccess) {
          Fail("hipMemset failed for stacked tensor gap");
          return d;
        }
        offset += gap_bytes;
      }
    }
    if (type != core::GgmlType::kF32) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      if (type != core::GgmlType::kQ4_0)
        max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  /// Decode twins (`--gguf-decode-override`): the layer's dense projections
  /// in the override's encoding, attached to `d` when the encoding differs
  /// and the stacking matches the primary upload.
  void DenseAlt(const LayerWeights& l, DeviceLayer& d,
                std::deque<DeviceTensor>& store) {
    const auto attach = [&](DeviceTensor& primary, const TensorRef& first,
                            const auto& make) {
      if (primary.data == nullptr || first.empty() ||
          first.type == primary.type)
        return;
      const DeviceTensor twin = make();
      if (twin.data == nullptr || twin.rows != primary.rows ||
          twin.cols != primary.cols)
        return;
      store.push_back(twin);
      primary.alt = &store.back();
    };
    if (l.linear() && Stackable({&l.ssm_qkv, &l.ssm_gate}))
      attach(d.ssm_in, l.ssm_qkv,
             [&] { return Stack({&l.ssm_qkv, &l.ssm_gate}); });
    attach(d.ssm_out, l.ssm_out, [&] { return Copy(l.ssm_out); });
    if (!l.linear() && Stackable({&l.attn_q, &l.attn_k, &l.attn_v}))
      attach(d.attn_qkv, l.attn_q,
             [&] { return Stack({&l.attn_q, &l.attn_k, &l.attn_v}); });
    attach(d.attn_out, l.attn_out, [&] { return Copy(l.attn_out); });
    attach(d.shexp_gate, l.shexp_gate, [&] { return Copy(l.shexp_gate); });
    attach(d.shexp_up, l.shexp_up, [&] { return Copy(l.shexp_up); });
  }

  DeviceLayer Layer(const LayerWeights& l, const FamilyTraits& family,
                    std::uint32_t num_experts) {
    DeviceLayer d;
    d.mixer = l.mixer;
    d.ffn = l.ffn;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one GEMV where they share
    // a quantization; otherwise they stay separate.
    if (l.mixer == MixerKind::kShortConv) {
      d.ssm_in = Copy(l.ssm_qkv);  // the short convolution's in_proj
    } else if (l.linear() && Stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.mixer == MixerKind::kGatedDeltaNet) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear() && Stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    if (l.ffn == FfnKind::kRoutedMoe) {
      // Without a shared expert, routing is on sigmoid scores with a
      // selection bias, where near ties are common: the router stays F32.
      // A pruned layer's router is padded with zero rows to num_experts;
      // routing reads only the first ffn_gate_exps.experts logits.
      d.router = family.shared_expert ? Stack({&l.router, &l.shexp_gate_inp},
                                              num_experts - l.router.rows)
                                      : Copy(l.router);
    }
    d.expert_bias = Copy(l.expert_bias);
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg, const ModelWeights* decode,
    const core::GgufReader* decode_reader) {
  // The routed gate/up kernels have no Q6_K; reject it before allocating
  // device weights. Q4_K/Q6_K down experts are re-encoded below, and the
  // embedding, head and dense projections read K-quants directly.
  const auto layer_supported = [&](const LayerWeights& l) {
    for (const TensorRef* t : {&l.ffn_gate_exps, &l.ffn_up_exps}) {
      if (t->type == core::GgmlType::kQ6_K) {
        if (error_msg != nullptr)
          *error_msg = "unsupported HIP routed expert format Q6_K: " +
                       std::string(t->name);
        return false;
      }
    }
    return true;
  };
  if (!std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  const auto decode_base = static_cast<std::uint32_t>(shards.size());
  if (decode != nullptr) {
    if (decode_reader == nullptr) {
      if (error_msg)
        *error_msg = "decode weights require their bound reader";
      return nullptr;
    }
    const auto extra = decode_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  if (decode != nullptr && w.output.data != w.token_embd.data &&
      decode->output.type != w.output.type) {
    // The head's calibrated twin (decode logits only; prefill reads the
    // primary for its last row).
    up.shard_base = decode_base;
    m->alt_.push_back(up.Copy(decode->output));
    up.shard_base = 0;
    m->output_.alt = &m->alt_.back();
  }
  m->hc_head_ = up.Mixer(w.hc_head);
  m->layers_.reserve(w.layers.size());
  for (std::size_t i = 0; i < w.layers.size(); ++i) {
    m->layers_.push_back(
        up.Layer(w.layers[i], w.config.Traits(), w.config.num_experts));
    if (decode != nullptr && i < decode->layers.size()) {
      up.shard_base = decode_base;
      up.DenseAlt(decode->layers[i], m->layers_.back(), m->alt_);
      up.shard_base = 0;
    }
    if (!up.ok) {
      return nullptr;
    }
  }
  if (decode != nullptr) {
    std::fprintf(
        stderr,
        "decode twins: %zu tensors from --gguf-decode-override, head %s\n",
        m->alt_.size(),
        decode->output.type != w.output.type ? "with head twin"
                                             : "no head twin");
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ =
        up.Layer(mtp->block, mtp->config.Traits(), mtp->config.num_experts);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  // Re-encodes one matrix on the device into `type` (`bytes` long).
  const auto replace = [&](DeviceTensor& t, core::GgmlType type,
                           std::size_t old_bytes, std::size_t bytes,
                           const auto& launch) {
    void* out = nullptr;
    if (hipMalloc(&out, bytes + kTailMargin) != hipSuccess)
      return false;
    (void)hipMemset(static_cast<std::uint8_t*>(out) + bytes, 0, kTailMargin);
    launch(t.data, out);
    (void)hipDeviceSynchronize();
    std::erase(m->allocations_, t.data);
    (void)hipFree(t.data);
    m->allocations_.push_back(out);
    m->bytes_ += bytes - old_bytes;
    t.data = out;
    t.type = type;
    return true;
  };
  // Q4_0 matrices are re-encoded exactly into formats the kernels read:
  // dense as Q8_0, routed down experts as Q5_1 (gate/up stay Q4_0 and are
  // packed below).
  const auto reencode = [&](DeviceTensor& t, bool q5_1) {
    const std::size_t blocks = std::size_t{t.rows} * t.experts * (t.cols / 32);
    if (!replace(t, q5_1 ? core::GgmlType::kQ5_1 : core::GgmlType::kQ8_0,
                 blocks * 18, blocks * (q5_1 ? 24 : 34),
                 [&](const void* in, void* out) {
                   ReencodeQ4_0(in, out, blocks, q5_1, nullptr);
                 }))
      return false;
    if (!q5_1)
      m->max_q8_cols_ = std::max<std::size_t>(m->max_q8_cols_, t.cols);
    return true;
  };
  // ponytail: llama.cpp's K-quant mixes store some routed down experts as
  // Q4_K or Q6_K, which the routed kernels do not read. Q4_K becomes Q5_1
  // (its F16 sub-block scale and min rounded), Q6_K becomes Q8_0 (codes
  // re-rounded per 32 values); native down kernels avoid that rounding if
  // QA shows it matters.
  const auto transcode_down = [&](DeviceTensor& t) {
    const std::size_t rows = std::size_t{t.rows} * t.experts;
    const std::size_t blocks = rows * (t.cols / 32);
    if (t.type == core::GgmlType::kQ4_K)
      return replace(t, core::GgmlType::kQ5_1, blocks / 8 * 144, blocks * 24,
                     [&](const void* in, void* out) {
                       TranscodeQ4KToQ5_1(in, out, rows, t.cols, nullptr);
                     });
    return replace(t, core::GgmlType::kQ8_0, blocks / 8 * 210, blocks * 34,
                   [&](const void* in, void* out) {
                     TranscodeQ6KToQ8_0(in, out, rows, t.cols, nullptr);
                   });
  };
  std::vector<DeviceLayer*> layers;
  for (auto& l : m->layers_)
    layers.push_back(&l);
  if (m->has_mtp_)
    layers.push_back(&m->mtp_);
  for (DeviceLayer* l : layers) {
    for (DeviceTensor* t :
         {&l->ssm_qkv, &l->ssm_gate, &l->ssm_out, &l->ssm_in, &l->attn_q,
          &l->attn_k, &l->attn_v, &l->attn_out, &l->attn_qkv, &l->shexp_gate,
          &l->shexp_up, &l->shexp_down, &l->ffn_down_exps}) {
      if (t->data != nullptr && t->type == core::GgmlType::kQ4_0 &&
          !reencode(*t, t->experts > 1)) {
        if (error_msg != nullptr)
          *error_msg = "Q4_0 re-encoding failed";
        return nullptr;
      }
    }
    DeviceTensor& down = l->ffn_down_exps;
    if (down.data != nullptr &&
        (down.type == core::GgmlType::kQ4_K ||
         down.type == core::GgmlType::kQ6_K) &&
        !transcode_down(down)) {
      if (error_msg != nullptr)
        *error_msg = "routed down expert re-encoding failed";
      return nullptr;
    }
  }
  // Routed experts are permuted in place into the packed layout (moe.hpp).
  // WeightType values are the GGUF type ids.
  const auto type = [](core::GgmlType t) { return static_cast<WeightType>(t); };
  const auto pack = [&](const DeviceLayer& l) {
    if (l.ffn_gate_exps.data == nullptr ||
        !RoutedPacked(type(l.ffn_gate_exps.type), type(l.ffn_down_exps.type)))
      return true;
    for (const DeviceTensor* t :
         {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps}) {
      if (!PackRoutedExperts(t->data, type(t->type),
                             std::size_t{t->rows} * t->experts, t->cols,
                             nullptr))
        return false;
    }
    return true;
  };
  if (!std::all_of(m->layers_.begin(), m->layers_.end(), pack) ||
      (m->has_mtp_ && !pack(m->mtp_))) {
    if (error_msg != nullptr)
      *error_msg = "routed expert packing failed";
    return nullptr;
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

bool DeviceModel::UploadDspark(const DsparkWeights& w,
                               const core::GgufReader& reader,
                               std::string* error_msg) {
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager)
    return false;
  std::vector<Conversion> conversions;
  Uploader up{*stager,        conversions,  allocations_, bytes_,
              max_half_cols_, max_q8_cols_, error_msg};
  auto& d = dspark_;
  d.block_size = w.block_size;
  d.heads = w.heads;
  d.kv_heads = w.kv_heads;
  d.head_dim = w.head_dim;
  d.rank = w.rank;
  d.mask_token = w.mask_token;
  d.rope_theta = w.rope_theta;
  d.rms_eps = w.rms_eps;
  d.conf_bias = *static_cast<const float*>(w.conf_b.data);
  d.target_layers = w.target_layers;
  d.fc = up.Copy(w.fc);
  d.enc_norm = up.Copy(w.enc_norm);
  d.out_norm = up.Copy(w.out_norm);
  d.markov_w1 = up.Copy(w.markov_w1);
  d.markov_w2 = up.Copy(w.markov_w2);
  d.conf_w = up.Copy(w.conf_w);
  for (const auto& b : w.blocks)
    d.blocks.push_back({up.Copy(b.attn_norm), up.Copy(b.q), up.Copy(b.k),
                        up.Copy(b.v), up.Copy(b.q_norm), up.Copy(b.k_norm),
                        up.Copy(b.out), up.Copy(b.ffn_norm), up.Copy(b.gate),
                        up.Copy(b.up), up.Copy(b.down)});
  if (!up.ok || !stager->Finish(error_msg)) {
    d.blocks.clear();
    return false;
  }
  // The projections run on the Q8_0 GEMV tier.
  const auto bf16_to_q8 = [&](DeviceTensor& t) {
    if (t.type != core::GgmlType::kBF16)
      return true;
    const std::size_t bytes = std::size_t{t.rows} * (t.cols / 32) * 34;
    void* q8 = nullptr;
    if (hipMalloc(&q8, bytes + kTailMargin) != hipSuccess)
      return false;
    (void)hipMemset(static_cast<std::uint8_t*>(q8) + bytes, 0, kTailMargin);
    TranscodeBf16ToQ8_0(t.data, q8, t.rows, t.cols, nullptr);
    (void)hipDeviceSynchronize();
    std::erase(allocations_, t.data);
    (void)hipFree(t.data);
    allocations_.push_back(q8);
    bytes_ += bytes - std::size_t{t.rows} * t.cols * 2;
    max_q8_cols_ = std::max<std::size_t>(max_q8_cols_, t.cols);
    t.data = q8;
    t.type = core::GgmlType::kQ8_0;
    return true;
  };
  bool ok = bf16_to_q8(d.fc);
  for (auto& b : d.blocks)
    for (DeviceTensor* t : {&b.q, &b.k, &b.v, &b.out, &b.gate, &b.up, &b.down})
      ok = ok && bf16_to_q8(*t);
  // Injection reads every layer's K and V of the same fused rows: stack
  // them into one GEMV ([k0 v0 k1 v1 ...] rows) and the K norms into one
  // [layers][head_dim] table for the fused store kernel.
  const std::size_t kv_bytes = std::size_t{d.blocks.front().k.rows} *
                               (d.blocks.front().k.cols / 32) * 34;
  const std::size_t all_bytes = 2 * d.blocks.size() * kv_bytes;
  ok = ok && d.blocks.front().k.type == core::GgmlType::kQ8_0 &&
       hipMalloc(&d.kv_all.data, all_bytes + kTailMargin) == hipSuccess &&
       hipMalloc(&d.k_norms.data,
                 d.blocks.size() * d.head_dim * sizeof(float)) == hipSuccess;
  if (ok) {
    allocations_.push_back(d.kv_all.data);
    allocations_.push_back(d.k_norms.data);
    bytes_ += all_bytes + kTailMargin + d.blocks.size() * d.head_dim * 4;
    auto* kv = static_cast<std::uint8_t*>(d.kv_all.data);
    auto* norms = static_cast<float*>(d.k_norms.data);
    for (std::size_t b = 0; b < d.blocks.size(); ++b) {
      const auto& blk = d.blocks[b];
      ok = ok &&
           hipMemcpy(kv + 2 * b * kv_bytes, blk.k.data, kv_bytes,
                     hipMemcpyDeviceToDevice) == hipSuccess &&
           hipMemcpy(kv + (2 * b + 1) * kv_bytes, blk.v.data, kv_bytes,
                     hipMemcpyDeviceToDevice) == hipSuccess &&
           hipMemcpy(norms + b * d.head_dim, blk.k_norm.data,
                     d.head_dim * sizeof(float),
                     hipMemcpyDeviceToDevice) == hipSuccess;
    }
    (void)hipMemset(kv + all_bytes, 0, kTailMargin);
    d.kv_all.type = core::GgmlType::kQ8_0;
    d.kv_all.cols = d.blocks.front().k.cols;
    d.kv_all.rows = static_cast<std::uint32_t>(2 * d.blocks.size() *
                                               d.blocks.front().k.rows);
  }
  if (!ok || hipDeviceSynchronize() != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "DSpark Q8_0 transcoding failed";
    d.blocks.clear();
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
