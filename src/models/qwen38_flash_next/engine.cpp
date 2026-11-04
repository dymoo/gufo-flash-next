#include "src/models/qwen38_flash_next/engine.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/parallel.hpp"
#include "src/models/qwen38_flash_next/dspark_policy.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtmd_vision.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/spec_trace.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

// Adds one prepared decode cycle to the executed step's cost shape.
template<class Pending>
void AddStepShape(StepShape& shape, const Pending& pending) {
  const auto rows = static_cast<std::uint32_t>(pending.chain.size());
  if (rows == 0)
    return;
  shape.sessions += 1;
  shape.rows += rows;
  if (pending.speculative && !pending.lookup_used) {
    shape.mtp_drafts += rows - 1;
    shape.rounds = std::max(shape.rounds, rows - 1);
  }
  shape.row_context +=
      static_cast<float>(rows) * static_cast<float>(pending.base) / 65536.0F;
}

// gfx1151 pp4096 at depths 0/4096: the 512/1024/2048/4096 sweep favored
// 2048; larger chunks used more scratch without improving throughput.
constexpr std::uint32_t kPrefillChunkTokens = 2048;

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  /// Session::Variant() of the captured context: the steering adapter, plus
  /// the fast model's kept experts << 16.
  std::uint32_t adapter{0};
  MtpLengthState lookup_policy;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

struct Model::SharedPages {
  struct Entry {
    std::vector<std::int32_t> tokens;
    std::vector<std::uint8_t> image;
    std::uint32_t adapter{0};
    std::vector<rocm::Executor::SnapshotPage> pages;
  };
  // ponytail: linear scan of a few recent contexts; index by prefix hash
  // if many distinct shared prompts must stay mappable.
  static constexpr std::size_t kCapacity = 32;
  std::mutex mutex;
  std::deque<Entry> entries;  ///< newest first
};

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  m->shared_pages_ = std::make_unique<SharedPages>();
  if (options.decode_concurrency == 0 ||
      options.decode_concurrency > rocm::Executor::kMaxBatchSessions) {
    AssignError(error_msg, "decode concurrency must be between one and 512");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() && !options.dspark_model_path.empty()) {
    AssignError(error_msg, "choose either the MTP or the DSpark drafter");
    return nullptr;
  }
  if ((!options.mtp_model_path.empty() || !options.dspark_model_path.empty()) &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  if (options.adapters.size() > rocm::Executor::kMaxAdapters) {
    AssignError(error_msg, "at most 255 steering adapters");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights =
      ModelWeights::Bind(*m->reader_, error_msg, options.layer_slice_first,
                         options.layer_slice_count, options.experts_used);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  for (const auto& a : options.adapters) {
    if (a.name.empty() || a.direction.size() != c.hidden_size) {
      AssignError(error_msg, "adapter '" + a.name + "' must name a " +
                                 std::to_string(c.hidden_size) +
                                 "-wide direction for this model");
      return nullptr;
    }
  }
  const auto routing =
      MoeRouting::Parse(options.moe_routing, c.num_experts_used, error_msg);
  if (!routing) {
    return nullptr;
  }
  if (routing->mode != MoeRouting::Mode::kExact &&
      (c.num_experts > 512 || c.num_experts_used > 32)) {
    AssignError(error_msg, "--moe-routing policies support 512 experts");
    return nullptr;
  }
  if (c.Traits().vision == VisionStack::kMtmd &&
      !options.vision_model_path.empty()) {
    m->mtmd_ = MtmdVision::Open(model_path, options.vision_model_path,
                                c.hidden_size, error_msg);
    if (!m->mtmd_)
      return nullptr;
  } else {
    try {
      m->vision_ = qwen::vision::Encoder::Open(
          model_path, options.vision_model_path, c.hidden_size);
    } catch (const std::exception& e) {
      AssignError(error_msg, e.what());
      return nullptr;
    }
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(
        m->reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp =
        MtpWeights::Bind(*m->mtp_reader_, c, error_msg, options.experts_used);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  if (!options.dspark_model_path.empty()) {
    m->dspark_reader_ =
        core::GgufReader::OpenFile(options.dspark_model_path, error_msg);
    if (!m->dspark_reader_)
      return nullptr;
    auto dspark = DsparkWeights::Bind(*m->dspark_reader_, c, error_msg);
    if (!dspark)
      return nullptr;
    m->dspark_weights_ = std::make_unique<DsparkWeights>(std::move(*dspark));
  }
  if (!options.gguf_decode_override.empty()) {
    core::GgufReader::SetDecodeTensorOverride(options.gguf_decode_override);
    m->decode_reader_ = core::GgufReader::OpenFile(
        model_path, error_msg, core::GgufReader::OverrideSlot::kDecode);
    core::GgufReader::SetDecodeTensorOverride({});
    if (!m->decode_reader_)
      return nullptr;
    auto decode = ModelWeights::Bind(
        *m->decode_reader_, error_msg, options.layer_slice_first,
        options.layer_slice_count, options.experts_used);
    if (!decode)
      return nullptr;
    m->decode_weights_ = std::make_unique<ModelWeights>(std::move(*decode));
  }
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, m->mtp_weights_.get(), m->mtp_reader_.get(),
      error_msg, m->decode_weights_.get(), m->decode_reader_.get());
  if (m->device_ && !options.gguf_decode_override.empty() &&
      m->device_->decode_twin_count() == 0) {
    AssignError(error_msg,
                "--gguf-decode-override holds no tensor of this model");
    return nullptr;
  }
  if (!m->device_ ||
      (m->dspark_weights_ &&
       !m->device_->UploadDspark(*m->dspark_weights_, *m->dspark_reader_,
                                 error_msg))) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.routing = *routing;
  if (options.routed_prefill_a4 != 0 && options.routed_prefill_a4 != 64 &&
      options.routed_prefill_a4 != 128 && options.routed_prefill_a4 != 256) {
    AssignError(error_msg, "--routed-prefill-a4 must be 0, 64, 128 or 256");
    return nullptr;
  }
  exec.routed_a4_group = static_cast<int>(options.routed_prefill_a4);
  exec.routed_down4 = options.routed_down_w4;
  if (options.routed_w4 != 0 && options.routed_w4 != 64 &&
      options.routed_w4 != 128) {
    AssignError(error_msg, "--routed-w4 must be 0, 64 or 128");
    return nullptr;
  }
  exec.routed_sym = static_cast<int>(options.routed_w4);
  exec.prefill_chunk =
      options.prefill_chunk != 0 ? options.prefill_chunk : 2048;
  exec.max_batch = m->PrefillCapacity();
  exec.max_logit_rows =
      m->mtp_weights_ || m->dspark_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  exec.kv_format = options.kv_cache;
  exec.gdn_state = options.gdn_state;
  exec.max_batch_sessions = options.decode_concurrency;
  exec.adapters = options.adapters;
  exec.dense_w4a4 = options.dense_prefill_w4a4;
  exec.dense_w4a4_weights = options.dense_prefill_weights;
  exec.decode_twin_rows = options.decode_twin_rows;
  exec.router_norm_k = options.router_norm_k;
  exec.fast_experts = options.fast_experts;
  exec.draft_vocab = options.mtp_draft_vocab;
  exec.draft_vocab_size = options.mtp_draft_vocab_size;
  SetSpecPolicy(options.mtp_policy_adaptive);
  exec.residual_f16 = options.prefill_residual_f16;
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::LossyIdentity() const {
  std::string id;
  const auto line = [&](std::string_view key, const std::string& value) {
    id.append(key).append("=").append(value).append("\n");
  };
  if (options_.dense_prefill_w4a4) {
    line("dense_prefill", "w4a4");
    if (!options_.dense_prefill_weights.empty())
      line("dense_prefill_weights", options_.dense_prefill_weights);
  }
  if (options_.gdn_state != GdnStateFormat::kF32)
    line("gdn_state", std::string(GdnStateFormatName(options_.gdn_state)));
  if (options_.routed_prefill_a4 != 0)
    line("routed_prefill_a4", std::to_string(options_.routed_prefill_a4));
  if (options_.routed_down_w4)
    line("routed_down_w4", "1");
  if (options_.routed_w4 != 0)
    line("routed_w4", std::to_string(options_.routed_w4));
  if (options_.prefill_residual_f16)
    line("prefill_residual", "f16");
  if (options_.experts_used != 0)
    line("experts_used", std::to_string(options_.experts_used));
  if (options_.router_norm_k != 0)
    line("router_norm_k", std::to_string(options_.router_norm_k));
  if (!options_.gguf_decode_override.empty()) {
    line("gguf_decode_override", options_.gguf_decode_override);
    line("decode_twin_rows", std::to_string(options_.decode_twin_rows));
  }
  return id;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(options_.prefill_capacity != 0 ? options_.prefill_capacity
                                                 : kPrefillChunkTokens,
                  options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

bool Model::HasDspark() const noexcept {
  return device_->dspark() != nullptr;
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionClaimBytes(core::SessionMode,
                                     std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ || mtmd_
          ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                    3 * sizeof(std::int32_t)) +
                64
          : 0;
  return rocm::Executor::UnpooledSessionBytes() + vision;
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      SessionClaimBytes(mode, context) - rocm::Executor::UnpooledSessionBytes();
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0) +
         vision;
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

std::size_t Model::KvBytes(core::SessionMode mode,
                           std::uint32_t context) const noexcept {
  return executor_->GrowableBytes(mode, context);
}

void Model::SetKvPool(std::size_t bytes) const {
  executor_->SetKvPool(bytes);
}

void Model::SetPaging(std::string directory, std::uint64_t ram_budget) const {
  executor_->SetPaging(std::move(directory), ram_budget);
}

std::size_t Model::KvPoolUsed() const {
  return executor_->kv_pool().used();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

std::size_t Session::CommittedKvBytes() const noexcept {
  return session_->CommittedKvBytes();
}

std::size_t Session::UnpooledBytes() const noexcept {
  return rocm::Executor::UnpooledSessionBytes() + session_->VisionBytes();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

bool Session::DsparkEnabled() const noexcept {
  return session_->dspark_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()),
      lookup_length_(model_->options_.max_draft_tokens,
                     model_->DecodeConcurrency()) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  lookup_length_.Reset();
  spec_cal_.Reset();
  spec_probe_.valid = false;
  spec_idle_ = 0;
  spec_idle_skip_ = 0;
  dspark_skip_ = 0;
  dspark_misses_ = 0;
  dspark_yield_ = kDsparkYieldPrior;
  dspark_probe_ = false;
  lookup_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const auto identity =
      prompt ? prompt->cache_identity : std::vector<std::uint8_t>{};
  if (!tokens_.empty() && identity != image_identity_)
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(std::move(prompt), model_->vision_,
                            model_->executor_->stream());
  image_identity_ = identity;
  valid_ = was_valid;
}

bool Session::SetAdapter(std::uint32_t adapter, std::string* error_msg) {
  if (adapter > model_->options_.adapters.size()) {
    AssignError(error_msg, "unknown steering adapter");
    return false;
  }
  if (adapter == Adapter())
    return true;
  Reset();
  session_->set_adapter(static_cast<std::uint8_t>(adapter));
  return true;
}

std::uint32_t Session::Adapter() const noexcept {
  return session_->adapter();
}

bool Session::SetFastModel(bool fast, std::string* error_msg) {
  if (fast && model_->options_.fast_experts == 0) {
    AssignError(error_msg,
                "no fast model: start the server with --fast-experts");
    return false;
  }
  const auto keep =
      static_cast<std::uint8_t>(fast ? model_->options_.fast_experts : 0);
  if (keep == session_->route_keep())
    return true;
  Reset();
  session_->set_route_keep(keep);
  return true;
}

bool Session::FastModel() const noexcept {
  return session_->route_keep() != 0;
}

std::uint32_t Session::Variant() const noexcept {
  return Adapter() | (std::uint32_t{session_->route_keep()} << 16);
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(), image_identity_.size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  if (!pin)
    return nullptr;
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), image_identity_.size());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes =
          static_cast<std::uint32_t>(image_identity_.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
      .adapter = Variant(),
      .lookup_policy = lookup_length_.State(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  if (!image_identity_.empty())
    std::memcpy(out, image_identity_.data(), image_identity_.size());
  out += image_identity_.size();
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  if (!model_->executor_->SaveSnapshot(
          *session_, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg)) {
    return nullptr;
  }
  std::vector<rocm::Executor::SnapshotPage> pages;
  model_->executor_->SharePages(*session_, &pages);
  if (!pages.empty()) {
    auto& shared = *model_->shared_pages_;
    const std::lock_guard lock(shared.mutex);
    std::erase_if(shared.entries, [&](const auto& entry) {
      return entry.tokens == tokens_ && entry.image == image_identity_ &&
             entry.adapter == Variant();
    });
    shared.entries.push_front(
        {tokens_, image_identity_, Variant(), std::move(pages)});
    if (shared.entries.size() > Model::SharedPages::kCapacity)
      shared.entries.pop_back();
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  if (!pin)
    return false;
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size,
                                                 header.image_identity_bytes) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  auto restored_lookup = lookup_length_;
  if (!restored_policy.Restore(header.draft_policy) ||
      !restored_lookup.Restore(header.lookup_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() && image_identity != image_identity_) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  if (header.adapter != Variant()) {
    AssignError(error_msg,
                "snapshot belongs to another steering adapter or model");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  if (image_identity.empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  const auto next_drafts =
      MtpEnabled() ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                            header.token_count)
      : DsparkEnabled() ? std::min(remaining ? remaining - 1 : 0,
                                   model_->options_.max_draft_tokens)
                        : 0;
  std::vector<rocm::Executor::SnapshotPage> pages;
  {
    auto& shared = *model_->shared_pages_;
    const std::lock_guard lock(shared.mutex);
    for (auto entry = shared.entries.begin(); entry != shared.entries.end();
         ++entry) {
      if (entry->tokens == tokens && entry->image == image_identity &&
          entry->adapter == header.adapter) {
        pages = entry->pages;
        // Most recently restored first: a system prompt that every new
        // conversation maps outlives one-off contexts.
        std::rotate(shared.entries.begin(), entry, std::next(entry));
        break;
      }
    }
  }
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg, next_drafts, pages)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  image_identity_ = std::move(image_identity);
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  lookup_length_ = restored_lookup;
  dspark_skip_ = 0;
  dspark_misses_ = 0;
  dspark_yield_ = kDsparkYieldPrior;
  dspark_probe_ = false;
  lookup_.Reset();
  stats_ = {};
  valid_ = true;
  return true;
}

std::vector<Session::SnapshotRegion> Session::DescribeSnapshot(
    std::span<const std::uint8_t> payload) {
  std::vector<SnapshotRegion> regions;
  SessionSnapshotHeader h{};
  if (payload.size() < sizeof(h))
    return regions;
  std::memcpy(&h, payload.data(), sizeof(h));
  const std::uint64_t host = SessionSnapshotHostBytes(
      h.token_count, h.vocab_size, h.image_identity_bytes);
  if (h.magic != kSessionSnapshotMagic || payload.size() < host)
    return regions;
  std::uint64_t offset = 0;
  const auto add = [&](std::string name, std::uint64_t bytes) {
    regions.push_back({std::move(name), offset, bytes});
    offset += bytes;
  };
  add("session header (tokens=" + std::to_string(h.token_count) +
          " hidden_rows=" + std::to_string(h.hidden_rows) + ")",
      sizeof(h));
  add("image identity", h.image_identity_bytes);
  add("tokens", std::uint64_t{h.token_count} * sizeof(std::int32_t));
  add("logits", std::uint64_t{h.vocab_size} * sizeof(float));
  for (auto& r : rocm::Executor::DescribeSnapshot(payload.subspan(host)))
    regions.push_back({std::move(r.name), host + r.offset, r.bytes});
  return regions;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {
  // Snapshot copies first-touch hundreds of MiB. Let Linux back the interior
  // with transparent huge pages instead of faulting one 4 KiB page at a time.
  // Advise only complete pages belonging to this allocation; this is optional
  // and does not pin memory or change the serialized payload.
  const long page = sysconf(_SC_PAGESIZE);
  if (page > 0) {
    const auto address = reinterpret_cast<std::uintptr_t>(data_.get());
    const auto skip = (page - address % page) % page;
    if (size > skip) {
      const auto length = (size - skip) / page * page;
      if (length != 0)
        (void)madvise(data_.get() + skip, length, MADV_HUGEPAGE);
    }
  }
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_) {
    return false;
  }
  std::memcpy(destination.data(), data_.get(), size_);
  return true;
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DsparkDraft(std::int32_t anchor, std::span<std::int32_t> tokens,
                          std::span<float> confidence, std::string* error_msg) {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  const std::array item{
      rocm::Executor::DsparkItem{session_.get(), anchor, tokens, confidence}};
  return pin && model_->executor_->DsparkDraft(item, error_msg);
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates, bool traced) {
  if (LookupOnly())
    return true;  // GUFO_LOOKUP_ONLY: the draft block never runs
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && (candidates == nullptr || traced) ? &draft_token_
                                                                 : nullptr,
           .candidates = candidates},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.empty() || LookupOnly())
    return true;
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill) {
  rocm::Executor& exec = *model_->executor_;
  for (std::size_t off = 0; off < tokens.size(); off += exec.max_batch()) {
    const std::size_t n =
        std::min<std::size_t>(exec.max_batch(), tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), mode, error_msg)) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n - std::min<std::size_t>(n, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  if (!pin)
    return false;
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  if (!pin)
    return false;
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  std::uint32_t budget{0};  ///< drafts this cycle may carry
  std::array<std::int32_t, kMaxMtpDraftTokens> lookup{};
  std::uint32_t lookup_size{0};
  std::uint32_t lookup_match{0};  ///< backward match length of the lookup
  bool lookup_used{false};
  /// DSpark proposals of this cycle (drafted with the batch or here).
  std::array<std::int32_t, kMaxMtpDraftTokens> dspark_tokens{};
  std::array<float, kMaxMtpDraftTokens> dspark_confidence{};
  bool dspark_drafted{false};
  bool dspark{false};  ///< the chain carries DSpark drafts
  // GUFO_SPEC_TRACE diagnostics: draft phase times and, for greedy MTP
  // chains, the compact candidates of every drafted depth.
  double catchup_ms{0};
  double rounds_ms{0};
  // Greedy C1 combined policy: per drafted token source and the confidence
  // (MTP) it was drafted at, the rounds paid and the planned tokens.
  bool spec2{false};
  bool gate_tail{false};  ///< batched greedy chain: candidates per depth
  const SpecCalibrator* cal{nullptr};  ///< calibration of gate_tail chains
  bool spec2_lookup{false};            ///< some draft came from prompt lookup
  bool spec2_all_lookup{false};
  std::uint32_t spec2_rounds{0};
  float spec2_expected{1.0F};
  std::array<DraftSource, kMaxMtpDraftTokens> spec2_source{};
  std::array<float, kMaxMtpDraftTokens> spec2_conf{};
  std::vector<MtpCandidateLogits> traced;
  mutable int reject_rank{-2};  ///< target token's rank at the rejection
};

namespace {

std::size_t CandidateCount(const MtpCandidateLogits& c) {
  return c.size == 0 || c.size > kMtpCandidates ? kMtpCandidates : c.size;
}

/// Top-1 probability of the draft within its compact candidates.
float CandidateConfidence(const MtpCandidateLogits& c) {
  const auto n = CandidateCount(c);
  float top = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i)
    top = std::max(top, c.logits[i]);
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += std::exp(static_cast<double>(c.logits[i] - top));
  return sum > 0 ? static_cast<float>(1.0 / sum) : 0.0F;
}

/// Rank of `token` among the candidates (0 = argmax), -1 when absent.
int CandidateRank(const MtpCandidateLogits& c, std::int32_t token) {
  const auto n = CandidateCount(c);
  for (std::size_t i = 0; i < n; ++i) {
    if (static_cast<std::int32_t>(c.ids[i]) != token)
      continue;
    int rank = 0;
    for (std::size_t j = 0; j < n; ++j)
      rank += c.logits[j] > c.logits[i] ? 1 : 0;
    return rank;
  }
  return -1;
}

/// Appends the trace fields of one finished cycle to `line`.
void TraceCycle(std::string& line, const char* source, std::size_t width,
                std::size_t keep, bool stop, std::uint32_t base,
                std::uint32_t lookup_size, std::uint32_t lookup_match,
                double catchup_ms, double rounds_ms,
                const std::vector<MtpCandidateLogits>& traced,
                int reject_rank) {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer),
                " src=%s w=%zu keep=%zu stop=%d base=%u lk=%u ml=%u cu=%.2f "
                "rd=%.2f",
                source, width, keep, stop ? 1 : 0, base, lookup_size,
                lookup_match, catchup_ms, rounds_ms);
  line += buffer;
  if (!traced.empty()) {
    line += " conf=";
    for (std::size_t d = 0; d < traced.size(); ++d) {
      std::snprintf(buffer, sizeof(buffer), d == 0 ? "%.3f" : ",%.3f",
                    CandidateConfidence(traced[d]));
      line += buffer;
    }
    std::snprintf(buffer, sizeof(buffer), " rank=%d", reject_rank);
    line += buffer;
  }
}

}  // namespace

bool Session::DsparkWants(const PendingDecode& pending,
                          std::uint32_t cohort) const noexcept {
  return DsparkEnabled() && !pending.chain.empty() && pending.budget != 0 &&
         dspark_skip_ == 0 &&
         (dspark_probe_ || DsparkPassPays(dspark_yield_, cohort));
}

void Session::DsparkPass(std::uint32_t accepted) noexcept {
  // Two unproductive passes in a row back off exponentially (2..32 tokens
  // of plain decoding) so unpredictable text stops paying for the drafter.
  dspark_probe_ = false;
  dspark_yield_ = 0.9F * dspark_yield_ + 0.1F * static_cast<float>(accepted);
  if (accepted != 0) {
    dspark_misses_ = 0;
    return;
  }
  if (++dspark_misses_ >= 2)
    dspark_skip_ = 1U << std::min(dspark_misses_ - 1, 5U);
}

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::BeginDecode(const DecodeRequest& request, PendingDecode* pending,
                          std::string* error_msg) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  if (cap == 0) {
    result->stop = true;
    return true;
  }
  // Drawing the anchor first consumes the same RNG as before: choosing a
  // width draws nothing.
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (request.stop_at_eos && model_->IsStopToken(anchor)) {
    result->stop = true;
    return true;
  }
  pending->chain = {anchor};
  pending->base = static_cast<std::uint32_t>(tokens_.size());
  pending->budget = MtpEnabled() || DsparkEnabled()
                        ? static_cast<std::uint32_t>(std::min<std::size_t>(
                              cap - 1, model_->options_.max_draft_tokens))
                        : 0;
  if (pending->budget != 0 && MtpEnabled()) {
    pending->lookup_size = static_cast<std::uint32_t>(lookup_.Find(
        tokens_, anchor, std::span(pending->lookup).first(pending->budget)));
    pending->lookup_match = lookup_.MatchLength();
  }
  return true;
}

MtpBatchController::Row Session::DraftRow(const DecodeRequest& request,
                                          const PendingDecode& pending) const {
  MtpBatchController::Row row{&draft_length_, &lookup_length_, pending.budget,
                              pending.lookup_size, pending.base};
  if (request.sampler->config().uses_random_sampling())
    row.fixed = ChooseSessionDraft(draft_length_, lookup_length_,
                                   pending.budget, pending.lookup_size,
                                   pending.base, draft_length_.concurrency());
  return row;
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head, std::optional<DraftPlan> plan,
                            std::uint32_t cohort) {
  if (pending->chain.empty())
    return true;  // BeginDecode stopped the request.
  if (pending->spec2)
    return PrepareSpecChain(request, pending, error_msg);
  auto& sampler = *request.sampler;
  rocm::Executor& exec = *model_->executor_;
  const auto anchor = pending->chain.front();
  if (DsparkEnabled()) {
    // One block proposes every draft; its confidence head picks the length.
    // The drafts are deterministic, so exact target draws decide them
    // (VerifyDraft): the point-mass case of speculative sampling. Output
    // never depends on the length, so it may follow the live cohort.
    if (pending->budget == 0)
      return true;
    if (!DsparkWants(*pending, cohort)) {
      // Backing off, or a cohort whose passes cost more than they yield:
      // decode plainly, then probe once to refresh the yield estimate.
      if (dspark_skip_ != 0) {
        --dspark_skip_;
      } else {
        dspark_skip_ = 128;
        dspark_probe_ = true;
      }
      return true;
    }
    const auto rows = pending->budget;
    const auto drafts = std::span(pending->dspark_tokens).first(rows);
    const auto confidence = std::span(pending->dspark_confidence).first(rows);
    if (!pending->dspark_drafted) {
      const std::array item{rocm::Executor::DsparkItem{session_.get(), anchor,
                                                       drafts, confidence}};
      if (!exec.DsparkDraft(item, error_msg))
        return false;
    }
    const std::size_t width =
        1 + (drafts[0] < 0 ? 0 : ChooseDsparkLength(confidence, cohort));
    if (width < 2) {
      DsparkPass(0);
      return true;
    }
    pending->dspark = true;
    pending->chain.insert(pending->chain.end(), drafts.begin(),
                          drafts.begin() + (width - 1));
    pending->width = width;
    pending->speculative = true;
    pending->gpu_greedy = sampler.config().can_use_unmodified_argmax();
    pending->gpu_verification = pending->gpu_greedy;
    if (!pending->gpu_greedy && verify_logits_.empty())
      verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
    return true;
  }
  const auto chosen =
      pending->budget == 0 ? DraftPlan{}
      : plan               ? *plan
                           : ChooseSessionDraft(draft_length_, lookup_length_,
                                                pending->budget, pending->lookup_size,
                                                pending->base, draft_length_.concurrency());
  const std::size_t width =
      1 + std::min({chosen.drafts, pending->budget,
                    chosen.lookup ? pending->lookup_size : pending->budget});
  auto& trace = SpecTrace::Get();
  const double t0 = trace.enabled() ? trace.Now() : 0.0;
  if (width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    if (trace.enabled())
      pending->catchup_ms = trace.Now() - t0;
    return true;
  }
  const bool gpu_greedy = sampler.config().can_use_unmodified_argmax();
  if (chosen.lookup) {
    // Deterministic drafts: exact target draws decide them (VerifyDraft),
    // so accepted and corrected tokens are the AR tokens for any sampler.
    if (!defer_head && !DraftCatchUp(anchor, false, error_msg))
      return false;
    if (trace.enabled())
      pending->catchup_ms = trace.Now() - t0;
    pending->chain.insert(pending->chain.end(), pending->lookup.begin(),
                          pending->lookup.begin() + (width - 1));
    pending->width = width;
    pending->speculative = true;
    pending->lookup_used = true;
    pending->gpu_greedy = gpu_greedy;
    pending->gpu_verification = gpu_greedy;
    if (!gpu_greedy && verify_logits_.empty())
      verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
    return true;
  }

  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_verification = gpu_greedy;
  // Experimental (GUFO_MTP_GATE=s): greedy chains draft up to the budget
  // and stop once the product of the drafts' top-1 confidences falls
  // below s. Verification is unchanged, so output is unchanged.
  const float gate = defer_head || sampled ? 0.0F : MtpConfidenceGate();
  const std::size_t max_width =
      gate > 0 ? 1 + std::size_t{pending->budget} : width;
  // Greedy traced or gated chains keep every depth's compact candidates.
  const auto& spec_knobs = SpecPolicyKnobs();
  const bool tail_gate = defer_head && !sampled && cohort >= 4 &&
                         spec_knobs.mode == 1 && spec_knobs.bgate &&
                         spec_knobs.mtp && !ForcedMtpDrafts();
  const bool traced = (trace.confidence() || gate > 0 || tail_gate) && !sampled;
  pending->gate_tail = tail_gate;
  pending->cal = tail_gate ? &spec_cal_ : nullptr;
  if (traced)
    pending->traced.assign(max_width - 1, MtpCandidateLogits{});
  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   sampled  ? &pending->candidates
                                   : traced ? pending->traced.data()
                                            : nullptr,
                                   traced)) {
    return false;
  }
  if (trace.enabled())
    pending->catchup_ms = trace.Now() - t0;
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler;
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = max_width;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  if (!defer_head) {
    float survival = 1.0F;
    while (pending->chain.size() < max_width) {
      if (gate > 0) {
        survival *=
            CandidateConfidence(pending->traced[pending->chain.size() - 1]);
        if (survival < gate)
          break;
      }
      AppendDraft(*pending);
      const auto depth = pending->chain.size() - 1;
      if (pending->chain.size() < max_width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = sampled  ? &pending->candidates
                             : traced ? &pending->traced[depth]
                                      : nullptr},
              error_msg)) {
        return false;
      }
    }
    if (gate > 0) {
      pending->width = pending->chain.size();
      pending->traced.resize(pending->chain.size() - 1);
      if (pending->width < 2)
        pending->speculative = false;
    }
    if (trace.enabled())
      pending->rounds_ms = trace.Now() - t0 - pending->catchup_ms;
  }
  return true;
}

bool Session::PrepareSpecChain(const DecodeRequest& request,
                               PendingDecode* pending, std::string* error_msg) {
  auto& sampler = *request.sampler;
  rocm::Executor& exec = *model_->executor_;
  const auto& knobs = SpecPolicyKnobs();
  const auto& costs = model_->spec_costs_;
  const bool exploring = knobs.explore != 0 && ++spec_explore_ >= knobs.explore;
  if (exploring)
    spec_explore_ = 0;
  const double aggr = knobs.aggr * (exploring ? 0.3 : 1.0);
  const auto anchor = pending->chain.front();
  const auto budget = pending->budget;
  const auto base = pending->base;
  const bool gpu_greedy = sampler.config().can_use_unmodified_argmax();
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto ms_since = [&](auto t) {
    return std::chrono::duration<double, std::milli>(now() - t).count();
  };
  const auto t0 = now();

  // The previous AR cycle's first draft was never verified; the anchor is the
  // token it predicted, a free label for the calibration.
  if (spec_probe_.valid) {
    if (spec_probe_.base + 1 == base) {
      if (spec_probe_.mtp_token >= 0)
        spec_cal_.ObserveMtp(spec_probe_.conf, anchor == spec_probe_.mtp_token);
      if (spec_probe_.lookup_token >= 0) {
        spec_cal_.ObserveLookup(spec_probe_.lookup_match, 1,
                                anchor == spec_probe_.lookup_token);
        if (spec_probe_.lookup_token == spec_probe_.mtp_token)
          spec_cal_.ObserveAgree(spec_probe_.lookup_match,
                                 anchor == spec_probe_.lookup_token);
      }
    }
    spec_probe_.valid = false;
  }

  const std::uint32_t lk_n = knobs.lookup ? pending->lookup_size : 0;
  const std::uint32_t ml = pending->lookup_match;
  const bool lookup_strong = lk_n != 0 && spec_cal_.Lookup(ml, 1) >= 0.5F;
  const bool head = knobs.mtp && (spec_idle_skip_ == 0 || lookup_strong);
  if (spec_idle_skip_ != 0)
    --spec_idle_skip_;

  const auto finish_ar = [&]() {
    pending->speculative = false;
    pending->chain.resize(1);
    pending->width = 0;
    pending->catchup_ms = ms_since(t0);
    return true;
  };

  if (!head) {
    // Hopeless drafts recently: plain decode keeps the draft block in sync
    // without its head.
    if (knobs.mtp && !DraftCatchUp(anchor, false, error_msg))
      return false;
    return finish_ar();
  }

  pending->traced.assign(std::max<std::uint32_t>(budget, 1),
                         MtpCandidateLogits{});
  if (!DraftCatchUp(anchor, true, error_msg, pending->traced.data(), true))
    return false;
  pending->catchup_ms = ms_since(t0);
  if (pending->traced[0].size == 0) {
    // The draft block had nothing to replay: no proposal this cycle.
    return finish_ar();
  }
  const std::int32_t mtp1 = draft_token_;
  const float conf1 = CandidateConfidence(pending->traced[0]);
  const float pm1 = spec_cal_.Mtp(conf1);

  // First draft: MTP, the lookup token, or both agreeing.
  std::int32_t first = mtp1;
  float p1 = pm1;
  auto source1 = DraftSource::kMtp;
  bool alive = false;    // every chain token so far equals the lookup's
  bool mtp_mode = true;  // MTP rounds can still extend the chain
  if (lk_n != 0) {
    if (pending->lookup[0] == mtp1) {
      p1 = spec_cal_.Agree(ml);
      source1 = DraftSource::kAgree;
      alive = true;
    } else if (const float pl = spec_cal_.Lookup(ml, 1); pl > pm1) {
      first = pending->lookup[0];
      p1 = pl;
      source1 = DraftSource::kLookup;
      alive = true;
      mtp_mode = false;
    }
  }
  const double c1 = costs.Cycle(1);
  if (p1 * c1 < aggr * (costs.Cycle(2) - c1)) {
    spec_probe_ = {true, base, mtp1, conf1, lk_n != 0 ? pending->lookup[0] : -1,
                   ml};
    if (knobs.idle && p1 < 0.1F)
      spec_idle_skip_ = std::min<std::uint32_t>(
          16, 1U << std::min<std::uint32_t>(spec_idle_++, 4));
    pending->spec2_expected = 1.0F;
    return finish_ar();
  }
  spec_idle_ = 0;

  auto& chain = pending->chain;
  chain.push_back(first);
  pending->spec2_source[0] = source1;
  pending->spec2_conf[0] = source1 == DraftSource::kMtp ? conf1 : p1;
  double survival = p1, expected = 1.0 + p1;
  std::uint32_t rounds = 0;
  double rounds_ms = 0;
  while (chain.size() - 1 < budget) {
    const auto rows = static_cast<std::uint32_t>(chain.size());
    const std::size_t k = rows - 1;  // drafts so far
    const double cost = costs.Cycle(rows) + rounds * costs.Round();
    const double rate = expected / cost;
    const double row_cost = costs.Cycle(rows + 1) - costs.Cycle(rows);
    const double need = aggr * row_cost * rate;
    if (alive && lk_n > k) {
      const float pl = spec_cal_.Lookup(ml, static_cast<std::uint32_t>(k + 1));
      if (pl >= knobs.lk_pref || !mtp_mode) {
        if (survival * pl < need)
          break;
        chain.push_back(pending->lookup[k]);
        pending->spec2_source[k] = DraftSource::kLookup;
        pending->spec2_conf[k] = pl;
        survival *= pl;
        expected += survival;
        mtp_mode = false;  // the draft block skipped these positions
        continue;
      }
    }
    if (!mtp_mode)
      break;
    if (survival * knobs.p_next < aggr * (costs.Round() + row_cost) * rate)
      break;
    const auto tr = now();
    pending->draft = chain.back();
    if (!exec.MtpForward(
            *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
            {.token = &pending->draft, .candidates = &pending->traced[k]},
            error_msg)) {
      return false;
    }
    rounds_ms += ms_since(tr);
    ++rounds;
    const std::int32_t token = pending->draft;
    const float conf = CandidateConfidence(pending->traced[k]);
    float p = spec_cal_.Mtp(conf);
    auto source = DraftSource::kMtp;
    if (alive && lk_n > k && pending->lookup[k] == token) {
      p = spec_cal_.Agree(ml);
      source = DraftSource::kAgree;
    } else {
      alive = false;
    }
    if (survival * p < need)
      break;  // the round is spent; its draft is not worth a row
    chain.push_back(token);
    pending->spec2_source[k] = source;
    pending->spec2_conf[k] = source == DraftSource::kMtp ? conf : p;
    survival *= p;
    expected += survival;
  }
  pending->traced.resize(std::max<std::size_t>(rounds + 1, 1));
  pending->spec2_rounds = rounds;
  pending->spec2_expected = static_cast<float>(expected);
  pending->rounds_ms = rounds_ms;
  const auto width = chain.size();
  if (width < 2)
    return finish_ar();
  bool any_lookup = false, all_lookup = true;
  for (std::size_t i = 0; i + 1 < width; ++i) {
    const bool l = pending->spec2_source[i] != DraftSource::kMtp;
    any_lookup = any_lookup || l;
    all_lookup = all_lookup && pending->spec2_source[i] == DraftSource::kLookup;
  }
  pending->spec2_lookup = any_lookup;
  pending->spec2_all_lookup = all_lookup;
  pending->lookup_used = any_lookup;
  pending->width = width;
  pending->speculative = true;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_greedy;
  if (!gpu_greedy && verify_logits_.empty())
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  return true;
}

void Session::ObserveSpecChain(const PendingDecode& pending,
                               std::uint32_t accepted, bool stopped) noexcept {
  const auto drafts = static_cast<std::uint32_t>(pending.chain.size() - 1);
  const auto ml = pending.lookup_match;
  // Draft i+1 was observed when every earlier draft was accepted.
  for (std::uint32_t i = 0;
       i < std::min(stopped ? accepted : accepted + 1, drafts); ++i) {
    const bool ok = i < accepted;
    switch (pending.spec2_source[i]) {
      case DraftSource::kMtp:
        spec_cal_.ObserveMtp(pending.spec2_conf[i], ok);
        break;
      case DraftSource::kLookup:
        spec_cal_.ObserveLookup(ml, i + 1, ok);
        break;
      case DraftSource::kAgree:
        spec_cal_.ObserveAgree(ml, ok);
        break;
    }
  }
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    lookup_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), error_msg)) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        if (!pending.spec2 && keep - 1 < pending.traced.size())
          pending.reject_rank =
              CandidateRank(pending.traced[keep - 1], prediction.index);
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      const auto verified =
          VerifyMtpProposal(std::span<const float>(verify_logits_)
                                .subspan((keep - 1) * vocab, vocab),
                            proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg,
                     gpu_verification ? logits_.data() : nullptr)) {
    return false;
  }
  if (!gpu_verification) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  if (pending.dspark) {
    DsparkPass(keep - 1);
  }
  if (pending.lookup_used) {
    stats_.lookup_drafted += k - 1;
    stats_.lookup_accepted += keep - 1;
  }
  if (pending.spec2)
    ObserveSpecChain(pending, keep - 1, result->stop);
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions. The idle source counts an AR token so
  // its back-off interval still elapses.
  auto& used = pending.lookup_used ? lookup_length_ : draft_length_;
  auto& idle = pending.lookup_used ? draft_length_ : lookup_length_;
  used.Observe(keep - 1, result->stop ? keep - 1 : k - 1, base);
  idle.ObserveArToken();

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  const rocm::Executor::Pin pin(*model_->executor_, *session_, error_msg);
  if (!pin)
    return false;
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  model_->executor_->SetDraftCompact(SpecPolicyKnobs().vocab);
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  if (!BeginDecode(request, &pending, error_msg))
    return false;
  // Greedy-equivalent requests use the timed batch policy (a batch of one);
  // sampled requests keep their deterministic per-session plan.
  auto& policy = model_->batch_policy_;
  std::optional<DraftPlan> plan;
  const bool greedy = !sampler.config().uses_random_sampling();
  if (!pending.chain.empty() && pending.budget != 0 && MtpEnabled() && greedy &&
      SpecPolicyKnobs().mode == 1 && !ForcedMtpDrafts() &&
      MtpConfidenceGate() == 0) {
    pending.spec2 = true;
  } else if (!pending.chain.empty() && pending.budget != 0 && MtpEnabled() &&
             greedy) {
    const std::array rows{DraftRow(request, pending)};
    std::array<DraftPlan, 1> plans{};
    policy.Choose(rows, plans);
    plan = plans[0];
    if (const auto forced = ForcedMtpDrafts())
      plan = DraftPlan{*forced, false};
    else if (MtpConfidenceGate() > 0 && !plan->lookup)
      plan = DraftPlan{pending.budget, false};
  }
  auto& trace = SpecTrace::Get();
  const auto start = std::chrono::steady_clock::now();
  if (!PrepareDecode(request, &pending, error_msg, false, plan)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  const double t_verify = trace.enabled() ? trace.Now() : 0.0;
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  if (!model_->executor_->Forward(
          *session_, pending.chain, pending.chain.size(), logits,
          pending.speculative ? rocm::Executor::ForwardMode::kVerify
                              : rocm::Executor::ForwardMode::kDecode,
          error_msg)) {
    return false;
  }
  const double t_finish = trace.enabled() ? trace.Now() : 0.0;
  const bool ok = FinishDecode(request, pending, error_msg);
  valid_ = ok;
  const float cycle_ms = std::chrono::duration<float, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  if (ok && MtpEnabled()) {
    StepShape shape;
    AddStepShape(shape, pending);
    policy.Observe(shape, cycle_ms);
  }
  // A cycle that skipped the draft head (hopeless drafts) is not an AR cost.
  if (ok && pending.spec2 && !result->stop &&
      (!pending.traced.empty() || pending.chain.size() > 1)) {
    auto& costs = model_->spec_costs_;
    costs.ObserveCycle(static_cast<std::uint32_t>(pending.chain.size()),
                       cycle_ms - pending.rounds_ms);
    if (pending.spec2_rounds != 0)
      costs.ObserveRound(pending.rounds_ms / pending.spec2_rounds);
  }
  if (ok && trace.enabled()) {
    const double now = trace.Now();
    char head[96];
    std::snprintf(head, sizeof(head), "%.1f c=1 vf=%.2f fin=%.2f tot=%.2f", now,
                  t_finish - t_verify, now - t_finish,
                  static_cast<double>(cycle_ms));
    std::string line = head;
    TraceCycle(line,
               !pending.speculative       ? "a"
               : pending.spec2_all_lookup ? "l"
               : pending.lookup_used      ? (pending.spec2 ? "h" : "l")
                                          : "m",
               pending.chain.size(), result->tokens.size(), result->stop,
               pending.base, pending.lookup_size, pending.lookup_match,
               pending.catchup_ms, pending.rounds_ms, pending.traced,
               pending.reject_rank);
    if (pending.spec2) {
      char extra[48];
      std::snprintf(extra, sizeof(extra), " ex=%.2f rn=%u",
                    static_cast<double>(pending.spec2_expected),
                    pending.spec2_rounds);
      line += extra;
    }
    trace.Line("%s", line.c_str());
  }
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg,
                               std::span<const PrefillRequest> prefills) {
  constexpr auto kMax = rocm::Executor::kMaxBatchSessions;
  if (requests.empty() || requests.size() > kMax) {
    AssignError(error_msg, "batch must contain 1..64 sessions");
    return false;
  }
  std::array<BatchOutcome, kMax> outcomes{};
  std::array<std::uint64_t, kMax> epochs{};
  std::array<sampling::SamplerState::DrawState, kMax> draws{};
  std::vector<Request> active;
  active.reserve(requests.size());
  // Batch members stay resident (paged in if needed) for the whole batch.
  std::deque<rocm::Executor::Pin> pins;
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    // Commit this step's KV pages before any peer state changes: an
    // exhausted pool fails this request alone.
    const auto& exec = *r.session->model_->executor_;
    if (!pins.emplace_back(exec, *r.session->session_, &outcome.error) ||
        !exec.Reserve(*r.session->session_,
                      r.session->Position() + exec.max_speculative() + 1,
                      &outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, &shared_error, prefills);
      else
        (void)EvaluateBatchImpl(active, &shared_error, prefills);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos);
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

template<class Request>
bool Session::WithPrefills(std::span<const Request> requests,
                           std::string* error_msg,
                           std::span<const PrefillRequest> prefills) {
  if (prefills.empty())
    return RunIsolatedBatch(requests, error_msg);
  constexpr auto kMax = rocm::Executor::kMaxBatchSessions;
  std::size_t rows = 0;
  for (const auto& r : requests) {
    if constexpr (std::is_same_v<Request, DecodeRequest>)
      rows += std::min<std::size_t>(
          r.max_tokens, r.session->model_->executor_->max_speculative());
    else
      rows += 1;
  }
  std::vector<PrefillRequest> fused;
  std::vector<std::uint64_t> epochs(prefills.size());
  std::vector<bool> was_valid(prefills.size());
  // Fused prompts stay resident (paged in if needed) for the whole batch.
  std::deque<rocm::Executor::Pin> pins;
  for (std::size_t i = 0; i < prefills.size(); ++i) {
    const auto& p = prefills[i];
    auto& s = *p.session;
    *p.outcome = {};
    epochs[i] = s.session_->MutationEpoch();
    was_valid[i] = s.valid_;
    const auto& exec = *s.model_->executor_;
    const auto n =
        p.prompt.size() - std::min(p.prompt.size(), s.tokens_.size());
    // Fuse a pure extension of a distinct session that fits the batch. A
    // lone prompt without decode rows takes the standalone path (Sync).
    bool fuse =
        s.valid_ && n != 0 && p.prompt.size() <= s.ContextSize() &&
        (!requests.empty() || prefills.size() > 1) &&
        rows + n <= exec.max_batch() && requests.size() + fused.size() < kMax &&
        std::equal(s.tokens_.begin(), s.tokens_.end(), p.prompt.begin()) &&
        std::none_of(
            requests.begin(), requests.end(),
            [&](const auto& r) { return r.session == &s; }) &&
        std::none_of(
            fused.begin(), fused.end(),
            [&](const auto& f) { return f.session == &s; });
    if (fuse &&
        !(pins.emplace_back(exec, *s.session_, &p.outcome->error) &&
          exec.Reserve(*s.session_, static_cast<std::uint32_t>(p.prompt.size()),
                       &p.outcome->error)))
      fuse = false;
    if (!fuse)
      continue;
    rows += n;
    s.valid_ = false;
    fused.push_back(p);
  }
  bool ok = true;
  if (requests.empty()) {
    std::string shared_error;
    try {
      if (!fused.empty())
        (void)DecodeBatchImpl({}, &shared_error, fused);
    } catch (const std::exception& exception) {
      shared_error = exception.what();
    }
    for (const auto& p : fused) {
      if (!p.outcome->completed && p.outcome->error.empty())
        p.outcome->error = shared_error;
    }
  } else {
    ok = RunIsolatedBatch(requests, error_msg,
                          std::span<const PrefillRequest>(fused));
  }
  for (std::size_t i = 0; i < prefills.size(); ++i) {
    const auto& p = prefills[i];
    auto& s = *p.session;
    auto& outcome = *p.outcome;
    if (outcome.completed)
      continue;
    if (s.session_->Cancelled()) {
      outcome.error = "generation cancelled";
    } else if (s.session_->MutationEpoch() == epochs[i]) {
      // Not fused, or the batch failed before touching this session.
      s.valid_ = was_valid[i];
      outcome.error.clear();
      try {
        outcome.completed = s.Sync(p.prompt, &outcome.error);
      } catch (const std::exception& exception) {
        outcome.error = exception.what();
      }
    } else if (outcome.error.empty()) {
      outcome.error = "batched prefill failed";
    }
    if (!outcome.completed) {
      if (ok)
        AssignError(error_msg, outcome.error);
      ok = false;
    }
  }
  return ok;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg,
                          std::span<const PrefillRequest> prefills) {
  return WithPrefills(requests, error_msg, prefills);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg,
                            std::span<const PrefillRequest> prefills) {
  return WithPrefills(requests, error_msg, prefills);
}

std::vector<Session::PrefillRequest> Session::StartPrefills(
    std::span<const PrefillRequest> prefills) {
  std::vector<PrefillRequest> live;
  for (const auto& p : prefills) {
    auto& s = *p.session;
    if (!s.session_->CheckCancellation(&p.outcome->error))
      continue;
    // The draft block consumes the previous frontier first, as Feed does.
    const auto chunk = p.prompt.subspan(s.tokens_.size());
    if (s.MtpEnabled() && !s.tokens_.empty() &&
        !s.DraftCatchUp(chunk[0], false, &p.outcome->error))
      continue;
    live.push_back(p);
  }
  return live;
}

template<class Items>
void Session::AppendPrefills(std::span<const PrefillRequest> prefills,
                             Items& items) {
  for (const auto& p : prefills) {
    items.push_back({.session = p.session->session_.get(),
                     .tokens = p.prompt.subspan(p.session->tokens_.size()),
                     .speculative = false,
                     .prefill = true});
  }
}

void Session::FinishPrefills(std::span<const PrefillRequest> prefills,
                             std::uint32_t row, bool download) {
  for (const auto& p : prefills) {
    auto& s = *p.session;
    auto& outcome = *p.outcome;
    const auto& exec = *s.model_->executor_;
    const auto chunk = p.prompt.subspan(s.tokens_.size());
    try {
      if (s.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (!download || exec.SelectBatchLogits(row, 1, s.logits_.data(),
                                                     &outcome.error)) {
        const auto n = chunk.size();
        s.hidden_base_ = static_cast<std::uint32_t>(
            s.tokens_.size() + n -
            std::min<std::size_t>(n, exec.max_speculative()));
        s.tokens_.insert(s.tokens_.end(), chunk.begin(), chunk.end());
        s.valid_ = true;
        outcome.completed = true;
      }
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
    ++row;
  }
}

/// Batched greedy decode with confidence (SpecKnobs::bgate): every session's
/// planned chain keeps the drafts whose calibrated survival pays for their
/// verify row. A row costs the cost model's marginal step time; it earns the
/// session's survival probability in tokens, against the step's tokens per
/// millisecond. One pass: the rate comes from the uncut chains.
void Session::CutBatchChains(std::vector<PendingDecode>& pending,
                             const MtpBatchController& policy) {
  const auto& knobs = SpecPolicyKnobs();
  StepShape shape;
  double expected = 0;
  std::size_t gated = 0;
  for (const auto& p : pending) {
    if (p.chain.empty())
      continue;
    AddStepShape(shape, p);
    float survival = 1.0F;
    expected += 1.0;
    if (p.gate_tail && p.speculative && p.chain.size() > 1) {
      ++gated;
      for (std::size_t d = 1; d < p.chain.size(); ++d) {
        survival *= p.cal->Mtp(CandidateConfidence(p.traced[d - 1]));
        expected += survival;
      }
    } else if (p.speculative) {
      expected += static_cast<double>(p.chain.size() - 1) * 0.5;  // unknown
    }
  }
  // Window 3: C2 prose lost 15% with the cut (n=2); C4/C8 gained 8-20%.
  if (gated == 0 || pending.size() < 4)
    return;
  const auto& model = policy.Model();
  auto less = shape;
  less.rows -= 1;
  less.mtp_drafts -= std::min(less.mtp_drafts, 1U);
  const double step = model.Predict(shape);
  const double row = std::max(step - model.Predict(less), 0.2);
  const double need = knobs.aggr * row * expected / step;
  for (auto& p : pending) {
    if (!p.gate_tail || !p.speculative || p.chain.size() < 2)
      continue;
    const auto drafts = p.chain.size() - 1;
    float survival = 1.0F;
    std::size_t keep = 0;
    for (std::size_t d = 0; d < drafts; ++d) {
      const float conf = CandidateConfidence(p.traced[d]);
      p.spec2_source[d] = DraftSource::kMtp;
      p.spec2_conf[d] = conf;
      survival *= p.cal->Mtp(conf);
      if (survival < need)
        break;
      keep = d + 1;
    }
    p.spec2 = true;  // learn from the verified outcome
    p.chain.resize(keep + 1);
    p.width = keep + 1;
    if (keep == 0)
      p.speculative = false;
  }
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg,
                              std::span<const PrefillRequest> prefills) {
  // Prompt chunks follow the decode rows.
  const auto live = StartPrefills(prefills);
  const bool fused = !live.empty();
  if (requests.empty()) {
    if (!fused)
      return true;
    std::vector<rocm::Executor::BatchItem> items;
    AppendPrefills(live, items);
    if (!live.front().session->model_->executor_->ForwardBatch(items,
                                                               error_msg))
      return false;
    FinishPrefills(live, 0, true);
    return true;
  }
  if (requests.size() == 1 && !fused) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->DecodeStep(
        r.max_tokens, *r.sampler, r.result, &r.outcome->error, r.stop_at_eos);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  auto& policy = requests.front().session->model_->batch_policy_;
  exec.SetDraftCompact(SpecPolicyKnobs().vocab);
  constexpr auto kMax = rocm::Executor::kMaxBatchSessions;
  std::vector<PendingDecode> pending(requests.size());
  std::array<MtpBatchController::Row, kMax> rows{};
  std::array<DraftPlan, kMax> plans{};
  std::array<std::size_t, kMax> planned{};
  std::size_t planned_rows = 0;
  // Sampling each session's next anchor is independent: draw them in
  // parallel, then plan in order.
  std::vector<char> begun(requests.size());
  ParallelFor(requests.size(), [&](std::size_t i) {
    const auto& r = requests[i];
    try {
      begun[i] = r.session->BeginDecode(r, &pending[i], &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  });
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!begun[i]) {
      pending[i] = {};
      continue;
    }
    try {
      if (!pending[i].chain.empty()) {
        rows[planned_rows] = r.session->DraftRow(r, pending[i]);
        planned[planned_rows++] = i;
      }
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  policy.Choose(std::span(rows).first(planned_rows),
                std::span(plans).first(planned_rows));
  if (const auto forced = ForcedMtpDrafts())
    for (std::size_t k = 0; k < planned_rows; ++k)
      if (!rows[k].fixed)
        plans[k] = DraftPlan{*forced, false};
  auto& trace = SpecTrace::Get();
  double t_catchup = 0, t_rounds = 0, t_verify = 0, t_finish = 0;
  std::vector<std::optional<DraftPlan>> chosen(requests.size());
  for (std::size_t k = 0; k < planned_rows; ++k)
    chosen[planned[k]] = plans[k];
  const auto cycle_start = std::chrono::steady_clock::now();
  // DSpark sessions draft together: one block pass reads the drafter and
  // the LM head once for every session's rows.
  std::vector<rocm::Executor::DsparkItem> sparks;
  std::vector<std::size_t> spark_index;
  std::uint32_t spark_rows = 0;
  const auto draft_sparks = [&] {
    if (sparks.empty())
      return;
    std::string spark_error;
    const bool ok = exec.DsparkDraft(sparks, &spark_error);
    for (const auto i : spark_index) {
      pending[i].dspark_drafted = ok;
      if (!ok)
        requests[i].outcome->error = spark_error;
    }
    sparks.clear();
    spark_index.clear();
    spark_rows = 0;
  };
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& p = pending[i];
    if (!requests[i].session->DsparkWants(
            p, static_cast<std::uint32_t>(requests.size())))
      continue;
    if (sparks.size() == rocm::Executor::kDsparkMaxItems ||
        spark_rows + p.budget > rocm::Executor::kDsparkMaxRows)
      draft_sparks();
    sparks.push_back({requests[i].session->session_.get(), p.chain.front(),
                      std::span(p.dspark_tokens).first(p.budget),
                      std::span(p.dspark_confidence).first(p.budget)});
    spark_index.push_back(i);
    spark_rows += p.budget;
  }
  draft_sparks();
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.outcome->error.empty()) {
      pending[i] = {};
      continue;
    }
    try {
      if (!r.session->PrepareDecode(
              r, &pending[i], &r.outcome->error, true, chosen[i],
              static_cast<std::uint32_t>(requests.size())))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty())
      catchup.push_back({requests[i].session, pending[i].chain.front()});
  }
  const double t0 = trace.enabled() ? trace.Now() : 0.0;
  if (!DraftCatchUpBatch(catchup, error_msg))
    return false;
  if (trace.enabled())
    t_catchup = trace.Now() - t0;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back(
            {requests[i].session->session_.get(),
             {.token = p.sampled ? nullptr : &p.draft,
              .candidates = p.sampled     ? &p.candidates
                            : p.gate_tail ? &p.traced[p.chain.size() - 1]
                                          : nullptr}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  if (trace.enabled())
    t_rounds = trace.Now() - t0 - t_catchup;
  CutBatchChains(pending, policy);
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative, pending[i].gpu_greedy});
    }
  }
  AppendPrefills(live, items);
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  if (trace.enabled())
    t_verify = trace.Now() - t0 - t_catchup - t_rounds;
  std::uint32_t offset = 0;
  std::vector<std::uint32_t> row_offsets(requests.size());
  std::vector<std::size_t> finish;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    row_offsets[i] = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    finish.push_back(i);
  }
  const bool any_speculative = std::ranges::any_of(
      finish, [&](std::size_t i) { return pending[i].speculative; });
  // Every session's kept rows join its drafter context in one pass.
  struct Flush {
    rocm::Executor& exec;
    std::vector<rocm::Session*> sessions;
    ~Flush() { (void)exec.DsparkFlush(sessions, nullptr); }
  } flush{exec, {}};
  for (const auto& r : requests)
    flush.sessions.push_back(r.session->session_.get());
  exec.DsparkDefer();
  // Autoregressive rows: one download for the whole step, then each
  // session copies its row and finishes in parallel.
  std::string download_error;
  const float* host = !any_speculative && offset != 0
                          ? exec.HostBatchLogits(offset, &download_error)
                          : nullptr;
  if (host != nullptr) {
    const std::size_t vocab = requests.front().session->model_->VocabSize();
    ParallelFor(finish.size(), [&](std::size_t k) {
      const std::size_t i = finish[k];
      const auto& r = requests[i];
      try {
        std::copy_n(host + std::size_t{row_offsets[i]} * vocab, vocab,
                    r.session->logits_.data());
        r.outcome->completed =
            r.session->FinishDecode(r, pending[i], &r.outcome->error);
      } catch (const std::exception& exception) {
        r.outcome->error = exception.what();
      }
    });
  } else {
    for (const std::size_t i : finish) {
      const auto& p = pending[i];
      const auto& r = requests[i];
      auto& session = *r.session;
      if (!download_error.empty()) {
        r.outcome->error = download_error;
        continue;
      }
      float* logits = !p.speculative       ? session.logits_.data()
                      : p.gpu_verification ? nullptr
                                           : session.verify_logits_.data();
      try {
        r.outcome->completed =
            exec.SelectBatchLogits(row_offsets[i], p.chain.size(), logits,
                                   &r.outcome->error) &&
            session.FinishDecode(r, p, &r.outcome->error);
      } catch (const std::exception& exception) {
        r.outcome->error = exception.what();
      }
    }
  }
  FinishPrefills(live, offset, true);
  if (trace.enabled()) {
    t_finish = trace.Now() - t0 - t_catchup - t_rounds - t_verify;
    char head[160];
    std::snprintf(head, sizeof(head),
                  "%.1f c=%zu fused=%zu cu=%.2f rd=%.2f vf=%.2f fin=%.2f "
                  "tot=%.2f rows=%u",
                  trace.Now(), requests.size(), live.size(), t_catchup,
                  t_rounds, t_verify, t_finish,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - cycle_start)
                      .count(),
                  offset);
    std::string line = head;
    line += " s=";
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& p = pending[i];
      char cell[48];
      std::snprintf(cell, sizeof(cell), "%s%s%zu/%zu", i == 0 ? "" : ",",
                    p.chain.empty()  ? "x"
                    : !p.speculative ? "a"
                    : p.lookup_used  ? "l"
                                     : "m",
                    p.chain.size(), requests[i].result->tokens.size());
      line += cell;
    }
    trace.Line("%s", line.c_str());
  }
  // A mixed step's time does not describe the decode rows alone.
  if (!fused && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; }) &&
      std::ranges::all_of(
          requests, [](const auto& r) { return r.session->MtpEnabled(); })) {
    StepShape shape;
    for (const auto& item : pending)
      AddStepShape(shape, item);
    policy.Observe(shape, std::chrono::duration<float, std::milli>(
                              std::chrono::steady_clock::now() - cycle_start)
                              .count());
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const PrefillRequest> prefills) {
  if (requests.empty())
    return DecodeBatchImpl({}, error_msg, prefills);
  const auto live = StartPrefills(prefills);
  if (requests.size() == 1 && live.empty()) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  // Narrow AR batches keep every session's state bit-exact with Evaluate,
  // draft block included. The shared catch-up regroups the draft
  // projections (tiled GEMMs past eight rows), which changes draft state
  // though never the target, so up to eight sessions catch up as Evaluate
  // does; wider AR batches keep the shared catch-up.
  constexpr std::size_t kExactSessions = 8;
  if (requests.size() <= kExactSessions) {
    for (const auto& r : requests) {
      auto& session = *r.session;
      // A cancelled session fails alone below, as in the shared catch-up.
      if (session.MtpEnabled() && !session.session_->Cancelled() &&
          !session.DraftCatchUp(r.token, false, error_msg) &&
          !session.session_->Cancelled())
        return false;
    }
  } else if (!DraftCatchUpBatch(requests, error_msg)) {
    return false;
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  AppendPrefills(live, items);
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  // One download and synchronization for every frontier row.
  std::vector<float*> frontiers(items.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!requests[i].session->session_->Cancelled())
      frontiers[i] = requests[i].session->logits_.data();
  }
  for (std::size_t i = 0; i < live.size(); ++i) {
    if (!live[i].session->session_->Cancelled())
      frontiers[requests.size() + i] = live[i].session->logits_.data();
  }
  // One download for the whole step; rows then copy out in parallel.
  std::string download_error;
  const float* host = exec.HostBatchLogits(
      static_cast<std::uint32_t>(frontiers.size()), &download_error);
  const bool downloaded = host != nullptr;
  if (downloaded) {
    const std::size_t vocab = requests.front().session->model_->VocabSize();
    ParallelFor(frontiers.size(), [&](std::size_t i) {
      if (frontiers[i] != nullptr)
        std::copy_n(host + i * vocab, vocab, frontiers[i]);
    });
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      if (!downloaded) {
        outcome.error = download_error;
        continue;
      }
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  if (downloaded) {
    FinishPrefills(live, 0, false);
  } else {
    for (const auto& p : live)
      p.outcome->error = download_error;
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
