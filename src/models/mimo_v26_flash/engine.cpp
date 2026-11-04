#include "src/models/mimo_v26_flash/engine.hpp"

#include <algorithm>
#include <cstring>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo_v26_flash/chat_template.hpp"
#include "src/models/mimo_v26_flash/kernels/rocm/executor.hpp"

namespace gufo::models::mimo_v26_flash {
namespace {

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

constexpr char kSnapshotMagic[8] = {'M', 'I', 'M', 'O', 'S', 'N', 'P', '1'};
struct SnapshotHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t positions;
  std::uint32_t vocab;
  std::uint32_t logits_valid;
  std::uint64_t kv_bytes;
};
static_assert(sizeof(SnapshotHeader) == 32);

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& path,
                                   const ModelOptions& options,
                                   std::string* error) {
  std::shared_ptr<Model> model(new Model());
  model->options_ = options;
  auto reader = core::GgufReader::OpenFile(path, error);
  if (reader == nullptr)
    return nullptr;
  model->reader_ = std::move(reader);
  auto weights = ModelWeights::Bind(*model->reader_, error);
  if (!weights || !ChatTemplate::Validate(*model->reader_, error))
    return nullptr;
  model->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.max_context < 2 ||
      options.max_context > model->weights_->config.native_context) {
    Fail(error, "MiMo context must be within the native " +
                    std::to_string(model->weights_->config.native_context));
    return nullptr;
  }
  model->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*model->reader_, error);
  if (model->tokenizer_ == nullptr)
    return nullptr;
  model->executor_ =
      rocm::Executor::Create(*model->weights_, *model->reader_,
                             options.max_context, options.kv_pool_bytes, error);
  if (model->executor_ == nullptr)
    return nullptr;
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error) {
  auto kv = executor_->CreateKv(error);
  if (kv == nullptr)
    return nullptr;
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(kv)));
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(token);
}
bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(token);
}
std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab;
}
std::uint32_t Model::PrefillCapacity() const noexcept {
  return rocm::Executor::kMaxRows;
}
std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("MiMo-V2.6-Flash"));
}
std::size_t Model::ResidentBytes() const noexcept {
  return executor_->resident_bytes();
}
std::size_t Model::KvPoolBytes() const noexcept {
  return executor_->pool_bytes();
}
std::size_t Model::KvPoolUsed() const noexcept {
  return executor_->pool_used_bytes();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::KvState> kv)
    : model_(std::move(model)), kv_(std::move(kv)) {}
Session::~Session() = default;

void Session::Reset() {
  kv_->Release();
  tokens_.clear();
  logits_valid_ = false;
}

std::size_t Session::CommittedBytes() const noexcept {
  return kv_->committed_bytes();
}

bool Session::Sync(std::span<const std::int32_t> prompt, std::string* error) {
  if (prompt.empty())
    return Fail(error, "empty prompt");
  const bool extends =
      prompt.size() >= tokens_.size() &&
      std::equal(tokens_.begin(), tokens_.end(), prompt.begin());
  if (!extends)
    Reset();
  if (prompt.size() == tokens_.size() && logits_valid_)
    return true;
  if (prompt.size() == tokens_.size())
    Reset();  // Same tokens without their logits: recompute.
  while (tokens_.size() < prompt.size()) {
    BatchOutcome outcome;
    const PrefillRequest request{this, prompt, model_->PrefillCapacity(),
                                 &outcome};
    if (!Step({}, std::span(&request, 1), error))
      return false;
    if (!outcome.completed)
      return Fail(error, outcome.error);
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error) {
  BatchOutcome outcome;
  const AdvanceRequest request{this, token, &outcome};
  return Step(std::span(&request, 1), {}, error) &&
         (outcome.completed || Fail(error, outcome.error));
}

bool Session::Step(std::span<const AdvanceRequest> advances,
                   std::span<const PrefillRequest> prefills,
                   std::string* error) {
  Model* model = !advances.empty()   ? advances[0].session->model_.get()
                 : !prefills.empty() ? prefills[0].session->model_.get()
                                     : nullptr;
  if (model == nullptr)
    return true;
  const auto vocab = model->VocabSize();
  auto& exec = *model->executor_;
  std::vector<rocm::StepItem> items;
  std::vector<Session*> sessions;
  std::vector<std::span<const std::int32_t>> appended;
  std::vector<bool> with_logits;
  std::vector<BatchOutcome*> accepted;
  std::size_t rows = 0;
  // A request's own failure (context, KV pool) fails it alone and restarts its
  // session, so the runner's position 0 matches the engine; peers continue.
  const auto reject = [](Session& s, BatchOutcome* o, std::string message) {
    s.Reset();
    if (o)
      *o = {false, std::move(message)};
  };
  std::lock_guard lock(model->device_mutex_);
  for (const auto& a : advances) {
    Session& s = *a.session;
    std::string why = "context is full";
    if (s.Position() >= model->MaxContext() ||
        !exec.Reserve(*s.kv_, s.Position() + 1, &why)) {
      reject(s, a.outcome, std::move(why));
      continue;
    }
    s.logits_.resize(vocab);
    items.push_back(
        {s.kv_.get(), std::span(&a.token, 1), s.Position(), s.logits_});
    sessions.push_back(&s);
    appended.push_back(std::span(&a.token, 1));
    with_logits.push_back(true);
    accepted.push_back(a.outcome);
    ++rows;
  }
  for (const auto& p : prefills) {
    Session& s = *p.session;
    const std::size_t have = s.tokens_.size();
    if (p.prompt.size() <= have ||
        !std::equal(s.tokens_.begin(), s.tokens_.end(), p.prompt.begin())) {
      reject(s, p.outcome, "prefill does not extend the session");
      continue;
    }
    if (p.prompt.size() > model->MaxContext()) {
      reject(s, p.outcome, "prompt exceeds the context");
      continue;
    }
    const std::size_t room = rocm::Executor::kMaxRows - rows;
    const std::size_t count =
        std::min({p.prompt.size() - have, p.max_tokens, room});
    std::string why;
    if (count != 0 &&
        !exec.Reserve(*s.kv_, static_cast<std::uint32_t>(have + count), &why)) {
      reject(s, p.outcome, std::move(why));
      continue;
    }
    accepted.push_back(p.outcome);
    if (count == 0)
      continue;
    const auto chunk = p.prompt.subspan(have, count);
    const bool last = have + count == p.prompt.size();
    if (last)
      s.logits_.resize(vocab);
    items.push_back({s.kv_.get(), chunk, static_cast<std::uint32_t>(have),
                     last ? std::span<float>(s.logits_) : std::span<float>{}});
    sessions.push_back(&s);
    appended.push_back(chunk);
    with_logits.push_back(last);
    rows += count;
  }
  const auto finish = [&](bool ok, const std::string& message) {
    for (auto* o : accepted)
      if (o)
        *o = {ok, ok ? std::string{} : message};
  };
  if (items.empty()) {
    finish(true, {});
    return true;
  }
  std::string message;
  if (!exec.Forward(items, &message)) {
    // The step may have written KV for some layers: nothing it touched is
    // trustworthy.
    for (Session* s : sessions)
      s->Reset();
    finish(false, "MiMo forward failed: " + message);
    return Fail(error, "MiMo forward failed: " + message);
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    s.tokens_.insert(s.tokens_.end(), appended[i].begin(), appended[i].end());
    s.logits_valid_ = with_logits[i];
  }
  finish(true, {});
  return true;
}

std::uint64_t Session::SnapshotBytes() const {
  return sizeof(SnapshotHeader) + tokens_.size() * 4 +
         static_cast<std::uint64_t>(model_->VocabSize()) * 4 +
         model_->executor_->KvSnapshotBytes(Position());
}

bool Session::SaveSnapshot(std::vector<std::uint8_t>* out,
                           std::string* error) const {
  if (tokens_.empty())
    return Fail(error, "nothing to snapshot");
  out->resize(SnapshotBytes());
  SnapshotHeader h{};
  std::memcpy(h.magic, kSnapshotMagic, 8);
  h.version = kSnapshotVersion;
  h.positions = Position();
  h.vocab = model_->VocabSize();
  h.logits_valid = logits_valid_ ? 1 : 0;
  h.kv_bytes = model_->executor_->KvSnapshotBytes(Position());
  std::uint8_t* p = out->data();
  std::memcpy(p, &h, sizeof h);
  p += sizeof h;
  std::memcpy(p, tokens_.data(), tokens_.size() * 4);
  p += tokens_.size() * 4;
  if (logits_valid_)
    std::memcpy(p, logits_.data(), static_cast<std::size_t>(h.vocab) * 4);
  p += static_cast<std::size_t>(h.vocab) * 4;
  std::lock_guard lock(model_->device_mutex_);
  return model_->executor_->SaveKv(*kv_, Position(), p, error);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error) {
  SnapshotHeader h{};
  if (payload.size() < sizeof h)
    return Fail(error, "snapshot truncated");
  std::memcpy(&h, payload.data(), sizeof h);
  if (std::memcmp(h.magic, kSnapshotMagic, 8) != 0 ||
      h.version != kSnapshotVersion || h.vocab != model_->VocabSize() ||
      h.positions == 0 || h.positions > model_->MaxContext() ||
      h.kv_bytes != model_->executor_->KvSnapshotBytes(h.positions) ||
      payload.size() !=
          sizeof h + h.positions * 4ULL + h.vocab * 4ULL + h.kv_bytes)
    return Fail(error, "snapshot does not belong to this model");
  const std::uint8_t* p = payload.data() + sizeof h;
  tokens_.resize(h.positions);
  std::memcpy(tokens_.data(), p, h.positions * 4ULL);
  p += h.positions * 4ULL;
  logits_.resize(h.vocab);
  std::memcpy(logits_.data(), p, h.vocab * 4ULL);
  logits_valid_ = h.logits_valid != 0;
  p += h.vocab * 4ULL;
  bool ok;
  {
    std::lock_guard lock(model_->device_mutex_);
    ok = model_->executor_->RestoreKv(*kv_, h.positions, p, error);
  }
  if (!ok)
    Reset();
  return ok;
}

}  // namespace gufo::models::mimo_v26_flash
