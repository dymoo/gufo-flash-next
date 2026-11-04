#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/prompt_lookup.hpp"
#include "src/models/qwen38_flash_next/spec_policy.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

class MtmdVision;

struct ModelWeights;
struct MtpWeights;
struct DsparkWeights;
class NgramTable;
struct MtpCandidateLogits;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional MTP draft sidecar (`mtp-...-shared-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Optional DSpark drafter (`dflash` GGUF); exclusive with the MTP path.
  std::string dspark_model_path;
  /// Maximum proposals per cycle; acceptance history selects the length.
  std::uint32_t max_draft_tokens = kMaxMtpDraftTokens;
  std::string vision_model_path;
  /// Fixed serving capacity used by the calibrated MTP cost model. Keeping
  /// it independent of scheduler timing preserves seeded request replay.
  std::uint32_t decode_concurrency = 1;
  /// `--moe-routing` (MoeRouting::Parse). Non-exact policies make a
  /// request's output depend on the requests decoded in the same batch.
  std::string moe_routing = "exact";
  /// Attention KV storage of every session; lossy formats are opt-in.
  KvCacheFormat kv_cache = KvCacheFormat::kF16;
  /// `--dense-prefill w4a4`: lossy int4 dense projections for prompt
  /// processing (rocm::Executor::Options::dense_w4a4); off is exact.
  bool dense_prefill_w4a4 = false;
  /// `--dense-prefill-weights DIR`: offline-calibrated S4G128 int4 weights
  /// for `--dense-prefill w4a4`; empty quantizes the Q8_0 weights at load.
  std::string dense_prefill_weights;
  /// `--experts-used K`: routed experts per token (trunk and MTP); 0 keeps
  /// the GGUF's expert_used_count. Lossy below it.
  std::uint32_t experts_used = 0;
  /// `--router-norm-k N`: routing weights are normalized over the top N
  /// router probabilities (>= the used count) instead of the kept ones;
  /// 0 keeps the GGUF semantics. Lossy.
  std::uint32_t router_norm_k = 0;
  /// `--fast-experts K`: sessions may select the fast model
  /// (Session::SetFastModel), whose decode rows keep their top K routed
  /// experts with the full set's weights; prompt rows stay exact. 0: no fast
  /// model. Lossy for fast sessions only.
  std::uint32_t fast_experts = 0;
  /// `--mtp-policy adaptive`: the combined greedy C1 speculative policy
  /// (compact draft head, calibrated confidence gate, longest-match prompt
  /// lookup). Lossless: greedy output equals autoregressive decoding.
  bool mtp_policy_adaptive = false;
  /// `--mtp-draft-vocab FILE` / `--mtp-draft-vocab-size N`: compact MTP draft
  /// head vocabulary (see rocm::Executor::Options::draft_vocab). Lossless.
  std::string mtp_draft_vocab;
  std::uint32_t mtp_draft_vocab_size = 32768;
  /// `--prefill-residual f16`: wide prefill batches keep the hyper-connection
  /// residual in F16 between combines (lossy); f32 is exact.
  bool prefill_residual_f16 = false;
  /// `--gguf-decode-override FILE`: a GGUF with requantized dense tensors
  /// (e.g. GPTQ Q4_K) uploaded as decode twins beside the primary tensors.
  /// Prefill and wide batches keep the primary encoding; decode batches of up
  /// to `decode_twin_rows` rows read the twins. Lossy (decode only).
  std::string gguf_decode_override;
  /// `--decode-twin-rows N`: batch rows up to which the twins are read.
  std::uint32_t decode_twin_rows = 4;
  /// GatedDeltaNet recurrent state storage (`--gdn-state`); lossy formats
  /// are opt-in.
  GdnStateFormat gdn_state = GdnStateFormat::kF32;
  /// `--routed-prefill-a4 GROUP`: prompt batches run the Q4_K routed
  /// gate/up on the iu4 WMMA with 4-bit activations in groups of GROUP
  /// (64, 128 or 256) elements. Lossy, off (0) by default.
  std::uint32_t routed_prefill_a4 = 0;
  /// `--prefill-capacity TOKENS`: largest prompt batch of one forward pass
  /// (sizes the prefill scratch); 0 keeps the default 2048. Larger batches
  /// stream every routed expert once for more rows.
  std::uint32_t prefill_capacity = 0;
  /// `--routed-down-w4`: re-quantize the Q5_1 routed down experts to 4-bit
  /// codes at load (lossy, every route) and, with routed_prefill_a4, run
  /// the prefill down projection on the iu4 WMMA.
  bool routed_down_w4 = false;
  /// `--routed-w4 GROUP` (64 or 128): re-quantize the Q4_K routed gate/up
  /// and Q5_1 routed down experts at load to symmetric int4 with one scale
  /// per GROUP K, in place (lossy, every route reads the same values), and
  /// run prompt batches on the iu4 WMMA with 4-bit activations of the same
  /// group. Replaces --routed-prefill-a4 / --routed-down-w4 when set.
  std::uint32_t routed_w4 = 0;
  /// `--prefill-phase-rows TOKENS`: rows the non-MoE prefill phases
  /// (projections, recurrent state, attention, residual combines) process at
  /// once inside a larger prefill capacity; the routed experts run over the
  /// whole batch. 0 keeps the default 2048.
  std::uint32_t prefill_chunk = 0;
  /// Diagnostics only: a non-zero count loads trunk layers [first,
  /// first + count) as the whole trunk (ModelWeights::Bind).
  std::uint32_t layer_slice_first = 0;
  std::uint32_t layer_slice_count = 0;
  /// Steering adapters sessions may select (Session::SetAdapter(i + 1)).
  std::vector<SteeringAdapter> adapters;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions with shared projection batches and independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] bool HasDspark() const noexcept;
  [[nodiscard]] std::uint32_t DecodeConcurrency() const noexcept {
    return options_.decode_concurrency;
  }
  [[nodiscard]] KvCacheFormat KvCache() const noexcept {
    return options_.kv_cache;
  }
  /// One `key=value\n` line per lossy option in effect, empty when the model
  /// is exact: state and KV written under different numerics must never be
  /// restored into one another (disk cache identity).
  [[nodiscard]] std::string LossyIdentity() const;
  [[nodiscard]] const std::string& MoeRouting() const noexcept {
    return options_.moe_routing;
  }
  /// ModelOptions::fast_experts (0: no fast model).
  [[nodiscard]] std::uint32_t FastExperts() const noexcept {
    return options_.fast_experts;
  }
  [[nodiscard]] const std::vector<SteeringAdapter>& Adapters() const noexcept {
    return options_.adapters;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state, including the configured rollback cap.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  /// Part of SessionBytes (KV caches, pooled block keys) that sessions
  /// commit as their context grows instead of at creation.
  /// Device bytes a session holds outside the shared pool (control block,
  /// worst-case image inputs); everything else pages in and out of it.
  [[nodiscard]] std::size_t SessionClaimBytes(
      core::SessionMode mode, std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t KvBytes(core::SessionMode mode,
                                    std::uint32_t context) const noexcept;
  /// Caps the KV bytes all sessions may commit together; a session whose
  /// growth would exceed it fails that forward cleanly. Unlimited by default.
  void SetKvPool(std::size_t bytes) const;
  [[nodiscard]] std::size_t KvPoolUsed() const;
  /// Idle sessions page out to host RAM up to `ram_budget` bytes, then to
  /// unlinked O_DIRECT files in `directory`; empty keeps them in RAM only.
  void SetPaging(std::string directory, std::uint64_t ram_budget) const;
  [[nodiscard]] const std::shared_ptr<qwen::vision::Encoder>& VisionEncoder()
      const noexcept {
    return vision_;
  }
  /// MiniCPM-V 4.7 (qwen35moe) image input through llama.cpp mtmd; null
  /// without --mmproj or in builds without GUFO_LLAMA_CPP_DIR.
  [[nodiscard]] const MtmdVision* Mtmd() const noexcept { return mtmd_.get(); }

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  std::shared_ptr<core::GgufReader> decode_reader_;
  std::unique_ptr<ModelWeights> decode_weights_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::shared_ptr<core::GgufReader> dspark_reader_;
  std::unique_ptr<DsparkWeights> dspark_weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<NgramTable> ngram_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::shared_ptr<qwen::vision::Encoder> vision_;
  std::unique_ptr<MtmdVision> mtmd_;
  MtpBatchController batch_policy_;
  SpecCosts spec_costs_;  ///< greedy C1 policy: measured cycle costs
  /// Device KV pages of recent snapshots, by exact context, so restoring
  /// the same context (in memory or from disk) maps them while live.
  struct SharedPages;
  std::unique_ptr<SharedPages> shared_pages_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Greedy decoding verifies deterministic drafts. Sampled decoding draws
  /// compact MTP proposals and uses target/draft rejection with a residual
  /// correction. History contains only committed tokens; stochastic RNG
  /// consumption includes proposal and verification draws. A
  /// one-token budget or a model without MTP uses ordinary decoding.
  /// Benchmarks may continue past EOS by setting stop_at_eos to false.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  struct BatchOutcome {
    bool completed{false};
    std::string error{};
  };
  struct DecodeRequest {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    BatchOutcome* outcome{nullptr};
  };
  /// A prompt extension (Sync semantics) sharing a batched step.
  struct PrefillRequest {
    Session* session;
    std::span<const std::int32_t> prompt;
    BatchOutcome* outcome;
  };
  /// A false return may include completed peers. Inspect each outcome; only
  /// sessions whose persistent state was partly mutated become invalid.
  ///
  /// `prefills` feed their new tokens in the same trunk batch, after the
  /// decode rows, when they fit (see rocm::Executor::ForwardBatch for the
  /// arithmetic); `requests` may then be empty. Any other prompt, or one
  /// the batch failed before touching, continues through Sync.
  [[nodiscard]] static bool DecodeBatch(
      std::span<const DecodeRequest> requests, std::string* error_msg = nullptr,
      std::span<const PrefillRequest> prefills = {});
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr,
      std::span<const PrefillRequest> prefills = {});
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Cancellation is checked at model, layer and snapshot-copy boundaries.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  [[nodiscard]] std::size_t CommittedKvBytes() const noexcept;
  /// Device bytes held outside the shared pool (see SessionClaimBytes).
  [[nodiscard]] std::size_t UnpooledBytes() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt);
  /// Selects the steering adapter of later forwards: 0 for none, i + 1 for
  /// ModelOptions::adapters[i]. A change drops the context, and snapshots
  /// restore only into a session of their own adapter.
  [[nodiscard]] bool SetAdapter(std::uint32_t adapter,
                                std::string* error_msg = nullptr);
  [[nodiscard]] std::uint32_t Adapter() const noexcept;
  /// Selects the fast model (ModelOptions::fast_experts) for later decode
  /// rows. A change drops the context, and snapshots restore only into a
  /// session of the same model; fails without a fast model.
  [[nodiscard]] bool SetFastModel(bool fast, std::string* error_msg = nullptr);
  [[nodiscard]] bool FastModel() const noexcept;
  /// DSpark proposals for the `tokens.size()` positions after `anchor` (the
  /// next token, not yet evaluated) and the drafter's confidence in each.
  /// Tokens are -1 when the drafter lacks this context (after a failure).
  [[nodiscard]] bool DsparkDraft(std::int32_t anchor,
                                 std::span<std::int32_t> tokens,
                                 std::span<float> confidence,
                                 std::string* error_msg = nullptr);
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy() noexcept {
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
  }

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    std::uint64_t lookup_drafted{0};  ///< prompt-lookup share of drafted
    std::uint64_t lookup_accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 17;
  /// Bytes a snapshot of the current context occupies.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  /// Captures the whole context (tokens, device caches and recurrent
  /// state, draft-block state, last logits) into host memory. The
  /// speculative length controller is preserved for stochastic replay.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Replaces this session's context with a snapshot of the same model.
  /// Image snapshots require ConfigureVision with the matching immutable
  /// prompt first; pixel data is not serialized. Text snapshots clear images.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);
  /// Diagnostic: the named byte ranges of a snapshot payload, in order.
  struct SnapshotRegion {
    std::string name;
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
  };
  [[nodiscard]] static std::vector<SnapshotRegion> DescribeSnapshot(
      std::span<const std::uint8_t> payload);

private:
  /// Adapter() | fast-model kept experts << 16: what snapshots and shared
  /// prefix pages must match, so no KV crosses models.
  [[nodiscard]] std::uint32_t Variant() const noexcept;
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
            bool prefill = false);
  /// Trunk rows the draft block may still read: [hidden_base_, size).
  [[nodiscard]] std::uint32_t KeptHiddenRows() const noexcept;
  /// `traced` (GUFO_SPEC_TRACE, greedy) proposes the greedy token and also
  /// downloads the compact candidates for diagnostics.
  bool DraftCatchUp(std::int32_t next_token, bool propose,
                    std::string* error_msg,
                    MtpCandidateLogits* candidates = nullptr,
                    bool traced = false);
  bool DraftReplay(std::int32_t next_token, std::vector<std::int32_t>* replay,
                   std::int32_t* hidden_row, std::string* error_msg) const;
  static bool DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  template<class Request>
  static bool RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg,
                               std::span<const PrefillRequest> prefills = {});
  /// Fuses the prompts that fit into the batch; the rest go through Sync.
  template<class Request>
  static bool WithPrefills(std::span<const Request> requests,
                           std::string* error_msg,
                           std::span<const PrefillRequest> prefills);
  static bool DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg,
                              std::span<const PrefillRequest> prefills);
  static bool EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const PrefillRequest> prefills);
  /// Catches each prompt's draft block up to its frontier (as Feed does)
  /// and returns the prompts still able to run.
  static std::vector<PrefillRequest> StartPrefills(
      std::span<const PrefillRequest> prefills);
  /// Appends each prompt's chunk to executor batch items.
  template<class Items>
  static void AppendPrefills(std::span<const PrefillRequest> prefills,
                             Items& items);
  /// Records each fused prompt after its batch; logit rows start at `row`.
  static void FinishPrefills(std::span<const PrefillRequest> prefills,
                             std::uint32_t row, bool download);
  struct PendingDecode;
  /// Validates, draws the anchor and finds prompt-lookup drafts. An empty
  /// chain afterwards means the request stops before this cycle.
  bool BeginDecode(const DecodeRequest& request, PendingDecode* pending,
                   std::string* error_msg);
  /// `plan` is the batch allocation; without it the session chooses from
  /// committed history alone (sampled requests, seeded replay).
  /// `cohort` is the number of sessions decoding in this step (DSpark).
  bool PrepareDecode(const DecodeRequest& request, PendingDecode* pending,
                     std::string* error_msg, bool defer_head = false,
                     std::optional<DraftPlan> plan = {},
                     std::uint32_t cohort = 1);
  /// Greedy C1 combined policy (`--mtp-policy adaptive`): MTP chain gated by
  /// calibrated acceptance against measured cycle costs, with prompt-lookup
  /// tokens competing for or extending the chain.
  bool PrepareSpecChain(const DecodeRequest& request, PendingDecode* pending,
                        std::string* error_msg);
  static void CutBatchChains(std::vector<PendingDecode>& pending,
                             const MtpBatchController& policy);
  void ObserveSpecChain(const PendingDecode& pending, std::uint32_t accepted,
                        bool stopped) noexcept;
  [[nodiscard]] MtpBatchController::Row DraftRow(
      const DecodeRequest& request, const PendingDecode& pending) const;
  static void AppendDraft(PendingDecode& pending);
  bool FinishDecode(const DecodeRequest& request, const PendingDecode& pending,
                    std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::int32_t draft_token_{0};
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  MtpLengthController draft_length_;
  MtpLengthController lookup_length_;  ///< prompt-lookup acceptance
  PromptLookup lookup_;
  // Greedy C1 combined policy (`--mtp-policy adaptive`): learned acceptance,
  // and the first draft of an AR cycle, labelled by the next cycle's anchor.
  SpecCalibrator spec_cal_;
  struct SpecProbe {
    bool valid{false};
    std::uint32_t base{0};
    std::int32_t mtp_token{-1};
    float conf{0};
    std::int32_t lookup_token{-1};
    std::uint32_t lookup_match{0};
  } spec_probe_;
  std::uint32_t spec_explore_{0};    ///< cycles since the last explore cycle
  std::uint32_t spec_idle_{0};       ///< consecutive hopeless-draft AR cycles
  std::uint32_t spec_idle_skip_{0};  ///< AR cycles left without the head
  SpeculativeStats stats_;
  std::vector<std::uint8_t> image_identity_;
  bool valid_{true};
  [[nodiscard]] bool MtpEnabled() const noexcept;
  [[nodiscard]] bool DsparkEnabled() const noexcept;
  /// Whether this cycle drafts with DSpark (not backing off).
  [[nodiscard]] bool DsparkWants(const PendingDecode& pending,
                                 std::uint32_t cohort) const noexcept;
  /// Records a drafting pass that verified `accepted` drafts (0: declined
  /// or rejected).
  void DsparkPass(std::uint32_t accepted) noexcept;
  static constexpr float kDsparkYieldPrior = 2.0F;
  std::uint32_t dspark_skip_{0};           ///< tokens to decode before drafting
  std::uint32_t dspark_misses_{0};         ///< unproductive passes in a row
  float dspark_yield_{kDsparkYieldPrior};  ///< accepted drafts per pass
  bool dspark_probe_{false};  ///< next pass runs whatever the yield
};

/// Immutable host copy of a session context. The same bytes restore in
/// memory and persist to disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot() = default;
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_.get(), size_};
  }
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::unique_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};

  friend class Session;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
