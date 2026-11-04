#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/core/hip/vmm.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/lfm.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class Executor;
struct ArgmaxCandidate;

/// Prompt arithmetic, independent of chunk width, for the scope's thread.
/// The output head keeps its logits-row arithmetic by leaving the scope.
inline thread_local bool prefill_phase = false;
struct PrefillPhase {
  bool previous;
  explicit PrefillPhase(bool enabled) : previous(prefill_phase) {
    prefill_phase = enabled;
  }
  ~PrefillPhase() { prefill_phase = previous; }
};
struct SnapshotHeader;

/// Per-sequence state on the device: recurrent SSM state, KV and indexer
/// caches, PLE conv history, plus the host-side n-gram window. A
/// speculative forward additionally keeps per-token snapshots of every
/// recurrent buffer so the batch can be cut back to its accepted prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  [[nodiscard]] bool dspark_enabled() const noexcept { return dspark_enabled_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next Forward starts at position 0.
  void Reset();
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  /// Fixed device bytes; the lazily committed caches are CommittedKvBytes.
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// Committed bytes of the context-proportional caches (KV, block keys).
  [[nodiscard]] std::size_t CommittedKvBytes() const noexcept;
  [[nodiscard]] std::size_t VisionBytes() const noexcept {
    return vision_input_.Bytes();
  }
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  void RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                           hipStream_t stream);
  [[nodiscard]] const qwen::vision::RopeLayout& VisionLayout() const noexcept {
    return vision_input_.layout();
  }
  /// Steering adapter of this session's trunk rows (0: none, i + 1: index i
  /// of Options::adapters). The caller drops the context on a change.
  [[nodiscard]] std::uint8_t adapter() const noexcept { return adapter_; }
  void set_adapter(std::uint8_t adapter) noexcept { adapter_ = adapter; }
  /// Routed experts each decode row of this session keeps (0: all; the fast
  /// model sets Options::fast_experts). Prompt rows always route exactly.
  /// The caller drops the context on a change.
  [[nodiscard]] std::uint8_t route_keep() const noexcept { return route_keep_; }
  void set_route_keep(std::uint8_t keep) noexcept { route_keep_ = keep; }

private:
  friend class Executor;
  Session() = default;

  struct LinearState {
    float* conv_state{nullptr};    ///< [kernel-1][channels]
    void* state{nullptr};          ///< [v_heads][d][d], Options::gdn_state
    RollbackRows conv_snapshots;   ///< [max_spec-1][kernel-1][channels]
    RollbackRows state_snapshots;  ///< [max_spec-1][v_heads][d][d]
  };
  struct AttentionState {
    const qwen::vision::DeviceRope* rope{nullptr};
    KvView kv;                 ///< [max_context] rows in the executor's format
    float* index_k{nullptr};   ///< [index_capacity_][indexer_dim] raw ring
    __half* block_k{nullptr};  ///< [max_context/ratio][indexer_dim]
  };
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t blocks;        ///< indexer blocks pooled so far
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
    std::uint32_t mtp_blocks;    ///< complete predictor indexer blocks
  };
  struct MtpState {
    KvView kv;
    float* index_k{nullptr};
    __half* block_k{nullptr};
    std::uint32_t blocks{0};
    float* h{nullptr};  ///< [hc_dim] wide residual handed to the next draft
    float* target_hidden{nullptr};  ///< last max_speculative trunk rows
    std::uint32_t position{0};
  };
  /// DSpark drafter context: K/V of the fused target features, indexed by
  /// trunk position. `position` trails the trunk only after a failure.
  struct DsparkState {
    std::vector<KvView> kv;
    std::uint32_t position{0};
    std::uint32_t feature_row{0};  ///< verify rows in the feature scratch
    /// Feature rows waiting for injection at `pending_pos` (DsparkFlush).
    std::uint32_t pending_row{0};
    std::uint32_t pending_rows{0};
    std::uint32_t pending_pos{0};
    void Pend(std::uint32_t row, std::uint32_t rows, std::uint32_t pos) {
      pending_row = row;
      pending_rows = rows;
      pending_pos = pos;
    }
  };

  mutable bool cancelled_{false};
  std::uint8_t adapter_{0};
  std::uint8_t route_keep_{0};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  qwen::vision::DeviceInput vision_input_;
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  bool dspark_enabled_{false};
  DsparkState dspark_;
  std::uint32_t index_capacity_{0};  ///< power-of-two raw indexer ring rows
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  float* ple_history_{nullptr};  ///< [PleConvHistory()][hc_dim]
  RollbackRows ple_snapshots_;   ///< [max_spec-1][PleConvHistory()][hc_dim]
  NgramHistory ngram_;
  std::vector<NgramHistory> ngram_snapshots_;
  std::uint32_t spec_base_{0};    ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};  ///< tokens of the pending speculative batch
  MtpState mtp_;
  Control* control_{nullptr};
  std::uint32_t blocks_{0};  ///< host shadow of Control::blocks
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  /// Context-proportional caches: address space for max_context_ tokens,
  /// pages committed by Executor::Reserve and released by Reset.
  struct Growable {
    gufo::hip::GrowableBuffer buffer;
    std::size_t row_bytes{0};
    std::uint32_t tokens_per_row{1};
    bool mtp{false};      ///< indexed by the draft block's positions
    std::size_t kept{0};  ///< shared bytes left mapped while paged out
  };
  std::vector<Growable> growable_;
  std::uint32_t reserved_tokens_{0};  ///< positions every Growable backs
  void ReleaseKv() noexcept;
  std::size_t rollback_bytes_{0};
  std::function<bool()> is_cancelled_;
  std::vector<gufo::hip::GrowableBuffer> rollback_allocations_;
  /// Fixed state (recurrent and conv state, PLE history, indexer rings,
  /// draft rows) in one pooled range, so an idle session pages out to host
  /// memory and back at the same addresses (captured graphs stay valid).
  gufo::hip::GrowableBuffer arena_;
  std::size_t arena_bytes_{0};
  /// While paged out: the arena and private live KV rows, in host memory
  /// or in an unlinked spill file (4 KiB-aligned segments, O_DIRECT).
  std::vector<std::uint8_t> paged_;
  int spill_fd_{-1};
  bool resident_{false};  ///< arena committed (a new session starts cold)
  std::uint32_t pins_{0};
  std::uint64_t last_use_{0};
  std::uint32_t rollback_depth_{0};
  void TrimRollback(std::uint32_t depth) noexcept;
};

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  /// Keeps a session resident while alive, paging it in if it was evicted
  /// (or never used); engine entry points that touch device state hold one.
  /// Under pool pressure unpinned sessions page out, least recently used
  /// first, to host memory; KV pages they share with others stay mapped.
  class Pin {
  public:
    Pin(const Executor& executor, Session& session, std::string* error_msg);
    ~Pin();
    Pin(const Pin&) = delete;
    Pin& operator=(const Pin&) = delete;
    [[nodiscard]] explicit operator bool() const noexcept { return ok_; }

  private:
    const Executor* executor_;
    Session* session_;
    bool ok_{false};
  };

  struct Options {
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the recurrent snapshot storage.
    std::uint32_t max_speculative{1};
    /// Attention KV storage of every session (trunk and draft layers).
    KvCacheFormat kv_format{KvCacheFormat::kF16};
    /// Storage of every session's GatedDeltaNet recurrent state.
    GdnStateFormat gdn_state{GdnStateFormat::kF32};
    /// Sessions a batch may pack; sizes the deferred logits reservation.
    std::uint32_t max_batch_sessions{1};
    /// Decode/verify routing policy (`--moe-routing`); prefill stays exact.
    MoeRouting routing;
    /// `--routed-prefill-a4`: activation group (64, 128, 256) of the lossy
    /// iu4 routed gate/up for prompt batches; 0 keeps the exact F16 route.
    int routed_a4_group{0};
    /// `--routed-down-w4`: the Q5_1 routed down experts are re-quantized to
    /// 4-bit codes at load (every route reads them; lossy) and prompt
    /// batches with routed_a4_group run the down projection on the iu4 WMMA
    /// with int4 SwiGLU rows.
    bool routed_down4{false};
    /// `--routed-w4`: group (64 or 128) of the symmetric int4 experts
    /// (RequantQ4KSym / RequantDownSym at load); prompt batches run them on
    /// the iu4 WMMA with 4-bit activations of the same group. 0 = off.
    int routed_sym{0};
    /// `--prefill-phase-rows`: rows per non-MoE prefill phase inside a larger
    /// max_batch (the routed experts run over the whole batch).
    std::uint32_t prefill_chunk{2048};
    /// Steering adapters sessions may select (Session::set_adapter).
    std::vector<SteeringAdapter> adapters;
    /// Lossy, opt-in (`--dense-prefill w4a4`): prompt processing runs the
    /// dense projections (SSM in/out, attention QKV/out, shared expert) on
    /// int4 weights and activations (dense_i4.hpp); decode, verification
    /// and short calls keep the Q8_0 weights.
    bool dense_w4a4{false};
    /// Directory of offline S4G128 weights for dense_w4a4 (empty: RTN g128
    /// of the Q8_0 weights at load).
    std::string dense_w4a4_weights;
    /// `--mtp-draft-vocab FILE` (token ids, one per line, most useful first)
    /// and `--mtp-draft-vocab-size N`: the MTP draft head scores only these
    /// rows of the Q8_0 output head (empty: the full head). Lossless.
    std::string draft_vocab;
    std::uint32_t draft_vocab_size{32768};
    /// `--router-norm-k`: see ModelOptions::router_norm_k (0: off).
    std::uint32_t router_norm_k{0};
    /// `--fast-experts`: routed experts a fast-model decode row keeps (0: no
    /// fast model). Enables the per-row Scratch::route_keep buffer.
    std::uint32_t fast_experts{0};
    /// `--decode-twin-rows`: batch rows up to which DeviceTensor::alt (the
    /// `--gguf-decode-override` twin) is read instead of the primary tensor.
    std::uint32_t decode_twin_rows{4};
    /// `--prefill-residual f16`: wide prefill batches keep the hyper-connection
    /// residual in F16 between combines (lossy;
    /// ModelOptions::prefill_residual_f16).
    bool residual_f16{false};
  };
  static constexpr std::size_t kMaxAdapters = 255;

  ~Executor();
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, NgramTable* ngram, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  /// Part of SessionBytes committed on demand (KV caches, pooled block
  /// keys) rather than at CreateSession.
  /// Part of SessionBytes that never comes from the pool (control block).
  [[nodiscard]] static std::size_t UnpooledSessionBytes() noexcept;
  [[nodiscard]] std::size_t GrowableBytes(
      core::SessionMode mode, std::uint32_t max_context) const noexcept;
  /// Commits the caches of `session` for `tokens` positions plus headroom.
  /// Forward, ForwardBatch, the MTP paths and RestoreSnapshot call it before
  /// mutating; a failure (pool or device memory exhausted) changes nothing.
  [[nodiscard]] bool Reserve(Session& session, std::uint32_t tokens,
                             std::string* error_msg) const;
  /// Caps the cache bytes all sessions may commit (unlimited by default).
  void SetKvPool(std::size_t bytes) const { kv_pool_.SetLimit(bytes); }
  /// Idle sessions page out to host RAM up to `ram_budget` bytes, then to
  /// unlinked O_DIRECT files in `directory` (NVMe); empty keeps RAM only.
  void SetPaging(std::string directory, std::uint64_t ram_budget) const {
    const std::lock_guard lock(residency_mutex_);
    spill_directory_ = std::move(directory);
    paging_ram_budget_ = ram_budget;
  }
  [[nodiscard]] const gufo::hip::PiecePool& kv_pool() const noexcept {
    return kv_pool_;
  }

  enum class ForwardMode { kDecode, kVerify, kPrefill };

  /// Appends `tokens` (at most max_batch) to the session and returns the
  /// logits of the last `n_logits` tokens in `logits` (n_logits * vocab
  /// floats, host memory). A null `logits` keeps the rows on the GPU for
  /// verification. The final wide residual of those tokens stays on
  /// the device for MtpForward. kVerify permits Rollback (at most
  /// max_speculative rows). kPrefill uses consistent prompt arithmetic at
  /// every chunk width.
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg) const;

  /// Sessions one batched call may pack (each with up to eight rows).
  static constexpr std::uint32_t kMaxBatchSessions =
      qwen38_flash_next::kMaxBatchSessions;
  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
    /// Greedy verification: GreedyMtpPredictions reads device argmaxes
    /// computed for every row without another synchronization.
    bool argmax{false};
    /// A prompt chunk (kPrefill; any length within max_batch). Prefill
    /// items follow every decode item; each yields one logit row (its last
    /// token) and catches up the draft block like a kPrefill Forward.
    bool prefill{false};
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own recurrent/KV state; up to eight rows per chain. Narrow
  /// batches keep the decode arithmetic. Wider ones read each shared weight
  /// once through the tiled GEMMs (same per-32 Q8 activation granularity or
  /// F16 activations), so they are not bit-identical to single decoding.
  ///
  /// Prompt chunks of several sessions share one pass: their rows are
  /// concatenated through the prefill arithmetic, which is row-count
  /// invariant, so each chunk's state and logits equal its standalone
  /// kPrefill Forward bit for bit. With decode rows present, the decode rows
  /// keep exactly the arithmetic of the same decode-only batch and both
  /// parts share one routed-expert pass (the device-planned route, which is
  /// batch invariant); prompt rows then differ from standalone prefill only
  /// in the routed experts.
  /// Logit rows remain in batch order until SelectBatchLogits is called:
  /// every decode row, then one row per prefill item.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;
  /// Downloads the first `rows` logit rows of the completed batch with one
  /// copy and returns them in host memory ([rows][vocab], valid until the
  /// next download), or null on failure.
  [[nodiscard]] const float* HostBatchLogits(std::uint32_t rows,
                                             std::string* error_msg) const;

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest. If `logits` is supplied, copies the kept frontier
  /// into it using the same synchronization as rollback.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg,
                              float* logits = nullptr) const;

  /// Runs the draft block over `tokens` (at most max_batch) at the session's
  /// MTP position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the draft block's own residual from the previous call.
  /// Only requested outputs are computed. Production requests a greedy
  /// token or compact candidate logits. Catch-up skips the output head
  /// when only the draft state is needed.
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
    MtpTrace* trace{nullptr};  ///< final-row diagnostic; disables graph capture
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
  };
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with shared weights.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// DSpark: proposes `tokens.size()` tokens after `anchor` (the sampled,
  /// not yet evaluated token at the session position) with the confidence
  /// of each. Sessions share one block pass; their K/V rows past the
  /// context are scratch. Tokens stay -1 where the drafter lacks context.
  struct DsparkItem {
    Session* session;
    std::int32_t anchor;
    std::span<std::int32_t> tokens;
    std::span<float> confidence;
  };
  [[nodiscard]] bool DsparkDraft(std::span<const DsparkItem> items,
                                 std::string* error_msg) const;
  /// Rollback leaves each session's kept rows pending until DsparkFlush,
  /// which fuses every listed session's rows in one pass. Call it before
  /// the next forward reuses the feature scratch.
  void DsparkDefer() const noexcept { dspark_defer_ = true; }
  [[nodiscard]] bool DsparkFlush(std::span<Session* const> sessions,
                                 std::string* error_msg) const;
  /// Block rows and sessions one DSpark pass drafts.
  static constexpr std::uint32_t kDsparkMaxRows = 64;
  static constexpr std::uint32_t kDsparkMaxItems = 16;

  /// Plain greedy verification keeps full logit rows on the GPU.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions, std::string* error_msg) const;

  /// A session's complete context as one host byte payload: recurrent and
  /// PLE state, KV and indexer caches up to the position, and the draft
  /// block's caches plus the `hidden_rows` most recent kept trunk rows.
  /// The payload restores into any session of this executor whose context
  /// holds the position; the pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] bool SaveSnapshot(const Session& session,
                                  std::uint32_t hidden_rows,
                                  std::span<std::uint8_t> payload,
                                  std::string* error_msg) const;
  /// Reuse at most the rollback rows needed by the restored operation;
  /// restoration never grows scratch. Callers derive this bound from the
  /// restored policy (or the concrete verifier width in a diagnostic).
  /// A device page of a snapshotted cache that a restore can map instead of
  /// copying, while some session still maps it (in-process, weak).
  struct SnapshotPage {
    std::uint32_t buffer{0};
    gufo::hip::PieceRef piece;
  };
  /// Pages of `session` whose rows are final: wholly below every append
  /// position, so sessions restoring its snapshot may share them.
  void SharePages(const Session& session,
                  std::vector<SnapshotPage>* pages) const;
  /// `pages` (from SharePages at the snapshot's capture) are mapped where
  /// still live; the rest of the caches is copied from the payload.
  [[nodiscard]] bool RestoreSnapshot(
      Session& session, std::span<const std::uint8_t> payload,
      SnapshotInfo* info, std::string* error_msg, std::uint32_t next_drafts = 0,
      std::span<const SnapshotPage> pages = {}) const;
  /// Diagnostic: the named byte ranges of an executor payload in order
  /// (header first, labelled with its positions). Empty if malformed.
  struct SnapshotRegion {
    std::string name;
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
  };
  [[nodiscard]] static std::vector<SnapshotRegion> DescribeSnapshot(
      std::span<const std::uint8_t> payload);

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const noexcept {
    if (!session.mtp_enabled_)
      return;  // no draft block (plain or DSpark sessions)
    session.mtp_.position = position;
    // Without an indexer (qwen35moe, lfm2moe) there are no pooled blocks.
    const auto ratio = config().compress_ratio;
    session.mtp_.blocks =
        ratio == 0 ? 0 : std::min(session.mtp_.blocks, position / ratio);
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_.position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  /// Draft with the compact head (`--mtp-draft-vocab`) or the full head;
  /// effective at the next draft evaluation. Scheduler thread only.
  void SetDraftCompact(bool on) noexcept { draft_compact_ = on; }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }
  [[nodiscard]] KvCacheFormat kv_format() const noexcept {
    return options_.kv_format;
  }
  [[nodiscard]] GdnStateFormat gdn_state() const noexcept {
    return options_.gdn_state;
  }

private:
  Executor() = default;

  /// Visits every device region of a snapshot in payload order with
  /// (device pointer or null when sizing, payload offset, bytes, name).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  static std::uint64_t WalkSnapshot(const SnapshotHeader& h,
                                    const Session* session, Visit&& visit);

  /// An activation batch quantized once for the decode GEMVs; `data` is
  /// null when the batch is wide enough for the tiled path.
  struct Q8Input {
    const float* x;
    const void* data;
    std::uint32_t n;
    std::uint32_t k;
  };
  bool Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                Q8Input* q, std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const Q8Input& q, float* out,
             std::string* error_msg) const;
  /// Decode GEMV of a Q8_0 or K-quant matrix over `rows` Q8_1 activation
  /// rows (gated when `gate` is set); returns the adapter's status.
  int VecPreq(const DeviceTensor& w, const void* gate, const void* x_q8,
              float* out, std::uint32_t rows) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// out = (up . x) * silu(gate . x); s_.shexp_gate is scratch.
  /// With `down` (the projection that consumes the result), a wide batch
  /// leaves only that projection's staged input form (F16 rows in
  /// s_.x_half or the tiled Q8 layout in s_.x_q8t, registered in the input
  /// cache); `out` then holds the gate projection, not the result.
  bool GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// W4A4 dense prefill route (Options::dense_w4a4): the packed int4 copy
  /// of `w` for a prompt-processing call of `n_tokens` rows, else null.
  [[nodiscard]] const void* I4Route(const DeviceTensor& w,
                                    std::uint32_t n_tokens) const;
  /// Rotates and quantizes F32 rows (read from s_.x_half when it holds
  /// them) or F16 rows into s_.x_i4.
  bool I4Activations(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error_msg) const;
  bool I4Activations(const __half* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error_msg) const;
  /// out [rows][w.rows] = packed `w` x the rows in s_.x_i4.
  bool I4Gemm(const void* packed, const DeviceTensor& w, float* out,
              std::uint32_t rows, std::string* error_msg) const;
  bool PackI4Weights(std::string* error_msg);
  void RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile
  /// hints for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  /// Routed gate/up projections followed by SwiGLU.
  bool GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t n_tokens, std::uint32_t n_used,
                    std::string* error_msg) const;
  bool Experts(const DeviceTensor& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t n_rows, std::uint32_t n_used,
               std::uint32_t n_tokens, std::string* error_msg) const;
  /// `normed` says the previous Combine already produced m's grouped norm
  /// of res (F32 in s_.xn, or F16 plus tiled Q8 on the wide route).
  /// qwen35moe pre-norm in place of a hyper-connection mixer.
  bool PlainNorm(const DeviceMixer& m, const float* res, float* mixed,
                 std::uint32_t rows) const;
  bool HcMix(const DeviceMixer& m, const float* res, bool normed, float* mixed,
             float* inject, std::uint32_t n_tokens,
             std::string* error_msg) const;
  /// Share mixer projections across decode requests without selecting the
  /// prefill arithmetic. Injection rows are compact across the whole batch.
  bool HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                  float* mixed, float* inject, std::uint32_t rows,
                  std::string* error) const;
  /// `moe_layer` names the trunk layer whose MoE output s_.block_out holds
  /// (-1: an attention output), for the adapters' routed-layer scope.
  void CombineBatch(float* res, const float* gamma, std::uint32_t rows,
                    std::int32_t moe_layer = -1) const;
  /// Ablates s_.block_out rows of adapter requests while a trunk forward
  /// with any (steer_) runs, or, in the MTP draft block (steer_writer_ =
  /// kMtp), rows of adapters that project it; no work otherwise.
  void Steer(std::uint32_t rows, std::int32_t moe_layer) const;
  [[nodiscard]] SteerRows SteerArgs(std::int32_t moe_layer) const noexcept;
  /// Ablates `rows` rows of `y` (`streams` equal copies each, see Steer)
  /// for the adapters among s_.steer that project `writer`
  /// (steer_writer::kEmbedding, kPle); no work without such an adapter.
  void SteerWriter(float* y, std::uint32_t rows, std::uint32_t streams,
                   std::uint8_t writer) const;
  /// Whether `adapter` (0: none) projects any MTP-block input or output.
  [[nodiscard]] bool SteersMtp(std::uint8_t adapter) const noexcept;
  /// Residual update by the block output plus the grouped norm for the next
  /// mixer (`gamma`); wide batches write it as F16 and tiled Q8. With the
  /// next mixer `next`, the wide route also emits its inject partials.
  /// `keep_f32` false lets the wide route keep the residual in F16
  /// (--prefill-residual); the default leaves the F32 residual authoritative.
  void Combine(float* res, const float* gamma, std::uint32_t n_tokens,
               std::int32_t moe_layer = -1, const DeviceMixer* next = nullptr,
               bool keep_f32 = true) const;
  /// The F16 HC projection route (HcDownF16Gemm + HcMixF16Gemm) applies to
  /// mixer `m` over this many rows.
  [[nodiscard]] bool FusedHcProjection(const DeviceMixer& m,
                                       std::uint32_t n_tokens) const;
  /// Hashes the batch's n-gram rows and starts reading them from disk, so
  /// the read overlaps the layers before the PLE one.
  bool PleFetch(Session& s, std::span<const std::int32_t> tokens,
                bool speculative, std::string* error_msg) const;
  bool WaitPle(std::string* error_msg) const;
  bool Ple(const DeviceLayer& l, Session& s, std::uint32_t n_tokens, float* res,
           bool speculative, std::string* error_msg,
           bool embeddings_ready = false) const;
  /// Without `project_output` the caller projects: F32 rows in s_.gdn_out,
  /// or, when HalfSsmOutput, F16 rows at `half_rows`.
  // lfm2moe blocks: short convolution, 64-wide GQA attention, and the FFN
  // (dense SwiGLU on the leading layers, sigmoid-routed experts after).
  bool LfmConv(const DeviceLayer& l, Session::LinearState& s, const float* x,
               float* out, std::uint32_t n_tokens,
               std::string* error_msg) const;
  bool LfmAttentionLayer(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         std::string* error_msg) const;
  bool LfmFfn(const DeviceLayer& l, const float* x, float* out,
              std::uint32_t n_tokens, std::string* error_msg) const;
  bool LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       bool speculative, std::string* error_msg,
                       bool projections_ready = false,
                       bool project_output = true,
                       __half* half_rows = nullptr) const;
  /// Whether prompt rows of `l` feed the SSM output projection as F16.
  [[nodiscard]] bool HalfSsmOutput(const DeviceLayer& l,
                                   std::uint32_t n_tokens) const noexcept;
  /// `pos`/`first_block` are device values; `start_pos` and `pool_grid`
  /// are their host-side counterparts for the eager-only decisions.
  bool Attention(const DeviceLayer& l, Session::AttentionState& s,
                 const float* x, float* out, std::uint32_t n_tokens,
                 const std::uint32_t* pos, const std::uint32_t* first_block,
                 std::uint32_t start_pos, std::uint32_t pool_grid,
                 std::uint32_t max_context, bool sparse, std::string* error_msg,
                 bool last_only = false, bool projections_ready = false,
                 bool project_output = true) const;
  /// Decode attention of trunk layer `il` for every batch item in one launch
  /// per stage: the stacked projection rows are in s_.qg (base scratch
  /// view); leaves the gated rows in s_.ctx (zero rows for cancelled items).
  /// Same per-row arithmetic as Attention for each session in isolation.
  bool AttentionBatchLayer(const DeviceLayer& l, std::uint32_t il,
                           std::span<const BatchItem> items,
                           std::span<const std::uint32_t> offsets,
                           std::uint32_t rows, std::string* error) const;
  /// One lfm2moe layer over every row of a multi-session batch (decode
  /// rows first, then prompt chunks): shared stages over all rows, short
  /// convolution state, KV and positions per item.
  bool LfmBatchLayer(const DeviceLayer& l, std::uint32_t il,
                     std::span<const BatchItem> items,
                     std::span<const std::uint32_t> offsets,
                     std::uint32_t decode_items, std::uint32_t rows,
                     std::string* error) const;
  bool LfmFfnBatch(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t rows, std::string* error) const;
  bool Moe(const DeviceLayer& l, const float* x, float* out,
           std::uint32_t n_tokens, std::string* error_msg,
           bool last_only = false) const;
  /// Router, routing and shared expert of Moe (s_.router, s_.ids,
  /// s_.weights, s_.shexp_out); `counts` starts the tiled route's
  /// expert-count download.
  bool MoeShared(const DeviceLayer& l, const float* x, std::uint32_t n_tokens,
                 bool counts, std::string* error_msg) const;
  /// MoeBatch's router, routing and shared expert for `rows` decode rows.
  bool MoeBatchShared(const DeviceLayer& l, const float* x, std::uint32_t rows,
                      std::string* error) const;
  /// Router logits (s_.router) -> s_.ids/s_.weights under options_.routing.
  void Route(const DeviceLayer& l, std::uint32_t rows) const;
  /// Runs routed experts after the router and shared expert are ready.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  /// Whether decode/verification rows of `l` take the device-planned
  /// routed experts (moe.hpp): no host synchronization, one arithmetic at
  /// every batch width. Prefill keeps the tiled route.
  [[nodiscard]] bool DeviceRouted(const DeviceLayer& l) const noexcept;
  /// DeviceRouted experts plus the slot-weighted epilogue into `out`.
  bool DeviceExperts(const DeviceLayer& l, const float* x, float* out,
                     std::uint32_t rows, std::string* error) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error) const;
  /// Selects the greedy token or compact candidates from full MTP logits.
  bool MtpHead(const DeviceMixer& head, const float* res, bool token,
               bool candidates, std::string* error_msg) const;
  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative, bool sparse,
                   std::uint32_t start_pos, std::uint32_t pool_grid,
                   std::uint32_t first_layer, std::uint32_t end_layer,
                   std::string* error_msg) const;

  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos, bool token,
               bool candidates, std::string* error_msg, std::uint32_t pool_grid,
               const float* hidden_source, MtpTrace* trace) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set. A prefix may leave its work queued so the host
  /// can wait for disk reads while the GPU computes it.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true) const;

  const DeviceModel* model_{nullptr};
  NgramTable* ngram_{nullptr};
  Options options_;
  hipStream_t stream_{nullptr};
  /// 1 MiB pieces: 1024 F16 K/V tokens; small caches waste at most a piece.
  mutable gufo::hip::PiecePool kv_pool_{std::size_t{1} << 20};
  /// Guards residency (pins, page-in/out) and the session registry.
  mutable std::recursive_mutex residency_mutex_;
  mutable std::vector<Session*> sessions_;
  mutable std::uint64_t use_clock_{0};
  static constexpr std::size_t kSpillChunk = std::size_t{8} << 20;
  mutable std::string spill_directory_;
  mutable std::array<void*, 2> spill_staging_{};
  mutable std::array<hipEvent_t, 2> spill_events_{};
  mutable std::uint64_t paged_ram_bytes_{0};
  mutable std::uint64_t paging_ram_budget_{std::uint64_t{16} << 30};
  std::vector<std::pair<char*, std::size_t>> PagedSegments(
      Session& session) const;
  bool SpillStaging(std::string* error_msg) const;
  void DropPaged(Session& session) const;
  bool PageIn(Session& session, std::string* error_msg) const;
  /// A session that was never paged in has no committed state arena, so
  /// running it faults the GPU ("illegal memory access" at the next host
  /// sync). Direct callers hold an Executor::Pin around the whole call.
  bool RequireResident(const Session& session, std::string* error_msg) const;
  bool PageOut(Session& session) const;
  bool EvictOne() const;
  friend class Session;
  /// GUFO_FLASH_NEXT_ROUTER_DUMP: raw router logits per MoE call (debug;
  /// synchronizes every layer and disables decode graphs).
  std::FILE* router_dump_{nullptr};
  /// Options::router_norm_k (`--router-norm-k`): routing weights are
  /// normalized over the
  /// top router_norm_k_ probabilities (>= num_experts_used) instead of the
  /// used ones, also for --moe-routing rewrites (equal to num_experts_used:
  /// exact routing unchanged, policies keep top-k weights instead of
  /// renormalizing); 0 keeps the GGUF semantics.
  std::uint32_t router_norm_k_{0};
  hipEvent_t counts_ready_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;   ///< activations narrowed to the weight's 16-bit type
    void* x_q8[2];  ///< Q8_1 activations of a decode batch, alternating
    /// Q8_1 copies a producer kernel writes beside its F32 output for the
    /// decode GEMV that reads it next: [0] the combine's norm (xn), [1] the
    /// mixer epilogue's mixed rows, [2] the GDN output (see PendingQ8).
    void* pre_q8[3];
    void* x_q8t;  ///< tiled Q8 activations of a wide batch (W8A8 route)
    void* x_i4;   ///< int4 activations of the W4A4 prefill route
    float* res;
    float* xn;
    __half* xn_half;   ///< xn as F16 on the F16 mixer input route
    __half* res_half;  ///< F16 residual (--prefill-residual)
    float* xn_scale;   ///< [t][4] norm scales when xn is not written
    void* xn_q8t;      ///< xn as tiled Q8 for the W8A8 mixer down projection
    float* lo;
    float* hc_gate;
    float* mixed;
    float* inject;
    float* block_out;
    // linear attention
    float* qkv;
    float* z;
    float* qkvz;  ///< [t][qkv | z] from the stacked projection
    float* alpha_beta;
    float* conv_scratch;
    float* qn;
    float* kn;
    float* gdn_raw;
    float* gdn_out;
    // attention
    float* qg;  ///< [t][q|gate (; k ; v)]
    float* q;
    float* attn_gate;
    float* k;
    float* v;
    float* iq;
    float* ik;
    std::uint32_t* mask;
    float* scores;
    float* ctx;
    float* attn_partials;  ///< split-key partials of a narrow batch
    // ple
    float* ple_emb;
    float* ple_key;
    float* ple_value;
    float* ple_query;
    float* ple_gated;
    float* ple_norm;
    float* ple_conv;
    float* ple_history_scratch;
    // moe
    float* router;
    std::int32_t* ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the tiled Q8 gathered rows.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;  ///< (expert | tile << 16) per launched tile
    float* weights;
    float* gate_e;
    float* up_e;
    float* down_e;
    float* shexp_gate;
    float* shexp_up;
    float* shexp_out;
    /// Shared-expert SwiGLU rows narrowed for the F16 down projection, so
    /// s_.x_half keeps the routed experts' token rows.
    __half* shexp_half;
    // head and hidden rows kept for the draft block
    float* logits;
    /// Per-row steering adapter (SteerRows::rows); null without adapters.
    std::uint8_t* steer;
    /// Per-row routed experts kept (TruncateRoutes); null without a fast
    /// model (Options::fast_experts).
    std::uint8_t* route_keep;
    // mtp
    float* mtp_h;
    float* mtp_embd;
    float* mtp_eproj;
    float* mtp_res;
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
  };
  mutable Scratch s_{};
  /// DSpark scratch (null without a drafter): target features of the
  /// current batch [max_batch][layers * hidden], their fusion, and the
  /// block's activations (kDsparkMaxRows rows).
  struct DsparkScratch {
    float* features{nullptr};
    float* g{nullptr};
    float* k{nullptr};
    float* v{nullptr};
    float* x{nullptr};
    float* xn{nullptr};
    float* q{nullptr};
    float* ctx{nullptr};
    float* o{nullptr};
    float* gate{nullptr};
    float* up{nullptr};
    float* logits{nullptr};
    std::uint32_t* cand_ids{nullptr};  ///< top-candidate workspaces
    std::uint32_t* cand_scratch{nullptr};
    float* cand_scores{nullptr};
    float* conf_in{nullptr};  ///< [rows][hidden | rank]
    float* conf{nullptr};
    float* partials{nullptr};      ///< attention split records
    std::int32_t* block{nullptr};  ///< [anchor, MASK...] per item
    std::int32_t* chain{nullptr};  ///< per item: anchor, then selections
    std::uint32_t* pos{nullptr};   ///< per item block start, last key
    float* kv{nullptr};            ///< stacked K/V of injected rows
    /// Injection descriptors, kStoreSlots rotating sets of sessions.
    static constexpr std::uint32_t kStoreSlots = 8;
    DsparkStoreItem* store_host{nullptr};
    DsparkStoreItem* store_device{nullptr};
    std::array<hipEvent_t, kStoreSlots> store_done{};
    std::uint32_t store_slot{0};
    std::int32_t* block_host{nullptr};
    std::int32_t* chain_host{nullptr};
    std::uint32_t* pos_host{nullptr};
    float* conf_host{nullptr};
    AttentionBatchItem* attn_host{nullptr};  ///< [block][row], mapped
    AttentionBatchItem* attn_device{nullptr};
  };
  mutable DsparkScratch ds_{};
  mutable bool dspark_defer_{false};
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  void UseScratch(const Scratch& scratch) const;
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  /// One tiled GEMM over every row (see DenseBatch's crossover).
  bool TiledDense(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  bool GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                       const float* x, float* out, std::uint32_t rows,
                       std::string* error) const;
  bool QuantizeBatch(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error) const;
  /// PLE with shared projections over every row and private histories.
  bool PleBatch(const DeviceLayer& l, std::span<const BatchItem> items,
                std::span<const std::uint32_t> offsets, std::uint32_t rows,
                std::string* error) const;
  /// Output projection of `rows` batch rows into BatchLogits(rows).
  /// `keep_moe_scratch` keeps the logits out of the MoE scratch alias
  /// because the draft block runs before the rows are read.
  /// `head` replaces the vocabulary head (the draft block's DraftHead()).
  bool BatchHead(const float* mixed, std::uint32_t rows, bool argmax,
                 std::string* error, bool keep_moe_scratch = false,
                 const DeviceTensor* head = nullptr) const;
  /// Greedy argmax of the first `rows` logit rows into batch_argmax_host_.
  bool BatchArgmax(std::uint32_t rows, std::string* error) const;
  /// Points batch_logits_ at storage for `rows` logit rows (see BatchHead).
  bool HeadLogits(std::uint32_t rows, bool keep_moe_scratch,
                  std::string* error) const;
  bool AllocateBatch(std::string* error_msg) const;
  /// Logit rows of the current batch: the idle MoE down-projection scratch
  /// when they fit (logit_alias_floats_), else a buffer grown on demand.
  mutable float* batch_logits_{nullptr};
  mutable float* batch_logits_owned_{nullptr};
  mutable std::uint32_t batch_logits_owned_rows_{0};
  std::size_t logit_alias_floats_{0};
  /// Pinned staging for HostBatchLogits, grown on demand.
  mutable float* batch_logits_host_{nullptr};
  mutable std::uint32_t batch_logits_host_rows_{0};
  mutable Session::Control* batch_controls_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  /// Per-row greedy argmax of the last ForwardBatch (argmax items) or
  /// MtpHeads; batch_argmax_rows_ is zero when absent.
  mutable ArgmaxCandidate* batch_argmax_host_{nullptr};
  mutable std::uint32_t batch_argmax_rows_{0};
  // Mapped descriptors, one slice per layer: GPU reads cannot race the host
  // preparing the next layer. ForwardBatch drains before reusing this table.
  mutable GdnBatchItem* batch_gdn_host_{nullptr};
  mutable GdnBatchItem* batch_gdn_{nullptr};
  /// Multi-session attention descriptors ([layer][session], mapped like
  /// the GDN table) and every item's split records.
  mutable AttentionBatchItem* batch_attn_host_{nullptr};
  /// lfm2 short-convolution items: one kMaxBatchSessions slice per layer.
  mutable LfmConvItem* lfm_conv_host_{nullptr};
  mutable LfmConvItem* lfm_conv_{nullptr};
  mutable AttentionBatchItem* batch_attn_{nullptr};
  mutable float* batch_attn_partials_{nullptr};
  mutable std::size_t batch_attn_partial_rows_{0};
  mutable std::uint32_t batch_rows_{0};
  mutable const float* selected_logits_{nullptr};
  mutable std::uint32_t selected_offset_{0};
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  std::uint32_t mask_words_{0};
  /// Queries per block-selection launch (its score scratch is chunk x
  /// max_blocks floats: 128 MB at the 262k context).
  std::uint32_t select_chunk_{512};
  std::vector<void*> allocations_;
  /// Pinned: the n-gram rows go up with hipMemcpyAsync, and a pageable
  /// source would not be ordered against the kernels behind it.
  float* host_emb_{nullptr};
  mutable std::vector<std::uint32_t> host_rows_;
  mutable bool ple_pending_{false};
  // Pinned host staging the launched (or captured) work reads and writes.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  mutable std::uint32_t routed_n_tiles_{0};   ///< down projection tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< appended 64-token tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_pair_rows_{64};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  /// Steering adapters on the device and per-row staging (see SteerRows).
  float* steer_directions_{nullptr};
  float* steer_alphas_{nullptr};
  std::uint64_t* steer_routed_layers_{nullptr};
  std::uint8_t* steer_writers_{nullptr};  ///< SteeringAdapter::writers
  std::uint8_t steer_writers_any_{0};     ///< union over the adapters
  std::uint8_t* steer_host_{nullptr};
  /// Pinned staging of Scratch::route_keep; set while a trunk forward with
  /// fast-model rows runs (Route then truncates decode rows).
  std::uint8_t* route_keep_host_{nullptr};
  mutable bool route_keep_{false};
  /// Set while a forward with adapter rows runs: the trunk (writer 0, every
  /// adapter's block outputs) or the MTP draft block (steer_writer::kMtp).
  mutable bool steer_{false};
  mutable std::uint8_t steer_writer_{0};
  struct RouteKeepScope {
    const Executor& executor;
    RouteKeepScope(const Executor& e, bool active) : executor(e) {
      executor.route_keep_ = active;
    }
    ~RouteKeepScope() { executor.route_keep_ = false; }
  };
  struct SteerScope {
    const Executor& executor;
    SteerScope(const Executor& e, bool active, std::uint8_t writer = 0)
        : executor(e) {
      executor.steer_ = active;
      executor.steer_writer_ = active ? writer : 0;
    }
    ~SteerScope() {
      executor.steer_ = false;
      executor.steer_writer_ = 0;
    }
  };
  std::int32_t* mtp_token_host_{nullptr};
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  /// Draft vocabulary (`--mtp-draft-vocab` ids file, `--mtp-draft-vocab-size`
  /// rows): a compact copy of the Q8 head rows of the listed tokens, read by
  /// the draft block instead of the full head. Draft logits of listed tokens
  /// are bit-identical; unlisted tokens are never proposed. Verification
  /// is unchanged, so output is unchanged (greedy) or distribution
  /// preserving (the proposal q is the compact candidates' distribution).
  DeviceTensor draft_head_{};
  std::vector<std::int32_t> draft_ids_;  ///< compact row -> token id
  /// Runtime switch (benchmarks, SpecKnobs::vocab, scheduler
  /// thread, latched once per decode step): draft with the compact head or
  /// the full head. Without `--mtp-draft-vocab` only the full head exists.
  bool draft_compact_{true};
  [[nodiscard]] bool DraftCompact() const noexcept {
    return draft_compact_ && !draft_head_.empty();
  }
  [[nodiscard]] const DeviceTensor& DraftHead() const noexcept {
    return DraftCompact() ? draft_head_ : model_->output();
  }
  [[nodiscard]] std::int32_t DraftToken(std::int32_t row) const noexcept {
    return !DraftCompact() || row < 0 ||
                   static_cast<std::size_t>(row) >= draft_ids_.size()
               ? row
               : draft_ids_[static_cast<std::size_t>(row)];
  }
  void MapDraftCandidates(MtpCandidateLogits& c) const noexcept {
    if (!DraftCompact())
      return;
    const auto n =
        c.size == 0 || c.size > kMtpCandidates ? kMtpCandidates : c.size;
    for (std::size_t i = 0; i < n; ++i)
      c.ids[i] = static_cast<sampling::TokenId>(
          DraftToken(static_cast<std::int32_t>(c.ids[i])));
  }
  bool LoadDraftVocabulary(std::string* error_msg);
  /// The model geometry allows the wide mixer route (see Combine).
  bool wide_mixer_{false};
  /// Set by Moe when its epilogue is left for the combine that follows.
  mutable bool moe_pending_{false};
  /// Set by GatedDense when s_.shexp_half holds the SwiGLU rows its F16
  /// down projection reads.
  mutable bool shexp_half_ready_{false};
  /// W4A4 route: packed int4 copies (dense_i4::WeightBytes) of the dense
  /// prefill projections, keyed by the Q8_0 tensor's device pointer.
  std::unordered_map<const void*, void*> i4_weights_;
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by a combine that wrote s_.xn_half / s_.xn_q8t instead of s_.xn.
  mutable bool xn_half_{false};
  /// Set by a combine that already wrote the next mixer's inject partials.
  mutable bool inject_ready_{false};
  /// Partial sums per inject logit the last HcMix left in s_.inject.
  mutable std::uint32_t inject_parts_{1};
  /// 0 F32 residual (default); 2 F16 residual between the wide prefill
  /// combines without the F16 norm (the HC up projection rebuilds it from the
  /// F16 residual and fuses the next mixer's inject dots into the combine).
  /// Lossy (`--prefill-residual f16`).
  int hc_streams_{0};
  /// The authoritative residual is s_.res_half (F32 s_.res is stale).
  mutable bool res_half_live_{false};
  /// The last combine wrote per-stream scales (s_.xn_scale) instead of xn.
  mutable bool xn_scale_live_{false};
  mutable unsigned q8_slot_{0};
  /// A Q8_1 copy of `src` ([rows][cols]) that its producer wrote, valid only
  /// for the next Quantize call, which consumes it if that call quantizes
  /// exactly these rows (and clears it either way). Producers set it right
  /// before the projection that reads their output.
  struct PendingQ8 {
    const float* src{nullptr};
    const void* q8{nullptr};
    std::uint32_t rows{0};
    std::uint32_t cols{0};
  };
  mutable PendingQ8 pending_q8_{};
  /// The pre_q8 slot a decode producer may fill for `rows` (null on wide
  /// batches, which take the tiled routes).
  [[nodiscard]] void* PreQ8(unsigned slot, std::uint32_t rows) const noexcept;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
