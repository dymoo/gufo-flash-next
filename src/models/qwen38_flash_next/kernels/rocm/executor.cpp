#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

#include <fcntl.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dense_i4.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/lfm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/routed_prefill.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/spike.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

using core::GgmlType;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* allocated_bytes = nullptr) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hipMalloc(&p, bytes) != hipSuccess) {
    AssignError(error_msg,
                "hipMalloc of " + std::to_string(bytes) + " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  if (allocated_bytes)
    *allocated_bytes += bytes;
  return static_cast<T*>(p);
}

WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next matrix format");
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;
bool MatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > kVecBatch;
}
bool ExpertMatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > 4 * kVecBatch;
}

/// Wide dense Q8_0 projections take the F16 WMMA GEMM (F16 activation
/// rows, weights dequantized as they are staged, no per-block scaling) when
/// its 256-row tiles fill the device and the K sweep is short enough that
/// the F16 rows stay cache-resident: measured per shape at 2048 tokens
/// (W8A8 -> F16 ms): 16384x2560 5.7 -> 4.6, 13312x2560 4.3 -> 3.9,
/// 10240x320 1.02 -> 0.79, 2560x2560 0.81 -> 0.74, 2560x640 0.24 -> 0.21;
/// against 2560x6144 1.8 -> 2.1+, 640x2560 0.19 -> 0.23 and 320x10240
/// 0.40 -> 1.3, which keep the int8 route.
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

/// W4A4 prefill route (Options::dense_w4a4): prompt calls of at least this
/// many rows, inputs up to this width, and the rotation both operands share.
/// The row threshold is the coverage the quality ladder passed (campaign
/// tuning override: GUFO_DENSE_I4_MIN_ROWS).
constexpr std::uint32_t kDenseI4MinRows = 2048;
constexpr std::uint32_t kDenseI4MaxCols = 6144;
constexpr dense_i4::Rotation kDenseI4Rotation = dense_i4::Rotation::kBlock128;
/// One scale per 128-wide K slab for activations (token) and weights (row),
/// each an integer multiple of a per-token / per-row scale, instead of one
/// per token and per row: per-token A4 and per-row RTN W4 measured far
/// outside the quality envelope (W2/Q2 ladders).
constexpr bool kDenseI4SlabScales = true;

std::uint32_t DenseI4MinRows() {
  static const std::uint32_t rows = [] {
    const char* v = std::getenv("GUFO_DENSE_I4_MIN_ROWS");
    return v != nullptr && *v != '\0'
               ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10))
               : kDenseI4MinRows;
  }();
  return rows;
}

/// Offline-quantized S4G128 tensors (dense-w4-calib quantize.py output
/// converted to <dir>/index.tsv + dense.bin): per tensor the int8 codes
/// [rows][cols], uint8 slab scales [rows][cols / 128] and F32 row scales.
class S4Weights {
public:
  struct Entry {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint64_t offset;
    std::uint64_t bytes;
  };

  bool Open(const std::string& dir, std::string* error_msg) {
    std::ifstream index(dir + "/index.tsv");
    if (!index) {
      AssignError(error_msg, "cannot open " + dir + "/index.tsv");
      return false;
    }
    bin_path_ = dir + "/dense.bin";
    std::string name;
    Entry e{};
    while (index >> name >> e.rows >> e.cols >> e.offset >> e.bytes)
      entries_.emplace(name, e);
    if (entries_.empty()) {
      AssignError(error_msg, dir + "/index.tsv holds no tensors");
      return false;
    }
    return true;
  }
  bool Loaded() const { return !entries_.empty(); }
  const Entry* Find(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
  }
  bool Read(const Entry& e, std::vector<char>* out,
            std::string* error_msg) const {
    std::ifstream bin(bin_path_, std::ios::binary);
    out->resize(e.bytes);
    bin.seekg(static_cast<std::streamoff>(e.offset));
    if (!bin.read(out->data(), static_cast<std::streamsize>(e.bytes))) {
      AssignError(error_msg, "short read of " + bin_path_);
      return false;
    }
    // Codes are signed int4 held as int8. PackS4 keeps the low nibble, so an
    // unsaturated sign fold (+8) would load silently as -8.
    const std::size_t n = std::size_t{e.rows} * e.cols;
    const auto* codes = reinterpret_cast<const std::int8_t*>(out->data());
    const auto* bad = n > out->size()
                          ? codes
                          : std::find_if(codes, codes + n, [](std::int8_t q) {
                              return q < -8 || q > 7;
                            });
    if (bad != codes + n) {
      AssignError(
          error_msg,
          bin_path_ + " at offset " + std::to_string(e.offset) +
              (n > out->size() ? ": entry smaller than its codes"
                               : ": int4 code " + std::to_string(int{*bad}) +
                                     " outside [-8, 7] at element " +
                                     std::to_string(bad - codes)));
      return false;
    }
    return true;
  }

private:
  std::string bin_path_;
  std::unordered_map<std::string, Entry> entries_;
};

/// Token rows per routed F16 expert GEMM tile: the narrow tile for batches
/// whose buckets pad to one or two 16-row tiles, the wide one when the
/// mean bucket fills most of it (the weight dequantization is per tile).
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

std::uint32_t RoutedTileRows(std::size_t slots, const Config& c) {
  return slots >= static_cast<std::size_t>(16) * c.num_experts
             ? kRoutedTileRowsWide
             : kRoutedTileRowsNarrow;
}

/// Upper bound on launched routed tiles: every 16-padded bucket contributes
/// at most one partial tile beyond its rows.
std::size_t RoutedTileCapacity(std::size_t slots, const Config& c) {
  return (slots + static_cast<std::size_t>(c.num_experts) * 15) /
             kRoutedTileRowsNarrow +
         c.num_experts + 1;
}

std::uint32_t IndexerCapacity(const Config& c, std::uint32_t batch,
                              std::uint32_t context) {
  return static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
      context, std::uint64_t{c.indexer_top_k} + batch)));
}

/// Reserve() commits this far past the requested position, for kernels
/// that load whole tiles past the last row.
constexpr std::uint32_t kReserveHeadroom = 256;

}  // namespace

Session::~Session() {
  {
    const std::lock_guard lock(owner_->residency_mutex_);
    std::erase(owner_->sessions_, this);
    owner_->DropPaged(*this);
  }
  ReleaseKv();
  TrimRollback(0);
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                                  hipStream_t stream) {
  layout.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.RestoreLayout(layout, stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt,
    std::shared_ptr<qwen::vision::Encoder> encoder, hipStream_t stream) {
  if (prompt)
    prompt->rope.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.Configure(std::move(prompt), std::move(encoder), stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  cancelled_ = false;
  is_cancelled_ = std::move(check);
  vision_input_.SetCancellationCheck(is_cancelled_);
}

bool Session::CheckCancellation(std::string* error) const {
  try {
    cancelled_ = cancelled_ || (is_cancelled_ && is_cancelled_());
  } catch (...) {
    cancelled_ = true;
  }
  if (cancelled_) {
    AssignError(error, "generation cancelled");
    return false;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return allocated_bytes_ + rollback_bytes_ + vision_input_.Bytes();
}

std::size_t Session::CommittedKvBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& g : growable_)
    bytes += g.buffer.committed();
  return bytes;
}

void Session::ReleaseKv() noexcept {
  // Queued kernels may still read the pages being unmapped.
  if (owner_ != nullptr && CommittedKvBytes() != 0)
    (void)hipStreamSynchronize(owner_->stream());
  for (auto& g : growable_) {
    g.buffer.Trim(0);
    g.kept = 0;
  }
  reserved_tokens_ = 0;
}

namespace {
/// Rows of a cache that hold live data: positions for K/V, pooled blocks
/// for block keys.
std::uint32_t LiveRows(std::uint32_t tokens_per_row, bool mtp,
                       std::uint32_t position, std::uint32_t blocks,
                       std::uint32_t mtp_position, std::uint32_t mtp_blocks) {
  return tokens_per_row == 1 ? (mtp ? mtp_position : position)
                             : (mtp ? mtp_blocks : blocks);
}
}  // namespace

Executor::Pin::Pin(const Executor& executor, Session& session,
                   std::string* error_msg)
    : executor_(&executor), session_(&session) {
  const std::lock_guard lock(executor.residency_mutex_);
  ++session.pins_;
  session.last_use_ = ++executor.use_clock_;
  ok_ = executor.PageIn(session, error_msg);
}

Executor::Pin::~Pin() {
  const std::lock_guard lock(executor_->residency_mutex_);
  --session_->pins_;
}

std::size_t Executor::UnpooledSessionBytes() noexcept {
  return sizeof(Session::Control);
}

std::vector<std::pair<char*, std::size_t>> Executor::PagedSegments(
    Session& session) const {
  std::vector<std::pair<char*, std::size_t>> segments{
      {static_cast<char*>(session.arena_.data()), session.arena_bytes_}};
  for (auto& g : session.growable_) {
    const std::size_t live =
        std::size_t{LiveRows(g.tokens_per_row, g.mtp, session.position_,
                             session.blocks_, session.mtp_.position,
                             session.mtp_.blocks)} *
        g.row_bytes;
    if (live > g.kept)
      segments.emplace_back(static_cast<char*>(g.buffer.data()) + g.kept,
                            live - g.kept);
  }
  return segments;
}

bool Executor::SpillStaging(std::string* error_msg) const {
  for (std::size_t i = 0; i < spill_staging_.size(); ++i) {
    if ((spill_staging_[i] == nullptr &&
         !Check(hipHostMalloc(&spill_staging_[i], kSpillChunk), "spill staging",
                error_msg)) ||
        (spill_events_[i] == nullptr &&
         !Check(
             hipEventCreateWithFlags(&spill_events_[i], hipEventDisableTiming),
             "spill event", error_msg)))
      return false;
  }
  return true;
}

bool Executor::PageIn(Session& session, std::string* error_msg) const {
  if (session.resident_)
    return true;
  // Commit zero-fills on stream_; the copies below are ordered after it.
  bool ok = true;
  {
    const gufo::hip::TlbBatch batch;  // arena and caches: one TLB flush
    ok = session.arena_.Commit(session.arena_bytes_, stream_, error_msg);
    const auto live = std::max(session.position_, session.mtp_.position);
    session.resident_ = ok;
    ok = ok && (live == 0 || Reserve(session, live, error_msg));
  }
  ok = kv_pool_.Settle(error_msg) && ok;
  if (!session.resident_)
    return false;
  if (ok && !session.paged_.empty()) {
    std::size_t offset = 0;
    for (const auto& [device, bytes] : PagedSegments(session)) {
      ok = ok && Check(hipMemcpy(device, session.paged_.data() + offset, bytes,
                                 hipMemcpyHostToDevice),
                       "session page-in", error_msg);
      offset += bytes;
    }
  } else if (ok && session.spill_fd_ >= 0) {
    // Double-buffered: the disk fills one pinned chunk while the GPU copies
    // the other. Segments start on 4 KiB file offsets (O_DIRECT).
    ok = SpillStaging(error_msg);
    std::uint64_t file = 0;
    unsigned slot = 0;
    for (const auto& [device, bytes] : PagedSegments(session)) {
      for (std::size_t done = 0; ok && done < bytes; done += kSpillChunk) {
        const std::size_t n = std::min(kSpillChunk, bytes - done);
        const std::size_t aligned = (n + 4095) / 4096 * 4096;
        ok = Check(hipEventSynchronize(spill_events_[slot]), "spill wait",
                   error_msg) &&
             ::pread(session.spill_fd_, spill_staging_[slot], aligned,
                     static_cast<off_t>(file)) >= static_cast<ssize_t>(n) &&
             Check(hipMemcpyAsync(device + done, spill_staging_[slot], n,
                                  hipMemcpyHostToDevice, stream_),
                   "spill page-in", error_msg) &&
             Check(hipEventRecord(spill_events_[slot], stream_), "spill event",
                   error_msg);
        if (!ok && error_msg != nullptr && error_msg->empty())
          *error_msg = "spill file read failed";
        file += aligned;
        slot ^= 1U;
      }
      file = (file + 4095) / 4096 * 4096;
    }
    ok = ok && Check(hipStreamSynchronize(stream_), "spill page-in", error_msg);
  }
  if (!ok) {
    // Stay paged out with the saved copy intact.
    (void)hipStreamSynchronize(stream_);
    for (auto& g : session.growable_)
      g.buffer.Trim(g.kept);
    session.reserved_tokens_ = 0;
    session.arena_.Trim(0);
    session.resident_ = false;
    return false;
  }
  DropPaged(session);
  return true;
}

void Executor::DropPaged(Session& session) const {
  paged_ram_bytes_ -= session.paged_.size();
  session.paged_ = {};
  if (session.spill_fd_ >= 0)
    (void)::close(session.spill_fd_);
  session.spill_fd_ = -1;
  for (auto& g : session.growable_)
    g.kept = 0;
}

bool Executor::PageOut(Session& session) const {
  if (!session.resident_ || session.pins_ != 0 || session.spec_tokens_ != 0)
    return false;
  // Never drain a stream another thread is capturing into a graph.
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(stream_, &capture) != hipSuccess ||
      capture != hipStreamCaptureStatusNone ||
      hipStreamSynchronize(stream_) != hipSuccess)
    return false;
  session.TrimRollback(0);
  // Shared prefix pages stay mapped: they free nothing and others read them.
  for (auto& g : session.growable_) {
    const std::size_t live =
        std::size_t{LiveRows(g.tokens_per_row, g.mtp, session.position_,
                             session.blocks_, session.mtp_.position,
                             session.mtp_.blocks)} *
        g.row_bytes;
    g.kept = std::min(g.buffer.SharedPrefixBytes(), live);
  }
  const auto segments = PagedSegments(session);
  std::uint64_t bytes = 0;
  for (const auto& segment : segments)
    bytes += segment.second;
  // Host RAM within the explicit paging budget (a container's meminfo can
  // overstate the host, which also backs GTT), then the NVMe spill.
  bool ok = false;
  if (paged_ram_bytes_ + bytes <= paging_ram_budget_) {
    try {
      session.paged_.resize(bytes);
      ok = true;
    } catch (const std::bad_alloc&) {
      session.paged_ = {};
    }
    std::size_t offset = 0;
    for (const auto& [device, n] : segments) {
      ok = ok && hipMemcpy(session.paged_.data() + offset, device, n,
                           hipMemcpyDeviceToHost) == hipSuccess;
      offset += n;
    }
    if (ok)
      paged_ram_bytes_ += bytes;
    else
      session.paged_ = {};
  }
  if (!ok && !spill_directory_.empty() && SpillStaging(nullptr)) {
    struct statvfs disk{};
    // Keep 64 GiB free, or a quarter of a smaller filesystem (which a fixed
    // 64 GiB reserve would never let spill).
    constexpr std::uint64_t kDiskReserve = std::uint64_t{64} << 30;
    int fd = -1;
    if (::statvfs(spill_directory_.c_str(), &disk) == 0 &&
        std::uint64_t{disk.f_bavail} * disk.f_frsize >=
            bytes + std::min(kDiskReserve, std::uint64_t{disk.f_blocks} *
                                               disk.f_frsize / 4)) {
      fd = ::open(spill_directory_.c_str(),
                  O_TMPFILE | O_RDWR | O_CLOEXEC | O_DIRECT, 0600);
    }
    ok = fd >= 0;
    std::uint64_t file = 0;
    for (const auto& [device, n] : segments) {
      for (std::size_t done = 0; ok && done < n; done += kSpillChunk) {
        const std::size_t chunk = std::min(kSpillChunk, n - done);
        const std::size_t aligned = (chunk + 4095) / 4096 * 4096;
        ok =
            hipMemcpy(spill_staging_[0], device + done, chunk,
                      hipMemcpyDeviceToHost) == hipSuccess &&
            ::pwrite(fd, spill_staging_[0], aligned,
                     static_cast<off_t>(file)) == static_cast<ssize_t>(aligned);
        file += aligned;
      }
      file = (file + 4095) / 4096 * 4096;
    }
    if (ok) {
      session.spill_fd_ = fd;
    } else if (fd >= 0) {
      (void)::close(fd);
    }
  }
  if (!ok) {
    for (auto& g : session.growable_)
      g.kept = 0;
    return false;
  }
  for (auto& g : session.growable_)
    g.buffer.Trim(g.kept);
  session.reserved_tokens_ = 0;
  session.arena_.Trim(0);
  session.resident_ = false;
  return true;
}

bool Executor::EvictOne() const {
  const std::lock_guard lock(residency_mutex_);
  Session* victim = nullptr;
  for (auto* candidate : sessions_) {
    if (candidate->resident_ && candidate->pins_ == 0 &&
        candidate->spec_tokens_ == 0 &&
        (victim == nullptr || candidate->last_use_ < victim->last_use_))
      victim = candidate;
  }
  return victim != nullptr && PageOut(*victim);
}

void Session::TrimRollback(std::uint32_t depth) noexcept {
  if (depth >= rollback_depth_)
    return;
  // Keep graphs whose rows still exist. Deeper verifier graphs capture
  // discarded pointers; ordinary decode and MTP graphs do not.
  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if ((it->first & (std::uint64_t{1} << 32)) != 0 &&
        (it->first & (std::uint64_t{1} << 40)) == 0 &&
        (it->first & 0xFFFFU) > depth + 1) {
      (void)hipGraphExecDestroy(it->second);
      it = graphs_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto row = depth; row < rollback_depth_; ++row) {
    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
    ple_snapshots_.rows[row] = nullptr;
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
  ngram_snapshots_.resize(depth);
}

void Session::Reset() {
  const std::lock_guard lock(owner_->residency_mutex_);
  TrimRollback(0);
  ReleaseKv();
  position_ = 0;
  spec_tokens_ = 0;
  ngram_.Reset();
  mtp_.position = 0;
  mtp_.blocks = 0;
  dspark_.position = 0;
  blocks_ = 0;
  if (!resident_) {
    // Paged out: the next page-in starts from zeroed pages.
    owner_->DropPaged(*this);
    return;
  }
  const Config& c = owner_->config();
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      (void)hipMemset(l.conv_state, 0,
                      static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                          c.SsmConvChannels() * sizeof(float));
      (void)hipMemset(l.state, 0,
                      static_cast<std::size_t>(c.ssm_num_v_heads) *
                          c.ssm_head_dim * c.ssm_head_dim *
                          GdnStateElementBytes(owner_->gdn_state()));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    (void)hipMemset(ple_history_, 0,
                    static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim() *
                        sizeof(float));
  }
}

Executor::~Executor() {
  // Readers write pinned staging memory; drain them before freeing it,
  // including when a forward failed before reaching PLE.
  if (ple_pending_) {
    (void)ngram_->WaitRead();
  }
  for (std::size_t i = 0; i < spill_staging_.size(); ++i) {
    (void)hipHostFree(spill_staging_[i]);
    (void)hipEventDestroy(spill_events_[i]);
  }
  (void)hipFree(batch_logits_owned_);
  (void)hipHostFree(batch_logits_host_);
  (void)hipHostFree(batch_argmax_host_);
  (void)hipHostFree(batch_gdn_host_);
  (void)hipHostFree(batch_attn_host_);
  (void)hipHostFree(lfm_conv_host_);
  (void)hipFree(batch_attn_partials_);
  (void)hipHostFree(batch_controls_);
  (void)hipHostFree(batch_candidates_host_);
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  for (void* p :
       {static_cast<void*>(host_emb_), static_cast<void*>(control_host_),
        static_cast<void*>(tokens_host_), static_cast<void*>(logits_host_),
        static_cast<void*>(mtp_token_host_), static_cast<void*>(counts_host_),
        static_cast<void*>(mtp_candidates_host_),
        static_cast<void*>(tiles_host_), static_cast<void*>(steer_host_),
        static_cast<void*>(route_keep_host_),
        static_cast<void*>(ds_.chain_host), static_cast<void*>(ds_.conf_host),
        static_cast<void*>(ds_.block_host), static_cast<void*>(ds_.pos_host),
        static_cast<void*>(ds_.attn_host),
        static_cast<void*>(ds_.store_host)}) {
    if (p != nullptr) {
      (void)hipHostFree(p);
    }
  }
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (counts_ready_ != nullptr) {
    (void)hipEventDestroy(counts_ready_);
  }
  for (auto event : ds_.store_done)
    if (event != nullptr)
      (void)hipEventDestroy(event);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
  if (router_dump_ != nullptr) {
    (void)std::fclose(router_dump_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {
  std::unique_ptr<Executor> e(new Executor());
  e->kv_pool_.SetReclaim([executor = e.get()] { return executor->EvictOne(); });
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  e->hc_streams_ = options.residual_f16 ? 2 : 0;
  if (const char* path = std::getenv("GUFO_FLASH_NEXT_ROUTER_DUMP")) {
    e->router_dump_ = std::fopen(path, "ab");
    if (e->router_dump_ == nullptr) {
      AssignError(error_msg, std::string("cannot open router dump ") + path);
      return nullptr;
    }
  }
  if (options.router_norm_k != 0) {
    const std::uint32_t used = model.config().num_experts_used;
    if (options.router_norm_k < used || options.router_norm_k > 32) {
      AssignError(error_msg,
                  "--router-norm-k must be " + std::to_string(used) + "..32");
      return nullptr;
    }
    e->router_norm_k_ = options.router_norm_k;
  }
  if (options.fast_experts != 0 &&
      options.fast_experts >= model.config().num_experts_used) {
    AssignError(error_msg,
                "--fast-experts must be 1.." +
                    std::to_string(model.config().num_experts_used - 1));
    return nullptr;
  }
  if (qfn_mmq_init(0) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  spike::Init(model, e->stream_);
  if (e->options_.routed_down4) {
    for (const DeviceLayer& l : model.layers()) {
      const DeviceTensor& d = l.ffn_down_exps;
      if (!d.empty() && d.type == core::GgmlType::kQ5_1 &&
          !routed_prefill::RequantDown4(d.data, WeightType::kQ5_1,
                                        std::size_t{d.rows} * d.experts, d.cols,
                                        e->stream_)) {
        AssignError(error_msg, "routed down re-quantization failed");
        return nullptr;
      }
    }
  }
  if (e->options_.routed_sym != 0) {
    // Symmetric int4 experts, in place: every route (decode included) reads
    // the same lossy values from the same Q4_K / Q5_1 containers.
    const int group = e->options_.routed_sym;
    for (const DeviceLayer& l : model.layers()) {
      for (const DeviceTensor* t : {&l.ffn_gate_exps, &l.ffn_up_exps}) {
        if (!t->empty() && t->type == core::GgmlType::kQ4_K &&
            !routed_prefill::RequantQ4KSym(t->data, WeightType::kQ4_K,
                                           std::size_t{t->rows} * t->experts,
                                           t->cols, group, e->stream_)) {
          AssignError(error_msg, "routed gate/up re-quantization failed");
          return nullptr;
        }
      }
      const DeviceTensor& d = l.ffn_down_exps;
      if (!d.empty() && d.type == core::GgmlType::kQ5_1 &&
          !routed_prefill::RequantDownSym(d.data, WeightType::kQ5_1,
                                          std::size_t{d.rows} * d.experts,
                                          d.cols, e->stream_)) {
        AssignError(error_msg, "routed down re-quantization failed");
        return nullptr;
      }
    }
    if (!Check(hipStreamSynchronize(e->stream_), "routed re-quantization",
               error_msg))
      return nullptr;
  }
  if (!Check(hipEventCreateWithFlags(&e->counts_ready_, hipEventDisableTiming),
             "expert counts event", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  // Defence in depth: model admission (src/core/model_admission.cpp) rejects
  // these KV formats before any weight upload. Only the plain-attention
  // (lfm2moe) kernels read tq8 and TurboQuant rows narrower than 256.
  const bool plain_attention =
      c.Traits().attention_mixer == MixerKind::kAttention;
  if (options.kv_format != KvCacheFormat::kF16 && c.head_dim != 256 &&
      !plain_attention) {
    AssignError(error_msg, "TurboQuant KV caches need 256-wide heads");
    return nullptr;
  }
  if (options.kv_format == KvCacheFormat::kTq8 && !plain_attention) {
    AssignError(error_msg, "the tq8 KV cache supports only LFM2 models");
    return nullptr;
  }
  const std::size_t T = e->options_.max_batch;
  e->blaslt_ = BlasLt::Create(e->stream_, error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  s.tokens = Alloc<std::int32_t>(a, T, error_msg);
  // The routed-expert WMMA route always narrows hidden-wide token rows here,
  // even when no F16 matrix is that wide (an F32 router).
  s.x_half = Alloc<std::uint16_t>(
      a, T * std::max<std::size_t>(model.max_half_cols(), hidden), error_msg);
  for (void*& slot : s.x_q8) {
    slot = Alloc<std::uint8_t>(
        a,
        qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                           static_cast<int>(model.max_q8_cols())),
        error_msg);
  }
  for (void*& slot : s.pre_q8) {
    slot = Alloc<std::uint8_t>(
        a,
        qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                           static_cast<int>(std::max<std::size_t>(
                               model.max_q8_cols(), hc_dim))),
        error_msg);
  }
  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.xn_half = Alloc<__half>(a, T * hc_dim, error_msg);
  if (e->hc_streams_ != 0) {
    s.res_half = Alloc<__half>(a, T * hc_dim, error_msg);
    s.xn_scale = f32(T * c.hc_count);
  }
  s.xn_q8t = Alloc<std::uint8_t>(a, Q8TiledBytes(T, hc_dim), error_msg);
  s.lo = f32(T * c.hc_low_rank);
  s.hc_gate = f32(T * hc_dim);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space,
  // as do a short convolution's B, C and x rows.
  const bool short_conv = c.Traits().linear_mixer == MixerKind::kShortConv;
  s.qkvz = f32(T * std::max<std::size_t>({c.SsmConvChannels() + c.SsmValueDim(),
                                          c.ple_layer >= 0 ? hc_dim : 0,
                                          short_conv ? 3 * hidden : 0}));
  s.qkv = s.qkvz;
  s.z = s.qkvz != nullptr ? s.qkvz + T * c.SsmConvChannels() : nullptr;
  s.alpha_beta = f32(T * 2 * c.ssm_num_v_heads);
  s.conv_scratch = f32((T + c.ssm_conv_kernel) * c.SsmConvChannels());
  s.qn = f32(T * c.SsmKeyDim());
  s.kn = f32(T * c.SsmKeyDim());
  s.gdn_raw = f32(T * c.SsmValueDim());
  s.gdn_out = f32(T * c.SsmValueDim());
  s.qg = f32(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = f32(T * c.AttentionQDim());
  s.attn_gate = f32(T * c.AttentionQDim());
  s.k = f32(T * c.AttentionKvDim());
  s.v = f32(T * c.AttentionKvDim());
  s.iq = f32(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = f32(T * c.indexer_head_dim);
  // Without an indexer (qwen35moe) attention is always dense and no mask
  // is read.
  const std::uint32_t max_blocks =
      c.HasIndexer()
          ? (c.context_length + c.compress_ratio - 1) / c.compress_ratio
          : 32;
  e->mask_words_ = (max_blocks + 31) / 32;
  s.mask = Alloc<std::uint32_t>(a, T * e->mask_words_, error_msg);
  s.scores =
      f32(static_cast<std::size_t>(e->select_chunk_) * e->mask_words_ * 32);
  s.ctx = f32(T * c.AttentionQDim());
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kMaxAttentionSplits * (c.head_dim + 2));
  if (c.ple_layer >= 0) {
    s.ple_emb = f32(T * c.PleEmbeddingDim());
    // PLE finishes before this layer's mixer/SSM. Its gate consumes key
    // and query before normalization/convolution reuse those two buffers.
    // The gated input stays separate until PleInject consumes both outputs.
    s.ple_key = s.hc_gate;
    s.ple_value = s.block_out;
    s.ple_query = s.xn;
    s.ple_gated = s.qkvz;
    s.ple_norm = s.hc_gate;
    s.ple_conv = s.xn;
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    void* pinned = nullptr;
    if (!Check(hipHostMalloc(&pinned, T * c.PleEmbeddingDim() * sizeof(float)),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    e->host_rows_.resize(T * c.ple_heads);
  }
  s.router = f32(T * (c.num_experts + 1));
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  {
    const std::size_t compact = RoutedCompactRows(slots, c.num_experts);
    s.routed_bounds = Alloc<std::int32_t>(a, c.num_experts + 1, error_msg);
    s.routed_cursors = Alloc<std::int32_t>(a, c.num_experts, error_msg);
    s.rows_token = Alloc<std::int32_t>(a, compact, error_msg);
    s.rows_slot = Alloc<std::int32_t>(a, compact, error_msg);
    s.routed_tiles =
        Alloc<std::int32_t>(a, 3 * RoutedTileCapacity(slots, c), error_msg);
  }
  s.weights = f32(slots);
  s.gate_e = f32(slots * c.expert_ff);
  s.up_e = f32(slots * c.expert_ff);
  s.down_e = f32(slots * hidden);
  e->logit_alias_floats_ = slots * hidden;  // batch logits reuse (batch.cpp)
  s.shexp_gate = f32(T * c.shared_expert_ff);
  s.shexp_up = f32(T * c.shared_expert_ff);
  s.shexp_out = f32(T * hidden);
  s.shexp_half = Alloc<__half>(a, T * c.shared_expert_ff, error_msg);
  if (e->options_.dense_w4a4) {
    s.x_i4 = Alloc<std::uint8_t>(
        a, dense_i4::ActivationBytes(T, kDenseI4MaxCols), error_msg);
    if (s.x_i4 == nullptr || !e->PackI4Weights(error_msg))
      return nullptr;
  }
  s.logits =
      f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  if (!c.Traits().shared_expert && s.router != nullptr &&
      s.shexp_out != nullptr) {
    // Without a shared expert, the gate column and shared output the routed
    // epilogues add stay zero for the executor's life (LfmFfn).
    (void)hipMemset(s.router, 0, T * (c.num_experts + 1) * sizeof(float));
    (void)hipMemset(s.shexp_out, 0, T * hidden * sizeof(float));
  }
  if (e->options_.fast_experts != 0) {
    void* staging = nullptr;
    s.route_keep = Alloc<std::uint8_t>(a, T, error_msg);
    if (s.route_keep == nullptr ||
        !Check(hipMemset(s.route_keep, 0, T), "fast-model rows", error_msg) ||
        !Check(hipHostMalloc(&staging, T), "pinned fast-model rows", error_msg))
      return nullptr;
    e->route_keep_host_ = static_cast<std::uint8_t*>(staging);
  }
  if (const auto& adapters = e->options_.adapters; !adapters.empty()) {
    std::vector<float> directions, alphas;
    std::vector<std::uint64_t> routed_layers;
    std::vector<std::uint8_t> writers;
    for (const auto& adapter : adapters) {
      directions.insert(directions.end(), adapter.direction.begin(),
                        adapter.direction.end());
      alphas.push_back(adapter.alpha);
      routed_layers.push_back(adapter.routed_layers);
      writers.push_back(adapter.writers);
      e->steer_writers_any_ |= adapter.writers;
    }
    s.steer = Alloc<std::uint8_t>(a, T, error_msg);
    e->steer_directions_ = f32(directions.size());
    e->steer_alphas_ = f32(alphas.size());
    e->steer_routed_layers_ =
        Alloc<std::uint64_t>(a, routed_layers.size(), error_msg);
    e->steer_writers_ = Alloc<std::uint8_t>(a, writers.size(), error_msg);
    void* staging = nullptr;
    if (e->steer_directions_ == nullptr || e->steer_alphas_ == nullptr ||
        e->steer_routed_layers_ == nullptr || e->steer_writers_ == nullptr ||
        !Check(hipMemcpy(e->steer_routed_layers_, routed_layers.data(),
                         routed_layers.size() * sizeof(std::uint64_t),
                         hipMemcpyHostToDevice),
               "adapter upload", error_msg) ||
        !Check(hipMemcpy(e->steer_writers_, writers.data(), writers.size(),
                         hipMemcpyHostToDevice),
               "adapter upload", error_msg) ||
        !Check(
            hipMemcpy(e->steer_directions_, directions.data(),
                      directions.size() * sizeof(float), hipMemcpyHostToDevice),
            "adapter upload", error_msg) ||
        !Check(hipMemcpy(e->steer_alphas_, alphas.data(),
                         alphas.size() * sizeof(float), hipMemcpyHostToDevice),
               "adapter upload", error_msg) ||
        !Check(hipHostMalloc(&staging, T), "pinned adapter rows", error_msg)) {
      return nullptr;
    }
    e->steer_host_ = static_cast<std::uint8_t*>(staging);
  }
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(hipHostMalloc(&control, sizeof(Session::Control)),
               "pinned control buffer", error_msg) ||
        !Check(hipHostMalloc(&tokens, T * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(hipHostMalloc(&logits, static_cast<std::size_t>(
                                          e->options_.max_logit_rows) *
                                          c.vocab_size * sizeof(float)),
               "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(hipHostMalloc(&counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    void* tiles = nullptr;
    if (!Check(hipHostMalloc(&tiles, 3 * RoutedTileCapacity(slots, c) *
                                         sizeof(std::int32_t)),
               "pinned routed tile map", error_msg)) {
      return nullptr;
    }
    e->tiles_host_ = static_cast<std::int32_t*>(tiles);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
  }
  // The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for the
  // W8A8 down projection) needs the four-stream geometry, 32-wide blocks and
  // a Q8_0 down projection.
  e->wide_mixer_ = c.hc_count == 4 && c.hidden_size % 32 == 0 &&
                   !model.layers().empty() &&
                   model.layers()[0].hc_ffn.down.type == GgmlType::kQ8_0;
  // DSpark verification reads the same greedy argmax scratch.
  if (model.has_mtp() || model.dspark() != nullptr) {
    // The trunk's kept rows are session-owned. Its transient residual and
    // mixer buffers are free while MTP constructs its input. The split
    // projections consume h/embd before the first mixer rewrites xn/mixed.
    s.mtp_h = s.xn;
    s.mtp_embd = s.mixed;
    s.mtp_eproj = s.qkvz;
    s.mtp_res = s.res;
    s.mtp_argmax = Alloc<ArgmaxCandidate>(a, kArgmaxParts, error_msg);
    s.mtp_token = Alloc<std::int32_t>(a, 1, error_msg);
    const auto candidate_ids = MtpCandidateWorkspaceSize(c.vocab_size);
    s.mtp_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    s.mtp_scratch_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    // Final selection consumes intermediate IDs before writing its scores.
    // Pack those scores behind the 64 returned IDs for one host transfer.
    s.mtp_scores = reinterpret_cast<float*>(s.mtp_ids + kMtpCandidates);
    if (!Check(hipHostMalloc(&e->mtp_token_host_, sizeof(std::int32_t)),
               "pinned draft token", error_msg) ||
        !Check(
            hipHostMalloc(&e->mtp_candidates_host_, sizeof(MtpCandidateLogits)),
            "pinned draft candidates", error_msg)) {
      return nullptr;
    }
    std::construct_at(e->mtp_candidates_host_);
    e->mtp_candidates_host_->size =
        std::min<std::size_t>(c.vocab_size, kMtpCandidates);
  }
  if (const auto* d = model.dspark()) {
    const std::size_t rows = kDsparkMaxRows;
    const std::size_t kv = std::size_t{d->kv_heads} * d->head_dim;
    const std::size_t ffn = d->blocks.front().gate.rows;
    auto& ds = e->ds_;
    ds.features = f32(T * d->target_layers.size() * hidden);
    ds.g = f32(T * hidden);
    ds.kv = f32(T * 2 * d->blocks.size() * kv);
    ds.k = f32(rows * kv);
    ds.v = f32(rows * kv);
    ds.x = f32(rows * hidden);
    ds.xn = f32(rows * hidden);
    ds.q = f32(rows * hidden);
    ds.ctx = f32(rows * hidden);
    ds.o = f32(rows * hidden);
    ds.gate = f32(rows * ffn);
    ds.up = f32(rows * ffn);
    ds.logits = f32(rows * c.vocab_size);
    const auto workspace = DsparkCandidateWorkspaceSize(c.vocab_size);
    ds.cand_ids = Alloc<std::uint32_t>(a, workspace, error_msg);
    ds.cand_scratch = Alloc<std::uint32_t>(a, workspace, error_msg);
    ds.cand_scores = f32(kDsparkCandidates);
    ds.conf_in = f32(rows * (hidden + d->rank));
    ds.conf = f32(rows);
    const std::size_t links = rows + kDsparkMaxItems;
    const std::size_t positions = 1 + 2 * kDsparkMaxItems;
    ds.block = Alloc<std::int32_t>(a, rows, error_msg);
    ds.chain = Alloc<std::int32_t>(a, links, error_msg);
    ds.pos = Alloc<std::uint32_t>(a, positions, error_msg);
    const std::size_t stores =
        std::size_t{DsparkScratch::kStoreSlots} * kMaxBatchSessions;
    ds.store_device = Alloc<DsparkStoreItem>(a, stores, error_msg);
    ds.partials =
        f32(rows * d->heads * kMaxAttentionSplits * (d->head_dim + 2));
    const std::size_t attn =
        d->blocks.size() * rows * sizeof(AttentionBatchItem);
    if (!Check(hipHostMalloc(&ds.block_host, rows * sizeof(std::int32_t)),
               "pinned DSpark block", error_msg) ||
        !Check(hipHostMalloc(&ds.chain_host, links * sizeof(std::int32_t)),
               "pinned DSpark tokens", error_msg) ||
        !Check(hipHostMalloc(&ds.pos_host, positions * sizeof(std::uint32_t)),
               "pinned DSpark positions", error_msg) ||
        !Check(hipHostMalloc(&ds.conf_host, rows * sizeof(float)),
               "pinned DSpark confidence", error_msg) ||
        !Check(hipHostMalloc(&ds.store_host, stores * sizeof(DsparkStoreItem)),
               "pinned DSpark descriptors", error_msg) ||
        !Check(hipHostMalloc(&ds.attn_host, attn, hipHostMallocMapped),
               "DSpark attention items", error_msg) ||
        !Check(hipHostGetDevicePointer(
                   reinterpret_cast<void**>(&ds.attn_device), ds.attn_host, 0),
               "DSpark attention items", error_msg)) {
      return nullptr;
    }
    for (auto& event : ds.store_done) {
      if (!Check(hipEventCreateWithFlags(&event, hipEventDisableTiming),
                 "DSpark store event", error_msg) ||
          !Check(hipEventRecord(event, e->stream_), "DSpark store event",
                 error_msg))
        return nullptr;
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  if (model.has_mtp() && !e->LoadDraftVocabulary(error_msg))
    return nullptr;
  return e;
}

bool Executor::LoadDraftVocabulary(std::string* error_msg) {
  const char* path = options_.draft_vocab.c_str();
  if (*path == '\0')
    return true;
  const DeviceTensor& head = model_->output();
  if (head.type != GgmlType::kQ8_0 || head.cols % 32 != 0) {
    AssignError(error_msg, "draft vocabulary needs a Q8_0 output head");
    return false;
  }
  const std::size_t limit = std::clamp<std::size_t>(
      options_.draft_vocab_size, std::size_t{64}, std::size_t{head.rows});
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) {
    AssignError(error_msg, std::string("cannot open draft vocabulary ") + path);
    return false;
  }
  std::vector<char> seen(head.rows, 0);
  long id = 0;
  while (draft_ids_.size() < limit && std::fscanf(file, "%ld", &id) == 1) {
    if (id < 0 || static_cast<std::size_t>(id) >= head.rows || seen[id] != 0)
      continue;
    seen[id] = 1;
    draft_ids_.push_back(static_cast<std::int32_t>(id));
  }
  (void)std::fclose(file);
  if (draft_ids_.size() < kMtpCandidates) {
    AssignError(error_msg, "draft vocabulary lists too few tokens");
    return false;
  }
  const std::size_t row_bytes = std::size_t{head.cols} / 32 * 34;
  auto* rows = Alloc<std::uint8_t>(allocations_, draft_ids_.size() * row_bytes,
                                   error_msg);
  if (rows == nullptr)
    return false;
  const auto* source = static_cast<const std::uint8_t*>(head.data);
  for (std::size_t i = 0; i < draft_ids_.size(); ++i) {
    if (!Check(hipMemcpyAsync(
                   rows + i * row_bytes,
                   source + static_cast<std::size_t>(draft_ids_[i]) * row_bytes,
                   row_bytes, hipMemcpyDeviceToDevice, stream_),
               "draft vocabulary rows", error_msg))
      return false;
  }
  if (!Check(hipStreamSynchronize(stream_), "draft vocabulary", error_msg))
    return false;
  draft_head_ = head;
  draft_head_.data = rows;
  draft_head_.rows = static_cast<std::uint32_t>(draft_ids_.size());
  // The copied descriptor still points at the head's decode twin
  // (`--gguf-decode-override`), which is indexed by token id: reading it for
  // a head indexed by draft rank scored the wrong tokens (acceptance < 3%).
  draft_head_.alt = nullptr;
  std::fprintf(stderr, "[draft vocabulary] %zu of %u head rows from %s\n",
               draft_ids_.size(), head.rows, path);
  return true;
}

std::unique_ptr<Session> Executor::CreateSession(core::SessionMode mode,
                                                 std::uint32_t max_context,
                                                 std::string* error_msg) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  const bool speculative = mode == core::SessionMode::kSpeculative;
  s->dspark_enabled_ = speculative && model_->dspark() != nullptr;
  s->mtp_enabled_ = speculative && !s->dspark_enabled_;
  if (s->mtp_enabled_ && !has_mtp()) {
    AssignError(error_msg, "MTP session requires a loaded predictor");
    return nullptr;
  }
  const Config& c = config();
  if (max_context == 0 || max_context > c.context_length) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  // Before sparse attention starts, pooling can lag by indexer_top_k
  // rows. Afterwards only an incomplete block precedes the current batch.
  // Completed block keys remain in block_k; their raw rows are dead.
  s->index_capacity_ = IndexerCapacity(c, options_.max_batch, max_context);
  s->linear_.resize(c.num_layers);
  s->attention_.resize(c.num_layers);
  auto& a = s->allocations_;
  // Context-proportional caches reserve address space only: Reserve()
  // commits their pages as positions grow and Reset() releases them, so a
  // slot costs memory for the context it holds, not for max_context.
  bool reserved = true;
  bool mtp_cache = false;
  const auto grow = [&](std::size_t rows, std::size_t row_bytes,
                        std::uint32_t tokens_per_row) {
    auto& g = s->growable_.emplace_back();
    g.row_bytes = row_bytes;
    g.tokens_per_row = tokens_per_row;
    g.mtp = mtp_cache;
    reserved =
        reserved && g.buffer.Reserve(rows * row_bytes, kv_pool_, error_msg);
    return g.buffer.data();
  };
  const std::size_t kv_row = c.AttentionKvDim();
  // TurboQuant codes and their FP16 scales grow as separate prefixes.
  const auto alloc_kv = [&](KvView& kv) {
    kv.format = options_.kv_format;
    if (kv.format == KvCacheFormat::kF16) {
      kv.k = grow(max_context, kv_row * sizeof(__half), 1);
      kv.v = grow(max_context, kv_row * sizeof(__half), 1);
      return;
    }
    // KvCodeBytes is per 256 values; lfm2's 64-wide rows take a quarter.
    const std::size_t codes =
        std::size_t{c.num_kv_heads} * KvCodeBytes(kv.format) * c.head_dim / 256;
    const std::size_t scales = std::size_t{c.num_kv_heads} * sizeof(__half);
    kv.k = grow(max_context, codes, 1);
    kv.v = grow(max_context, codes, 1);
    kv.k_scale = static_cast<__half*>(grow(max_context, scales, 1));
    kv.v_scale = static_cast<__half*>(grow(max_context, scales, 1));
  };
  // Fixed state is placed in the session's pooled arena (Session::arena_),
  // committed on first use and paged out when idle under pool pressure.
  std::vector<std::function<void(char*)>> place;
  const auto fixed_bytes = [&](auto*& pointer, std::size_t bytes) {
    const std::size_t offset = s->arena_bytes_;
    s->arena_bytes_ += (std::max<std::size_t>(bytes, 4) + 255) / 256 * 256;
    place.emplace_back([&pointer, offset](char* base) {
      pointer = reinterpret_cast<std::remove_reference_t<decltype(pointer)>>(
          base + offset);
    });
  };
  const auto fixed = [&](float*& pointer, std::size_t count) {
    fixed_bytes(pointer, count * sizeof(float));
  };
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      auto& l = s->linear_[il];
      fixed(l.conv_state, conv_elems);
      fixed_bytes(l.state,
                  state_elems * GdnStateElementBytes(options_.gdn_state));
    } else {
      auto& at = s->attention_[il];
      alloc_kv(at.kv);
      if (c.HasIndexer()) {
        fixed(at.index_k, static_cast<std::size_t>(s->index_capacity_) *
                              c.indexer_head_dim);
        at.block_k = static_cast<__half*>(
            grow(max_context / c.compress_ratio + 1,
                 c.indexer_head_dim * sizeof(__half), c.compress_ratio));
      }
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist =
        static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    fixed(s->ple_history_, hist);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);
  if (s->mtp_enabled_) {
    fixed(s->mtp_.target_hidden,
          static_cast<std::size_t>(options_.max_speculative) * c.HcDim());
    mtp_cache = true;
    alloc_kv(s->mtp_.kv);
    fixed(s->mtp_.index_k,
          std::size_t{s->index_capacity_} * c.indexer_head_dim);
    s->mtp_.block_k = static_cast<__half*>(
        grow(max_context / c.compress_ratio + 1,
             c.indexer_head_dim * sizeof(__half), c.compress_ratio));
    fixed(s->mtp_.h, c.HcDim());
  }
  if (s->dspark_enabled_) {
    // ponytail: F16 draft K/V whatever --kv-cache says (small: 5 layers of
    // two heads); TurboQuant rows would need the rotated query path.
    const auto* d = model_->dspark();
    const std::size_t row = std::size_t{d->kv_heads} * d->head_dim;
    mtp_cache = false;
    s->dspark_.kv.resize(d->blocks.size());
    for (auto& kv : s->dspark_.kv) {
      kv.format = KvCacheFormat::kF16;
      kv.k = grow(max_context, row * sizeof(__half), 1);
      kv.v = grow(max_context, row * sizeof(__half), 1);
    }
  }
  reserved =
      reserved && s->arena_.Reserve(s->arena_bytes_, kv_pool_, error_msg);
  if (!reserved)
    return nullptr;
  for (const auto& set : place)
    set(static_cast<char*>(s->arena_.data()));
  s->allocated_bytes_ += s->arena_bytes_;
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  const std::lock_guard lock(residency_mutex_);
  sessions_.push_back(s.get());
  return s;
}

bool Executor::EnsureRollback(Session& session, std::uint32_t depth,
                              std::string* error_msg) const {
  if (depth <= session.rollback_depth_)
    return true;
  if (depth >= options_.max_speculative ||
      depth > std::size(session.ple_snapshots_.rows) ||
      session.spec_tokens_ != 0) {
    AssignError(error_msg, "invalid rollback allocation depth");
    return false;
  }
  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear = c.num_layers - c.AttentionLayerCount();
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  session.rollback_allocations_.reserve(depth);
  session.ngram_snapshots_.resize(depth);
  // New row buffers flush the GPU TLB once for all rows.
  bool ok = true;
  {
    const gufo::hip::TlbBatch batch;
    for (auto row = session.rollback_depth_; ok && row < depth; ++row) {
      const auto state = GdnRollbackRowFloats(
          row, c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim);
      const auto bytes = (linear * (conv + state) + ple) * sizeof(float);
      gufo::hip::GrowableBuffer row_buffer;
      ok = row_buffer.Reserve(bytes, kv_pool_, error_msg) &&
           row_buffer.Commit(bytes, stream_, error_msg);
      if (!ok)
        break;
      auto* allocation = static_cast<float*>(row_buffer.data());
      session.rollback_allocations_.push_back(std::move(row_buffer));
      session.rollback_bytes_ += bytes;
      auto* next = allocation;
      for (std::uint32_t il = 0; il < c.num_layers; ++il) {
        if (!c.IsLinearLayer(il))
          continue;
        session.linear_[il].conv_snapshots.rows[row] = next;
        next += conv;
        session.linear_[il].state_snapshots.rows[row] = next;
        next += state;
      }
      if (ple != 0) {
        session.ple_snapshots_.rows[row] = next;
      }
      session.rollback_depth_ = row + 1;
    }
  }
  return kv_pool_.Settle(error_msg) && ok;
}

std::size_t Executor::SessionBytes(
    core::SessionMode mode, std::uint32_t max_context,
    std::uint32_t rollback_depth) const noexcept {
  const auto& c = config();
  const std::size_t attention = c.AttentionLayerCount();
  const std::size_t linear = c.num_layers - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const std::size_t ring =
      c.HasIndexer()
          ? std::size_t{IndexerCapacity(c, options_.max_batch, max_context)} *
                c.indexer_head_dim * sizeof(float)
          : 0;
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  return ((linear * conv + ple) * (rollback_depth + 1) +
          linear * rollback_state) *
             sizeof(float) +
         linear * state * GdnStateElementBytes(options_.gdn_state) +
         attention * ring + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? ring + std::size_t{options_.max_speculative + 1} * c.HcDim() *
                           sizeof(float)
              : 0) +
         GrowableBytes(mode, max_context);
}

std::size_t Executor::GrowableBytes(core::SessionMode mode,
                                    std::uint32_t max_context) const noexcept {
  const auto& c = config();
  const std::size_t layers = c.AttentionLayerCount() +
                             (mode == core::SessionMode::kSpeculative ? 1 : 0);
  const std::size_t kv =
      std::size_t{max_context} *
      KvTokenBytes(options_.kv_format, c.num_kv_heads, c.head_dim);
  const std::size_t blocks =
      c.HasIndexer() ? std::size_t{max_context / c.compress_ratio + 1} *
                           c.indexer_head_dim * sizeof(__half)
                     : 0;
  const auto* d = model_->dspark();
  const std::size_t dspark =
      mode == core::SessionMode::kSpeculative && d != nullptr
          ? d->blocks.size() * std::size_t{max_context} * 2 * d->kv_heads *
                d->head_dim * sizeof(__half)
          : 0;
  return layers * (kv + blocks) + dspark;
}

bool Executor::RequireResident(const Session& session,
                               std::string* error_msg) const {
  if (session.resident_)
    return true;
  AssignError(error_msg,
              "session is not resident: hold an Executor::Pin while it runs");
  return false;
}

bool Executor::Reserve(Session& session, std::uint32_t tokens,
                       std::string* error_msg) const {
  if (tokens <= session.reserved_tokens_)
    return true;
  const auto target = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      session.max_context_, std::uint64_t{tokens} + kReserveHeadroom));
  std::string error;
  bool ok = true;
  {
    // One TLB flush for every cache buffer, not one each.
    const gufo::hip::TlbBatch batch;
    for (auto& g : session.growable_) {
      ok = g.buffer.Commit(
          (std::size_t{target / g.tokens_per_row} + 1) * g.row_bytes, stream_,
          &error);
      if (!ok)
        break;
    }
  }
  // Pieces a failed commit mapped stay committed, so flush either way.
  if (!kv_pool_.Settle(&error) || !ok) {
    AssignError(error_msg, "KV cache commit for " + std::to_string(target) +
                               " tokens failed: " + error);
    return false;
  }
  session.reserved_tokens_ = target;
  return true;
}

std::size_t Executor::DeferredScratchBytes() const {
  // Batch logits reuse idle MoE scratch; only wider verification batches
  // grow a device buffer (pinned host staging holds the AR rows).
  const std::size_t vocab = config().vocab_size;
  const std::size_t rows =
      std::size_t{std::min(kMaxBatchSessions, options_.max_batch_sessions)} *
      std::min(8U, options_.max_logit_rows);
  const std::size_t device =
      rows * vocab > logit_alias_floats_ && batch_logits_owned_rows_ < rows
          ? rows * vocab * sizeof(float)
          : 0;
  const std::size_t host =
      batch_logits_host_rows_ == 0 && options_.max_logit_rows == 1
          ? std::min<std::size_t>(rows, kMaxBatchSessions) * vocab *
                sizeof(float)
          : 0;
  return device + host;
}

/// Column-tile width for the routed expert GEMMs: the tile at or above twice
/// the mean bucket, so most experts fit one tile with little padding.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

bool Executor::Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                        Q8Input* q, std::string* error_msg) const {
  q->x = x;
  q->data = nullptr;
  q->n = n_tokens;
  q->k = k;
  const PendingQ8 pending = pending_q8_;
  pending_q8_ = {};
  if (MatrixRows(n_tokens)) {
    return true;  // the tiled path quantizes per call
  }
  if (pending.src == x && pending.rows == n_tokens && pending.cols == k) {
    // The producer of x already wrote these Q8_1 bytes (same quantizer).
    q->data = pending.q8;
    return true;
  }
  // Two slots alternate, so an input stays valid across one other
  // quantization; captured graphs replay the same alternation.
  void* slot = s_.x_q8[q8_slot_];
  q8_slot_ ^= 1u;
  if (qfn_mmq_quantize_q8_1(x, slot, static_cast<int>(n_tokens),
                            static_cast<int>(k), stream_) != 0) {
    AssignError(error_msg, "activation quantization failed");
    return false;
  }
  q->data = slot;
  return true;
}

void* Executor::PreQ8(unsigned slot, std::uint32_t rows) const noexcept {
  return rows > 0 && rows <= kVecBatch && !MatrixRows(rows) ? s_.pre_q8[slot]
                                                            : nullptr;
}

bool Executor::Dense(const DeviceTensor& w, const Q8Input& q, float* out,
                     std::string* error_msg) const {
  if ((w.type == GgmlType::kQ8_0 || KQuant(w.type)) && q.data != nullptr) {
    if (w.cols != q.k) {
      AssignError(error_msg, "quantized input width mismatch");
      return false;
    }
    if (VecPreq(w, nullptr, q.data, out, q.n) != 0) {
      AssignError(error_msg, "quantized GEMV failed");
      return false;
    }
    return true;
  }
  return Dense(w, q.x, out, q.n, error_msg);
}

int Executor::VecPreq(const DeviceTensor& w, const void* gate, const void* x_q8,
                      float* out, std::uint32_t rows) const {
  const int m = static_cast<int>(w.rows);
  const int n = static_cast<int>(rows);
  const int k = static_cast<int>(w.cols);
  return w.type == GgmlType::kQ8_0
             ? qfn_mmq_q8_0_dense_vec_preq(w.data, gate, x_q8, out, m, n, k,
                                           stream_)
             : qfn_mmq_k_dense_vec_preq(static_cast<int>(w.type), w.data, gate,
                                        x_q8, out, m, n, k, stream_);
}

bool Executor::GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                          const float* x, float* out, std::uint32_t n_tokens,
                          const DeviceTensor* down,
                          std::string* error_msg) const {
  if (up.alt != nullptr && gate.alt != nullptr && !MatrixRows(n_tokens) &&
      n_tokens <= options_.decode_twin_rows)
    return GatedDense(*up.alt, *gate.alt, x, out, n_tokens, down, error_msg);
  // Decode and verification use the same fused projections and activation.
  // A different SwiGLU rounding can change later activation quantization.
  if (!MatrixRows(n_tokens) &&
      (up.type == GgmlType::kQ8_0 || KQuant(up.type)) && gate.type == up.type &&
      up.rows == gate.rows && up.cols == gate.cols) {
    // Spike, MIN_ROWS_MODE=act: the GEMV reads the W4 shadows of
    // gate and up (same role, same rotation) and rows rotated to match.
    DeviceTensor up_w = up;
    DeviceTensor gate_w = gate;
    const void* uw = nullptr;
    const float* xs = nullptr;
    if (!spike::GemvOperands(up.data, x, n_tokens, up.cols, &uw, &xs,
                             stream_)) {
      AssignError(error_msg, "spike GEMV rotation failed");
      return false;
    }
    up_w.data = const_cast<void*>(uw);
    gate_w.data = const_cast<void*>(spike::GemvWeight(gate.data));
    if ((up_w.data != up.data) != (gate_w.data != gate.data)) {
      AssignError(error_msg, "spike gate/up shadows disagree");
      return false;
    }
    Q8Input xq;
    if (!Quantize(xs, n_tokens, up.cols, &xq, error_msg)) {
      return false;
    }
    if (VecPreq(up_w, gate_w.data, xq.data, out, n_tokens) != 0) {
      AssignError(error_msg, "gated quantized GEMV failed");
      return false;
    }
    return true;
  }
  // Swiglu is in place over its first operand. Both int4 projections read
  // one quantization of x.
  const void* i4_gate = I4Route(gate, n_tokens);
  const void* i4_up = I4Route(up, n_tokens);
  if (i4_gate != nullptr && i4_up != nullptr && gate.cols == up.cols) {
    if (!I4Activations(x, n_tokens, gate.cols, error_msg) ||
        !I4Gemm(i4_gate, gate, out, n_tokens, error_msg) ||
        !I4Gemm(i4_up, up, s_.shexp_gate, n_tokens, error_msg))
      return false;
  } else if (!Dense(gate, x, out, n_tokens, error_msg) ||
             !Dense(up, x, s_.shexp_gate, n_tokens, error_msg)) {
    return false;
  }
  // A wide batch writes the down projection's staged input directly (the
  // F32 rows are read by nothing else): F16 rows for the F16 route, else
  // the tiled Q8 layout; the cache lets Dense skip its activation pass.
  if (down != nullptr && down->cols == up.rows && MatrixRows(n_tokens) &&
      n_tokens <= options_.max_batch) {
    if (DenseF16Route(*down, n_tokens)) {
      // A private F16 buffer: s_.x_half keeps the narrowed token rows the
      // router left there, which the routed experts read next.
      SwigluHalf(out, s_.shexp_gate, s_.shexp_half,
                 static_cast<std::size_t>(n_tokens) * up.rows, stream_);
      shexp_half_ready_ = true;
      return true;
    }
    if (down->type == GgmlType::kQ8_0 &&
        SwigluQ8Tiled(out, s_.shexp_gate, s_.x_q8t, n_tokens, up.rows,
                      stream_)) {
      q8t_src_ = out;
      q8t_rows_ = n_tokens;
      q8t_cols_ = up.rows;
      return true;
    }
  }
  Swiglu(out, s_.shexp_gate, static_cast<std::size_t>(n_tokens) * up.rows,
         stream_);
  return true;
}

void Executor::PrepareHalfInput(const float* x, std::uint32_t rows,
                                std::uint32_t cols) const {
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols &&
      !half_bf16_) {
    return;
  }
  NarrowActivations(x, s_.x_half, false, static_cast<std::size_t>(rows) * cols,
                    stream_);
  half_src_ = x;
  half_rows_ = rows;
  half_cols_ = cols;
  half_bf16_ = false;
}

bool Executor::DenseF16Route(const DeviceTensor& w,
                             std::uint32_t n_tokens) const {
  return w.type == GgmlType::kQ8_0 && MatrixRows(n_tokens) &&
         w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols &&
         w.cols <= model_->max_half_cols();
}

const void* Executor::I4Route(const DeviceTensor& w,
                              std::uint32_t n_tokens) const {
  if (!prefill_phase || i4_weights_.empty() || n_tokens < DenseI4MinRows() ||
      n_tokens > options_.max_batch)
    return nullptr;
  const auto it = i4_weights_.find(w.data);
  return it == i4_weights_.end() ? nullptr : it->second;
}

bool Executor::I4Activations(const float* x, std::uint32_t rows,
                             std::uint32_t cols, std::string* error_msg) const {
  // A producer (or an earlier projection) may have left these rows in F16.
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols && !half_bf16_)
    return I4Activations(static_cast<const __half*>(s_.x_half), rows, cols,
                         error_msg);
  if (!dense_i4::Quantize(x, rows, cols, kDenseI4Rotation, s_.x_i4, stream_,
                          kDenseI4SlabScales)) {
    AssignError(error_msg, "int4 activation quantization failed");
    return false;
  }
  return true;
}

bool Executor::I4Activations(const __half* x, std::uint32_t rows,
                             std::uint32_t cols, std::string* error_msg) const {
  if (!dense_i4::Quantize(x, rows, cols, kDenseI4Rotation, s_.x_i4, stream_,
                          kDenseI4SlabScales)) {
    AssignError(error_msg, "int4 activation quantization failed");
    return false;
  }
  return true;
}

bool Executor::I4Gemm(const void* packed, const DeviceTensor& w, float* out,
                      std::uint32_t rows, std::string* error_msg) const {
  if (!dense_i4::Gemm(
          dense_i4::WeightView(packed, w.rows, w.cols, kDenseI4SlabScales),
          dense_i4::ActivationView(s_.x_i4, rows, w.cols, kDenseI4SlabScales),
          out, stream_)) {
    AssignError(error_msg, "int4 dense GEMM failed");
    return false;
  }
  return true;
}

bool Executor::PackI4Weights(std::string* error_msg) {
  // Offline-calibrated weights (dense-w4-calib S4G128 output, converted by
  // tools/qwen-flash/s4_index.py) replace the load-time
  // RTN copy of every projection they hold.
  S4Weights offline;
  if (!options_.dense_w4a4_weights.empty() &&
      !offline.Open(options_.dense_w4a4_weights.c_str(), error_msg))
    return false;
  void* staging = nullptr;
  std::size_t staging_bytes = 0;
  struct StagingFree {
    void*& p;
    ~StagingFree() {
      if (p != nullptr)
        (void)hipFree(p);
    }
  } staging_free{staging};
  std::vector<char> host;
  std::size_t layer_index = 0;
  for (const DeviceLayer& l : model_->layers()) {
    const std::string prefix = "blk." + std::to_string(layer_index++) + ".";
    const struct {
      const DeviceTensor* tensor;
      std::vector<const char*> parts;  // stacked source tensors, in row order
    } sources[] = {
        {&l.ssm_in, {"attn_qkv.weight", "attn_gate.weight"}},
        {&l.attn_qkv, {"attn_q.weight", "attn_k.weight", "attn_v.weight"}},
        {&l.ssm_out, {"ssm_out.weight"}},
        {&l.attn_out, {"attn_output.weight"}},
        {&l.shexp_gate, {"ffn_gate_shexp.weight"}},
        {&l.shexp_up, {"ffn_up_shexp.weight"}},
        {&l.shexp_down, {"ffn_down_shexp.weight"}}};
    for (const auto& [t, parts] : sources) {
      if (t->empty() || t->type != GgmlType::kQ8_0 ||
          t->cols > kDenseI4MaxCols ||
          !dense_i4::Supported(t->cols, kDenseI4Rotation) ||
          i4_weights_.contains(t->data))
        continue;
      void* packed = Alloc<std::uint8_t>(
          allocations_,
          dense_i4::WeightBytes(t->rows, t->cols, kDenseI4SlabScales),
          error_msg);
      if (packed == nullptr)
        return false;
      if (offline.Loaded()) {
        if (!kDenseI4SlabScales || t->cols % 128 != 0) {
          AssignError(error_msg, "offline int4 weights need 128-wide slabs");
          return false;
        }
        if (hipMemsetAsync(packed, 0,
                           dense_i4::WeightBytes(t->rows, t->cols, true),
                           stream_) != hipSuccess) {
          AssignError(error_msg, "int4 weight clear failed");
          return false;
        }
        std::uint32_t row0 = 0;
        for (const char* part : parts) {
          const S4Weights::Entry* e = offline.Find(prefix + part);
          if (e == nullptr || e->cols != t->cols || row0 + e->rows > t->rows ||
              e->bytes != static_cast<std::uint64_t>(e->rows) *
                              (e->cols + (e->cols / 128) + 4)) {
            AssignError(error_msg, "offline int4 weights: bad or missing " +
                                       prefix + part);
            return false;
          }
          if (!offline.Read(*e, &host, error_msg))
            return false;
          if (staging_bytes < host.size()) {
            if (staging != nullptr)
              (void)hipFree(staging);
            staging = nullptr;
            if (!Check(hipMalloc(&staging, host.size()), "int4 staging",
                       error_msg))
              return false;
            staging_bytes = host.size();
          }
          if (!Check(hipMemcpy(staging, host.data(), host.size(),
                               hipMemcpyHostToDevice),
                     "int4 staging copy", error_msg))
            return false;
          const auto* base = static_cast<const std::uint8_t*>(staging);
          const std::size_t codes = std::size_t{e->rows} * e->cols;
          const std::size_t slabs = std::size_t{e->rows} * (e->cols / 128);
          if (!dense_i4::PackS4(
                  reinterpret_cast<const std::int8_t*>(base), base + codes,
                  reinterpret_cast<const float*>(base + codes + slabs), e->rows,
                  e->cols, row0, t->rows, packed, stream_) ||
              !Check(hipStreamSynchronize(stream_), "int4 repack", error_msg)) {
            AssignError(error_msg, "offline int4 weight repack failed");
            return false;
          }
          row0 += e->rows;
        }
        if (row0 != t->rows) {
          AssignError(error_msg, "offline int4 weights: " + prefix +
                                     parts.front() + " rows do not add up");
          return false;
        }
      } else if (!dense_i4::PackQ8_0(t->data, t->rows, t->cols,
                                     kDenseI4Rotation, packed, stream_, 1,
                                     kDenseI4SlabScales)) {
        AssignError(error_msg, "int4 weight packing failed");
        return false;
      }
      i4_weights_.emplace(t->data, packed);
    }
  }
  // The HC up projections stay on the F16 mixer route: K = 320 leaves the
  // int4 GEMM memory- and epilogue-bound (dense_i4_bench hc_up, 2048 rows:
  // F16 504 us, int4 477 us), for ~2.6 ms per prompt against W4A4 noise in
  // every layer's mixed rows.
  return Check(hipStreamSynchronize(stream_), "int4 weight packing", error_msg);
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  // `--gguf-decode-override`: narrow decode batches read the calibrated twin.
  if (w.alt != nullptr && !MatrixRows(n_tokens) &&
      n_tokens <= options_.decode_twin_rows)
    return Dense(*w.alt, x, out, n_tokens, error_msg);
  if (w.type == GgmlType::kQ8_0) {
    if (!MatrixRows(n_tokens)) {
      // Spike, MIN_ROWS_MODE=act: decode reads the W4 shadow.
      DeviceTensor ws = w;
      const void* wd = nullptr;
      const float* xs = nullptr;
      if (!spike::GemvOperands(w.data, x, n_tokens, w.cols, &wd, &xs,
                               stream_)) {
        AssignError(error_msg, "spike GEMV rotation failed");
        return false;
      }
      ws.data = const_cast<void*>(wd);
      Q8Input q;
      return Quantize(xs, n_tokens, w.cols, &q, error_msg) &&
             Dense(ws, q, out, error_msg);
    }
    if (const void* packed = I4Route(w, n_tokens)) {
      return I4Activations(x, n_tokens, w.cols, error_msg) &&
             I4Gemm(packed, w, out, n_tokens, error_msg);
    }
    // Wide batches: the F16 route for the shapes it wins (F16 rows in
    // s_.x_half, often left there by a producer), else activations
    // quantized per 32-wide block into the tiled layout and the int8 WMMA
    // GEMM. Both staging buffers hold max_batch rows; a wider call (the
    // draft block folds its streams into rows) runs in pieces.
    const bool f16 = DenseF16Route(w, n_tokens);
    const std::uint32_t piece = static_cast<std::uint32_t>(options_.max_batch);
    for (std::uint32_t r0 = 0; r0 < n_tokens; r0 += piece) {
      const std::uint32_t rows = std::min(piece, n_tokens - r0);
      // The staging buffer may already hold this input (a producer wrote
      // it, or the previous projection read the same rows).
      const float* src = x + static_cast<std::size_t>(r0) * w.cols;
      if (f16) {
        PrepareHalfInput(src, rows, w.cols);
        if (!spike::DenseF16Gemm(prefill_phase, w.data,
                                 static_cast<const __half*>(s_.x_half),
                                 out + static_cast<std::size_t>(r0) * w.rows,
                                 rows, w.rows, w.cols, stream_)) {
          AssignError(error_msg, "dense F16 GEMM failed");
          return false;
        }
        continue;
      }
      // Spike: registered prefill weights whose shape keeps the
      // int8 route (attn_out 2560x6144, shared gate/up 640x2560) would never
      // reach the emulation; they take the spike's F16 path instead.
      if (spike::Covers(prefill_phase, w.data, rows, stream_)) {
        if (!spike::DenseF32Gemm(prefill_phase, w.data, src,
                                 out + static_cast<std::size_t>(r0) * w.rows,
                                 rows, w.rows, w.cols, stream_)) {
          AssignError(error_msg, "spike dense GEMM failed");
          return false;
        }
        continue;
      }
      if (!(q8t_src_ == src && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
        QuantizeQ8Tiled(src, s_.x_q8t, rows, w.cols, stream_);
        q8t_src_ = src;
        q8t_rows_ = rows;
        q8t_cols_ = w.cols;
      }
      if (!W8A8Gemm(w.data, s_.x_q8t,
                    out + static_cast<std::size_t>(r0) * w.rows, rows, w.rows,
                    w.cols, stream_)) {
        AssignError(error_msg, "W8A8 GEMM failed");
        return false;
      }
    }
    return true;
  }
  if (KQuant(w.type)) {
    if (!MatrixRows(n_tokens)) {
      Q8Input q;
      return Quantize(x, n_tokens, w.cols, &q, error_msg) &&
             Dense(w, q, out, error_msg);
    }
    // ponytail: wide K-quant batches take the vendored MMQ tiles; a
    // dequantize-to-F16 WMMA route like Q8_0's if prefill time matters.
    if (qfn_mmq_k_dense(static_cast<int>(w.type), w.data, x, out,
                        static_cast<int>(w.rows), static_cast<int>(n_tokens),
                        static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "K-quant GEMM failed");
      return false;
    }
    return true;
  }
  if (!MatrixRows(n_tokens)) {
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  // Wide batches of the unquantized projections (router, alpha/beta,
  // indexer): out[t][m] = sum_k w[m][k] x[t][k].
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     static_cast<const float*>(w.data), k, x, k, &beta, out,
                     m) != HIPBLAS_STATUS_SUCCESS) {
      AssignError(error_msg, "hipBLAS GEMM failed");
      return false;
    }
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  if (!(half_src_ == x && half_rows_ == n_tokens && half_cols_ == w.cols &&
        half_bf16_ == bf16)) {
    NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = w.cols;
    half_bf16_ = bf16;
  }
  return blaslt_->Gemm(w.data, s_.x_half, out, type, m, n, k, error_msg);
}

void Executor::RoutedHints(const DeviceTensor& w,
                           std::uint32_t n_tokens) const {
  // The column grid is bounded by the largest expert bucket and the tile
  // width fitted to the whole distribution (see RouteHints); the fallback
  // bound is the token count with a tile near twice the mean bucket.
  if (routed_max_rows_ > 0) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    qfn_mmq_set_routed_tile_cols(routed_tile_cols_);
    return;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  qfn_mmq_set_routed_tile_cols(
      RoutedTileCols(n_tokens, config().num_experts_used, w.experts));
}

bool Executor::RouteHints(std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the grid is cut to the real largest
  // bucket. Counts are downloaded before the shared expert; wait only for
  // that download while the shared expert continues on the same stream.
  const Config& c = config();
  routed_max_rows_ = 0;
  routed_64_tiles_ = 0;
  routed_pair_tiles_ = 0;
  routed_pair_offset_ = 0;
  routed_pair_rows_ = 64;
  if (!ExpertMatrixRows(n_tokens)) {
    return true;
  }
  if (!Check(hipEventSynchronize(counts_ready_), "expert counts", error_msg)) {
    return false;
  }
  // The F16 expert GEMM launches one block per (expert, row tile of its
  // 16-padded bucket): the map is built here and uploaded ahead of the
  // launches on the same stream.
  std::uint32_t max_rows = 0;
  std::uint32_t n_tiles = 0;
  routed_tile_rows_ = RoutedTileRows(
      static_cast<std::size_t>(n_tokens) * c.num_experts_used, c);
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
    max_rows = std::max(max_rows, counts_host_[e]);
    for (std::uint32_t j = 0;
         j < (padded + routed_tile_rows_ - 1) / routed_tile_rows_; ++j) {
      tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  routed_n_tiles_ = n_tiles;
  // Append the 64-token map for gate/up and eligible down projections.
  // Wide expert buckets also get a 128-token gate/up map: it amortizes
  // weight decoding, while short buckets retain the cheaper 64-token tile.
  // Every prompt batch gets the paired map: pairing gate/up streams the
  // same weights with half the launches and no F32 gate round trip, 1-3%
  // faster at 256-512 tokens with identical output.
  {
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 63u) / 64u; ++j)
        tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    routed_64_tiles_ = n_tiles - routed_n_tiles_;
    routed_pair_offset_ = routed_n_tiles_;
    routed_pair_tiles_ = routed_64_tiles_;
    std::uint32_t tiles_128 = 0;
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      tiles_128 += (counts_host_[e] + 127u) / 128u;
    }
    if (tiles_128 * 4 <= routed_64_tiles_ * 3) {
      routed_pair_rows_ = 128;
      routed_pair_offset_ = n_tiles;
      routed_pair_tiles_ = tiles_128;
      for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        for (std::uint32_t j = 0; j < (counts_host_[e] + 127u) / 128u; ++j) {
          tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
        }
      }
    }
  }
  routed_tile_cols_ = qfn_mmq_routed_tile_cols_for_counts(
      counts_host_, static_cast<int>(c.num_experts));
  return n_tiles == 0 || Check(hipMemcpyAsync(s_.routed_tiles, tiles_host_,
                                              n_tiles * sizeof(std::int32_t),
                                              hipMemcpyHostToDevice, stream_),
                               "routed tile map upload", error_msg);
}

bool Executor::Experts(const DeviceTensor& w, const float* x,
                       const std::int32_t* ids, float* out,
                       std::uint32_t n_rows, std::uint32_t n_used,
                       std::uint32_t n_tokens, std::string* error_msg) const {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(n_rows);
  const int U = static_cast<int>(n_used);
  // Decode rows take RoutedExperts (DeviceRouted); prompts tile.
  RoutedHints(w, n_tokens);
  int rc = -1;
  {
    switch (w.type) {
      case GgmlType::kQ4_K:
        rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_K:
        rc = qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_1:
        rc = qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ8_0:
        rc = qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      default:
        break;
    }
  }
  if (rc != 0) {
    AssignError(error_msg, "expert GEMM failed");
    return false;
  }
  return true;
}

bool Executor::GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                            const float* x, const std::int32_t* ids, float* out,
                            std::uint32_t n_tokens, std::uint32_t n_used,
                            std::string* error_msg) const {
  if (!Experts(a, x, ids, out, n_tokens, n_used, n_tokens, error_msg) ||
      !Experts(b, x, ids, s_.up_e, n_tokens, n_used, n_tokens, error_msg))
    return false;
  Swiglu(out, s_.up_e, static_cast<std::size_t>(n_tokens) * n_used * a.rows,
         stream_);
  return true;
}

bool Executor::FusedHcProjection(const DeviceMixer& m,
                                 std::uint32_t n_tokens) const {
  const Config& c = config();
  return n_tokens >= 96 && n_tokens <= options_.max_batch &&
         c.hidden_size <= model_->max_half_cols() && c.hc_count == 4 &&
         c.hidden_size == 2560 && c.hc_low_rank == 320 &&
         m.up.type == GgmlType::kQ8_0 && m.up.rows == c.HcDim() &&
         m.up.cols == c.hc_low_rank;
}

void Executor::Combine(float* res, const float* gamma, std::uint32_t n_tokens,
                       std::int32_t moe_layer, const DeviceMixer* next,
                       bool keep_f32) const {
  const Config& c = config();
  inject_ready_ = false;
  xn_scale_live_ = false;
  // Wide batches hand the next mixer an F16 norm for its epilogue and the
  // same norm quantized into the tiled Q8 layout for its W8A8 down
  // projection: half the bytes for the combine and the epilogue, and no
  // separate activation pass for the projection.
  xn_half_ = wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr &&
             c.HyperConnections();
  // Only the F16 residual without a norm (`--hc-streams f16nx`) fuses the
  // next mixer's F32 inject dots into the combine (it has no F16 norm for
  // the mixer to read them from); the default keeps the mixer's own pass.
  const float* next_inject =
      hc_streams_ == 2 && xn_half_ && next != nullptr &&
              !next->inject.empty() && next->inject.type == GgmlType::kF32 &&
              c.hc_count == 4 &&
              inject_parts_ == HcInjectPartsVec4(c.hidden_size)
          ? next->inject.f32()
          : nullptr;
  // --prefill-residual: keep the residual in F16 between combines. Everything
  // else (PLE, the last layer, narrow batches, steering) needs the F32 rows,
  // so a live F16 residual is widened first.
  HcResArgs ra;
  const bool want16 = hc_streams_ != 0 && !keep_f32 && xn_half_ && !steer_ &&
                      n_tokens <= options_.max_batch && c.hc_count == 4 &&
                      c.hidden_size % 128 == 0 && c.hidden_size <= 2560 &&
                      s_.res_half != nullptr;
  if (!want16 && res_half_live_) {
    HcResHalfToF32(s_.res_half, res,
                   static_cast<std::size_t>(n_tokens) * c.HcDim(), stream_);
    res_half_live_ = false;
  }
  if (want16) {
    ra.mode = res_half_live_ ? 2 : 1;
    ra.res_half = s_.res_half;
    if (hc_streams_ == 2 && next != nullptr && next_inject != nullptr &&
        FusedHcProjection(*next, n_tokens)) {
      ra.xn_scale = s_.xn_scale;
    }
  }
  if (moe_pending_) {
    // The MoE epilogue was deferred to this combine (see Moe).
    moe_pending_ = false;
    const auto* down = reinterpret_cast<const __half*>(s_.down_e);
    const SteerRows steer = SteerArgs(moe_layer);
    // The fused kernel ablates every adapter row; writer-scoped rows (the
    // MTP block) take the epilogue and Steer below.
    if (xn_half_ && steer.writers == nullptr &&
        HcCombineMoeF16(
            res, down, s_.weights, s_.shexp_out, s_.router + c.num_experts,
            c.num_experts + 1, c.num_experts_used, s_.inject, inject_parts_,
            gamma, s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count,
            c.rms_eps, stream_, steer_ ? &steer : nullptr, next_inject,
            next_inject != nullptr ? s_.inject : nullptr, ra)) {
      inject_ready_ = next_inject != nullptr;
      res_half_live_ = want16;
      xn_scale_live_ = ra.xn_scale != nullptr;
      return;
    }
    MoeEpilogueVec4F16(down, s_.weights, s_.shexp_out,
                       s_.router + c.num_experts, c.num_experts + 1,
                       s_.block_out, n_tokens, c.num_experts_used,
                       c.hidden_size, stream_);
  }
  Steer(n_tokens, moe_layer);
  if (!c.HyperConnections()) {
    MtpAddEmbedding(s_.block_out, res, n_tokens, c.hidden_size, 1, stream_);
    return;
  }
  if (xn_half_) {
    HcCombineF16(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn_half,
                 s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count, c.rms_eps,
                 stream_, next_inject,
                 next_inject != nullptr ? s_.inject : nullptr, ra);
    inject_ready_ = next_inject != nullptr;
    res_half_live_ = want16;
    xn_scale_live_ = ra.xn_scale != nullptr;
    return;
  }
  // A decode norm also leaves the next mixer's down projection its Q8_1
  // input, so that projection skips its quantization launch.
  void* xn_q81 =
      gamma != nullptr && c.HcDim() % 512 == 0 ? PreQ8(0, n_tokens) : nullptr;
  HcCombine(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn, n_tokens,
            c.hidden_size, c.hc_count, c.rms_eps, stream_, false, xn_q81);
  pending_q8_ = xn_q81 != nullptr
                    ? PendingQ8{s_.xn, xn_q81, n_tokens, c.HcDim()}
                    : PendingQ8{};
}

SteerRows Executor::SteerArgs(std::int32_t moe_layer) const noexcept {
  SteerRows steer{s_.steer, steer_directions_, steer_alphas_};
  if (moe_layer >= 0) {
    steer.routed_layers = steer_routed_layers_;
    steer.layer = static_cast<std::uint32_t>(moe_layer);
    steer.shared = s_.shexp_out;
    steer.gate = s_.router + config().num_experts;
    steer.gate_stride = config().num_experts + 1;
  }
  if (steer_writer_ != 0) {
    steer.writers = steer_writers_;
    steer.writer = steer_writer_;
  }
  return steer;
}

void Executor::Steer(std::uint32_t rows, std::int32_t moe_layer) const {
  if (steer_)
    rocm::Steer(s_.block_out, SteerArgs(moe_layer), rows, config().hidden_size,
                stream_);
}

void Executor::SteerWriter(float* y, std::uint32_t rows, std::uint32_t streams,
                           std::uint8_t writer) const {
  if (!steer_ || (steer_writers_any_ & writer) == 0)
    return;
  SteerRows steer{s_.steer, steer_directions_, steer_alphas_};
  steer.writers = steer_writers_;
  steer.writer = writer;
  rocm::Steer(y, steer, rows, config().hidden_size, stream_, streams);
}

bool Executor::SteersMtp(std::uint8_t adapter) const noexcept {
  return adapter != 0 && (options_.adapters[adapter - 1].writers &
                          (steer_writer::kEmbedding | steer_writer::kMtp)) != 0;
}

bool Executor::PlainNorm(const DeviceMixer& m, const float* res, float* mixed,
                         std::uint32_t rows) const {
  // One residual stream (qwen35moe): the block input is the RMS-normed
  // residual. Nothing downstream may reuse an earlier F16/Q8 activation.
  xn_half_ = false;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  inject_parts_ = 1;
  RmsNormRows(res, m.norm.f32(), mixed, rows, config().hidden_size, 1,
              config().rms_eps, stream_);
  return true;
}

bool Executor::HcMix(const DeviceMixer& m, const float* res, bool normed,
                     float* mixed, float* inject, std::uint32_t n_tokens,
                     std::string* error_msg) const {
  const Config& c = config();
  if (!c.HyperConnections())
    return PlainNorm(m, res, mixed, n_tokens);
  // Without `normed` this mixer's grouped norm of res is computed here (F32);
  // otherwise the previous combine produced it, as F16 plus tiled Q8 on the
  // wide route or as F32 in s_.xn.
  if (!normed) {
    xn_half_ = false;
    pending_q8_ = {};  // a combine's Q8_1 norm belongs to another gamma
    RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  }
  void* mixed_q81 = nullptr;
  const bool extras = n_tokens <= options_.max_batch &&
                      c.hidden_size <= model_->max_half_cols();
  const bool fused_projection = xn_half_ && FusedHcProjection(m, n_tokens);
  // The combine wrote the F16 residual and per-stream scales instead of xn.
  const bool no_xn = xn_scale_live_ && normed;
  xn_scale_live_ = false;
  if (no_xn && !fused_projection) {
    AssignError(error_msg, "F16 residual without the fused HC projection");
    return false;
  }
  if (fused_projection) {
    // The up projection writes x_half; keep its input in the gate buffer.
    if (!HcDownF16Gemm(m.down.data, s_.xn_q8t,
                       reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                       stream_)) {
      AssignError(error_msg, "fused HC down projection failed");
      return false;
    }
  } else {
    if (xn_half_) {
      if (!W8A8Gemm(m.down.data, s_.xn_q8t, s_.lo, n_tokens, m.down.rows,
                    m.down.cols, stream_)) {
        AssignError(error_msg, "W8A8 mixer down projection failed");
        return false;
      }
    } else if (!Dense(m.down, s_.xn, s_.lo, n_tokens, error_msg)) {
      return false;
    }
    SiluScale(s_.lo, 1.0F / static_cast<float>(c.hc_count),
              static_cast<std::size_t>(n_tokens) * c.hc_low_rank, stream_);
    if (!Dense(m.up, s_.lo, s_.hc_gate, n_tokens, error_msg)) {
      return false;
    }
  }
  const bool fused_inject =
      inject != nullptr && !m.inject.empty() && m.inject.type == GgmlType::kF32;
  // The combine already wrote these partials (same values and layout).
  const bool inject_done =
      inject_ready_ && xn_half_ && fused_inject && inject == s_.inject;
  inject_ready_ = false;
  const float* mix_inject_w =
      fused_inject && !inject_done ? m.inject.f32() : nullptr;
  const float* xn = s_.xn;
  const bool vectorized =
      xn_half_ ||
      (MatrixRows(n_tokens) && c.hc_count == 4 && c.hidden_size % 4 == 0);
  // `mixed` is being rewritten: whatever the input caches held of it is
  // stale. The wide F16 route also emits the F16 and tiled Q8 copies the
  // projections that follow read.
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (xn_half_) {
    if (fused_projection) {
      if (!HcMixF16Gemm(m.up.data, reinterpret_cast<const __half*>(s_.hc_gate),
                        no_xn ? s_.res_half : s_.xn_half, mix_inject_w, mixed,
                        static_cast<__half*>(s_.x_half), s_.x_q8t, inject,
                        n_tokens, c.hidden_size, c.hc_low_rank, stream_,
                        no_xn ? s_.xn_scale : nullptr,
                        no_xn ? m.norm.f32() : nullptr)) {
        AssignError(error_msg, "fused HC projection failed");
        return false;
      }
    } else {
      HcMixEpilogueVec4F16(s_.xn_half, s_.hc_gate, mix_inject_w, mixed,
                           extras ? static_cast<__half*>(s_.x_half) : nullptr,
                           extras ? s_.x_q8t : nullptr, inject, n_tokens,
                           c.hidden_size, stream_);
    }
    if (extras) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
      q8t_src_ = mixed;
      q8t_rows_ = n_tokens;
      q8t_cols_ = c.hidden_size;
    }
  } else if (vectorized) {
    HcMixEpilogueVec4(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                      mixed, inject, n_tokens, c.hidden_size, c.hc_count,
                      stream_);
  } else {
    mixed_q81 = c.hidden_size % 32 == 0 ? PreQ8(1, n_tokens) : nullptr;
    HcMixEpilogue(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                  mixed, inject, n_tokens, c.hidden_size, c.hc_count, stream_,
                  mixed_q81);
  }
  inject_parts_ = fused_inject ? (vectorized ? HcInjectPartsVec4(c.hidden_size)
                                             : HcInjectParts(c.hidden_size))
                               : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    // A quantized inject projection (the draft block's) reads an F32 norm;
    // on the wide route the combine only produced F16, so norm again.
    if (xn_half_) {
      RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                  c.rms_eps, stream_);
    }
    if (!Dense(m.inject, xn, inject, n_tokens, error_msg)) {
      return false;
    }
  }
  // The block's first projection reads the mixed rows next.
  pending_q8_ = mixed_q81 != nullptr
                    ? PendingQ8{mixed, mixed_q81, n_tokens, c.hidden_size}
                    : PendingQ8{};
  return true;
}

bool Executor::PleFetch(Session& session, std::span<const std::int32_t> tokens,
                        bool speculative, std::string* error_msg) const {
  if (ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  // Only proper prefixes need snapshots; the full batch keeps its live state.
  if (speculative) {
    for (std::uint32_t i = 0; i < n; ++i) {
      HashNgramRows(c, session.ngram_, tokens.subspan(i, 1),
                    std::span<std::uint32_t>(
                        host_rows_.data() + i * c.ple_heads, c.ple_heads));
      if (i + 1 < n) {
        session.ngram_snapshots_[i] = session.ngram_;
      }
    }
  } else {
    HashNgramRows(c, session.ngram_, tokens,
                  std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
  }
  ple_pending_ = ngram_->StartRead(
      std::span<const std::uint32_t>(host_rows_.data(), n * c.ple_heads),
      std::span<float>(host_emb_,
                       static_cast<std::size_t>(n) * c.PleEmbeddingDim()));
  if (!ple_pending_) {
    AssignError(error_msg, "n-gram table read could not start");
  }
  return ple_pending_;
}

bool Executor::WaitPle(std::string* error_msg) const {
  const bool ok = ple_pending_ && ngram_->WaitRead();
  ple_pending_ = false;
  if (!ok) {
    AssignError(error_msg, "n-gram table read failed");
  }
  return ok;
}

bool Executor::Ple(const DeviceLayer& l, Session& session, std::uint32_t n,
                   float* res, bool speculative, std::string* error_msg,
                   bool embeddings_ready) const {
  const Config& c = config();
  const std::size_t emb_count =
      static_cast<std::size_t>(n) * c.PleEmbeddingDim();
  if (!embeddings_ready &&
      (!WaitPle(error_msg) ||
       !Check(hipMemcpyAsync(s_.ple_emb, host_emb_, emb_count * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              "n-gram upload", error_msg))) {
    return false;
  }
  const std::uint32_t hc_dim = c.HcDim();
  Q8Input emb;
  if (!Quantize(s_.ple_emb, n, c.PleEmbeddingDim(), &emb, error_msg) ||
      !Dense(l.ple_key, emb, s_.ple_key, error_msg) ||
      !Dense(l.ple_value, emb, s_.ple_value, error_msg)) {
    return false;
  }
  SteerWriter(s_.ple_value, n, 1, steer_writer::kPle);
  RmsNormRows(s_.ple_key, l.ple_norm_key.f32(), s_.ple_key, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(res, l.ple_norm_query.f32(), s_.ple_query, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  PleGate(s_.ple_key, s_.ple_query, s_.ple_value, s_.ple_gated, n,
          c.hidden_size, c.hc_count, stream_);
  RmsNormRows(s_.ple_gated, l.ple_norm_conv.f32(), s_.ple_norm, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  PleConv(s_.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
          s_.ple_history_scratch, s_.ple_conv,
          speculative ? session.ple_snapshots_ : RollbackRows{}, n, hc_dim,
          c.ple_conv_kernel, c.ple_ngram_size, stream_);
  PleInject(res, s_.ple_gated, s_.ple_conv,
            static_cast<std::size_t>(n) * hc_dim, stream_);
  return true;
}

bool Executor::HalfSsmOutput(const DeviceLayer& l,
                             std::uint32_t n_tokens) const noexcept {
  return MatrixRows(n_tokens) && l.ssm_out.type == GgmlType::kQ8_0 &&
         n_tokens <= options_.max_batch && l.ssm_out.rows == 2560 &&
         l.ssm_out.cols == 6144;
}

bool Executor::LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                               const float* x, float* out,
                               std::uint32_t n_tokens, bool speculative,
                               std::string* error_msg, bool projections_ready,
                               bool project_output, __half* half_rows) const {
  const Config& c = config();
  const std::uint32_t channels = c.SsmConvChannels();
  const float* qkv = s_.qkv;
  const float* z = s_.z;
  std::uint32_t qkv_stride = channels;
  std::uint32_t z_stride = c.SsmValueDim();
  bool convolved = false;
  if (!l.ssm_in.empty()) {
    // Wide prefill convolves QKV in the projection's LDS tile. The raw
    // boundary rows remain available for the rolling history update.
    const void* i4 = projections_ready || speculative
                         ? nullptr
                         : I4Route(l.ssm_in, n_tokens);
    if (i4 != nullptr && n_tokens >= 1024) {
      if (!I4Activations(x, n_tokens, l.ssm_in.cols, error_msg))
        return false;
      convolved = dense_i4::SsmGemm(
          dense_i4::WeightView(i4, l.ssm_in.rows, l.ssm_in.cols,
                               kDenseI4SlabScales),
          dense_i4::ActivationView(s_.x_i4, n_tokens, l.ssm_in.cols,
                                   kDenseI4SlabScales),
          l.ssm_conv1d.f32(), s.conv_state, s_.qkvz, s_.conv_scratch, channels,
          c.ssm_conv_kernel, stream_);
    } else if (i4 == nullptr && !projections_ready && !speculative &&
               n_tokens >= 1024 && n_tokens <= options_.max_batch &&
               DenseF16Route(l.ssm_in, n_tokens)) {
      PrepareHalfInput(x, n_tokens, l.ssm_in.cols);
      convolved = spike::DenseF16SsmGemm(
          prefill_phase, l.ssm_in.data, static_cast<const __half*>(s_.x_half),
          l.ssm_conv1d.f32(), s.conv_state, s_.qkvz, s_.conv_scratch, n_tokens,
          l.ssm_in.rows, l.ssm_in.cols, channels, c.ssm_conv_kernel, stream_);
    }
    if (!projections_ready && !convolved &&
        !Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg)) {
      return false;
    }
    qkv = s_.qkvz;
    z = s_.qkvz + channels;
    qkv_stride = z_stride = l.ssm_in.rows;
  } else if (!projections_ready) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.ssm_qkv, xq, s_.qkv, error_msg) ||
        !Dense(l.ssm_gate, xq, s_.z, error_msg)) {
      return false;
    }
  }
  if (!projections_ready &&
      !Dense(l.ssm_alpha_beta, x, s_.alpha_beta, n_tokens, error_msg)) {
    return false;
  }
  // Keep the same activation precision across prefill chunk boundaries.
  // The F16 epilogue reuses the F32 output allocation. Switching this
  // projection to Q8 for a short tail changes every subsequent layer.
  // A caller projecting the output itself reads F16 rows at `half_rows` on
  // the F16 route, else F32 rows.
  const bool tiled = MatrixRows(n_tokens) &&
                     l.ssm_out.type == GgmlType::kQ8_0 &&
                     n_tokens <= options_.max_batch;
  const bool half_output = HalfSsmOutput(l, n_tokens);
  auto* out_half = !half_output     ? nullptr
                   : project_output ? reinterpret_cast<__half*>(s_.gdn_out)
                                    : half_rows;
  // A decode row's output projection reads the epilogue's Q8_1 copy.
  void* out_q81 =
      !tiled && !half_output && project_output && c.SsmValueDim() % 512 == 0 &&
              (l.ssm_out.type == GgmlType::kQ8_0 || KQuant(l.ssm_out.type))
          ? PreQ8(2, n_tokens)
          : nullptr;
  GatedDeltaNet(qkv, qkv_stride, z, z_stride, s_.alpha_beta, l.ssm_conv1d.f32(),
                l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), s.conv_state,
                s_.conv_scratch, s_.qn, s_.kn, s_.gdn_raw, s.state, s_.gdn_out,
                tiled && !half_output && project_output ? s_.x_q8t : nullptr,
                speculative ? s.state_snapshots : RollbackRows{},
                speculative ? s.conv_snapshots : RollbackRows{}, n_tokens,
                c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
                c.ssm_conv_kernel, MatrixRows(n_tokens) && !speculative,
                convolved, c.rms_eps, stream_, out_half,
                c.Traits().gdn_gate == GdnGate::kSilu, options_.gdn_state,
                out_q81);
  pending_q8_ = out_q81 != nullptr
                    ? PendingQ8{s_.gdn_out, out_q81, n_tokens, c.SsmValueDim()}
                    : PendingQ8{};
  if (!project_output) {
    return true;
  }
  if (half_output) {
    if (const void* i4 = I4Route(l.ssm_out, n_tokens)) {
      return I4Activations(out_half, n_tokens, l.ssm_out.cols, error_msg) &&
             I4Gemm(i4, l.ssm_out, out, n_tokens, error_msg);
    }
    if (!spike::DenseF16Gemm(prefill_phase, l.ssm_out.data, out_half, out,
                             n_tokens, l.ssm_out.rows, l.ssm_out.cols,
                             stream_)) {
      AssignError(error_msg, "SSM output F16 GEMM failed");
      return false;
    }
    return true;
  }
  if (tiled) {
    q8t_src_ = nullptr;
    if (!W8A8Gemm(l.ssm_out.data, s_.x_q8t, out, n_tokens, l.ssm_out.rows,
                  l.ssm_out.cols, stream_)) {
      AssignError(error_msg, "W8A8 GEMM failed");
      return false;
    }
    return true;
  }
  return Dense(l.ssm_out, s_.gdn_out, out, n_tokens, error_msg);
}

bool Executor::LfmConv(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       std::string* error_msg) const {
  const Config& c = config();
  // in_proj -> [B | C | x] rows, gated causal convolution -> y, out_proj.
  if (!Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg))
    return false;
  LfmShortConv(s_.qkvz, l.ssm_in.rows, l.ssm_conv1d.f32(), s.conv_state, s_.ctx,
               n_tokens, c.hidden_size, stream_);
  return Dense(l.ssm_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::LfmAttentionLayer(const DeviceLayer& l,
                                 Session::AttentionState& s, const float* x,
                                 float* out, std::uint32_t n_tokens,
                                 const std::uint32_t* pos,
                                 std::string* error_msg) const {
  const Config& c = config();
  if (l.attn_qkv.empty()) {
    AssignError(error_msg, "lfm2 attention needs stacked Q8_0 q/k/v");
    return false;
  }
  if (!Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg))
    return false;
  LfmPrepareAttention(s_.qg, l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q,
                      s.kv, n_tokens, pos, c.rope_theta, c.rms_eps, stream_);
  LfmAttention(s_.q, s.kv, s_.ctx, s_.attn_partials, n_tokens, pos, stream_);
  return Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::LfmFfn(const DeviceLayer& l, const float* x, float* out,
                      std::uint32_t n_tokens, std::string* error_msg) const {
  const Config& c = config();
  if (l.ffn == FfnKind::kDense) {
    // Leading dense SwiGLU layer, on the shared-expert path and scratch.
    shexp_half_ready_ = false;
    if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                    &l.shexp_down, error_msg)) {
      return false;
    }
    if (shexp_half_ready_) {
      shexp_half_ready_ = false;
      if (!spike::DenseF16Gemm(prefill_phase, l.shexp_down.data, s_.shexp_half,
                               out, n_tokens, l.shexp_down.rows,
                               l.shexp_down.cols, stream_)) {
        AssignError(error_msg, "dense FFN F16 GEMM failed");
        return false;
      }
      return true;
    }
    return Dense(l.shexp_down, s_.shexp_up, out, n_tokens, error_msg);
  }
  // The routed epilogue adds sigmoid(gate) * shared expert: both stay zero
  // (s_.router's gate column and s_.shexp_out are cleared at creation and
  // never written without a shared expert), so routing goes through s_.qg.
  LfmRouter(x, l.router.f32(), l.expert_bias.f32(), s_.ids, s_.weights,
            n_tokens, c.num_experts_used, stream_);
  if (ExpertMatrixRows(n_tokens) && !DeviceRouted(l)) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts,
                 c.num_experts_used, stream_);
    if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg)) {
      return false;
    }
  }
  if (!MoeExperts(l, x, out, n_tokens, error_msg))
    return false;
  if (moe_pending_) {
    // One residual stream: no fused combine to defer the routed epilogue
    // to, and callers combine with CombineBatch or Combine alike.
    moe_pending_ = false;
    MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                       s_.shexp_out, s_.router + c.num_experts,
                       c.num_experts + 1, out, n_tokens, c.num_experts_used,
                       c.hidden_size, stream_);
  }
  return true;
}

bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         const std::uint32_t* first_block,
                         std::uint32_t start_pos, std::uint32_t pool_grid,
                         std::uint32_t max_context, bool sparse,
                         std::string* error_msg, bool last_only,
                         bool projections_ready, bool project_output) const {
  const Config& c = config();
  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::uint32_t index_capacity =
      IndexerCapacity(c, options_.max_batch, max_context);
  bool prepared = false;
  if (!l.attn_qkv.empty()) {
    const bool fused_shape = l.attn_qkv.rows == 13312 &&
                             l.attn_qkv.cols == 2560 && c.num_heads == 24 &&
                             c.num_kv_heads == 2 && c.head_dim == 256 &&
                             c.rotary_dim == 64;
    const void* i4 = projections_ready || n_tokens < 1024 || !fused_shape
                         ? nullptr
                         : I4Route(l.attn_qkv, n_tokens);
    const bool fused_projection =
        i4 == nullptr && !projections_ready && n_tokens >= 1024 &&
        n_tokens <= options_.max_batch && DenseF16Route(l.attn_qkv, n_tokens) &&
        !(prefill_phase && spike::Handles(l.attn_qkv.data)) && fused_shape;
    if (i4 != nullptr) {
      prepared =
          I4Activations(x, n_tokens, l.attn_qkv.cols, error_msg) &&
          dense_i4::AttentionGemm(
              dense_i4::WeightView(i4, l.attn_qkv.rows, l.attn_qkv.cols,
                                   kDenseI4SlabScales),
              dense_i4::ActivationView(s_.x_i4, n_tokens, l.attn_qkv.cols,
                                       kDenseI4SlabScales),
              l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q, s_.attn_gate,
              s.kv, pos, c.rope_theta, c.rms_eps, stream_, s.rope);
      if (!prepared) {
        AssignError(error_msg, "fused int4 attention projection failed");
        return false;
      }
    } else if (fused_projection) {
      PrepareHalfInput(x, n_tokens, l.attn_qkv.cols);
      prepared = spike::AttentionF16Gemm(
          prefill_phase, l.attn_qkv.data, static_cast<const __half*>(s_.x_half),
          l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q, s_.attn_gate, s.kv,
          n_tokens, pos, c.rope_theta, c.rms_eps, stream_, s.rope);
      if (!prepared) {
        AssignError(error_msg, "fused attention projection failed");
        return false;
      }
    } else {
      if (!projections_ready &&
          !Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
        return false;
      }
      prepared = PrepareAttention(s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(),
                                  l.attn_k_norm.f32(), s_.q, s_.attn_gate, s.kv,
                                  n_tokens, c.num_heads, c.num_kv_heads,
                                  c.head_dim, c.rotary_dim, pos, c.rope_theta,
                                  c.rms_eps, stream_, s.rope, prefill_phase);
      if (!prepared) {
        UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                    n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
      }
    }
  } else {
    Q8Input xq;
    if (!projections_ready &&
        (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
         !Dense(l.attn_q, xq, s_.qg, error_msg) ||
         !Dense(l.attn_k, xq, s_.k, error_msg) ||
         !Dense(l.attn_v, xq, s_.v, error_msg))) {
      return false;
    }
    UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
  }
  if (!prepared) {
    RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    if (s.kv.format == KvCacheFormat::kF16) {
      StoreKv(s_.k, static_cast<__half*>(s.kv.k), n_tokens, kv_row, pos,
              stream_);
      StoreKv(s_.v, static_cast<__half*>(s.kv.v), n_tokens, kv_row, pos,
              stream_);
    } else {
      QuantizeKv(s_.k, s_.v, s.kv, n_tokens, c.num_kv_heads, pos, stream_);
    }
  }

  // Raw keys survive only until pooling. The ring holds the initial
  // pre-budget backlog and a batch, independently for trunk and predictor.
  if (s.index_k != nullptr) {
    if (!Dense(l.indexer_k, x, s_.ik, n_tokens, error_msg)) {
      return false;
    }
    StoreRows(s_.ik, s.index_k, n_tokens, c.indexer_head_dim, pos,
              index_capacity, stream_);
  }
  const std::uint32_t* mask = nullptr;
  if (sparse) {
    if (!Dense(l.indexer_q, x, s_.iq, n_tokens, error_msg)) {
      return false;
    }
    RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq,
                n_tokens * c.indexer_heads, c.indexer_head_dim, 1, c.rms_eps,
                stream_);
    Rope(s_.iq, n_tokens, c.indexer_heads, c.indexer_head_dim, c.rotary_dim,
         pos, c.rope_theta, stream_, s.rope);
    PoolIndexerBlocks(s.index_k, l.indexer_k_norm.f32(), s.block_k, first_block,
                      pos, n_tokens, pool_grid, c.compress_ratio,
                      c.indexer_head_dim, c.rotary_dim, c.rope_theta, c.rms_eps,
                      index_capacity, stream_, s.rope);
    // Align score rows to full cache lines. The selector still considers
    // only complete causal blocks, so padding cannot change the ranking.
    const std::uint32_t blocks =
        (max_context + c.compress_ratio - 1) / c.compress_ratio;
    const std::uint32_t max_blocks = (blocks + 31) / 32 * 32;
    // Catch-up consumes only the final attention tile. Keep all its query
    // masks (sparse attention packs four queries; dense tiles hold sixteen)
    // but avoid scoring the unused prefix against the complete context.
    const auto first_query = last_only ? (n_tokens - 1) / 16 * 16 : 0U;
    // 256 query rows of scores (<= 32 MiB at 131K context) stay in the MALL
    // between scoring and marking; 512 rows spill at depth.
    const std::uint32_t chunk = std::min(select_chunk_, 256U);
    for (std::uint32_t t0 = first_query; t0 < n_tokens; t0 += chunk) {
      const std::uint32_t n = std::min(chunk, n_tokens - t0);
      SelectBlocks(s_.iq + static_cast<std::size_t>(t0) * c.indexer_heads *
                               c.indexer_head_dim,
                   s.block_k,
                   s_.mask + static_cast<std::size_t>(t0) * mask_words_,
                   s_.scores, n, pos, t0, c.indexer_heads, c.indexer_head_dim,
                   c.compress_ratio, c.indexer_top_k / c.compress_ratio,
                   mask_words_, max_blocks, stream_);
    }
    mask = s_.mask;
  }
  // Wide batches run the fused WMMA kernel (never inside a graph: the kv
  // extent is a host value), output gate included, skipping the key tiles
  // no query of a block selected; the per-token kernel covers the rest.
  if (last_only && !Check(hipMemsetAsync(s_.ctx, 0,
                                         static_cast<std::size_t>(n_tokens) *
                                             c.AttentionQDim() * sizeof(float),
                                         stream_),
                          "draft attention output initialization", error_msg)) {
    return false;
  }
  // TurboQuant caches attend in the rotated basis: rotate the queries, then
  // rotate the output back and apply the gate there.
  const bool rotated = s.kv.format != KvCacheFormat::kF16;
  const std::uint32_t head_rows = n_tokens * c.num_heads;
  if (rotated)
    RotateHeads(s_.q, head_rows, stream_);
  if (MatrixRows(n_tokens) &&
      WmmaCausalAttention(s_.q, rotated ? nullptr : s_.attn_gate, s.kv, mask,
                          mask_words_, s_.ctx, n_tokens, start_pos, c.num_heads,
                          c.num_kv_heads, c.head_dim, c.compress_ratio, stream_,
                          last_only)) {
    if (rotated)
      UnrotateHeads(s_.ctx, s_.attn_gate, head_rows, stream_);
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  // Narrow batches split each row's key tiles so a decode step at depth
  // fills the device; a multi-session batch uses the same splits per row.
  const bool split = !MatrixRows(n_tokens);
  rocm::Attention(s_.q, s.kv, mask, mask_words_, s_.ctx,
                  split ? s_.attn_partials : nullptr, AttentionSplits(n_tokens),
                  n_tokens, pos, c.num_heads, c.num_kv_heads, c.head_dim,
                  c.compress_ratio, stream_);
  if (rotated)
    UnrotateHeads(s_.ctx, s_.attn_gate, head_rows, stream_);
  else
    SigmoidMul(s_.ctx, s_.attn_gate,
               static_cast<std::size_t>(n_tokens) * c.AttentionQDim(), stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::AttentionBatchLayer(const DeviceLayer& l, std::uint32_t il,
                                   std::span<const BatchItem> items,
                                   std::span<const std::uint32_t> offsets,
                                   std::uint32_t rows,
                                   std::string* error) const {
  const Config& c = config();
  if (l.attn_qkv.empty() || items.size() > kMaxBatchSessions) {
    AssignError(error, "multi-session attention needs the stacked projection");
    return false;
  }
  if (batch_attn_host_ == nullptr) {
    const std::size_t bytes = std::size_t{c.num_layers} * kMaxBatchSessions *
                              sizeof(AttentionBatchItem);
    if (!Check(hipHostMalloc(&batch_attn_host_, bytes, hipHostMallocMapped),
               "attention batch table", error) ||
        !Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&batch_attn_),
                                       batch_attn_host_, 0),
               "attention batch table", error))
      return false;
  }
  // Split records of every live row; grown on demand (rows x heads x 8).
  if (std::size_t{rows} > batch_attn_partial_rows_) {
    if (batch_attn_partials_ != nullptr &&
        !Check(hipStreamSynchronize(stream_), "attention partials", error))
      return false;
    (void)hipFree(batch_attn_partials_);
    batch_attn_partials_ = nullptr;
    batch_attn_partial_rows_ = 0;
    if (!Check(hipMalloc(&batch_attn_partials_,
                         std::size_t{rows} * c.num_heads * kMaxAttentionSplits *
                             (c.head_dim + 2) * sizeof(float)),
               "attention batch partials", error))
      return false;
    batch_attn_partial_rows_ = rows;
  }
  // One table slice per layer: the GPU may still read the previous layer's.
  AttentionBatchItem* table =
      batch_attn_host_ + std::size_t{il} * kMaxBatchSessions;
  const AttentionBatchItem* device_table =
      batch_attn_ + std::size_t{il} * kMaxBatchSessions;
  std::uint32_t max_rows = 0, max_pool = 0, partial = 0, max_blocks = 32;
  bool any_sparse = false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    Session& session = *items[i].session;
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const bool live = session.CheckCancellation(nullptr);
    const auto end = session.position_ + n;
    const bool sparse = c.compress_ratio > 0 && end > c.indexer_top_k;
    const auto complete = c.compress_ratio > 0 ? end / c.compress_ratio : 0;
    const auto& at = session.attention_[il];
    table[i] = AttentionBatchItem{
        .kv = at.kv,
        .index_k = at.index_k,
        .block_k = at.block_k,
        .position = &session.control_->position,
        .first_block = &session.control_->blocks,
        .rope = at.rope,
        .index_capacity = session.index_capacity_,
        .row = offsets[i],
        .rows = live ? n : 0,
        .pool = sparse && complete > session.blocks_
                    ? complete - session.blocks_
                    : 0,
        .sparse = sparse ? 1U : 0U,
        .splits = AttentionSplits(n),
        .partial = partial,
    };
    if (!live)
      continue;
    partial += n * c.num_heads * table[i].splits;
    max_rows = std::max(max_rows, n);
    max_pool = std::max(max_pool, table[i].pool);
    any_sparse |= sparse;
    const auto blocks =
        c.HasIndexer()
            ? (session.max_context_ + c.compress_ratio - 1) / c.compress_ratio
            : 0;
    max_blocks = std::max(max_blocks, (blocks + 31) / 32 * 32);
  }
  const auto count = static_cast<std::uint32_t>(items.size());
  PrepareAttentionBatch(s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(),
                        l.attn_k_norm.f32(), s_.q, s_.attn_gate,
                        options_.kv_format, device_table, count, max_rows,
                        c.num_heads, c.num_kv_heads, c.head_dim, c.rotary_dim,
                        c.rope_theta, c.rms_eps, stream_);
  if (c.HasIndexer()) {
    if (!DenseBatch(l.indexer_k, s_.mixed, s_.ik, rows, error))
      return false;
    StoreIndexerBatch(s_.ik, l.indexer_k_norm.f32(), device_table, count,
                      max_rows, max_pool, c.compress_ratio, c.indexer_head_dim,
                      c.rotary_dim, c.rope_theta, c.rms_eps, stream_);
  }
  if (any_sparse) {
    if (!DenseBatch(l.indexer_q, s_.mixed, s_.iq, rows, error))
      return false;
    RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq, rows * c.indexer_heads,
                c.indexer_head_dim, 1, c.rms_eps, stream_);
    // The score scratch holds select_chunk_ rows of the model's widest
    // context; a fused prompt step packs up to max_batch rows, so deep
    // session contexts score the rows in windows of what the scratch holds.
    const std::size_t score_floats =
        std::size_t{select_chunk_} * mask_words_ * 32;
    if (!SelectBlocksBatch(
            s_.iq, s_.mask, s_.scores, device_table, count, max_rows, rows,
            static_cast<std::uint32_t>(score_floats / max_blocks),
            c.indexer_heads, c.indexer_head_dim, c.rotary_dim, c.rope_theta,
            c.compress_ratio, c.indexer_top_k / c.compress_ratio, mask_words_,
            max_blocks, stream_)) {
      AssignError(error, "multi-session block selection scratch too small");
      return false;
    }
  }
  const bool rotated = options_.kv_format != KvCacheFormat::kF16;
  if (rotated)
    RotateHeads(s_.q, rows * c.num_heads, stream_);
  AttentionBatch(s_.q, options_.kv_format, device_table, count, max_rows,
                 s_.mask, mask_words_, s_.ctx, batch_attn_partials_,
                 c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio,
                 stream_);
  if (rotated)
    UnrotateHeads(s_.ctx, s_.attn_gate, rows * c.num_heads, stream_);
  else
    SigmoidMul(s_.ctx, s_.attn_gate, std::size_t{rows} * c.AttentionQDim(),
               stream_);
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (table[i].rows == 0 &&
        !Check(hipMemsetAsync(
                   s_.ctx + std::size_t{offsets[i]} * c.AttentionQDim(), 0,
                   items[i].tokens.size() * c.AttentionQDim() * sizeof(float),
                   stream_),
               "cancelled attention rows", error))
      return false;
  }
  return true;
}

bool Executor::MoeShared(const DeviceLayer& l, const float* x,
                         std::uint32_t n_tokens, bool counts,
                         std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  // Router logits and the shared-expert gate come out of one GEMM.
  if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  Route(l, n_tokens);
  if (counts) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts, used,
                 stream_);
    if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg)) {
      return false;
    }
  }
  // The shared expert (gated by the last router row) does not depend on
  // routing. Queue its GEMMs after the count download, then prepare the
  // routed dispatch on the CPU without waiting for these GEMMs to finish.
  shexp_half_ready_ = false;
  if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                  &l.shexp_down, error_msg)) {
    return false;
  }
  if (shexp_half_ready_) {
    shexp_half_ready_ = false;
    if (const void* i4 = I4Route(l.shexp_down, n_tokens)) {
      if (!I4Activations(s_.shexp_half, n_tokens, l.shexp_down.cols,
                         error_msg) ||
          !I4Gemm(i4, l.shexp_down, s_.shexp_out, n_tokens, error_msg))
        return false;
    } else if (!spike::DenseF16Gemm(prefill_phase, l.shexp_down.data,
                                    s_.shexp_half, s_.shexp_out, n_tokens,
                                    l.shexp_down.rows, l.shexp_down.cols,
                                    stream_)) {
      AssignError(error_msg, "shared expert F16 GEMM failed");
      return false;
    }
  } else if (!Dense(l.shexp_down, s_.shexp_up, s_.shexp_out, n_tokens,
                    error_msg)) {
    return false;
  }
  return true;
}

bool Executor::Moe(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg,
                   bool last_only) const {
  const Config& c = config();
  if (!MoeShared(l, x, n_tokens, ExpertMatrixRows(n_tokens) && !DeviceRouted(l),
                 error_msg))
    return false;
  if (last_only && l.ffn_gate_exps.type == GgmlType::kQ8_0 &&
      l.ffn_up_exps.type == GgmlType::kQ8_0 &&
      l.ffn_down_exps.type == GgmlType::kQ8_0) {
    // Catch-up retains only the final predictor residual. Keep the router
    // and shared-expert shapes, then run its routed experts with the same
    // tiled quantization and accumulation as the complete prompt batch.
    const auto base = s_;
    UseScratch(RowScratch(base, n_tokens - 1));
    PrefillPhase phase(true);
    try {
      const std::size_t offset =
          static_cast<std::size_t>(n_tokens - 1) * c.hidden_size;
      const bool ok = MoeExperts(l, x + offset, out + offset, 1, error_msg);
      UseScratch(base);
      return ok;
    } catch (...) {
      UseScratch(base);
      throw;
    }
  }
  return MoeExperts(l, x, out, n_tokens, error_msg);
}

bool Executor::DeviceRouted(const DeviceLayer& l) const noexcept {
  const auto gate = l.ffn_gate_exps.type;
  const auto down = l.ffn_down_exps.type;
  return !prefill_phase &&
         (gate == GgmlType::kQ4_K || gate == GgmlType::kQ5_K ||
          gate == GgmlType::kQ8_0 || gate == GgmlType::kQ4_0 ||
          gate == GgmlType::kIQ4_XS || gate == GgmlType::kIQ3_S ||
          gate == GgmlType::kIQ3_XXS) &&
         l.ffn_up_exps.type == gate &&
         (down == GgmlType::kQ5_1 || down == GgmlType::kQ8_0 ||
          down == GgmlType::kIQ4_NL) &&
         ((config().hidden_size == 2560 && config().expert_ff == 640) ||
          // qwen35moe and lfm2: K-quant/Q8_0 (lfm2 also Q4_0) gate/up and
          // Q5_1/Q8_0 down only.
          (config().hidden_size == 2048 &&
           (config().expert_ff == 512 || config().expert_ff == 1536) &&
           (gate == GgmlType::kQ4_K || gate == GgmlType::kQ5_K ||
            gate == GgmlType::kQ8_0 || gate == GgmlType::kQ4_0) &&
           down != GgmlType::kIQ4_NL));
}

bool Executor::DeviceExperts(const DeviceLayer& l, const float* x, float* out,
                             std::uint32_t rows, std::string* error) const {
  const Config& c = config();
  // The routed plan borrows the gate scratch, which only the older routes
  // use (it holds rows * used * expert_ff floats; the plan needs less).
  // WeightType values are the GGUF type ids (DeviceRouted checked them).
  if (!RoutedExperts(
          l.ffn_gate_exps.data, l.ffn_up_exps.data,
          static_cast<WeightType>(l.ffn_gate_exps.type), l.ffn_down_exps.data,
          static_cast<WeightType>(l.ffn_down_exps.type), c.num_experts, x,
          s_.ids, s_.down_e, rows, c.num_experts_used, s_.gate_e, stream_,
          nullptr, c.hidden_size, c.expert_ff,
          spike::RoutedRotation(l.ffn_gate_exps.data))) {
    AssignError(error, "routed experts failed");
    return false;
  }
  MoeEpilogueVec4(s_.down_e, s_.weights, s_.shexp_out,
                  s_.router + c.num_experts, c.num_experts + 1, out, rows,
                  c.num_experts_used, c.hidden_size, stream_);
  return true;
}

void Executor::Route(const DeviceLayer& l, std::uint32_t rows) const {
  const Config& c = config();
  const std::uint32_t stride = c.num_experts + 1;
  if (router_dump_ != nullptr) {
    // Header: layer (the MTP block is num_layers), rows, stride, prefill.
    const auto& layers = model_->layers();
    const auto layer = static_cast<std::uint32_t>(
        &l >= layers.data() && &l < layers.data() + layers.size()
            ? &l - layers.data()
            : layers.size());
    const std::uint32_t header[] = {layer, rows, stride, prefill_phase};
    std::vector<float> logits(std::size_t{rows} * stride);
    if (hipMemcpyAsync(logits.data(), s_.router, logits.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream_) == hipSuccess &&
        hipStreamSynchronize(stream_) == hipSuccess) {
      std::fwrite(header, sizeof(header), 1, router_dump_);
      std::fwrite(logits.data(), sizeof(float), logits.size(), router_dump_);
    }
  }
  // A pruned layer stores fewer experts; its padded router logits past them
  // are never candidates, so ids stay inside the stored expert tensors.
  const std::uint32_t n_experts =
      l.ffn_gate_exps.experts != 0 ? l.ffn_gate_exps.experts : c.num_experts;
  RouterTopK(s_.router, stride, s_.ids, s_.weights, rows, n_experts,
             c.num_experts_used, stream_, router_norm_k_);
  const MoeRouting& r = options_.routing;
  if (r.mode != MoeRouting::Mode::kExact && !prefill_phase) {
    RoutePolicy(s_.router, stride, s_.ids, s_.weights, rows, n_experts,
                c.num_experts_used, static_cast<std::uint32_t>(r.mode), r.keep,
                r.value, r.budget, stream_, router_norm_k_);
  }
  // Fast-model decode rows keep their top Options::fast_experts slots (prompt
  // rows route exactly, so every model id shares the prompt's arithmetic).
  if (route_keep_ && !prefill_phase)
    TruncateRoutes(s_.ids, s_.weights, s_.route_keep, rows, c.num_experts_used,
                   stream_);
}

bool Executor::MoeExperts(const DeviceLayer& l, const float* x, float* out,
                          std::uint32_t n_tokens,
                          std::string* error_msg) const {
  if (DeviceRouted(l))
    return DeviceExperts(l, x, out, n_tokens, error_msg);
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t slots = n_tokens * used;
  if (!RouteHints(n_tokens, error_msg)) {
    return false;
  }
  // Tiled batches take the WMMA route: assignments compacted by expert
  // into 16-row padded buckets, token rows narrowed to F16 once, then the
  // F16 matrix-core GEMM per (expert, row tile).
  // WeightType values are the GGUF type ids; the packed expert layouts
  // (RoutedPacked) are the ones the WMMA route reads.
  const auto gate_type = static_cast<WeightType>(l.ffn_gate_exps.type);
  const auto down_type = static_cast<WeightType>(l.ffn_down_exps.type);
  const bool wmma_experts = ExpertMatrixRows(n_tokens) &&
                            RoutedPacked(gate_type, down_type) &&
                            l.ffn_up_exps.type == l.ffn_gate_exps.type &&
                            c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
  // Q4_0 experts have no MMQ route: short prompt chunks take the decode
  // kernels, which handle any row count. So do rotated experts (spike,
  // GUFO_SPIKE_ROUTED_HAD): the MMQ route has no activation rotation.
  if (!wmma_experts && (gate_type == WeightType::kQ4_0 ||
                        spike::RoutedRotation(l.ffn_gate_exps.data) != 0))
    return DeviceExperts(l, x, out, n_tokens, error_msg);
  if (wmma_experts) {
    RoutedCompact(s_.ids, s_.expert_counts, s_.routed_bounds, s_.routed_cursors,
                  s_.rows_token, s_.rows_slot, n_tokens, used, c.num_experts,
                  stream_);
    // The GEMMs read F16 token rows: the router's F16 GEMM (or the mix)
    // usually left them in s_.x_half already.
    if (!(half_src_ == x && half_rows_ == n_tokens &&
          half_cols_ == c.hidden_size && !half_bf16_)) {
      NarrowActivations(x, s_.x_half, false,
                        static_cast<std::size_t>(n_tokens) * c.hidden_size,
                        stream_);
      half_src_ = x;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
    const auto* x_half = static_cast<const __half*>(s_.x_half);
    // The paired gate/up GEMM applies SwiGLU without materializing the
    // gate (every tiled batch builds its map in RouteHints).
    auto* up_half = reinterpret_cast<__half*>(s_.up_e);
    // --routed-prefill-a4 (lossy): Q4_K gate/up on the iu4 WMMA with 4-bit
    // activations (routed_prefill.hpp), on the 64-row map; with
    // --routed-down-w4 the gate/up writes int4 SwiGLU rows and the 4-bit
    // down experts run on the iu4 WMMA too. The spike's experiments keep
    // their own route.
    const bool sym = options_.routed_sym != 0;
    const int a4_group = sym ? options_.routed_sym : options_.routed_a4_group;
    const bool iu4_gate_up = a4_group != 0 && gate_type == WeightType::kQ4_K &&
                             routed_64_tiles_ != 0 && c.expert_ff % 128 == 0 &&
                             c.hidden_size % 256 == 0 && !spike::Enabled();
    const bool iu4_down = iu4_gate_up && (sym || options_.routed_down4) &&
                          down_type == WeightType::kQ5_1 &&
                          c.hidden_size % 256 == 0 && c.expert_ff <= 768;
    if (iu4_gate_up) {
      auto q =
          routed_prefill::CarveA4(s_.gate_e, n_tokens, c.hidden_size, a4_group);
      if (sym)
        q.sums = nullptr;
      const auto h4 = routed_prefill::CarveA4(
          s_.up_e, static_cast<std::size_t>(n_tokens) * used, c.expert_ff, 128);
      const std::int32_t* map64 = s_.routed_tiles + routed_n_tiles_;
      if (!routed_prefill::QuantizeA4(x_half, n_tokens, q, stream_) ||
          !routed_prefill::GateUpIu4(
              l.ffn_gate_exps.data, l.ffn_up_exps.data, gate_type, q, map64,
              routed_64_tiles_, 64, s_.routed_bounds, s_.rows_token,
              s_.rows_slot, iu4_down ? nullptr : up_half,
              iu4_down ? &h4 : nullptr, c.expert_ff, sym, stream_) ||
          (iu4_down &&
           !(sym ? routed_prefill::DownSym(
                       l.ffn_down_exps.data, down_type, h4, map64,
                       routed_64_tiles_, 64, s_.routed_bounds, s_.rows_slot,
                       s_.rows_slot, reinterpret_cast<__half*>(s_.down_e),
                       c.hidden_size, stream_)
                 : routed_prefill::DownIu4(
                       l.ffn_down_exps.data, down_type, h4, map64,
                       routed_64_tiles_, 64, s_.routed_bounds, s_.rows_slot,
                       s_.rows_slot, reinterpret_cast<__half*>(s_.down_e),
                       c.hidden_size, stream_)))) {
        AssignError(error_msg, "routed iu4 GEMM failed");
        return false;
      }
    }
    const bool gated_ok =
        iu4_gate_up ||
        (routed_pair_tiles_ != 0
             ? spike::RoutedGatedF16Gemm(
                   prefill_phase, n_tokens, l.ffn_gate_exps.data,
                   l.ffn_up_exps.data, gate_type, x_half,
                   s_.routed_tiles + routed_pair_offset_, routed_pair_tiles_,
                   routed_pair_rows_, s_.routed_bounds, s_.rows_token,
                   s_.rows_slot, up_half, c.expert_ff, c.hidden_size, stream_)
             : (spike::RoutedF16Gemm(
                    prefill_phase, n_tokens, l.ffn_gate_exps.data, gate_type,
                    x_half, s_.routed_tiles, routed_n_tiles_, routed_tile_rows_,
                    s_.routed_bounds, s_.rows_token, s_.rows_slot, nullptr,
                    s_.gate_e, nullptr, c.expert_ff, c.hidden_size, stream_) &&
                spike::RoutedF16Gemm(
                    prefill_phase, n_tokens, l.ffn_up_exps.data, gate_type,
                    x_half, s_.routed_tiles, routed_n_tiles_, routed_tile_rows_,
                    s_.routed_bounds, s_.rows_token, s_.rows_slot, s_.gate_e,
                    nullptr, up_half, c.expert_ff, c.hidden_size, stream_)));
    if (!gated_ok) {
      AssignError(error_msg, "routed F16 gate/up GEMM failed");
      return false;
    }
    // Larger down tiles amortize weight decoding. Reuse the 64-token map
    // when it has no more padded rows than the 48-token map.
    const bool wide_down = routed_tile_rows_ == 48 && routed_64_tiles_ != 0 &&
                           routed_64_tiles_ * 4 <= routed_n_tiles_ * 3;
    // The down projection's rows are F16 too: the epilogue reads half the
    // bytes of the largest routed intermediate.
    if (!iu4_down &&
        !spike::RoutedF16Gemm(
            prefill_phase, static_cast<std::size_t>(n_tokens) * used,
            l.ffn_down_exps.data, down_type, up_half,
            s_.routed_tiles + (wide_down ? routed_n_tiles_ : 0),
            wide_down ? routed_64_tiles_ : routed_n_tiles_,
            wide_down ? 64 : routed_tile_rows_, s_.routed_bounds, s_.rows_slot,
            s_.rows_slot, nullptr, nullptr,
            reinterpret_cast<__half*>(s_.down_e), c.hidden_size, c.expert_ff,
            stream_)) {
      AssignError(error_msg, "routed F16 down GEMM failed");
      return false;
    }
  } else {
    if (!GatedExperts(l.ffn_gate_exps, l.ffn_up_exps, x, s_.ids, s_.gate_e,
                      n_tokens, used, error_msg)) {
      return false;
    }
    // The down projection sees one (token, slot) row per expert id.
    if (!Experts(l.ffn_down_exps, s_.gate_e, s_.ids, s_.down_e, slots, 1,
                 n_tokens, error_msg)) {
      return false;
    }
  }
  if (wmma_experts) {
    // The combine that follows folds this epilogue into its own pass when
    // it takes the F16 route (Combine); otherwise it runs here.
    moe_pending_ = out == s_.block_out;
    if (!moe_pending_) {
      MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                         s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, n_tokens, used, c.hidden_size,
                         stream_);
    }
  } else if (MatrixRows(n_tokens)) {
    MoeEpilogueVec4(s_.down_e, s_.weights, s_.shexp_out,
                    s_.router + c.num_experts, c.num_experts + 1, out, n_tokens,
                    used, c.hidden_size, stream_);
  } else {
    MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
                c.num_experts + 1, out, n_tokens, used, c.hidden_size, stream_);
  }
  return true;
}

bool Executor::CopyTrunkHidden(const Session& session, std::span<float> hidden,
                               std::string* error_msg) const {
  if (!session.mtp_enabled_ || session.owner_ != this ||
      session.position_ == 0 || hidden.empty() ||
      hidden.size() % config().HcDim() != 0 ||
      hidden.size() / config().HcDim() >
          std::min(session.position_, options_.max_speculative)) {
    AssignError(error_msg, "invalid trunk hidden diagnostic input");
    return false;
  }
  return Check(hipMemcpyAsync(hidden.data(), session.mtp_.target_hidden,
                              hidden.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "trunk hidden download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "trunk hidden", error_msg);
}

bool Executor::MtpHead(const DeviceMixer& head, const float* res, bool token,
                       bool candidates, std::string* error_msg) const {
  const DeviceTensor& output = DraftHead();
  if (!HcMix(head, res, false, s_.mixed, nullptr, 1, error_msg) ||
      !Dense(output, s_.mixed, s_.logits, 1, error_msg)) {
    return false;
  }
  if (candidates)
    MtpTopCandidates(s_.logits, s_.mtp_ids, s_.mtp_scratch_ids, s_.mtp_scores,
                     output.rows, stream_);
  if (token) {
    Argmax(s_.logits, s_.mtp_argmax, s_.mtp_token, 1, output.rows, stream_);
    if (!Check(
            hipMemcpyAsync(mtp_token_host_, s_.mtp_token, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "draft token download", error_msg)) {
      return false;
    }
  }
  if (candidates) {
    const auto count = std::min<std::size_t>(output.rows, kMtpCandidates);
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    const auto bytes =
        offsetof(MtpCandidateLogits, logits) + count * sizeof(float);
    if (!Check(hipMemcpyAsync(mtp_candidates_host_, s_.mtp_ids, bytes,
                              hipMemcpyDeviceToHost, stream_),
               "draft candidates download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body, std::string* error_msg,
                   bool synchronize) const {
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    if (const auto it = session.graphs_.find(key);
        it != session.graphs_.end()) {
      exec = it->second;
    } else {
      hipGraph_t captured = nullptr;
      // Frozen peer sessions can copy snapshots on independent nonblocking
      // streams while the scheduler records this session's decode graph.
      if (!Check(
              hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
              "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        return false;
      }
      session.graphs_.emplace(key, exec);
    }
    if (!Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return !synchronize ||
         Check(hipStreamSynchronize(stream_), "forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg) const {
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  if (!RequireResident(session, error_msg))
    return false;
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_.position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  if ((speculative && !EnsureRollback(session, n - 1, error_msg)) ||
      !Reserve(session, start_pos + n, error_msg))
    return false;
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens, the control block (positions
  // the kernels read on the device) and the n-gram rows.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->mtp_blocks = session.mtp_.blocks;
  control_host_->hidden_row = -1;
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    if (!PleFetch(session, tokens, speculative, error_msg)) {
      return false;
    }
  }
  // Sparse selection only changes the result once a query can see more
  // than the token budget; every layer of this model shares one ratio.
  const bool sparse = c.HasIndexer() && start_pos + n > c.indexer_top_k;
  const std::uint32_t complete =
      c.compress_ratio > 0 ? (start_pos + n) / c.compress_ratio : 0;
  const std::uint32_t pool_grid =
      sparse && complete > session.blocks_ ? complete - session.blocks_ : 0;
  // Decode-sized batches replay as graphs; a pooling backlog (the first
  // batch past the budget) needs the wider eager grid.
  const std::uint32_t graph_pool_grid = n / std::max(c.compress_ratio, 1u) + 1;
  const bool graph = !prefill_phase && router_dump_ == nullptr &&
                     n <= kVecBatch && pool_grid <= graph_pool_grid &&
                     session.position_ >= session.VisionLayout().PrefixLength();
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (static_cast<std::uint64_t>(sparse) << 33) |
                            (std::uint64_t{logits != nullptr} << 34) |
                            (std::uint64_t{session.adapter_} << 36) |
                            (std::uint64_t{session.route_keep_} << 56);
  // PLE first consumes the disk rows at its injection layer. Queue the
  // preceding layers before waiting, including on captured graph replay.
  // Both pieces use the same stream and arithmetic as the unsplit graph.
  const std::uint32_t first_layer =
      graph && c.ple_layer > 0 ? static_cast<std::uint32_t>(c.ple_layer) : 0;
  if (first_layer > 0) {
    const auto prefix = [&] {
      return ForwardBody(session, n, 0, false, speculative, sparse, start_pos,
                         graph_pool_grid, 0, first_layer, error_msg);
    };
    constexpr std::uint64_t kPrefixKey = std::uint64_t{1} << 35;
    if (!Run(session, key | kPrefixKey, graph, prefix, error_msg, false)) {
      return false;
    }
  }
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       sparse, start_pos, graph ? graph_pool_grid : pool_grid,
                       first_layer, c.num_layers, error_msg);
  };
  // Only the suffix waits for its pinned n-gram rows. During eager
  // execution and capture, Ple performs this wait at the same boundary.
  if (graph && session.graphs_.contains(key) && ple_pending_ &&
      !WaitPle(error_msg)) {
    return false;
  }
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }

  session.position_ += n;
  if (sparse) {
    session.blocks_ = complete;
  }
  // Verified rows join the drafter's context once Rollback knows them.
  session.dspark_.feature_row = 0;
  if (session.dspark_enabled_ && !speculative) {
    session.dspark_.Pend(0, n, start_pos);
    Session* const one = &session;
    if (!DsparkFlush({&one, 1}, error_msg))
      return false;
  }
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch. MtpBody reads that
    // residual into xn before reusing res for the predictor output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {}, error_msg, s_.res))
      return false;
  }
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, bool sparse,
                           std::uint32_t start_pos, std::uint32_t pool_grid,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg) const {
  const Config& c = config();
  // Trunk block outputs only; the draft layer runs outside this scope.
  const SteerScope steer(*this, session.adapter_ != 0);
  // Fast-model decode rows (Session::route_keep_); the draft layer and
  // prompt rows route exactly.
  const RouteKeepScope route_keep(*this, session.route_keep_ != 0);
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg) ||
        (steer_ &&
         !Check(hipMemsetAsync(s_.steer, session.adapter_, n, stream_),
                "adapter rows", error_msg)) ||
        (route_keep_ &&
         !Check(hipMemsetAsync(s_.route_keep, session.route_keep_, n, stream_),
                "fast-model rows", error_msg))) {
      return false;
    }
    EmbedTokens(model_->token_embd().data,
                static_cast<WeightType>(model_->token_embd().type), s_.tokens,
                s_.res, n, c.hidden_size, c.hc_count, stream_);
    // Text embedding rows only: image rows replace them below, as image
    // features replace a folded table's rows.
    SteerWriter(s_.res, n, c.hc_count, steer_writer::kEmbedding);
    session.vision_input_.Inject(s_.res, start_pos, n, c.hidden_size,
                                 c.hc_count, stream_);
  }
  const auto& layers = model_->layers();
  // F16 residual between the wide combines (lossy, --prefill-residual); DSpark
  // reads the F32 residual at its target layers.
  const bool res16 = hc_streams_ != 0 && !session.dspark_enabled_;
  res_half_live_ = false;
  xn_scale_live_ = false;
  bool normed = false;  ///< xn holds the next mixer's grouped norm of res
  // Macro batch (--prefill-phase-rows): a prompt batch wider than the chunk
  // runs every non-MoE phase per `chunk` rows (their efficient shapes, state
  // and position carried chunk to chunk exactly as consecutive Forward calls
  // would) but each layer's routed experts once over all rows, so every
  // expert streams its weights once per batch. Layer order per layer: the
  // pre-MoE phases of chunks 0..k-1 (a chunk's mixer needs only the previous
  // chunks' state of THIS layer, which their pre-MoE phases already wrote),
  // the routed experts over the whole batch, the combines per chunk.
  const std::uint32_t chunk =
      std::min(options_.prefill_chunk, options_.max_batch);
  const bool macro = prefill_phase && !speculative && first_layer == 0 &&
                     end_layer == c.num_layers && chunk >= 256 &&
                     chunk % 128 == 0 && n > chunk && !steer_ &&
                     c.HyperConnections() && c.Traits().shared_expert &&
                     (n + chunk - 1) / chunk <= kMaxBatchSessions &&
                     AllocateBatch(error_msg);
  struct MacroChunk {
    std::uint32_t off;
    std::uint32_t rows;
    std::uint32_t pool;
    bool sparse;
    bool xn_half{false};
    std::uint32_t inject_parts{1};
  };
  std::vector<MacroChunk> chunks;
  if (macro) {
    std::uint32_t pooled = session.blocks_;
    for (std::uint32_t off = 0; off < n; off += chunk) {
      const std::uint32_t rows = std::min(chunk, n - off);
      const std::uint32_t end = start_pos + off + rows;
      const bool sp = c.HasIndexer() && end > c.indexer_top_k;
      const std::uint32_t complete =
          c.compress_ratio > 0 ? end / c.compress_ratio : 0;
      batch_controls_[chunks.size()] = {start_pos + off, pooled,
                                        session.mtp_.position, -1,
                                        session.mtp_.blocks};
      chunks.push_back(
          {off, rows, sp && complete > pooled ? complete - pooled : 0, sp});
      if (sp)
        pooled = complete;
    }
  }
  const auto macro_layer = [&](std::uint32_t il, const DeviceLayer& l,
                               const float* next_norm) -> bool {
    const Scratch base = s_;
    const std::size_t used = c.num_experts_used;
    const auto view = [&](const MacroChunk& m) {
      Scratch v = RowScratch(base, m.off);
      // The routed down rows are F16 (RowScratch strides them as F32).
      v.down_e =
          reinterpret_cast<float*>(reinterpret_cast<__half*>(base.down_e) +
                                   std::size_t{m.off} * used * c.hidden_size);
      v.xn_q8t = static_cast<std::uint8_t*>(base.xn_q8t) +
                 Q8TiledBytes(m.off, c.HcDim());
      return v;
    };
    for (auto& m : chunks) {
      UseScratch(view(m));
      xn_half_ = m.xn_half;
      if (!HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, m.rows,
                 error_msg))
        return false;
      bool ok = false;
      const auto index = static_cast<std::size_t>(&m - chunks.data());
      if (l.mixer == MixerKind::kGatedDeltaNet) {
        ok = LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out,
                             m.rows, false, error_msg);
      } else if (l.mixer == MixerKind::kGatedAttention) {
        ok = Check(hipMemcpyAsync(session.control_, batch_controls_ + index,
                                  sizeof(Session::Control),
                                  hipMemcpyHostToDevice, stream_),
                   "chunk control", error_msg) &&
             Attention(l, session.attention_[il], s_.mixed, s_.block_out,
                       m.rows, &session.control_->position,
                       &session.control_->blocks, start_pos + m.off, m.pool,
                       session.max_context_, m.sparse, error_msg);
      } else {
        AssignError(error_msg, "macro prefill: unsupported mixer");
      }
      if (!ok)
        return false;
      Combine(s_.res, l.hc_ffn.norm.f32(), m.rows);
      if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, m.rows,
                 error_msg) ||
          !MoeShared(l, s_.mixed, m.rows, false, error_msg))
        return false;
      m.inject_parts = inject_parts_;
    }
    UseScratch(base);
    ExpertCounts(base.ids, base.expert_counts, n, c.num_experts,
                 c.num_experts_used, stream_);
    if (!Check(hipMemcpyAsync(counts_host_, base.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg) ||
        !MoeExperts(l, base.mixed, base.block_out, n, error_msg))
      return false;
    const bool pending = moe_pending_;
    for (auto& m : chunks) {
      UseScratch(view(m));
      moe_pending_ = pending;
      inject_parts_ = m.inject_parts;
      Combine(s_.res, next_norm, m.rows, static_cast<std::int32_t>(il));
      m.xn_half = xn_half_;
    }
    UseScratch(base);
    return true;
  };
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (!session.CheckCancellation(error_msg))
      return false;
    const DeviceLayer& l = layers[il];
    if (session.dspark_enabled_) {
      // DSpark reads each target layer's input residual.
      const auto& targets = model_->dspark()->target_layers;
      const auto hit = std::find(targets.begin(), targets.end(), il);
      const std::size_t row = targets.size() * c.hidden_size * sizeof(float);
      if (hit != targets.end() &&
          !Check(hipMemcpy2DAsync(
                     ds_.features + (hit - targets.begin()) * c.hidden_size,
                     row, s_.res, c.HcDim() * sizeof(float),
                     c.hidden_size * sizeof(float), n, hipMemcpyDeviceToDevice,
                     stream_),
                 "DSpark feature capture", error_msg))
        return false;
    }
    if (c.IsPleLayer(il) &&
        !Ple(l, session, n, s_.res, speculative, error_msg)) {
      return false;
    }
    if (macro) {
      const float* macro_norm =
          il + 1 < c.num_layers
              ? (c.IsPleLayer(il + 1) ? nullptr
                                      : layers[il + 1].hc_attn.norm.f32())
              : model_->hc_head().norm.f32();
      if (!macro_layer(il, l, macro_norm))
        return false;
      normed = macro_norm != nullptr;
      continue;
    }
    if (!HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, n, error_msg)) {
      return false;
    }
    bool mixer_ok = false;
    switch (l.mixer) {
      case MixerKind::kGatedDeltaNet:
        mixer_ok = LinearAttention(l, session.linear_[il], s_.mixed,
                                   s_.block_out, n, speculative, error_msg);
        break;
      case MixerKind::kShortConv:
        mixer_ok = LfmConv(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           error_msg);
        break;
      case MixerKind::kGatedAttention:
        mixer_ok = Attention(l, session.attention_[il], s_.mixed, s_.block_out,
                             n, &session.control_->position,
                             &session.control_->blocks, start_pos, pool_grid,
                             session.max_context_, sparse, error_msg);
        break;
      case MixerKind::kAttention:
        mixer_ok =
            LfmAttentionLayer(l, session.attention_[il], s_.mixed, s_.block_out,
                              n, &session.control_->position, error_msg);
        break;
    }
    if (!mixer_ok) {
      return false;
    }

    // Each combine also norms the residual for the mixer that follows it,
    // unless PLE rewrites the residual first.
    Combine(s_.res, l.hc_ffn.norm.f32(), n, -1, &l.hc_ffn, !res16);
    if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, n, error_msg) ||
        !(c.Traits().shared_expert
              ? Moe(l, s_.mixed, s_.block_out, n, error_msg)
              : LfmFfn(l, s_.mixed, s_.block_out, n, error_msg))) {
      return false;
    }

    const float* next_norm =
        il + 1 < c.num_layers
            ? (c.IsPleLayer(il + 1) ? nullptr
                                    : layers[il + 1].hc_attn.norm.f32())
            : model_->hc_head().norm.f32();
    Combine(s_.res, next_norm, n, static_cast<std::int32_t>(il),
            il + 1 < c.num_layers && next_norm != nullptr
                ? &layers[il + 1].hc_attn
                : nullptr,
            !res16 || il + 1 >= end_layer);
    normed = next_norm != nullptr;
  }
  if (macro && !Check(hipMemcpyAsync(session.control_, control_host_,
                                     sizeof(Session::Control),
                                     hipMemcpyHostToDevice, stream_),
                      "control restore", error_msg))
    return false;
  if (end_layer < c.num_layers) {
    return true;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(n, options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.mtp_.target_hidden,
                 s_.res + static_cast<std::size_t>(n - kept) * c.HcDim(),
                 static_cast<std::size_t>(kept) * c.HcDim() * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    // The head mixer norms its tail rows itself: the last combine's norm is
    // laid out for the whole batch (and tiled on the wide route), so a row
    // offset into it is not addressable.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    const DeviceMixer& head = model_->hc_head();
    if (!HcMix(head, s_.res + skip * c.HcDim(), false, s_.mixed, nullptr,
               n_logits, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg, float* logits) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  session.spec_tokens_ = 0;
  if (session.dspark_enabled_) {
    // A decode batch injects every session's kept rows together after its
    // last Rollback (DsparkDefer); otherwise they join here.
    session.dspark_.Pend(session.dspark_.feature_row, keep, session.spec_base_);
    Session* const one = &session;
    if (!dspark_defer_ && !DsparkFlush({&one, 1}, error_msg))
      return false;
  }
  if (logits != nullptr &&
      !Check(hipMemcpyAsync(
                 logits_host_,
                 VerificationLogits() +
                     static_cast<std::size_t>(keep - 1) * c.vocab_size,
                 c.vocab_size * sizeof(float), hipMemcpyDeviceToHost, stream_),
             "frontier download", error_msg)) {
    return false;
  }
  if (keep == n && logits == nullptr) {
    return true;
  }
  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      RestoreGdnState(l.state, l.state_snapshots, keep, c.ssm_num_k_heads,
                      c.ssm_num_v_heads, stream_, options_.gdn_state);
      if (!Check(hipGetLastError(), "state rollback", error_msg) ||
          !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    if (session.ple_history_ != nullptr) {
      const std::size_t hist =
          static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
      if (!Check(hipMemcpyAsync(
                     session.ple_history_, session.ple_snapshots_.rows[slot],
                     hist * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                 "PLE rollback", error_msg)) {
        return false;
      }
      session.ngram_ = session.ngram_snapshots_[slot];
    }
    session.position_ = session.spec_base_ + keep;
    // Pooled block keys past the kept prefix are stale; they are rebuilt
    // from the raw keys when needed.
    session.blocks_ =
        c.compress_ratio == 0
            ? 0
            : std::min(session.blocks_, session.position_ / c.compress_ratio);
  }
  if (!Check(hipStreamSynchronize(stream_), "rollback", error_msg)) {
    return false;
  }
  if (logits != nullptr) {
    std::copy_n(logits_host_, c.vocab_size, logits);
  }
  return true;
}

constexpr std::array<char, 8> kSnapshotMagic{'Q', 'F', 'N', 'S',
                                             'N', 'A', 'P', '8'};

/// Fixed header ahead of the section bytes. It carries every geometry
/// value the section sizes derive from, so a payload of another artifact
/// or executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint64_t attention_layers;  ///< bit l: layer l is attention
  std::uint32_t conv_elems;
  /// Recurrent state elements per layer; bits 28-31 carry a non-F32
  /// GdnStateFormat (F32 snapshots keep their original header bytes).
  std::uint32_t state_elems;
  std::uint32_t kv_row;
  std::uint32_t kv_heads;
  std::uint32_t kv_format;  ///< KvCacheFormat of the attention caches
  std::uint32_t indexer_head_dim;
  std::uint32_t compress_ratio;
  std::uint32_t hc_dim;
  std::uint32_t ple_elems;
  std::uint32_t has_mtp;
  std::uint32_t dspark_layers;  ///< drafter K/V caches (F16 rows)
  std::uint32_t dspark_kv_row;
  std::uint32_t position;
  std::uint32_t blocks;
  std::uint32_t mtp_position;
  std::uint32_t mtp_blocks;
  std::uint32_t hidden_rows;
  std::uint32_t image_count;
  std::uint32_t explicit_rows;  ///< explicit RoPE triples (mtmd prompts)
  std::int32_t explicit_delta;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> ngram_prev;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);

namespace {

constexpr std::uint32_t kStateFormatShift = 28;

std::uint64_t StateRegionBytes(const SnapshotHeader& h) {
  const auto format =
      static_cast<GdnStateFormat>(h.state_elems >> kStateFormatShift);
  return std::uint64_t{h.state_elems & ((1U << kStateFormatShift) - 1)} *
         GdnStateElementBytes(format);
}

SnapshotHeader MakeSnapshotHeader(const Config& c, bool has_mtp,
                                  const Session& session,
                                  KvCacheFormat kv_format,
                                  GdnStateFormat gdn_state,
                                  const DeviceDspark* dspark) {
  SnapshotHeader h{};
  h.magic = kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.full_attention_interval = c.full_attention_interval;
  h.attention_layers = c.AttentionMask();
  h.conv_elems = (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  h.state_elems = c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim |
                  static_cast<std::uint32_t>(gdn_state) << kStateFormatShift;
  h.kv_row = c.AttentionKvDim();
  h.kv_heads = c.num_kv_heads;
  h.kv_format = static_cast<std::uint32_t>(kv_format);
  h.indexer_head_dim = c.indexer_head_dim;
  h.compress_ratio = c.compress_ratio;
  h.hc_dim = c.HcDim();
  h.ple_elems = c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0;
  h.has_mtp = has_mtp ? 1 : 0;
  if (dspark != nullptr && session.dspark_enabled()) {
    h.dspark_layers = static_cast<std::uint32_t>(dspark->blocks.size());
    h.dspark_kv_row = dspark->kv_heads * dspark->head_dim;
  }
  h.position = session.position();
  h.image_count = session.VisionLayout().images.size();
  h.explicit_rows = static_cast<std::uint32_t>(
      session.VisionLayout().explicit_positions.size());
  h.explicit_delta = session.VisionLayout().explicit_delta;
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.full_attention_interval == mine.full_attention_interval &&
         h.attention_layers == mine.attention_layers &&
         h.conv_elems == mine.conv_elems && h.state_elems == mine.state_elems &&
         h.kv_row == mine.kv_row && h.kv_heads == mine.kv_heads &&
         h.kv_format == mine.kv_format &&
         h.indexer_head_dim == mine.indexer_head_dim &&
         h.compress_ratio == mine.compress_ratio && h.hc_dim == mine.hc_dim &&
         h.ple_elems == mine.ple_elems && h.has_mtp == mine.has_mtp &&
         h.dspark_layers == mine.dspark_layers &&
         h.dspark_kv_row == mine.dspark_kv_row;
}

}  // namespace

template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session, Visit&& visit) {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what) {
    if (bytes != 0 && !visit(device, offset, bytes, what)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const auto linear = [&](std::uint32_t il) -> const Session::LinearState* {
    return session != nullptr ? &session->linear_[il] : nullptr;
  };
  const auto attention =
      [&](std::uint32_t il) -> const Session::AttentionState* {
    return session != nullptr ? &session->attention_[il] : nullptr;
  };
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((h.attention_layers >> il) & 1U) == 0) {
      const auto* l = linear(il);
      if (!region(l != nullptr ? l->conv_state : nullptr,
                  std::uint64_t{h.conv_elems} * sizeof(float), "conv state") ||
          !region(l != nullptr ? l->state : nullptr, StateRegionBytes(h),
                  "recurrent state")) {
        return 0;
      }
    }
  }
  if (!region(session != nullptr ? session->ple_history_ : nullptr,
              std::uint64_t{h.ple_elems} * sizeof(float), "PLE history")) {
    return 0;
  }
  // The first `positions` rows of one K or V cache: F16 rows, or TurboQuant
  // codes then scales (256-wide heads).
  const auto kv_rows = [&](const KvView* kv, bool value,
                           std::uint32_t positions, const char* codes_name,
                           const char* scales_name) {
    const auto format = static_cast<KvCacheFormat>(h.kv_format);
    void* data = kv == nullptr ? nullptr : value ? kv->v : kv->k;
    if (format == KvCacheFormat::kF16)
      return region(data, std::uint64_t{positions} * h.kv_row * sizeof(__half),
                    codes_name);
    // Codes are KvCodeBytes per 256 values; one scale per (token, head).
    const std::uint64_t code_rows = std::uint64_t{positions} * (h.kv_row / 256);
    return region(data, code_rows * KvCodeBytes(format), codes_name) &&
           region(kv == nullptr ? nullptr
                  : value       ? kv->v_scale
                                : kv->k_scale,
                  std::uint64_t{positions} * h.kv_heads * sizeof(__half),
                  scales_name);
  };
  const std::uint64_t index_row_bytes =
      std::uint64_t{h.indexer_head_dim} * sizeof(float);
  const std::uint32_t index_begin = h.blocks * h.compress_ratio;
  const std::uint32_t index_rows = h.position - index_begin;
  const std::uint64_t block_bytes =
      std::uint64_t{h.blocks} * h.indexer_head_dim * sizeof(__half);
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((h.attention_layers >> il) & 1U) != 0) {
      const auto* at = attention(il);
      const KvView* kv = at != nullptr ? &at->kv : nullptr;
      if (!kv_rows(kv, false, h.position, "K cache", "K scales") ||
          !kv_rows(kv, true, h.position, "V cache", "V scales")) {
        return 0;
      }
      // Serialize only unpooled rows, in chronological order. The physical
      // ring may wrap, and a destination session may have another capacity.
      if (session == nullptr) {
        if (!region(nullptr, index_rows * index_row_bytes, "indexer keys"))
          return 0;
      } else {
        const auto first = index_begin & (session->index_capacity_ - 1);
        const auto tail =
            std::min(index_rows, session->index_capacity_ - first);
        if (!region(at->index_k + std::size_t{first} * h.indexer_head_dim,
                    tail * index_row_bytes, "indexer tail") ||
            !region(at->index_k, (index_rows - tail) * index_row_bytes,
                    "indexer wrap"))
          return 0;
      }
      if (!region(at != nullptr ? at->block_k : nullptr, block_bytes,
                  "pooled block keys"))
        return 0;
    }
  }
  if (h.has_mtp != 0) {
    const auto* mtp = session != nullptr ? &session->mtp_ : nullptr;
    const std::uint32_t begin = h.mtp_blocks * h.compress_ratio;
    const std::uint32_t rows = h.mtp_position - begin;
    if (session == nullptr) {
      if (!region(nullptr, rows * index_row_bytes, "draft indexer keys"))
        return 0;
    } else {
      const auto first = begin & (session->index_capacity_ - 1);
      const auto tail = std::min(rows, session->index_capacity_ - first);
      if (!region(mtp->index_k + std::size_t{first} * h.indexer_head_dim,
                  tail * index_row_bytes, "draft indexer tail") ||
          !region(mtp->index_k, (rows - tail) * index_row_bytes,
                  "draft indexer wrap"))
        return 0;
    }
    if (!region(
            mtp ? mtp->block_k : nullptr,
            std::uint64_t{h.mtp_blocks} * h.indexer_head_dim * sizeof(__half),
            "draft pooled keys"))
      return 0;
    const KvView* mtp_kv = mtp != nullptr ? &mtp->kv : nullptr;
    if (!kv_rows(mtp_kv, false, h.mtp_position, "draft K cache",
                 "draft K scales") ||
        !kv_rows(mtp_kv, true, h.mtp_position, "draft V cache",
                 "draft V scales") ||
        !region(mtp != nullptr ? mtp->h : nullptr,
                std::uint64_t{h.hc_dim} * sizeof(float), "draft residual") ||
        !region(mtp != nullptr ? mtp->target_hidden : nullptr,
                std::uint64_t{h.hidden_rows} * h.hc_dim * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  for (std::uint32_t il = 0; il < h.dspark_layers; ++il) {
    const auto* kv = session != nullptr ? &session->dspark_.kv[il] : nullptr;
    const std::uint64_t bytes =
        std::uint64_t{h.position} * h.dspark_kv_row * sizeof(__half);
    if (!region(kv ? kv->k : nullptr, bytes, "DSpark K cache") ||
        !region(kv ? kv->v : nullptr, bytes, "DSpark V cache"))
      return 0;
  }
  return offset +
         std::uint64_t{h.image_count} * sizeof(qwen::vision::ImageGrid) +
         std::uint64_t{h.explicit_rows} * sizeof(std::array<std::int32_t, 3>);
}

std::uint64_t Executor::SnapshotBytes(const Session& session,
                                      std::uint32_t hidden_rows) const {
  SnapshotHeader h = MakeSnapshotHeader(config(), session.mtp_enabled_, session,
                                        options_.kv_format, options_.gdn_state,
                                        model_->dspark());
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(
      h, nullptr,
      [](void*, std::uint64_t, std::uint64_t, const char*) { return true; });
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t hidden_rows,
                            std::span<std::uint8_t> payload,
                            std::string* error_msg) const {
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (session.spec_tokens_ != 0) {
    AssignError(error_msg,
                "snapshot with a pending speculative batch; roll back first");
    return false;
  }
  if (hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    AssignError(error_msg, "kept trunk rows exceed the position or batch");
    return false;
  }
  SnapshotHeader h = MakeSnapshotHeader(config(), session.mtp_enabled_, session,
                                        options_.kv_format, options_.gdn_state,
                                        model_->dspark());
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  h.ngram_prev = session.ngram_.prev;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);
  if (payload.size() != h.payload_bytes) {
    AssignError(error_msg, "snapshot buffer size does not match the payload");
    return false;
  }
  gufo::hip::SnapshotTransfer transfer;
  std::memcpy(payload.data(), &h, sizeof(h));
  const auto grid_bytes =
      std::size_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
  const auto explicit_bytes =
      std::size_t{h.explicit_rows} * sizeof(std::array<std::int32_t, 3>);
  if (grid_bytes != 0) {
    std::memcpy(payload.data() + payload.size() - explicit_bytes - grid_bytes,
                session.VisionLayout().images.data(), grid_bytes);
  }
  if (explicit_bytes != 0) {
    std::memcpy(payload.data() + payload.size() - explicit_bytes,
                session.VisionLayout().explicit_positions.data(),
                explicit_bytes);
  }
  return WalkSnapshot(h, &session,
                      [&](void* device, std::uint64_t offset,
                          std::uint64_t bytes, const char*) {
                        if (!session.CheckCancellation(error_msg))
                          return false;
                        transfer.Copy(payload.data() + offset, device, bytes);
                        return true;
                      }) != 0;
}

std::vector<Executor::SnapshotRegion> Executor::DescribeSnapshot(
    std::span<const std::uint8_t> payload) {
  std::vector<SnapshotRegion> regions;
  SnapshotHeader h{};
  if (payload.size() < sizeof(h))
    return regions;
  std::memcpy(&h, payload.data(), sizeof(h));
  if (h.magic != kSnapshotMagic || h.attention_layers == 0)
    return regions;
  regions.push_back({"executor header (position=" + std::to_string(h.position) +
                         " blocks=" + std::to_string(h.blocks) +
                         " mtp_position=" + std::to_string(h.mtp_position) +
                         " mtp_blocks=" + std::to_string(h.mtp_blocks) +
                         " hidden_rows=" + std::to_string(h.hidden_rows) + ")",
                     0, sizeof(h)});
  // Per-layer regions repeat their name; number them by occurrence.
  std::unordered_map<std::string_view, std::uint32_t> seen;
  std::uint64_t end = sizeof(h);
  (void)WalkSnapshot(
      h, nullptr,
      [&](void*, std::uint64_t offset, std::uint64_t bytes, const char* what) {
        regions.push_back(
            {std::string(what) + " #" + std::to_string(seen[what]++), offset,
             bytes});
        end = offset + bytes;
        return true;
      });
  if (end < payload.size())
    regions.push_back({"image grids", end, payload.size() - end});
  return regions;
}

void Executor::SharePages(const Session& session,
                          std::vector<SnapshotPage>* pages) const {
  // Speculative rollback and draft rewinds stay within a few rows of the
  // append positions; rows further below are never rewritten.
  constexpr std::uint32_t kFinalMargin = 64;
  const auto final_rows = [](std::uint32_t rows, std::uint32_t margin) {
    return rows > margin ? rows - margin : 0;
  };
  std::vector<gufo::hip::PieceRef> refs;
  for (std::uint32_t i = 0; i < session.growable_.size(); ++i) {
    const auto& g = session.growable_[i];
    const auto rows =
        g.tokens_per_row == 1
            ? final_rows(g.mtp ? session.mtp_.position : session.position_,
                         kFinalMargin)
            : final_rows(g.mtp ? session.mtp_.blocks : session.blocks_,
                         kFinalMargin / g.tokens_per_row);
    refs.clear();
    g.buffer.Share(std::size_t{rows} * g.row_bytes, &refs);
    for (const auto& ref : refs)
      pages->push_back({i, ref});
  }
}

bool Executor::RestoreSnapshot(Session& session,
                               std::span<const std::uint8_t> payload,
                               SnapshotInfo* info, std::string* error_msg,
                               std::uint32_t next_drafts,
                               std::span<const SnapshotPage> pages) const {
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (payload.size() < sizeof(SnapshotHeader)) {
    AssignError(error_msg, "snapshot payload is truncated");
    return false;
  }
  SnapshotHeader h{};
  std::memcpy(&h, payload.data(), sizeof(h));
  const SnapshotHeader mine = MakeSnapshotHeader(
      config(), session.mtp_enabled_, session, options_.kv_format,
      options_.gdn_state, model_->dspark());
  if (!SameGeometry(h, mine)) {
    AssignError(error_msg, "snapshot was taken with another model geometry");
    return false;
  }
  const std::uint32_t max_blocks =
      h.compress_ratio == 0 ? 0 : h.position / h.compress_ratio;
  if (h.image_count > 256 || h.position == 0 ||
      h.position > session.max_context_ || h.blocks > max_blocks ||
      h.mtp_position > h.position ||
      h.mtp_blocks >
          (h.compress_ratio == 0 ? 0 : h.mtp_position / h.compress_ratio) ||
      (h.compress_ratio != 0 &&
       h.mtp_position - h.mtp_blocks * h.compress_ratio >
           session.index_capacity_) ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  // Without an indexer (qwen35moe) no raw rows wait for pooling.
  if (h.compress_ratio != 0 &&
      h.position - h.blocks * h.compress_ratio > session.index_capacity_) {
    AssignError(error_msg, "unpooled indexer rows exceed the ring capacity");
    return false;
  }
  if (h.payload_bytes != payload.size() ||
      WalkSnapshot(h, nullptr,
                   [](void*, std::uint64_t, std::uint64_t, const char*) {
                     return true;
                   }) != payload.size()) {
    AssignError(error_msg, "snapshot payload size does not match its header");
    return false;
  }
  qwen::vision::RopeLayout layout;
  layout.images.resize(h.image_count);
  layout.explicit_positions.resize(h.explicit_rows);
  layout.explicit_delta = h.explicit_delta;
  const auto grid_bytes =
      layout.images.size() * sizeof(qwen::vision::ImageGrid);
  const auto explicit_bytes =
      layout.explicit_positions.size() * sizeof(std::array<std::int32_t, 3>);
  if (grid_bytes != 0) {
    std::memcpy(layout.images.data(),
                payload.data() + payload.size() - explicit_bytes - grid_bytes,
                grid_bytes);
  }
  if (explicit_bytes != 0) {
    std::memcpy(layout.explicit_positions.data(),
                payload.data() + payload.size() - explicit_bytes,
                explicit_bytes);
  }
  try {
    layout.Validate(session.max_context_);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return false;
  }
  const bool has_layout =
      !layout.images.empty() || !layout.explicit_positions.empty();
  if (has_layout && layout != session.VisionLayout()) {
    AssignError(error_msg,
                "image snapshot layout does not match its prompt attachment");
    return false;
  }
  if (!has_layout)
    session.ConfigureVision(nullptr, nullptr, stream_);
  // The session's queued work targets buffers the copies overwrite.
  if (!Check(hipStreamSynchronize(stream_), "restore drain", error_msg))
    return false;
  // Map the snapshot's still-live prefix pages instead of copying them.
  // Retain them before releasing this session's pages (it may be restoring
  // its own snapshot); the copies then write only private pages.
  std::vector<std::vector<std::uint32_t>> retained(session.growable_.size());
  std::vector<bool> broken(session.growable_.size());
  for (const auto& page : pages) {
    if (page.buffer >= retained.size() || broken[page.buffer])
      continue;
    if (kv_pool_.Retain(page.piece))
      retained[page.buffer].push_back(page.piece.index);
    else
      broken[page.buffer] = true;
  }
  session.ReleaseKv();
  std::vector<std::size_t> shared(session.growable_.size());
  bool committed = true;
  {
    const gufo::hip::TlbBatch batch;  // attached and new pages: one flush
    for (std::size_t i = 0; i < retained.size(); ++i) {
      auto& buffer = session.growable_[i].buffer;
      shared[i] = buffer.Attach(retained[i], nullptr) * kv_pool_.piece_bytes();
    }
    committed = Reserve(session, h.position, error_msg);
  }
  // New pages' zero fills land before the copies.
  if (!kv_pool_.Settle(error_msg) || !committed ||
      !Check(hipStreamSynchronize(stream_), "restore commit", error_msg)) {
    session.Reset();
    return false;
  }
  // Every live state region is overwritten below. Retain only scratch that
  // the restored operation can use; this never allocates new rollback rows.
  // Explicit reset and failed restoration still release all scratch.
  const auto remaining = session.max_context_ - h.position;
  session.TrimRollback(std::min({next_drafts, options_.max_speculative - 1,
                                 remaining ? remaining - 1 : 0}));
  if (WalkSnapshot(h, &session,
                   [&](void* device, std::uint64_t offset, std::uint64_t bytes,
                       const char* what) {
                     std::uint64_t skip = 0;
                     for (std::size_t i = 0; i < shared.size(); ++i) {
                       if (session.growable_[i].buffer.data() == device)
                         skip = std::min<std::uint64_t>(bytes, shared[i]);
                     }
                     return session.CheckCancellation(error_msg) &&
                            (skip == bytes ||
                             Check(
                                 hipMemcpy(static_cast<char*>(device) + skip,
                                           payload.data() + offset + skip,
                                           bytes - skip, hipMemcpyHostToDevice),
                                 what, error_msg));
                   }) == 0) {
    session.Reset();
    return false;
  }
  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.blocks_ = h.blocks;
  session.mtp_.position = h.mtp_position;
  session.mtp_.blocks = h.mtp_blocks;
  session.dspark_.position = h.dspark_layers != 0 ? h.position : 0;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  session.ngram_.prev = h.ngram_prev;
  if (info != nullptr) {
    info->position = h.position;
    info->hidden_rows = h.hidden_rows;
  }
  return true;
}

bool Executor::GreedyMtpPredictions(std::span<ArgmaxCandidate> predictions,
                                    std::string* error_msg) const {
  const auto rows = predictions.size();
  if (rows == 0 || rows > options_.max_logit_rows || rows > kArgmaxParts ||
      s_.mtp_ids == nullptr) {
    AssignError(error_msg, "invalid greedy MTP verification request");
    return false;
  }
  if (selected_logits_ != nullptr && batch_argmax_rows_ != 0) {
    // ForwardBatch already reduced every row with this argmax.
    if (selected_offset_ + rows > batch_argmax_rows_) {
      AssignError(error_msg, "greedy MTP rows outside the decode batch");
      return false;
    }
    std::copy_n(batch_argmax_host_ + selected_offset_, rows,
                predictions.data());
    return true;
  }
  // Proposal selection has finished. Its two ID buffers and argmax scratch
  // can be reused until the next draft head overwrites them.
  const auto vocab = config().vocab_size;
  gufo::hip::LaunchBatchedGPUArgmax(
      VerificationLogits(), s_.mtp_ids, rows, vocab,
      {reinterpret_cast<float*>(s_.mtp_scratch_ids),
       MtpCandidateWorkspaceSize(vocab)},
      stream_);
  GatherArgmaxCandidates(VerificationLogits(), s_.mtp_ids, s_.mtp_argmax,
                         static_cast<std::uint32_t>(rows), vocab, stream_);
  return Check(hipMemcpyAsync(predictions.data(), s_.mtp_argmax,
                              predictions.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "greedy MTP predictions download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "greedy MTP verification",
               error_msg);
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, MtpOutput output,
                          std::string* error_msg,
                          const float* hidden_source) const {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (session.owner_ != this ||
      std::any_of(tokens.begin(), tokens.end(), [&](auto t) {
        return t < 0 || static_cast<std::uint32_t>(t) >= config().vocab_size;
      })) {
    AssignError(error_msg, "invalid MTP session or token");
    return false;
  }
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!session.mtp_enabled_) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }

  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n >
           (hidden_source ? options_.max_batch : options_.max_speculative))) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  if (output.trace != nullptr &&
      !output.trace->Valid(config().hidden_size, config().HcDim())) {
    AssignError(error_msg, "invalid MTP trace destinations");
    return false;
  }
  const std::uint32_t pos = session.mtp_.position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  if (!RequireResident(session, error_msg))
    return false;
  if (!Reserve(session, pos + n, error_msg))
    return false;
  ++session.mutation_epoch_;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  control_host_->mtp_blocks = session.mtp_.blocks;
  const auto& c = config();
  const bool sparse = pos + n > c.indexer_top_k;
  const auto complete = (pos + n) / c.compress_ratio;
  const auto pool = sparse ? complete - session.mtp_.blocks : 0;
  const auto graph_pool = n / c.compress_ratio + 1;
  const bool graph = output.trace == nullptr && hidden_source == nullptr &&
                     router_dump_ == nullptr && n <= kVecBatch &&
                     pool <= graph_pool &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43) |
      (std::uint64_t{sparse} << 44) | (std::uint64_t{DraftCompact()} << 45) |
      (std::uint64_t{SteersMtp(session.adapter_) ? session.adapter_ : 0U}
       << 48);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg,
                   graph ? graph_pool : pool, hidden_source, output.trace);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = DraftToken(*mtp_token_host_);
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
    MapDraftCandidates(*output.candidates);
  }
  session.mtp_.position = pos + n;
  if (sparse)
    session.mtp_.blocks = complete;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       std::uint32_t pool_grid, const float* hidden_source,
                       MtpTrace* trace) const {
  const Config& c = config();
  const DeviceLayer& l = model_->mtp();
  const std::uint32_t hc_dim = c.HcDim();
  const std::size_t final_row = static_cast<std::size_t>(n - 1) * hc_dim;
  const auto copy_trace = [&](const float* source,
                              std::span<float> destination) {
    return destination.empty() ||
           Check(hipMemcpyAsync(destination.data(), source,
                                destination.size_bytes(), hipMemcpyDeviceToHost,
                                stream_),
                 "MTP trace download", error_msg);
  };
  // An adapter that projects the embedding or the draft block's outputs
  // steers this block's rows as its folded weights would.
  const bool steer = SteersMtp(session.adapter_);
  const SteerScope steer_scope(*this, steer, steer_writer::kMtp);
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg) ||
      (steer && !Check(hipMemsetAsync(s_.steer, session.adapter_, n, stream_),
                       "adapter rows", error_msg))) {
    return false;
  }
  EmbedTokens(model_->token_embd().data,
              static_cast<WeightType>(model_->token_embd().type), s_.tokens,
              s_.mtp_embd, n, c.hidden_size, 1, stream_);
  SteerWriter(s_.mtp_embd, n, 1, steer_writer::kEmbedding);
  // MTP embeds shifted token IDs. Visual information arrives in the trunk
  // hidden stream; image embeddings belong only to the target input.
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  MtpHidden(hidden_source ? hidden_source : session.mtp_.target_hidden,
            session.mtp_.h, &session.control_->hidden_row, s_.mtp_h, n, hc_dim,
            stream_);
  // Unlike the HC mixers, this normalization spans the complete HC * H row.
  RmsNormRows(s_.mtp_h, l.nextn_hnorm.f32(), s_.mtp_h, n, hc_dim, 1, c.rms_eps,
              stream_);
  if (trace && !copy_trace(s_.mtp_h + final_row, trace->normalized_hidden))
    return false;
  if (!Dense(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n, error_msg) ||
      !Dense(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
             error_msg)) {
    return false;
  }
  MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, n, c.hidden_size, c.hc_count,
                  stream_);
  if (trace && !copy_trace(s_.mtp_res + final_row, trace->fused))
    return false;
  Session::AttentionState attn;
  attn.rope = session.vision_input_.rope();
  attn.kv = session.mtp_.kv;
  attn.index_k = session.mtp_.index_k;
  attn.block_k = session.mtp_.block_k;
  // The draft block's attention runs at its own position.
  // Catch-up exports KV for every row but only carries its final residual.
  // Preserve that row's original query tile; earlier attention results
  // never contribute to the carried state.
  const bool last_only =
      !token && !candidates && n > 32 && control_host_->hidden_row >= 0;
  // Keep the final 128-column tile (and its predecessor for short tails).
  // At least 96 rows retain the wide mixer/projection arithmetic. Attention
  // still writes every KV/indexer row before the scratch view is narrowed.
  const auto skipped = last_only && n >= 224 ? (n - 96) / 128 * 128 : 0U;
  if (!HcMix(l.hc_attn, s_.mtp_res, false, s_.mixed, s_.inject, n, error_msg) ||
      !Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, &session.control_->mtp_blocks,
                 pos, pool_grid, session.max_context_,
                 pos + n > c.indexer_top_k, error_msg, last_only, false,
                 skipped == 0)) {
    return false;
  }
  struct RestoreScratch {
    const Executor* executor;
    Scratch scratch;
    bool changed;
    ~RestoreScratch() {
      if (changed)
        executor->UseScratch(scratch);
    }
  } restore{this, s_, skipped != 0};
  const auto tail_rows = n - skipped;
  const auto tail_last = static_cast<std::size_t>(tail_rows - 1) * hc_dim;
  if (skipped != 0) {
    auto tail = RowScratch(s_, skipped);
    // Keep the producer's inject stride: it may use vectorized partials or
    // a single quantized projection rather than RowScratch's scalar layout.
    tail.inject = s_.inject + static_cast<std::size_t>(skipped) * c.hc_count *
                                  inject_parts_;
    UseScratch(tail);
    if (!Dense(l.attn_out, s_.ctx, s_.block_out, tail_rows, error_msg))
      return false;
  }
  Combine(s_.mtp_res, l.hc_ffn.norm.f32(), tail_rows);
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->attention))
    return false;
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!HcMix(l.hc_ffn, s_.mtp_res, true, s_.mixed, s_.inject, tail_rows,
             error_msg) ||
      (trace && !copy_trace(s_.mixed + static_cast<std::size_t>(tail_rows - 1) *
                                           c.hidden_size,
                            trace->ffn_input)) ||
      !Moe(l, s_.mixed, s_.block_out, tail_rows, error_msg, last_only)) {
    return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace &&
      !copy_trace(s_.block_out +
                      static_cast<std::size_t>(tail_rows - 1) * c.hidden_size,
                  trace->ffn_output))
    return false;
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->hidden))
    return false;
  if (trace && !trace->head.empty()) {
    if (!HcMix(l.nextn_head, s_.mtp_res + tail_last, false, s_.mixed, nullptr,
               1, error_msg) ||
        !copy_trace(s_.mixed, trace->head))
      return false;
  }
  const float* last = s_.mtp_res + tail_last;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, hc_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(l.nextn_head, last, token, candidates, error_msg);
}

bool Executor::DsparkFlush(std::span<Session* const> sessions,
                           std::string* error_msg) const {
  dspark_defer_ = false;
  const DeviceDspark* dp = model_->dspark();
  if (dp == nullptr)
    return true;
  const DeviceDspark& d = *dp;
  const Config& c = config();
  // Sessions whose drafter context continues exactly at the pending rows; a
  // gap (a failed step, a snapshot without drafter rows) leaves the drafter
  // idle until the next Reset.
  std::vector<Session*> live;
  std::uint32_t rows = 0;
  for (Session* s : sessions) {
    if (s == nullptr || s->dspark_.pending_rows == 0)
      continue;
    auto& p = s->dspark_;
    if (p.position == p.pending_pos && live.size() < kMaxBatchSessions) {
      live.push_back(s);
      rows = std::max(rows, p.pending_row + p.pending_rows);
    } else {
      p.pending_rows = 0;
    }
  }
  if (live.empty())
    return true;
  const std::uint32_t row_dim = d.kv_heads * d.head_dim;
  const auto blocks = static_cast<std::uint32_t>(d.blocks.size());
  // One fc pass and one stacked K/V GEMV over every feature row up to the
  // last pending one (rejected verify rows ride along, never stored), then
  // one launch normalizes, rotates and stores every session's rows.
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (!Dense(d.fc, ds_.features, ds_.g, rows, error_msg))
    return false;
  RmsNormRows(ds_.g, d.enc_norm.f32(), ds_.g, rows, c.hidden_size, 1, d.rms_eps,
              stream_);
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (!Dense(d.kv_all, ds_.g, ds_.kv, rows, error_msg))
    return false;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  // Descriptor slots rotate; a slot is rewritten only after the launch
  // that read it completed.
  const auto slot = ds_.store_slot;
  ds_.store_slot = (slot + 1) % DsparkScratch::kStoreSlots;
  if (!Check(hipEventSynchronize(ds_.store_done[slot]), "DSpark descriptors",
             error_msg))
    return false;
  auto* host = ds_.store_host + std::size_t{slot} * kMaxBatchSessions;
  auto* device = ds_.store_device + std::size_t{slot} * kMaxBatchSessions;
  std::uint32_t max_rows = 0;
  for (std::size_t k = 0; k < live.size(); ++k) {
    const auto& p = live[k]->dspark_;
    host[k] = {
        .row = p.pending_row, .rows = p.pending_rows, .pos = p.pending_pos};
    for (std::uint32_t b = 0; b < blocks; ++b) {
      host[k].k[b] = static_cast<__half*>(p.kv[b].k);
      host[k].v[b] = static_cast<__half*>(p.kv[b].v);
    }
    max_rows = std::max(max_rows, p.pending_rows);
  }
  if (!Check(hipMemcpyAsync(device, host, live.size() * sizeof(*host),
                            hipMemcpyHostToDevice, stream_),
             "DSpark descriptors", error_msg))
    return false;
  DsparkStoreKv(ds_.kv, 2 * blocks * row_dim, d.k_norms.f32(), device,
                static_cast<std::uint32_t>(live.size()), max_rows, blocks,
                d.kv_heads, d.rope_theta, d.rms_eps, stream_);
  if (!Check(hipEventRecord(ds_.store_done[slot], stream_), "DSpark store",
             error_msg))
    return false;
  for (Session* s : live) {
    s->dspark_.position += s->dspark_.pending_rows;
    s->dspark_.pending_rows = 0;
  }
  return Check(hipGetLastError(), "DSpark injection", error_msg);
}

bool Executor::DsparkDraft(std::span<const DsparkItem> items,
                           std::string* error_msg) const {
  const DeviceDspark* dp = model_->dspark();
  const Config& c = config();
  // Rows of every drafting item, stacked; chain rows add each anchor.
  std::uint32_t rows = 0;
  std::uint32_t links = 0;
  std::vector<std::size_t> live;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& it = items[i];
    const auto n = static_cast<std::uint32_t>(it.tokens.size());
    if (dp == nullptr || it.session == nullptr ||
        !it.session->dspark_enabled_ || it.session->owner_ != this || n == 0 ||
        n > dp->block_size || it.confidence.size() != n || it.anchor < 0 ||
        static_cast<std::uint32_t>(it.anchor) >= c.vocab_size ||
        it.session->spec_tokens_ != 0 || rows + n > kDsparkMaxRows ||
        live.size() >= kDsparkMaxItems) {
      AssignError(error_msg, "invalid DSpark draft request");
      return false;
    }
    std::fill(it.tokens.begin(), it.tokens.end(), -1);
    std::fill(it.confidence.begin(), it.confidence.end(), 0.0F);
    // Without the complete context the block would read stale rows. Image
    // prompts (planned M-RoPE positions) decode without drafts.
    auto& s = *it.session;
    const auto& layout = s.VisionLayout();
    if (s.Cancelled() || s.dspark_.position != s.position_ ||
        s.position_ + n > s.max_context_ || !layout.images.empty() ||
        !layout.explicit_positions.empty())
      continue;
    // Block rows past the context are scratch: no persistent state moves.
    if (!Reserve(s, s.position_ + n, error_msg))
      return false;
    live.push_back(i);
    rows += n;
    links += n + 1;
  }
  if (live.empty())
    return true;
  const DeviceDspark& d = *dp;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t vocab = c.vocab_size;
  const std::uint32_t row_dim = d.kv_heads * d.head_dim;
  const std::uint32_t ffn = d.blocks.front().gate.rows;
  const std::uint32_t conf_dim = hidden + d.rank;
  const auto fresh = [&] {
    q8t_src_ = nullptr;
    half_src_ = nullptr;
  };
  // Per item: block [anchor, MASK...] at pos0.. (pos[1 + 2k]) and the last
  // block key (pos[2 + 2k]): every row attends to the whole context and
  // the whole block. Chains hold each anchor, then its selections.
  std::vector<std::uint32_t> first(live.size()), chain(live.size());
  std::uint32_t row = 0, link = 0;
  for (std::size_t k = 0; k < live.size(); ++k) {
    const auto& it = items[live[k]];
    const auto n = static_cast<std::uint32_t>(it.tokens.size());
    const auto pos0 = it.session->position_;
    first[k] = row;
    chain[k] = link;
    ds_.pos_host[1 + 2 * k] = pos0;
    ds_.pos_host[2 + 2 * k] = pos0 + n - 1;
    ds_.chain_host[link] = it.anchor;
    for (std::uint32_t r = 0; r < n; ++r)
      ds_.block_host[row + r] = r == 0 ? it.anchor : d.mask_token;
    row += n;
    link += n + 1;
  }
  if (!Check(
          hipMemcpyAsync(ds_.block, ds_.block_host, rows * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "DSpark block upload", error_msg) ||
      !Check(hipMemcpyAsync(ds_.chain, ds_.chain_host,
                            links * sizeof(std::int32_t), hipMemcpyHostToDevice,
                            stream_),
             "DSpark chain upload", error_msg) ||
      !Check(hipMemcpyAsync(ds_.pos + 1, ds_.pos_host + 1,
                            2 * live.size() * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream_),
             "DSpark position upload", error_msg))
    return false;
  const DeviceTensor& embd = model_->token_embd();
  EmbedTokens(embd.data, static_cast<WeightType>(embd.type), ds_.block, ds_.x,
              rows, hidden, 1, stream_);
  for (std::size_t b = 0; b < d.blocks.size(); ++b) {
    const auto& blk = d.blocks[b];
    RmsNormRows(ds_.x, blk.attn_norm.f32(), ds_.xn, rows, hidden, 1, d.rms_eps,
                stream_);
    fresh();
    if (!Dense(blk.q, ds_.xn, ds_.q, rows, error_msg) ||
        !Dense(blk.k, ds_.xn, ds_.k, rows, error_msg) ||
        !Dense(blk.v, ds_.xn, ds_.v, rows, error_msg))
      return false;
    RmsNormRows(ds_.q, blk.q_norm.f32(), ds_.q, rows * d.heads, d.head_dim, 1,
                d.rms_eps, stream_);
    RmsNormRows(ds_.k, blk.k_norm.f32(), ds_.k, rows * d.kv_heads, d.head_dim,
                1, d.rms_eps, stream_);
    // One single-row attention item per block row, all at the block's last
    // key: the kernel's causal limit then admits the whole block.
    AttentionBatchItem* attn = ds_.attn_host + b * kDsparkMaxRows;
    for (std::size_t k = 0; k < live.size(); ++k) {
      const auto& it = items[live[k]];
      const auto n = static_cast<std::uint32_t>(it.tokens.size());
      const auto& kv = it.session->dspark_.kv[b];
      const auto* pos = ds_.pos + 1 + 2 * k;
      Rope(ds_.q + std::size_t{first[k]} * hidden, n, d.heads, d.head_dim,
           d.head_dim, pos, d.rope_theta, stream_);
      Rope(ds_.k + std::size_t{first[k]} * row_dim, n, d.kv_heads, d.head_dim,
           d.head_dim, pos, d.rope_theta, stream_);
      StoreKv(ds_.k + std::size_t{first[k]} * row_dim,
              static_cast<__half*>(kv.k), n, row_dim, pos, stream_);
      StoreKv(ds_.v + std::size_t{first[k]} * row_dim,
              static_cast<__half*>(kv.v), n, row_dim, pos, stream_);
      for (std::uint32_t r = 0; r < n; ++r) {
        const auto t = first[k] + r;
        attn[t] =
            AttentionBatchItem{.kv = kv,
                               .position = ds_.pos + 2 + 2 * k,
                               .row = t,
                               .rows = 1,
                               .splits = AttentionSplits(1),
                               .partial = t * d.heads * AttentionSplits(1)};
      }
    }
    AttentionBatch(ds_.q, KvCacheFormat::kF16,
                   ds_.attn_device + b * kDsparkMaxRows, rows, 1, nullptr, 0,
                   ds_.ctx, ds_.partials, d.heads, d.kv_heads, d.head_dim, 1,
                   stream_);
    fresh();
    if (!Dense(blk.out, ds_.ctx, ds_.o, rows, error_msg))
      return false;
    MtpAddEmbedding(ds_.o, ds_.x, rows, hidden, 1, stream_);
    RmsNormRows(ds_.x, blk.ffn_norm.f32(), ds_.xn, rows, hidden, 1, d.rms_eps,
                stream_);
    fresh();
    if (!Dense(blk.gate, ds_.xn, ds_.gate, rows, error_msg) ||
        !Dense(blk.up, ds_.xn, ds_.up, rows, error_msg))
      return false;
    Swiglu(ds_.gate, ds_.up, std::size_t{rows} * ffn, stream_);
    fresh();
    if (!Dense(blk.down, ds_.gate, ds_.o, rows, error_msg))
      return false;
    MtpAddEmbedding(ds_.o, ds_.x, rows, hidden, 1, stream_);
  }
  // Base logits from the target head over the normalized block rows; the
  // confidence head reads the same rows next to W1[previous token].
  RmsNormRows(ds_.x, d.out_norm.f32(), ds_.xn, rows, hidden, 1, d.rms_eps,
              stream_);
  fresh();
  if (!Dense(model_->output(), ds_.xn, ds_.logits, rows, error_msg) ||
      !Check(hipMemcpy2DAsync(ds_.conf_in, conf_dim * sizeof(float), ds_.xn,
                              hidden * sizeof(float), hidden * sizeof(float),
                              rows, hipMemcpyDeviceToDevice, stream_),
             "DSpark confidence input", error_msg))
    return false;
  // Markov chain: row i adds W2 . W1[previous] and selects greedily; the
  // anchor precedes row 0. Selections stay on the device.
  // ponytail: the bias is scored on each row's top-256 base logits, not the
  // whole vocabulary (one W2 pass per row costs more than the LM head);
  // probe acceptance stayed within a few percent of the full chain.
  for (std::size_t k = 0; k < live.size(); ++k) {
    const auto n = static_cast<std::uint32_t>(items[live[k]].tokens.size());
    for (std::uint32_t i = 0; i < n; ++i) {
      const auto t = first[k] + i;
      DsparkTopCandidates(ds_.logits + std::size_t{t} * vocab, ds_.cand_ids,
                          ds_.cand_scratch, ds_.cand_scores, vocab, stream_);
      DsparkMarkovStep(
          d.markov_w1.data, d.markov_w2.data, d.rank, ds_.cand_ids,
          ds_.cand_scores, std::min<std::uint32_t>(vocab, kDsparkCandidates),
          ds_.chain + chain[k], i,
          ds_.conf_in + std::size_t{t} * conf_dim + hidden, stream_);
    }
  }
  // Eight-row groups keep the one-row head on the small GEMV.
  for (std::uint32_t r = 0; r < rows; r += 8) {
    if (!Dense(d.conf_w, ds_.conf_in + std::size_t{r} * conf_dim, ds_.conf + r,
               std::min(8U, rows - r), error_msg))
      return false;
  }
  fresh();
  if (!Check(hipMemcpyAsync(ds_.chain_host, ds_.chain,
                            links * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                            stream_),
             "DSpark draft download", error_msg) ||
      !Check(hipMemcpyAsync(ds_.conf_host, ds_.conf, rows * sizeof(float),
                            hipMemcpyDeviceToHost, stream_),
             "DSpark confidence download", error_msg) ||
      !Check(hipStreamSynchronize(stream_), "DSpark draft", error_msg))
    return false;
  for (std::size_t k = 0; k < live.size(); ++k) {
    const auto& it = items[live[k]];
    for (std::size_t i = 0; i < it.tokens.size(); ++i) {
      it.tokens[i] = ds_.chain_host[chain[k] + 1 + i];
      it.confidence[i] =
          1.0F /
          (1.0F + std::exp(-(ds_.conf_host[first[k] + i] + d.conf_bias)));
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
