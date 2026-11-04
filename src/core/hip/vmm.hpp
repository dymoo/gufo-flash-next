#ifndef GUFO_CORE_HIP_VMM_HPP_
#define GUFO_CORE_HIP_VMM_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gufo::hip {

namespace vmm_detail {
inline void Assign(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}
inline bool Ok(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess)
    return true;
  Assign(error, std::string(what) + " failed: " + hipGetErrorString(status));
  return false;
}
/// ROCm 7.2.3 on gfx1151 (Linux 7.0) can leave a stale GPU TLB entry for an
/// address re-mapped to another physical piece: kernels keep reaching the old
/// piece while copies and fills reach the new one, so pooled pages alias
/// across sessions (tests/core/hip_vmm_test.hip, re-mapped slots). The KFD
/// map behind a real device allocation flushes the TLB.
// ponytail: 2 MiB is the smallest hipMalloc ROCm does not suballocate; drop
// this once hipMemMap flushes by itself (that test fails without it).
inline bool FlushTlb(std::string* error) {
  void* p = nullptr;
  return Ok(hipMalloc(&p, std::size_t{2} << 20), "TLB flush hipMalloc",
            error) &&
         Ok(hipFree(p), "TLB flush hipFree", error);
}
/// Open TlbBatch scopes on this thread; see TlbBatch.
inline thread_local int tlb_batch_depth = 0;
inline hipMemAllocationProp Prop() {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  int device = 0;
  (void)hipGetDevice(&device);
  prop.location.id = device;
  return prop;
}
}  // namespace vmm_detail

/// Defers PiecePool::Settle for every Commit/Attach on this thread while in
/// scope; the caller then calls Settle once. Each flush is a device
/// allocation and free (the free synchronizes the device), so a session's
/// first growth flushes once instead of once per cache buffer. Pages mapped
/// inside the scope must not be used until that Settle.
class TlbBatch {
public:
  TlbBatch() noexcept { ++vmm_detail::tlb_batch_depth; }
  ~TlbBatch() { --vmm_detail::tlb_batch_depth; }
  TlbBatch(const TlbBatch&) = delete;
  TlbBatch& operator=(const TlbBatch&) = delete;
};

/// A weak reference to a pool piece: valid while the piece is still mapped
/// somewhere and has not been recycled since the reference was taken.
struct PieceRef {
  std::uint32_t index{0};
  std::uint64_t generation{0};
};

/// Fixed-size, reference-counted physical pieces shared by growable
/// buffers; the limit caps the bytes of pieces in use. A piece mapped into
/// several buffers (a shared prompt prefix) counts once. ROCm 7.2 returns a
/// mapped piece's pages only when every address reservation it was mapped
/// into is freed (tools/bench/vmm_probe.hip), so unreferenced pieces are
/// recycled rather than handed back: the footprint is the high-water mark.
class PiecePool {
public:
  explicit PiecePool(std::size_t piece_bytes) : piece_(piece_bytes) {}
  PiecePool(const PiecePool&) = delete;
  PiecePool& operator=(const PiecePool&) = delete;
  ~PiecePool() {
    for (const auto& piece : pieces_)
      (void)hipMemRelease(piece.handle);
  }

  void SetLimit(std::size_t bytes) {
    const std::lock_guard lock(mutex_);
    limit_ = bytes;
  }
  /// Called (without the pool lock) when the limit is reached; returns
  /// whether it released anything, e.g. by paging an idle holder out.
  void SetReclaim(std::function<bool()> reclaim) {
    reclaim_ = std::move(reclaim);
  }
  [[nodiscard]] std::size_t piece_bytes() const noexcept { return piece_; }
  /// Records a new mapping at `at`; `stale` if its address may still have a
  /// GPU TLB entry for another piece. `zero` bytes at `at` are zero-filled
  /// on `stream` after the flush: fills and kernels issued before it reach
  /// the stale piece, so nothing may run on the mapping until Settle.
  void Mapped(void* at, bool stale, std::size_t zero, hipStream_t stream) {
    const std::lock_guard lock(mutex_);
    stale_ = stale_ || stale;
    if (zero != 0)
      fills_.push_back({static_cast<char*>(at), zero, stream});
  }
  /// Drops pending fills in [begin, end) (pages unmapped before Settle).
  void Unmapped(const void* begin, const void* end) {
    const std::lock_guard lock(mutex_);
    std::erase_if(fills_, [&](const Fill& f) {
      return f.at >= static_cast<const char*>(begin) &&
             f.at < static_cast<const char*>(end);
    });
  }
  /// Flushes the GPU TLB after new mappings, then issues their zero fills;
  /// deferred while a TlbBatch is open on this thread.
  [[nodiscard]] bool Settle(std::string* error) {
    if (vmm_detail::tlb_batch_depth > 0)
      return true;
    std::vector<Fill> fills;
    bool stale = false;
    {
      const std::lock_guard lock(mutex_);
      std::swap(stale, stale_);
      fills.swap(fills_);
    }
    if (stale)
      ++flushes_;
    bool ok = !stale || vmm_detail::FlushTlb(error);
    for (const auto& f : fills)
      ok = ok && vmm_detail::Ok(hipMemsetAsync(f.at, 0, f.bytes, f.stream),
                                "hipMemsetAsync", error);
    if (!ok) {
      const std::lock_guard lock(mutex_);
      stale_ = stale_ || stale;
      fills_.insert(fills_.end(), fills.begin(), fills.end());
    }
    return ok;
  }
  /// TLB flushes so far (diagnostics and tests).
  [[nodiscard]] std::uint64_t flushes() const noexcept { return flushes_; }
  [[nodiscard]] std::size_t limit() const {
    const std::lock_guard lock(mutex_);
    return limit_;
  }
  /// Bytes of referenced pieces (shared ones once, idle ones excluded).
  [[nodiscard]] std::size_t used() const {
    const std::lock_guard lock(mutex_);
    return in_use_ * piece_;
  }

  /// An unmapped piece with one reference: recycled, or new within the
  /// limit.
  [[nodiscard]] bool Acquire(std::uint32_t* index, std::string* error) {
    std::unique_lock lock(mutex_);
    while (in_use_ * piece_ > limit_ || piece_ > limit_ - in_use_ * piece_) {
      lock.unlock();
      const bool reclaimed = reclaim_ && reclaim_();
      lock.lock();
      if (!reclaimed) {
        vmm_detail::Assign(error, "device memory pool exhausted (" +
                                      std::to_string((in_use_ * piece_) >> 20) +
                                      " of " + std::to_string(limit_ >> 20) +
                                      " MiB in use)");
        return false;
      }
    }
    if (idle_.empty()) {
      hipMemGenericAllocationHandle_t handle{};
      const auto prop = vmm_detail::Prop();
      if (!vmm_detail::Ok(hipMemCreate(&handle, piece_, &prop, 0),
                          "hipMemCreate", error))
        return false;
      idle_.push_back(static_cast<std::uint32_t>(pieces_.size()));
      pieces_.push_back({handle, 0, 0});
    }
    *index = idle_.back();
    idle_.pop_back();
    pieces_[*index].refs = 1;
    ++in_use_;
    return true;
  }
  /// Adds a reference if `ref` still names live contents.
  [[nodiscard]] bool Retain(PieceRef ref) {
    const std::lock_guard lock(mutex_);
    if (ref.index >= pieces_.size())
      return false;
    auto& piece = pieces_[ref.index];
    if (piece.refs == 0 || piece.generation != ref.generation)
      return false;
    ++piece.refs;
    return true;
  }
  /// Drops a reference; the last one recycles the piece.
  void Release(std::uint32_t index) {
    const std::lock_guard lock(mutex_);
    auto& piece = pieces_[index];
    if (--piece.refs == 0) {
      ++piece.generation;
      --in_use_;
      idle_.push_back(index);
    }
  }
  [[nodiscard]] std::uint32_t Refs(std::uint32_t index) const {
    const std::lock_guard lock(mutex_);
    return pieces_[index].refs;
  }
  [[nodiscard]] PieceRef Ref(std::uint32_t index) const {
    const std::lock_guard lock(mutex_);
    return {index, pieces_[index].generation};
  }
  [[nodiscard]] hipMemGenericAllocationHandle_t Handle(
      std::uint32_t index) const {
    const std::lock_guard lock(mutex_);
    return pieces_[index].handle;
  }

private:
  struct Piece {
    hipMemGenericAllocationHandle_t handle;
    std::uint32_t refs;
    std::uint64_t generation;
  };
  const std::size_t piece_;
  mutable std::mutex mutex_;
  std::size_t limit_{std::numeric_limits<std::size_t>::max()};
  std::size_t in_use_{0};
  std::vector<Piece> pieces_;
  std::vector<std::uint32_t> idle_;
  std::function<bool()> reclaim_;
  struct Fill {
    char* at;
    std::size_t bytes;
    hipStream_t stream;
  };
  bool stale_{false};
  std::vector<Fill> fills_;
  std::atomic<std::uint64_t> flushes_{0};
};

/// Device memory at a fixed virtual address whose physical pages are
/// committed on demand from a PiecePool (HIP VMM). Pointers into the range,
/// including those captured by graph kernel nodes, stay valid while the
/// committed prefix grows and shrinks. One reservation covers the range, so
/// copies may span pieces (copies across reservations are rejected).
///
/// Leading pieces may be shared with other buffers (Attach). Their owners
/// must treat them as read-only: callers only share pieces wholly below
/// every holder's append position, and replace a buffer's pages (Trim, then
/// Attach/Commit) before rewriting rows below it.
class GrowableBuffer {
public:
  GrowableBuffer() = default;
  GrowableBuffer(const GrowableBuffer&) = delete;
  GrowableBuffer& operator=(const GrowableBuffer&) = delete;
  GrowableBuffer(GrowableBuffer&& other) noexcept { *this = std::move(other); }
  GrowableBuffer& operator=(GrowableBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      va_ = std::exchange(other.va_, nullptr);
      capacity_ = std::exchange(other.capacity_, 0);
      pool_ = std::exchange(other.pool_, nullptr);
      pieces_ = std::move(other.pieces_);
      slots_ = std::move(other.slots_);
    }
    return *this;
  }
  ~GrowableBuffer() { Free(); }

  /// Reserves address space for `capacity` bytes; commits nothing.
  [[nodiscard]] bool Reserve(std::size_t capacity, PiecePool& pool,
                             std::string* error) {
    Free();
    const std::size_t piece = pool.piece_bytes();
    const std::size_t bytes =
        (std::max<std::size_t>(capacity, 1) + piece - 1) / piece * piece;
    if (!vmm_detail::Ok(hipMemAddressReserve(&va_, bytes, 0, nullptr, 0),
                        "hipMemAddressReserve", error)) {
      va_ = nullptr;
      return false;
    }
    capacity_ = bytes;
    pool_ = &pool;
    slots_.assign(bytes / piece, kNoPiece);
    // The range may reuse addresses a freed reservation mapped: flush once
    // before its first use.
    pool.Mapped(va_, true, 0, nullptr);
    return true;
  }

  /// Backs at least [0, bytes) and zero-fills new pages on `stream`, which
  /// must not be capturing: after the TLB flush, at the end of this call or
  /// of the open TlbBatch. On failure (pool or device memory exhausted) the
  /// committed prefix and its contents are unchanged.
  [[nodiscard]] bool Commit(std::size_t bytes, hipStream_t stream,
                            std::string* error) {
    if (pool_ == nullptr)
      return bytes == 0;
    const std::size_t want =
        (std::min(bytes, capacity_) + piece() - 1) / piece();
    bool ok = true;
    while (ok && pieces_.size() < want) {
      std::uint32_t index = 0;
      ok = pool_->Acquire(&index, error) && Map(index, error, true, stream);
    }
    // Pieces mapped before a failure stay committed, so settle them too.
    return pool_->Settle(error) && ok;
  }

  /// Maps pieces the caller already Retained, in order, onto an empty
  /// buffer; returns how many were mapped (the rest are released).
  [[nodiscard]] std::size_t Attach(std::span<const std::uint32_t> retained,
                                   std::string* error) {
    bool ok = pool_ != nullptr && pieces_.empty();
    std::size_t mapped = 0;
    for (const auto index : retained) {
      // A failed Map releases its piece; later ones would leave a hole.
      if (ok && (pieces_.size() + 1) * piece() <= capacity_) {
        ok = Map(index, error, false, nullptr);
        mapped += ok ? 1 : 0;
      } else {
        ok = false;
        if (pool_ != nullptr)
          pool_->Release(index);
      }
    }
    if (pool_ != nullptr && !pool_->Settle(error)) {
      Trim(0);
      return 0;
    }
    return mapped;
  }

  /// Weak references to the pieces wholly inside [0, bytes).
  void Share(std::size_t bytes, std::vector<PieceRef>* out) const {
    if (pool_ == nullptr)
      return;
    const std::size_t n = std::min(bytes / piece(), pieces_.size());
    for (std::size_t i = 0; i < n; ++i)
      out->push_back(pool_->Ref(pieces_[i]));
  }

  /// Bytes of the leading pieces other buffers also map.
  [[nodiscard]] std::size_t SharedPrefixBytes() const {
    std::size_t n = 0;
    while (n < pieces_.size() && pool_->Refs(pieces_[n]) > 1)
      ++n;
    return n * (pool_ == nullptr ? 0 : piece());
  }

  /// Releases the pieces wholly beyond [0, bytes). No pending GPU work may
  /// still access them.
  void Trim(std::size_t bytes) noexcept {
    if (pool_ == nullptr)
      return;
    const std::size_t keep = (bytes + piece() - 1) / piece();
    if (pieces_.size() > keep)
      pool_->Unmapped(Address(keep), Address(pieces_.size()));
    while (pieces_.size() > keep) {
      (void)hipMemUnmap(Address(pieces_.size() - 1), piece());
      pool_->Release(pieces_.back());
      pieces_.pop_back();
    }
  }

  [[nodiscard]] void* data() const noexcept { return va_; }
  [[nodiscard]] std::size_t committed() const noexcept {
    return pool_ == nullptr ? 0 : pieces_.size() * piece();
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
  [[nodiscard]] std::size_t piece() const noexcept {
    return pool_->piece_bytes();
  }
  [[nodiscard]] char* Address(std::size_t piece_index) const noexcept {
    return static_cast<char*>(va_) + piece_index * piece();
  }
  /// Maps a referenced piece after the committed prefix; on failure the
  /// reference is released. With `zero`, Settle zero-fills it on `stream`.
  bool Map(std::uint32_t index, std::string* error, bool zero,
           hipStream_t stream) {
    char* at = Address(pieces_.size());
    hipMemAccessDesc access{};
    access.location = vmm_detail::Prop().location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    if (!vmm_detail::Ok(hipMemMap(at, piece(), 0, pool_->Handle(index), 0),
                        "hipMemMap", error)) {
      pool_->Release(index);
      return false;
    }
    if (!vmm_detail::Ok(hipMemSetAccess(at, piece(), &access, 1),
                        "hipMemSetAccess", error)) {
      (void)hipMemUnmap(at, piece());
      pool_->Release(index);
      return false;
    }
    // A slot that held another piece is stale. First maps are not: the
    // flush after Reserve (the flush is device-wide) cleared any entries a
    // freed reservation left at these addresses.
    auto& last = slots_[pieces_.size()];
    pool_->Mapped(at, last != kNoPiece && last != index, zero ? piece() : 0,
                  stream);
    last = index;
    pieces_.push_back(index);
    return true;
  }
  void Free() noexcept {
    if (va_ == nullptr)
      return;
    Trim(0);
    (void)hipMemAddressFree(va_, capacity_);
    va_ = nullptr;
    capacity_ = 0;
    pool_ = nullptr;
  }

  void* va_{nullptr};
  std::size_t capacity_{0};
  PiecePool* pool_{nullptr};
  std::vector<std::uint32_t> pieces_;
  static constexpr std::uint32_t kNoPiece =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> slots_;  ///< piece last mapped at each slot
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_VMM_HPP_
