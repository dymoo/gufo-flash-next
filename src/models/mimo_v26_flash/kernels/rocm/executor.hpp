#ifndef GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo_v26_flash/weights.hpp"

namespace gufo::models::mimo_v26_flash::rocm {

class Executor;

/// Device context of one conversation: full-attention KV pages from the
/// executor's pool and a private sliding-window ring.
class KvState {
public:
  ~KvState();
  KvState(const KvState&) = delete;
  KvState& operator=(const KvState&) = delete;

  [[nodiscard]] std::uint32_t capacity() const noexcept {
    return static_cast<std::uint32_t>(pages_.size()) * kTokensPerPage;
  }
  [[nodiscard]] std::size_t committed_bytes() const noexcept;
  /// Returns every page to the pool; the ring is kept.
  void Release() noexcept;

  static constexpr std::uint32_t kTokensPerPage = 256;

private:
  friend class Executor;
  explicit KvState(Executor* owner) : owner_(owner) {}

  Executor* owner_;
  std::vector<std::int32_t> pages_;
  std::int32_t* device_pages_{nullptr};  ///< [max_pages]
  std::uint32_t device_page_count_{0};   ///< entries already uploaded
  std::uint16_t* ring_{nullptr};
};

/// One run of consecutive positions of one session in a forward step.
struct StepItem {
  KvState* kv;
  std::span<const std::int32_t> tokens;
  std::uint32_t start;
  /// Destination for the logits of the item's last position, or empty.
  std::span<float> logits;
};

class Executor {
public:
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::uint32_t max_context, std::size_t kv_pool_bytes, std::string* error);

  /// Rows one step may carry (prefill chunks plus decode rows).
  static constexpr std::uint32_t kMaxRows = 2048;

  [[nodiscard]] std::unique_ptr<KvState> CreateKv(std::string* error);
  /// Runs one forward over every item; items must not share a session.
  [[nodiscard]] bool Forward(std::span<const StepItem> items,
                             std::string* error);

  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return weight_bytes_;
  }
  [[nodiscard]] std::size_t page_bytes() const noexcept;
  [[nodiscard]] std::size_t ring_bytes() const noexcept;
  [[nodiscard]] std::size_t pool_used_bytes() const noexcept;
  [[nodiscard]] std::size_t pool_bytes() const noexcept;
  [[nodiscard]] std::uint32_t vocab() const noexcept { return config_.vocab; }

  /// Host copies of a session's device state for snapshots.
  [[nodiscard]] bool SaveKv(const KvState& kv, std::uint32_t positions,
                            std::uint8_t* dst, std::string* error);
  [[nodiscard]] bool RestoreKv(KvState& kv, std::uint32_t positions,
                               const std::uint8_t* src, std::string* error);
  [[nodiscard]] std::size_t KvSnapshotBytes(std::uint32_t positions) const;
  /// Pages for `positions` tokens; idempotent (Forward reserves too). Lets a
  /// caller reject one request without failing its batch.
  [[nodiscard]] bool Reserve(KvState& kv, std::uint32_t positions,
                             std::string* error);

private:
  friend class KvState;
  struct Impl;
  explicit Executor(Config config);
  void ReturnPages(std::vector<std::int32_t>& pages) noexcept;

  Config config_;
  std::size_t weight_bytes_{0};
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::mimo_v26_flash::rocm

#endif  // GUFO_MODELS_MIMO_V26_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
