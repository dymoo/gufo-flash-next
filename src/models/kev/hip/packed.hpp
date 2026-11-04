#ifndef GUFO_MODELS_KEV_HIP_PACKED_HPP_
#define GUFO_MODELS_KEV_HIP_PACKED_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/kev/encoding.hpp"

namespace gufo::hip {
class QwenGpuModel;
}

namespace gufo::models::kev {

struct PointerHead;

struct PackedOptions {
  std::uint32_t max_tokens{8192};  ///< Tokens per pass.
  std::uint32_t state_slots{16};   ///< Cached or in-flight states.
  std::uint32_t kv_slots{16384};   ///< State tokens the bank holds.
};

struct PackedStats {
  std::uint64_t passes{0};
  std::uint64_t pass_tokens{0};
  std::uint64_t cache_hits{0};
  std::uint64_t cache_misses{0};
  std::uint32_t cached_states{0};
  double gpu_ms{0.0};   ///< Layer stack, gather and copy back.
  double head_ms{0.0};  ///< Pointer head on the readout rows.
};

/// Kev's batched forward: every state and question row of a batch packed into
/// one varlen buffer per pass, weights read once per pass. States are kept in
/// a bank (DeltaNet state, conv window and attention KV) that doubles as the
/// prefix cache; question rows continue from their state's bank entry.
class PackedExecutor {
public:
  ~PackedExecutor();
  [[nodiscard]] static std::unique_ptr<PackedExecutor> Create(
      std::shared_ptr<const ::gufo::hip::QwenGpuModel> model,
      const PointerHead& head, const PackedOptions& options,
      std::string* error);

  /// Probabilities per request, per question.
  [[nodiscard]] std::vector<std::vector<std::vector<double>>> Score(
      std::span<const Encoding* const> batch);

  [[nodiscard]] PackedStats stats() const noexcept { return stats_; }
  [[nodiscard]] const PackedOptions& options() const noexcept {
    return options_;
  }
  /// Device bytes held by the bank and pass buffers.
  [[nodiscard]] std::size_t ReservedBytes() const noexcept;

  /// Called with layer 0 when a pass starts and, after its stream has
  /// drained, before every `layers`-th layer: the caller paces Kev against
  /// other GPU work there.
  void SetSliceHook(std::function<void(std::uint32_t layer)> hook,
                    std::uint32_t layers);

private:
  struct Impl;
  explicit PackedExecutor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  PackedOptions options_;
  PackedStats stats_;
};

}  // namespace gufo::models::kev

#endif  // GUFO_MODELS_KEV_HIP_PACKED_HPP_
