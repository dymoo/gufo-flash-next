#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Dense formats the decode GEMV and the MMQ tier read directly, beside
/// Q8_0 (whose projections also have the WMMA routes).
[[nodiscard]] constexpr bool KQuant(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ6_K;
}

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  /// The same matrix in a cheaper encoding that single-token decode reads
  /// (`--gguf-decode-override`, up to `--decode-twin-rows` rows); prefill and
  /// wide batches keep this one.
  const DeviceTensor* alt{nullptr};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceMixer {
  DeviceTensor norm;
  DeviceTensor down;
  DeviceTensor up;
  DeviceTensor inject;
};

struct DeviceLayer {
  MixerKind mixer{MixerKind::kGatedAttention};
  FfnKind ffn{FfnKind::kRoutedMoe};
  [[nodiscard]] bool linear() const noexcept { return IsLinearMixer(mixer); }
  DeviceMixer hc_attn;
  DeviceMixer hc_ffn;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm,
      indexer_q, indexer_k, indexer_q_norm, indexer_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  DeviceTensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv,
      ple_conv1d;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16; without a shared expert the F32
  /// [hidden -> num_experts] router alone.
  DeviceTensor router;
  /// Without a shared expert: [num_experts] F32 bias added to the sigmoid
  /// scores for selection.
  DeviceTensor expert_bias;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden;
  DeviceMixer nextn_head;
};

/// The DSpark drafter (DsparkWeights) on the device.
struct DeviceDspark {
  struct Block {
    DeviceTensor attn_norm, q, k, v, q_norm, k_norm, out, ffn_norm, gate, up,
        down;
  };
  std::uint32_t block_size{0}, heads{0}, kv_heads{0}, head_dim{0}, rank{0};
  std::int32_t mask_token{0};
  float rope_theta{0}, rms_eps{0}, conf_bias{0};
  std::vector<std::uint32_t> target_layers;
  DeviceTensor fc, enc_norm, out_norm, markov_w1, markov_w2, conf_w;
  std::vector<Block> blocks;
  /// Injection: every block's K and V rows stacked ([k0 v0 k1 v1 ...]) and
  /// the K norms as [blocks][head_dim] F32.
  DeviceTensor kv_all, k_norms;
};

/// The trunk (and optionally the MTP draft block) uploaded to the GPU. The
/// n-gram table is never uploaded: it is read from disk per token.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr, const ModelWeights* decode = nullptr,
      const core::GgufReader* decode_reader = nullptr);
  /// Adds the DSpark drafter (projections transcoded to Q8_0) before the
  /// executor is created.
  [[nodiscard]] bool UploadDspark(const DsparkWeights& weights,
                                  const core::GgufReader& reader,
                                  std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceMixer& hc_head() const noexcept { return hc_head_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  /// Null without a DSpark drafter.
  const DeviceDspark* dspark() const noexcept {
    return dspark_.blocks.empty() ? nullptr : &dspark_;
  }
  /// Decode twins uploaded from `--gguf-decode-override` (0 without one).
  [[nodiscard]] std::size_t decode_twin_count() const noexcept {
    return alt_.size();
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 and K-quant matrices (decode activation
  /// quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceMixer hc_head_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  DeviceDspark dspark_;
  std::vector<void*> allocations_;
  /// Decode-encoding twins that DeviceTensor::alt points into (stable).
  std::deque<DeviceTensor> alt_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEVICE_MODEL_HPP_
