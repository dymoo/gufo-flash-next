#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::cli {

struct BenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  std::string speculative_backend{""};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  std::string kv_cache{"f16"};         ///< Qwen3.8-Flash-Next KV storage
  std::string dense_prefill{"f16"};    ///< Flash-Next prompt projections
  std::uint32_t routed_prefill_a4{0};  ///< Flash-Next lossy iu4 routed gate/up
  std::uint32_t prefill_capacity{0};   ///< Flash-Next prompt batch (0: 2048)
  bool routed_down_w4{false};          ///< Flash-Next 4-bit routed down experts
  std::uint32_t routed_w4{0};          ///< Flash-Next symmetric int4 experts
  std::uint32_t prefill_chunk{0};  ///< Flash-Next non-MoE phase rows (0: 2048)
  std::string gdn_state{"f32"};    ///< Flash-Next recurrent state storage
  std::string dense_prefill_weights;  ///< Flash-Next offline S4G128 weights
  std::string mtp_policy{"off"};      ///< Flash-Next MTP draft policy
  std::string mtp_draft_vocab;        ///< Flash-Next compact draft vocabulary
  std::uint32_t mtp_draft_vocab_size{32768};
  std::string gguf_override;            ///< tensor override GGUF
  std::uint32_t experts_used{0};        ///< Flash-Next routed experts per token
  std::uint32_t router_norm_k{0};       ///< Flash-Next routing weight norm
  std::string prefill_residual{"f32"};  ///< Flash-Next prefill residual
  std::string gguf_decode_override;     ///< decode twin override GGUF
  std::uint32_t decode_twin_rows{4};    ///< rows that read the decode twins
  sampling::SamplingConfig sampling{.seed = 0};
  bool verbose{false};
};

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
