#include "src/cli/serve/serve.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/systemone_api.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/grammar/grammar.hpp"
#include "src/core/model_admission.hpp"
#include "src/models/kev/model.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::cli {
namespace {

class ModelLoadLog {
public:
  ModelLoadLog(std::string kind, const std::filesystem::path& artifact)
      : kind_(std::move(kind)), start_(std::chrono::steady_clock::now()) {
    server::Logger::Info("loader", "event=load_started kind=" + kind_ +
                                       " artifact=" + artifact.string() + " " +
                                       server::Logger::MemoryStatus());
  }
  ~ModelLoadLog() {
    if (!completed_) {
      try {
        server::Logger::Error("loader", "event=load_failed kind=" + kind_ +
                                            " " +
                                            server::Logger::MemoryStatus());
      } catch (...) {
      }
    }
  }
  void Complete(std::string_view details, bool gpu_loaded = true) {
    completed_ = true;
    std::ostringstream out;
    out << "event=load_completed kind=" << kind_ << " elapsed_ms="
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start_)
               .count()
        << ' ' << details << ' ' << server::Logger::MemoryStatus();
#if defined(ENGINE_ENABLE_HIP)
    std::size_t free = 0, total = 0;
    if (gpu_loaded && hipMemGetInfo(&free, &total) == hipSuccess) {
      out << " gpu_device_used_mib=" << (total - free) / (1024 * 1024)
          << " gpu_device_total_mib=" << total / (1024 * 1024);
    }
#else
    (void)gpu_loaded;
#endif
    server::Logger::Info("loader", out.str());
  }

private:
  std::string kind_;
  std::chrono::steady_clock::time_point start_;
  bool completed_{false};
};

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

// Server-level options are accepted on either side of the modality
// subcommand. Registering them from one place keeps `gufo serve --help`, every
// `gufo serve <modality> --help`, and the parser that actually consumes them
// from drifting apart.
void AddServerOptions(gufo::cli::ArgParser& parser, std::string* host,
                      int* port, std::size_t* session_count,
                      std::size_t* max_connections,
                      std::size_t* max_request_body_bytes, std::string* api_key,
                      std::string* api_key_file, std::string* admin_key_file,
                      std::string* metrics_address,
                      std::vector<std::string>* allowed_origins,
                      bool* verbose) {
  parser.AddOption("-i", "--host", "IP", "Bind address", "Server", host);
  parser.AddOption("-p", "--port", "N", "Port to listen on", "Server", port);
  if (session_count != nullptr)
    parser.AddOption("-j", "--sessions", "N",
                     "Preallocated GPU request sessions", "Server",
                     session_count);
  parser.AddOption("", "--max-connections", "N",
                   "Maximum simultaneous HTTP connections", "Server",
                   max_connections);
  parser.AddOption("", "--max-request-bytes", "N",
                   "Maximum HTTP request body bytes", "Server",
                   max_request_body_bytes);
  parser.AddOption("", "--api-key", "KEY", "API key", "Server", api_key);
  parser.AddOption("", "--api-key-file", "PATH",
                   "Read the API key from a file (keeps it out of argv)",
                   "Server", api_key_file);
  parser.AddOption("", "--admin-key-file", "PATH",
                   "Read the only key that may POST /v1/runtime/drain from a "
                   "file; without it only loopback clients may drain",
                   "Server", admin_key_file);
  parser.AddOption(
      "", "--metrics-address", "HOST:PORT",
      "Bind an additional, unauthenticated listener serving only GET "
      "/metrics and /health (e.g. for a Prometheus scrape network); "
      "disabled by default",
      "Server", metrics_address);
  parser.AddCustomOption(
      "", "--allow-origin", "ORIGIN",
      "Browser origin allowed to call the server, matched exactly "
      "(repeatable; * allows any). Requests with another Origin get 403; "
      "default: none",
      "Server",
      [allowed_origins](std::string_view, std::string_view value,
                        std::string*) {
        allowed_origins->emplace_back(value);
        return true;
      });
  parser.AddFlag("-v", "--verbose", "Verbose logging", "General", verbose);
}

// Help-only sinks for AddServerOptions, so subcommand help can list the
// server options it shares with `gufo serve` without parsing into them.
struct ServerOptionHelpTargets {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  std::string api_key_file;
  std::string admin_key_file;
  std::string metrics_address;
  std::vector<std::string> allowed_origins;
  bool verbose = false;
};

void AddServerOptionsForHelp(gufo::cli::ArgParser& parser,
                             ServerOptionHelpTargets* targets,
                             bool include_sessions = true) {
  AddServerOptions(parser, &targets->host, &targets->port,
                   include_sessions ? &targets->session_count : nullptr,
                   &targets->max_connections, &targets->max_request_body_bytes,
                   &targets->api_key, &targets->api_key_file,
                   &targets->admin_key_file, &targets->metrics_address,
                   &targets->allowed_origins, &targets->verbose);
}

// First line of a key file without trailing whitespace; empty when the file
// cannot be read.
std::string ReadKeyFile(const std::string& path) {
  std::ifstream file(path);
  std::string key;
  std::getline(file, key);
  while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back())))
    key.pop_back();
  return key;
}

// A browser sends Origin as SCHEME://HOST[:PORT] in lowercase, without a path
// or trailing slash. Reject spellings that could never match one.
bool ValidAllowedOrigin(std::string_view origin) {
  if (origin == "*")
    return true;
  const auto separator = origin.find("://");
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 3 == origin.size())
    return false;
  const auto lower = [](char c) {
    return std::islower(static_cast<unsigned char>(c)) != 0;
  };
  const auto scheme = origin.substr(0, separator);
  return lower(scheme.front()) &&
         std::ranges::all_of(
             scheme,
             [&](char c) {
               return lower(c) ||
                      std::isdigit(static_cast<unsigned char>(c)) != 0 ||
                      c == '+' || c == '-' || c == '.';
             }) &&
         std::ranges::none_of(origin.substr(separator + 3), [](char c) {
           return c == '/' ||
                  std::isupper(static_cast<unsigned char>(c)) != 0 ||
                  std::isspace(static_cast<unsigned char>(c)) != 0;
         });
}

// Split a `NAME=VALUE` CLI spec. Returns false when either side is empty.
bool SplitNameValue(std::string_view spec, std::string_view flag,
                    std::string* name, std::string* value, std::string* error) {
  const std::size_t separator = spec.find('=');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 >= spec.size()) {
    *error = std::string(flag) + " expects NAME=VALUE";
    return false;
  }
  *name = std::string(spec.substr(0, separator));
  *value = std::string(spec.substr(separator + 1));
  return true;
}

std::optional<std::string> ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::nullopt;
  }
  std::string text{std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>()};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

// Resolve a `--voice-text` value: an existing file is read for its contents,
// anything else is taken as the transcript itself.
std::string ResolveReferenceText(const std::string& value) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(std::filesystem::path(value), ec)) {
    if (const auto text = ReadTextFile(std::filesystem::path(value))) {
      return *text;
    }
  }
  return value;
}

// Build the registered voices from `--voice NAME=WAV` and the optional
// `--voice-text NAME=<text|path>` overrides. Resolution happens after parsing
// so the two flags may appear in any order.
bool BuildVoicePresets(
    const std::vector<std::pair<std::string, std::string>>& voice_specs,
    const std::map<std::string, std::string>& text_specs,
    const std::map<std::string, std::string>& language_specs,
    std::map<std::string, server::TtsVoicePreset>* presets,
    std::string* error) {
  for (const auto& [name, wav] : voice_specs) {
    if (presets->contains(name)) {
      *error = "duplicate --voice name '" + name + "'";
      return false;
    }
    const std::filesystem::path wav_path(wav);
    std::ifstream wav_file(wav_path, std::ios::binary);
    if (!wav_file) {
      *error = "cannot open voice reference " + wav_path.string();
      return false;
    }
    const std::vector<char> wav_bytes{std::istreambuf_iterator<char>(wav_file),
                                      std::istreambuf_iterator<char>()};

    server::TtsVoicePreset preset;
    std::string decode_error;
    if (!models::qwen3_tts::DecodeWav(
            std::as_bytes(std::span<const char>(wav_bytes)),
            &preset.reference_audio, &decode_error)) {
      *error = wav_path.string() + ": " + decode_error;
      return false;
    }

    // An explicit --voice-text wins; otherwise fall back to a `.txt` sidecar
    // beside the WAV. With neither, the voice clones from the speaker
    // embedding alone, which needs no transcript.
    if (const auto override_text = text_specs.find(name);
        override_text != text_specs.end()) {
      preset.reference_text = ResolveReferenceText(override_text->second);
    } else {
      std::filesystem::path sidecar = wav_path;
      sidecar.replace_extension(".txt");
      if (const auto text = ReadTextFile(sidecar)) {
        preset.reference_text = *text;
      }
    }
    preset.speaker_embedding_only = preset.reference_text.empty();
    if (const auto language = language_specs.find(name);
        language != language_specs.end()) {
      preset.language = language->second;
    }
    presets->emplace(name, std::move(preset));
  }

  for (const auto& [name, unused] : text_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-text names unknown voice '" + name + "'";
      return false;
    }
  }
  for (const auto& [name, unused] : language_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-lang names unknown voice '" + name + "'";
      return false;
    }
  }
  return true;
}

std::optional<ReasoningOptions> ResolveReasoningDefaults(
    std::string_view mode, std::string_view effort, std::string_view preserve,
    std::string* error) {
  ReasoningOptions options;
  if (mode == "on") {
    options.enabled = true;
  } else if (mode == "off") {
    options.enabled = false;
  } else if (mode != "auto") {
    *error = "--think must be on, off, or auto";
    return std::nullopt;
  }

  if (effort != "auto") {
    const auto parsed = ParseReasoningEffort(effort);
    if (!parsed.has_value()) {
      *error =
          "--reasoning-effort must be auto, minimal, low, medium, high, "
          "xhigh, or max";
      return std::nullopt;
    }
    if (options.enabled == false) {
      *error = "--reasoning-effort cannot be set while --think is off";
      return std::nullopt;
    }
    options.enabled = true;
    options.effort = parsed;
  }

  if (preserve == "on") {
    options.preserve_thinking = true;
  } else if (preserve == "off") {
    options.preserve_thinking = false;
  } else if (preserve != "auto") {
    *error = "--preserve-thinking must be on, off, or auto";
    return std::nullopt;
  }
  return options;
}

struct SpeechOptions {
  std::filesystem::path model;
  std::size_t context;
  std::string served_model_name;
  std::vector<std::pair<std::string, std::string>> voice_specs;
  std::map<std::string, std::string> voice_text_specs;
  std::map<std::string, std::string> voice_lang_specs;
};

void AddSpeechOptions(ArgParser& parser, bool tts, SpeechOptions* options) {
  parser.AddOption("-m", "--model", "DIR",
                   tts ? "Qwen3-TTS 12Hz 1.7B model directory"
                       : "Qwen3-ASR 1.7B model directory",
                   "Model", &options->model);
  parser.AddOption("-c", "--context", "N",
                   tts ? "Context capacity (default: 4096)"
                       : "Context capacity per audio chunk (default: 1024)",
                   "Model", &options->context);
  parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                   "Model", &options->served_model_name);
  if (!tts)
    return;
  parser.AddCustomOption(
      "", "--voice", "NAME=PATH",
      "Register a named Qwen3-TTS Base voice from a reference WAV "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string path;
        if (!SplitNameValue(value, "--voice", &name, &path, err)) {
          return false;
        }
        options->voice_specs.emplace_back(std::move(name), std::move(path));
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-lang", "NAME=LANGUAGE",
      "Language a --voice speaks; used when a request omits 'language' "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string language;
        if (!SplitNameValue(value, "--voice-lang", &name, &language, err)) {
          return false;
        }
        if (!options->voice_lang_specs
                 .emplace(std::move(name), std::move(language))
                 .second) {
          *err = "duplicate --voice-lang name";
          return false;
        }
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-text", "NAME=TEXT|PATH",
      "Reference transcript for a --voice, given inline or as a file path; "
      "defaults to a .txt sidecar beside the WAV (repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string text;
        if (!SplitNameValue(value, "--voice-text", &name, &text, err)) {
          return false;
        }
        if (!options->voice_text_specs.emplace(std::move(name), std::move(text))
                 .second) {
          *err = "duplicate --voice-text name";
          return false;
        }
        return true;
      });
}

struct SystemOneOptions {
  std::filesystem::path model;
  std::string name{"kev-4b"};
  std::uint32_t context{40960};
  double share{0.25};
};

void AddSystemOneOptions(ArgParser& parser, SystemOneOptions* options,
                         bool standalone) {
  if (!standalone)
    parser.AddOption("", "--systemone-share", "FRACTION",
                     "Kev's share of GPU time while the text model is busy "
                     "(default: 0.25)",
                     "System One", &options->share);
  parser.AddOption(standalone ? "-m" : "",
                   standalone ? "--model" : "--systemone-model", "FILE",
                   "Kev GGUF served at /v1/systemone", "System One",
                   &options->model);
  parser.AddOption("", standalone ? "--served-model-name" : "--systemone-name",
                   "NAME", "System One model ID (default: kev-4b)",
                   "System One", &options->name);
  parser.AddOption("", standalone ? "--context" : "--systemone-context", "N",
                   "Longest System One row in tokens, state included "
                   "(default: 40960: a 32768-token state plus an 8192-token "
                   "question)",
                   "System One", &options->context);
}

/// `gufo serve llm` options; help and the real parser register the same set.
struct LlmOptions {
  std::string model;
  std::string served_model_name;
  std::uint32_t max_context = 0;
  std::int64_t max_tokens = -1;
  sampling::SamplingConfig sampling_config;
  std::string reasoning_mode = "auto";
  std::string reasoning_effort = "auto";
  std::string preserve_thinking = "auto";
  std::string speculative_backend;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::string mtp_model_path;
  std::string mtp_policy = "off";
  std::string mtp_draft_vocab;
  std::uint32_t mtp_draft_vocab_size = 32768;
  std::string moe_routing = "exact";
  std::string vision_model_path;
  std::string kv_cache = "f16";
  std::string dense_prefill = "f16";
  std::string gdn_state = "f32";
  std::uint32_t routed_prefill_a4 = 0;
  std::uint32_t prefill_capacity = 0;
  bool routed_down_w4 = false;
  std::uint32_t routed_w4 = 0;
  std::uint32_t prefill_chunk = 0;
  std::string dense_prefill_weights;
  std::string gguf_override;
  std::string gguf_decode_override;
  std::uint32_t decode_twin_rows = 4;
  std::uint32_t experts_used = 0;
  std::uint32_t router_norm_k = 0;
  std::uint32_t fast_experts = 0;
  std::string prefill_residual = "f32";
  std::vector<std::string> adapters;
  std::size_t draft_tokens = 7;
  std::size_t min_draft_tokens = 1;
  std::size_t prefill_chunk_tokens = server::kDefaultDecodeActivePrefillTokens;
  std::size_t max_itl_ms =
      static_cast<std::size_t>(server::kDefaultMaxItl.count());
  std::size_t max_pending_requests = 16;
  std::size_t max_pending_requests_per_client = 4;
  std::size_t flex_sessions = 2;
  std::string admission = "queue";
  std::uint64_t request_timeout_ms = 0;
  std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
  std::size_t max_buffered_output_bytes =
      server::kDefaultMaxBufferedOutputBytes;
  std::size_t max_buffered_output_bytes_total =
      server::kDefaultMaxBufferedOutputBytesTotal;
  std::filesystem::path cache_disk_directory;
  std::size_t cache_disk_bytes =
      server::TextRunnerDiskCacheOptions::kDefaultCapacityBytes;
  std::size_t cache_disk_staging_bytes = 0;
  std::size_t cache_pin_bytes =
      server::TextRunnerDiskCacheOptions{}.pin_capacity_bytes;
  std::size_t cache_pin_conversations =
      server::TextRunnerDiskCacheOptions{}.pin_min_conversations;
  std::size_t host_memory_budget = 0;
  std::size_t grammar_cache_bytes = grammar::Limits{}.cache_bytes;
  std::size_t grammar_compile_timeout_ms =
      static_cast<std::size_t>(grammar::Limits{}.compile_timeout.count());
  bool log_progress = false;
};

void AddLlmOptions(ArgParser& parser, LlmOptions* options,
                   SystemOneOptions* systemone) {
  parser.AddOption("-m", "--model", "PATH",
                   "Path to GGUF model file (required)", "Model",
                   &options->model);
  parser.AddOption("", "--mmproj", "PATH",
                   "Qwen BF16 vision sidecar (auto-discovered beside model)",
                   "Model", &options->vision_model_path);
  parser.AddOption("", "--served-model-name", "ID",
                   "Model identifier exposed by the OpenAI API", "Model",
                   &options->served_model_name);
  parser.AddOption(
      "-c", "--context", "N",
      "Context tokens per session (default: 0 = model native context)", "Model",
      &options->max_context);
  parser.AddOption("", "--kv-cache", "MODE", core::kKvCacheHelp, "Model",
                   &options->kv_cache);
  parser.AddOption("", "--dense-prefill", "MODE", core::kDensePrefillHelp,
                   "Model", &options->dense_prefill);
  parser.AddOption(
      "", "--gdn-state", "MODE",
      "GatedDeltaNet recurrent state storage: f32 (exact, default), f16 or "
      "bf16 (lossy, opt-in: half the state traffic of every decode step)",
      "Model", &options->gdn_state);
  parser.AddOption(
      "", "--routed-prefill-a4", "GROUP",
      "Flash-Next: lossy iu4 routed gate/up for prompt batches with "
      "4-bit activations in groups of 64/128/256 (default: 0 = off)",
      "Model", &options->routed_prefill_a4);
  parser.AddOption("", "--prefill-capacity", "TOKENS",
                   "Flash-Next: largest prompt batch per forward pass "
                   "(default: 0 = 2048)",
                   "Model", &options->prefill_capacity);
  parser.AddFlag("", "--routed-down-w4",
                 "Flash-Next: re-quantize the routed down experts to 4-bit "
                 "codes (lossy) and run their prefill GEMM on the iu4 WMMA",
                 "Model", &options->routed_down_w4);
  parser.AddOption("", "--routed-w4", "GROUP",
                   "Flash-Next: re-quantize the routed experts to symmetric "
                   "int4 with one scale per 64/128 K (lossy, every route) and "
                   "run prompt batches on the iu4 WMMA (default: 0 = off)",
                   "Model", &options->routed_w4);
  parser.AddOption("", "--prefill-phase-rows", "TOKENS",
                   "Flash-Next: rows per non-MoE prefill phase inside a larger "
                   "--prefill-capacity (default: 0 = 2048)",
                   "Model", &options->prefill_chunk);
  parser.AddOption(
      "", "--dense-prefill-weights", "DIR",
      "Flash-Next: offline S4G128 int4 weights for --dense-prefill "
      "w4a4 (default: quantize the Q8_0 weights at load)",
      "Model", &options->dense_prefill_weights);
  parser.AddOption("", "--gguf-override", "PATH",
                   "Flash-Next: lossy, GGUF whose tensors replace same-named "
                   "tensors of the model (requantized variants)",
                   "Model", &options->gguf_override);
  parser.AddOption("", "--gguf-decode-override", "PATH",
                   "Flash-Next: lossy, GGUF of requantized dense tensors "
                   "uploaded as decode twins beside the primary tensors "
                   "(prefill unchanged; decode batches of --decode-twin-rows "
                   "or fewer rows read the twins)",
                   "Model", &options->gguf_decode_override);
  parser.AddOption("", "--decode-twin-rows", "N",
                   "Flash-Next: batch rows up to which --gguf-decode-override "
                   "twins are read (default: 4)",
                   "Model", &options->decode_twin_rows);
  parser.AddOption("", "--experts-used", "K",
                   "Flash-Next: lossy, route every token to its top K experts "
                   "instead of the model's expert_used_count (default: 0 = "
                   "model)",
                   "Model", &options->experts_used);
  parser.AddOption("", "--prefill-residual", "MODE",
                   "Flash-Next: hyper-connection residual of wide prefill "
                   "batches: f32 (exact, default) or f16 (lossy)",
                   "Model", &options->prefill_residual);
  parser.AddOption("", "--router-norm-k", "N",
                   "Flash-Next: lossy, normalize routing weights over the top "
                   "N router probabilities (default: 0 = model)",
                   "Model", &options->router_norm_k);
  parser.AddOption("", "--fast-experts", "K",
                   "Flash-Next: also serve every model id's fast model as "
                   "<id>-fast, a lossy model whose decode rows keep their top "
                   "K routed experts (prompts stay exact; default: 0 = none)",
                   "Model", &options->fast_experts);
  parser.AddCustomOption(
      "", "--adapter", "NAME=PATH",
      "Qwen3.8-Flash-Next steering adapter served as model id NAME from "
      "the same weights (repeatable; PATH is its JSON metadata)",
      "Model",
      [options](std::string_view, std::string_view value, std::string*) {
        options->adapters.emplace_back(value);
        return true;
      });
  parser.AddOption(
      "-n", "--max-tokens", "N",
      "Default new-token limit (default: -1 = until EOS or context full)",
      "Sampling Defaults", &options->max_tokens);
  RegisterSamplingOptions(parser, &options->sampling_config,
                          "Sampling Defaults");
  parser.AddOption(
      "", "--think", "MODE",
      "Default reasoning mode: on, off, or auto (default: model template)",
      "Reasoning Defaults", &options->reasoning_mode);
  parser.AddOption(
      "", "--reasoning-effort", "LEVEL",
      "Default effort: auto, minimal, low, medium, high, xhigh, or max",
      "Reasoning Defaults", &options->reasoning_effort);
  parser.AddOption("", "--preserve-thinking", "MODE",
                   "Replay prior reasoning: on, off, or auto",
                   "Reasoning Defaults", &options->preserve_thinking);
  parser.AddOption("", "--speculative", "MODE",
                   "HTTP draft backend: dspark, dflash2, mtp, or off",
                   "Speculative", &options->speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to Qwen DFlash2 GGUF file", "Speculative",
                   &options->dflash_model_path);
  parser.AddOption(
      "", "--draft-policy", "POLICY",
      "DFlash2 block length: fixed or adaptive (default: adaptive)",
      "Speculative", &options->draft_policy);
  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to DeepSeek V4 Flash DSpark support GGUF file",
                   "Speculative", &options->dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the "
                   "mtp-...-shared-*.gguf sidecar)",
                   "Speculative", &options->mtp_model_path);
  parser.AddOption("", "--mtp-policy", "MODE",
                   "Qwen3.8-Flash-Next MTP draft policy: off (default) or "
                   "adaptive (lossless; compact draft head, calibrated "
                   "confidence gate, longest-match prompt lookup)",
                   "Speculative", &options->mtp_policy);
  parser.AddOption("", "--mtp-draft-vocab", "FILE",
                   "Qwen3.8-Flash-Next MTP: token ids (one per line, most "
                   "useful first) the draft head scores instead of the full "
                   "vocabulary (lossless)",
                   "Speculative", &options->mtp_draft_vocab);
  parser.AddOption("", "--mtp-draft-vocab-size", "N",
                   "Draft vocabulary rows used from --mtp-draft-vocab "
                   "(default: 32768)",
                   "Speculative", &options->mtp_draft_vocab_size);
  parser.AddOption(
      "-d", "--draft-tokens", "N",
      "Maximum speculative draft tokens evaluated per step (default: 7)",
      "Speculative", &options->draft_tokens);
  parser.AddOption("", "--min-draft-tokens", "N",
                   "Adaptive draft floor (default: 1)", "Speculative",
                   &options->min_draft_tokens);
  parser.AddOption("", "--moe-routing", "POLICY", core::kMoeRoutingHelp,
                   "Model", &options->moe_routing);
  parser.AddOption(
      "", "--prefill-chunk", "N",
      "Prompt tokens between active decode rounds, or the starting prompt "
      "rows of a fused prefill+decode step (default: 512)",
      "Scheduling", &options->prefill_chunk_tokens);
  parser.AddOption(
      "", "--max-itl-ms", "MS",
      "Target fused prefill+decode step time; prompt rows per step adapt to "
      "it (default: 1500, 0 keeps --prefill-chunk rows)",
      "Scheduling", &options->max_itl_ms);
  parser.AddOption("", "--max-pending", "N",
                   "Maximum queued generation requests (default: 16)",
                   "Scheduling", &options->max_pending_requests);
  parser.AddOption("", "--max-pending-per-client", "N",
                   "Maximum queued requests per client IP (default: 4)",
                   "Scheduling", &options->max_pending_requests_per_client);
  parser.AddOption("", "--flex-sessions", "N",
                   "Sessions service_tier=flex requests may hold at once "
                   "(default: 2)",
                   "Scheduling", &options->flex_sessions);
  parser.AddOption("", "--admission", "MODE",
                   "When every session is busy: queue (wait, up to "
                   "--max-pending) or refuse (503 capacity_exhausted at once; "
                   "X-Gufo-Queue: 1 waits) (default: queue)",
                   "Scheduling", &options->admission);
  AddSystemOneOptions(parser, systemone, false);
  parser.AddOption("", "--request-timeout-ms", "MS",
                   "Queue plus generation timeout; 0 disables it (default: 0)",
                   "Scheduling", &options->request_timeout_ms);
  parser.AddOption("", "--max-output-bytes", "N",
                   "Maximum generated bytes per request (default: 1048576)",
                   "Scheduling", &options->max_output_bytes);
  parser.AddOption("", "--max-buffered-output-bytes", "N",
                   "Maximum queued stream bytes per request (default: 65536)",
                   "Scheduling", &options->max_buffered_output_bytes);
  parser.AddOption(
      "", "--max-buffered-output-total", "N",
      "Maximum queued stream bytes across requests (default: 262144)",
      "Scheduling", &options->max_buffered_output_bytes_total);
  parser.AddOption("", "--cache-disk", "DIR",
                   "Opt-in restart-safe continuation cache directory", "Cache",
                   &options->cache_disk_directory);
  parser.AddOption("", "--cache-disk-bytes", "N",
                   "Retained disk-cache byte budget (default: " +
                       std::to_string(options->cache_disk_bytes) + ")",
                   "Cache", &options->cache_disk_bytes);
  parser.AddOption("", "--cache-disk-staging-bytes", "N",
                   "RAM limit for queued snapshots and each disk read "
                   "(default: 0 = auto, at most 1 GiB and 1/8 available RAM)",
                   "Cache", &options->cache_disk_staging_bytes);
  parser.AddOption(
      "", "--cache-pin-bytes", "N",
      "Shared-prefix snapshot bytes exempt from eviction, per tier (RAM, "
      "disk; each at most half its budget; 0 disables; default: " +
          std::to_string(options->cache_pin_bytes) + ")",
      "Cache", &options->cache_pin_bytes);
  parser.AddOption(
      "", "--cache-pin-conversations", "N",
      "Distinct conversations sharing a system prompt before it is pinned "
      "(default: " +
          std::to_string(options->cache_pin_conversations) + ")",
      "Cache", &options->cache_pin_conversations);
  parser.AddOption(
      "", "--host-memory-budget", "N",
      "Host RAM bytes for retained snapshots (1/2), idle sessions paged "
      "out of the GPU pool (1/4) and disk staging (1/4); beyond it they "
      "go to --cache-disk (default: 0 = available RAM minus 8 GiB, or a "
      "quarter of it when that is smaller; set it "
      "where the container view overstates host RAM)",
      "Cache", &options->host_memory_budget);
  parser.AddOption(
      "", "--grammar-cache-bytes", "N",
      "Host RAM bytes for compiled structured-output grammars (default: " +
          std::to_string(options->grammar_cache_bytes) + ")",
      "Structured outputs", &options->grammar_cache_bytes);
  parser.AddOption(
      "", "--grammar-compile-timeout-ms", "N",
      "Longest a response_format or tool_choice grammar may compile before "
      "the request is rejected with HTTP 400 (default: " +
          std::to_string(options->grammar_compile_timeout_ms) + ")",
      "Structured outputs", &options->grammar_compile_timeout_ms);
  parser.AddFlag("", "--log-progress", "Log live prefill and decode progress",
                 "Logging", &options->log_progress);
}

std::shared_ptr<server::SystemOneService> LoadSystemOne(
    const SystemOneOptions& options, std::size_t max_connections) {
  if (options.context < 2048) {
    std::cerr << "Error: the System One context must be at least 2048\n";
    return nullptr;
  }
  if (!(options.share > 0.0 && options.share <= 1.0)) {
    std::cerr << "Error: --systemone-share must be in (0, 1]\n";
    return nullptr;
  }
  ModelLoadLog load_log("kev", options.model);
  models::kev::ModelOptions model_options;
  model_options.limits.max_row = options.context;
  model_options.limits.max_state =
      options.context > 16384 ? options.context - 8192 : options.context / 2;
  // The bank must hold the longest state it admits.
  model_options.packed.kv_slots =
      static_cast<std::uint32_t>(std::max<std::size_t>(
          model_options.packed.kv_slots, model_options.limits.max_state));
  std::string error;
  auto model = models::kev::Model::Load(options.model, model_options, &error);
  if (model == nullptr) {
    std::cerr << "Error loading Kev '" << options.model.string()
              << "': " << error << "\n";
    return nullptr;
  }
  // A waiting System One request holds an HTTP connection; leave half of
  // them for chat, health and runtime requests.
  auto service = std::make_shared<server::SystemOneService>(
      std::move(model),
      server::SystemOneService::Options{
          .model_id = options.name,
          .max_pending = std::max<std::size_t>(1, max_connections / 2),
          .gpu_share = options.share});
  load_log.Complete("model=" + options.name +
                    " context_tokens=" + std::to_string(options.context));
  return service;
}

// `-flex` (and `-fast` with --fast-experts) end alias ids, so a served id may
// not end in one: its literal name would parse as another model's alias.
std::optional<std::string_view> ReservedModelIdSuffix(std::string_view id,
                                                      std::uint32_t fast) {
  if (id.ends_with(server::TextGenerationBackend::kFlexSuffix))
    return server::TextGenerationBackend::kFlexSuffix;
  if (fast != 0 && id.ends_with(server::TextGenerationBackend::kFastSuffix))
    return server::TextGenerationBackend::kFastSuffix;
  return std::nullopt;
}

}  // namespace

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand) {
  if (subcommand == "systemone") {
    SystemOneOptions options;
    ArgParser parser(std::string(program_name) + " serve systemone",
                     "Serve Kev (TypeSafe System One) at /v1/systemone.");
    AddSystemOneOptions(parser, &options, true);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }
  if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser(std::string(program_name) + " serve image",
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }
  if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve video",
        "Start the MiniMax H3 text-to-video HTTP generation server.");
    parser.AddOption("-m", "--model", "DIR",
                     "Operator-supplied MiniMax H3 directory", "Model",
                     &video_model);
    parser.AddOption(
        "", "--root", "DIR",
        "Storage root for persistent video jobs (default: video-jobs)",
        "Storage", &video_root);
    parser.AddOption("", "--manifest", "PATH", "Pinned H3 manifest override",
                     "Storage", &video_manifest);
    parser.AddOption("", "--ttl", "SEC",
                     "Completed-artifact TTL in seconds (default: 3600)",
                     "Storage", &video_ttl_seconds);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "tts" || subcommand == "asr") {
    const bool tts = subcommand == "tts";
    SpeechOptions options{.context = tts ? 4096U : 1024U};
    ArgParser parser(
        std::string(program_name) + " serve " + std::string(subcommand),
        tts ? "Serve Qwen3-TTS speech synthesis."
            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, tts, &options);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "llm") {
    LlmOptions options;
    SystemOneOptions systemone_help;
    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    AddLlmOptions(parser, &options, &systemone_help);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  std::cout
      << "Usage: " << program_name
      << " serve [SERVER_OPTIONS] [COMMAND] [OPTIONS]\n\n"
      << "Start the OpenAI-compatible HTTP server for a specific model "
         "modality.\n\n"
      << "Commands:\n"
      << "  llm       Serve text LLM endpoints (/v1/chat/completions, "
         "/v1/completions) [default]\n"
      << "  video     Serve MiniMax H3 video generation endpoint "
         "(/v1/video/generations)\n"
      << "  image     Serve Qwen-Image-2.1 (/v1/images/generations, "
         "/v1/images/edits)\n"
      << "  tts       Serve Qwen3-TTS speech synthesis (/v1/audio/speech)\n"
      << "  asr       Serve Qwen3-ASR transcription "
         "(/v1/audio/transcriptions)\n"
      << "  systemone Serve Kev, TypeSafe System One (/v1/systemone)\n\n"
      << "Server Options:\n"
      << "  -i, --host <IP>        Bind address (default: 127.0.0.1)\n"
      << "  -p, --port <N>         Port to listen on (default: 8080)\n"
      << "  -j, --sessions <N>     Preallocated GPU request sessions (default: "
         "1)\n"
      << "      --max-connections <N>\n"
      << "                         Maximum simultaneous HTTP connections "
         "(default: 16)\n"
      << "      --max-request-bytes <N>\n"
      << "                         Maximum HTTP request body bytes (default: "
         "8388608)\n"
      << "      --api-key <KEY>    Require Bearer authorization for requests\n"
      << "      --admin-key-file <PATH>\n"
      << "                         Key for POST /v1/runtime/drain (default: "
         "loopback clients only)\n"
      << "      --metrics-address <HOST:PORT>\n"
      << "                         Bind an additional, unauthenticated "
         "metrics-only listener\n"
      << "      --allow-origin <ORIGIN>\n"
      << "                         Browser origin allowed to call the server "
         "(repeatable; default: none)\n"
      << "  -v, --verbose          Print detailed server metrics and request "
         "traces\n"
      << "  -h, --help             Print help\n";
}

int RunServe(std::span<const char* const> args) {
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  std::string host = "127.0.0.1";
  // The variable that chose the bind address, named by the exposure warning.
  const char* host_variable = nullptr;
  if (const char* env_host = std::getenv("HOST");
      env_host != nullptr && *env_host != '\0') {
    host = env_host;
    host_variable = "HOST";
  } else if (const char* env_strix_host = std::getenv("GUFO_HOST");
             env_strix_host != nullptr && *env_strix_host != '\0') {
    host = env_strix_host;
    host_variable = "GUFO_HOST";
  }
  const std::string environment_host = host;

  int port = 8080;
  if (const char* env_port = std::getenv("PORT");
      env_port != nullptr && *env_port != '\0') {
    const std::string_view sv(env_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  } else if (const char* env_strix_port = std::getenv("GUFO_PORT");
             env_strix_port != nullptr && *env_strix_port != '\0') {
    const std::string_view sv(env_strix_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  }

  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  std::string api_key_file;
  std::string admin_key_file;
  std::string admin_key;
  std::string metrics_address;
  std::vector<std::string> allowed_origins;
  bool verbose = false;

  // Identify the modality only after leading server options. The selected
  // parser consumes every option together, so --served-model-name -v and
  // --model audio cannot be mistaken for server flags or subcommands.
  ArgParser server_parser("gufo serve");
  const auto add_server_options = [&](ArgParser& parser,
                                      bool include_sessions = true) {
    AddServerOptions(
        parser, &host, &port, include_sessions ? &session_count : nullptr,
        &max_connections, &max_request_body_bytes, &api_key, &api_key_file,
        &admin_key_file, &metrics_address, &allowed_origins, &verbose);
  };
  add_server_options(server_parser);
  std::string subcommand = "llm";
  std::vector<const char*> sub_args(args.begin(), args.end());
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    const auto equals = arg.find('=');
    const auto* option = server_parser.FindOption(arg.substr(0, equals));
    if (option != nullptr) {
      if (!option->is_flag && equals == std::string_view::npos) {
        ++i;
      }
      continue;
    }
    if (arg == "llm" || arg == "video" || (arg == "tts" || arg == "asr") ||
        arg == "image" || arg == "systemone") {
      subcommand = arg;
      sub_args.erase(sub_args.begin() + static_cast<std::ptrdiff_t>(i));
    } else if (arg == "--help" || arg == "-h" || arg == "help") {
      const std::string_view topic =
          arg == "help" && i + 1 < args.size() ? args[i + 1] : "";
      if (!topic.empty() && topic != "llm" && topic != "video" &&
          topic != "tts" && topic != "asr" && topic != "image" &&
          topic != "systemone") {
        std::cerr << "Error: unknown serve command '" << topic << "'\n";
        return 2;
      }
      PrintServeHelp("gufo", topic);
      return 0;
    }
    break;
  }
  const auto valid_server_options = [&] {
    in_addr address{};
    if (::inet_pton(AF_INET, host.c_str(), &address) != 1) {
      std::cerr << "Error: --host must be an IPv4 address\n";
      return false;
    }
    if (port < 0 || port > 65535) {
      std::cerr << "Error: --port must be between 0 and 65535\n";
      return false;
    }
    if (session_count == 0 || max_connections == 0 ||
        max_request_body_bytes == 0) {
      std::cerr << "Error: server limits must be positive\n";
      return false;
    }
    for (const auto& origin : allowed_origins) {
      if (!ValidAllowedOrigin(origin)) {
        std::cerr << "Error: --allow-origin expects * or SCHEME://HOST[:PORT], "
                     "not '"
                  << origin << "'\n";
        return false;
      }
    }
    if (!api_key_file.empty()) {
      if (!api_key.empty()) {
        std::cerr << "Error: use --api-key or --api-key-file, not both\n";
        return false;
      }
      api_key = ReadKeyFile(api_key_file);
      if (api_key.empty()) {
        std::cerr << "Error: --api-key-file has no readable key\n";
        return false;
      }
    }
    if (!admin_key_file.empty()) {
      admin_key = ReadKeyFile(admin_key_file);
      if (admin_key.empty()) {
        std::cerr << "Error: --admin-key-file has no readable key\n";
        return false;
      }
      if (admin_key == api_key) {
        std::cerr << "Error: --admin-key-file must hold a different key from "
                     "the API key\n";
        return false;
      }
    }
    if (api_key.empty() && !server::IsLoopbackAddress(host)) {
      std::cerr << "Warning: listening on " << host;
      if (host_variable != nullptr && host == environment_host)
        std::cerr << " (from the " << host_variable << " environment variable)";
      std::cerr << " without --api-key or --api-key-file: anyone who can "
                   "reach this address can use the server\n";
    }
    return true;
  };
  std::string parse_err;

  std::shared_ptr<server::InferenceBackend> backend;
  std::shared_ptr<server::VideoJobService> video_jobs;
  std::shared_ptr<server::TtsService> tts;
  std::shared_ptr<server::AsrService> asr;
  std::shared_ptr<server::ImageService> images;
  std::shared_ptr<server::SystemOneService> systemone;
  SystemOneOptions systemone_options;

  if (subcommand == "systemone") {
    ArgParser parser("gufo serve systemone",
                     "Serve Kev (TypeSafe System One) at /v1/systemone.");
    AddSystemOneOptions(parser, &systemone_options, true);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "systemone");
      return 0;
    }
    if (!valid_server_options())
      return 2;
    if (systemone_options.model.empty()) {
      std::cerr << "Error: --model <FILE> is required for System One\n";
      return 2;
    }
    systemone = LoadSystemOne(systemone_options, max_connections);
    if (systemone == nullptr)
      return 1;
  } else if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser("gufo serve image",
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "image");
      return 0;
    }
    if (!valid_server_options())
      return 2;
    if (model.empty()) {
      std::cerr << "Error: --model <DIR> is required for image server\n";
      return 2;
    }
    ModelLoadLog load_log("qwen_image_21", model);
    try {
      images = std::make_shared<server::ImageService>(model, name);
    } catch (const std::exception& error) {
      std::cerr << "Error loading Qwen-Image-2.1: " << error.what() << '\n';
      return 1;
    }
    load_log.Complete("model=" + name + " weights=mapped upload=on_demand");
  } else if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser video_parser(
        "gufo serve video", "Start the MiniMax H3 video generation server.");
    video_parser.AddOption("-m", "--model", "DIR",
                           "Operator-supplied MiniMax H3 directory", "Model",
                           &video_model);
    video_parser.AddOption("", "--root", "DIR",
                           "Storage root for persistent video jobs", "Storage",
                           &video_root);
    video_parser.AddOption("", "--manifest", "PATH",
                           "Pinned H3 manifest override", "Storage",
                           &video_manifest);
    video_parser.AddOption("", "--ttl", "SEC",
                           "Completed-artifact TTL in seconds", "Storage",
                           &video_ttl_seconds);

    add_server_options(video_parser);
    if (!video_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "video");
      return 2;
    }
    if (video_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "video");
      return 0;
    }
    if (!valid_server_options()) {
      return 2;
    }

    if (video_ttl_seconds == 0 ||
        video_ttl_seconds >
            static_cast<std::uint64_t>(std::chrono::seconds::max().count())) {
      std::cerr << "Error: --ttl must be a positive duration\n";
      return 2;
    }

    if (video_model.empty()) {
      std::cerr << "Error: --model <DIR> is required for video server\n";
      PrintServeHelp("gufo", "video");
      return 2;
    }

    ModelLoadLog load_log("video_inventory", video_model);
    video_jobs = std::make_shared<server::VideoJobService>(
        server::VideoJobServiceOptions{
            .model_root = video_model,
            .source_manifest = video_manifest,
            .storage_root = video_root,
            .queue_capacity = 1,
            .artifact_ttl = std::chrono::seconds(
                static_cast<std::chrono::seconds::rep>(video_ttl_seconds)),
            .validate_model_inventory = true,
            .id_factory = {},
            .now = {},
            .runner = {},
        });
    if (!video_jobs->ready()) {
      std::cerr << "Error enabling MiniMax H3 video service: "
                << video_jobs->initialization_error() << '\n';
      return 1;
    }
    load_log.Complete(
        "model=minimax-h3 sessions=1 queue_capacity=1 weights=lazy", false);
  } else if (subcommand == "tts" || subcommand == "asr") {
    const bool is_tts = subcommand == "tts";
    SpeechOptions options{.context = is_tts ? 4096U : 1024U};
    ArgParser parser("gufo serve " + subcommand,
                     is_tts ? "Serve Qwen3-TTS speech synthesis."
                            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, is_tts, &options);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp("gufo", subcommand);
      return 0;
    }
    if (!valid_server_options())
      return 2;
    if (options.model.empty()) {
      std::cerr << "Error: --model <DIR> is required for " << subcommand
                << " server\n";
      return 2;
    }
    if (options.context < (is_tts ? 1U : 32U)) {
      std::cerr << "Error: --context must be at least " << (is_tts ? 1 : 32)
                << '\n';
      return 2;
    }
    ModelLoadLog load_log(subcommand, options.model);
    if (is_tts) {
      std::map<std::string, server::TtsVoicePreset> voice_presets;
      if (!BuildVoicePresets(options.voice_specs, options.voice_text_specs,
                             options.voice_lang_specs, &voice_presets,
                             &parse_err)) {
        std::cerr << "Error: " << parse_err << '\n';
        return 2;
      }
      tts = std::make_shared<server::TtsService>(server::TtsServiceOptions{
          .model_root = options.model,
          .native_context_tokens = options.context,
          .validate_model = true,
          .model_id = options.served_model_name,
          .voices = {},
          .voice_presets = std::move(voice_presets),
          .runner = {},
      });
      if (!tts->ready()) {
        std::cerr << "Error enabling Qwen3-TTS service: "
                  << tts->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + tts->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    } else {
      asr = std::make_shared<server::AsrService>(server::AsrServiceOptions{
          .model_root = options.model,
          .native_context_tokens = options.context,
          .validate_model = true,
          .model_id = options.served_model_name,
          .runner = {},
      });
      if (!asr->ready()) {
        std::cerr << "Error enabling Qwen3-ASR service: "
                  << asr->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + asr->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    }
  } else {
    // Default to LLM server
    LlmOptions llm;
    gufo::cli::ArgParser llm_parser(
        "gufo serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    AddLlmOptions(llm_parser, &llm, &systemone_options);
    add_server_options(llm_parser);
    if (!llm_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "llm");
      return 2;
    }
    server::SetHostMemoryBudget(llm.host_memory_budget);
    core::GgufReader::SetTensorOverride(llm.gguf_override);
    {
      auto limits = grammar::DefaultLimits();
      limits.cache_bytes = llm.grammar_cache_bytes;
      limits.compile_timeout =
          std::chrono::milliseconds(llm.grammar_compile_timeout_ms);
      grammar::SetDefaultLimits(limits);
    }
    if (llm_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "llm");
      return 0;
    }
    if (!valid_server_options()) {
      return 2;
    }
    bool sampling_valid = true;
    try {
      llm.sampling_config.Validate();
    } catch (const std::invalid_argument&) {
      sampling_valid = false;
    }
    if (llm.max_tokens < -1 || llm.max_tokens == 0 ||
        llm.max_tokens > std::numeric_limits<std::uint32_t>::max() ||
        llm.prefill_chunk_tokens == 0 || llm.max_pending_requests == 0 ||
        llm.max_pending_requests_per_client == 0 ||
        llm.max_pending_requests_per_client > llm.max_pending_requests ||
        llm.max_output_bytes == 0 || llm.max_buffered_output_bytes == 0 ||
        llm.max_buffered_output_bytes_total == 0 ||
        (!llm.cache_disk_directory.empty() && llm.cache_disk_bytes == 0) ||
        llm.request_timeout_ms >
            static_cast<std::uint64_t>(
                std::chrono::milliseconds::max().count()) ||
        !sampling_valid || llm.sampling_config.temperature > 2.0F) {
      std::cerr << "Error: sampling and scheduling limits are invalid\n";
      return 2;
    }
    if (llm.admission != "queue" && llm.admission != "refuse") {
      std::cerr << "Error: --admission must be queue or refuse\n";
      return 2;
    }
    if (llm.draft_tokens == 0 || llm.min_draft_tokens == 0 ||
        llm.min_draft_tokens > llm.draft_tokens ||
        llm.draft_tokens > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Error: speculative draft limits are invalid\n";
      return 2;
    }
    const auto reasoning_defaults =
        ResolveReasoningDefaults(llm.reasoning_mode, llm.reasoning_effort,
                                 llm.preserve_thinking, &parse_err);
    if (!reasoning_defaults.has_value()) {
      std::cerr << "Error: " << parse_err << "\n";
      return 2;
    }

    server::TextSpeculativeConfig speculative_config;
    if (llm.speculative_backend.empty() && !llm.dspark_model_path.empty()) {
      llm.speculative_backend = "dspark";
    }
    if (llm.speculative_backend.empty() || llm.speculative_backend == "off") {
      speculative_config.backend = server::TextSpeculativeBackend::kDisabled;
    } else if (llm.speculative_backend == "dflash2") {
      speculative_config.backend = server::TextSpeculativeBackend::kDFlash;
    } else if (llm.speculative_backend == "dspark") {
      speculative_config.backend = server::TextSpeculativeBackend::kDSpark;
    } else if (llm.speculative_backend == "mtp") {
      speculative_config.backend = server::TextSpeculativeBackend::kMtp;
    } else {
      std::cerr << "Error: speculative backend '" << llm.speculative_backend
                << "' is not supported by the HTTP server\n";
      return 2;
    }
    try {
      if (!llm.draft_policy.empty() &&
          speculative_config.backend != server::TextSpeculativeBackend::kDFlash)
        throw std::invalid_argument("--draft-policy requires DFlash2");
      speculative_config.dflash_policy =
          speculative::ParseDFlashDraftPolicy(llm.draft_policy);
    } catch (const std::invalid_argument& exception) {
      std::cerr << "Error: " << exception.what() << '\n';
      return 2;
    }
    if (speculative_config.backend == server::TextSpeculativeBackend::kDFlash &&
        (llm.dflash_model_path.empty() || llm.min_draft_tokens != 1)) {
      std::cerr << "Error: DFlash2 requires --dflash-model and "
                   "--min-draft-tokens 1; bound blocks with --draft-tokens\n";
      return 2;
    }
    speculative_config.draft_model_path =
        speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? llm.dspark_model_path
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? llm.mtp_model_path
            : llm.dflash_model_path;
    if (llm.mtp_policy != "off" && llm.mtp_policy != "adaptive") {
      std::cerr << "Error: --mtp-policy must be off or adaptive\n";
      return 2;
    }
    speculative_config.mtp_adaptive = llm.mtp_policy == "adaptive";
    speculative_config.mtp_draft_vocab = llm.mtp_draft_vocab;
    speculative_config.mtp_draft_vocab_size = llm.mtp_draft_vocab_size;
    speculative_config.max_draft_tokens =
        static_cast<std::uint32_t>(llm.draft_tokens);
    speculative_config.min_draft_tokens =
        static_cast<std::uint32_t>(llm.min_draft_tokens);
    if (llm.model.empty()) {
      std::cerr << "Error: --model <PATH> is required\n";
      return 2;
    }
    for (const auto& adapter : llm.adapters) {
      if (!llm.served_model_name.empty() &&
          adapter.starts_with(llm.served_model_name + "=")) {
        std::cerr << "Error: --adapter must not reuse --served-model-name\n";
        return 2;
      }
      // `<id>-flex` is reserved for the flex alias of every served id.
      const std::string_view name =
          std::string_view(adapter).substr(0, adapter.find('='));
      const std::string_view flex = server::TextGenerationBackend::kFlexSuffix;
      if (name.ends_with(flex) ||
          (!llm.served_model_name.empty() &&
           llm.served_model_name == std::string(name) + std::string(flex))) {
        std::cerr << "Error: --adapter name '" << name
                  << "' collides with a '-flex' model alias\n";
        return 2;
      }
      // With --fast-experts, `<id>-fast` is reserved for the fast model.
      const std::string_view fast = server::TextGenerationBackend::kFastSuffix;
      if (llm.fast_experts != 0 && name.ends_with(fast)) {
        std::cerr << "Error: --adapter name '" << name
                  << "' collides with a '-fast' model id\n";
        return 2;
      }
    }
    // The model id's suffixes are reserved too, or a literal id such as
    // `workhorse-fast-flex` would parse as an alias of another model.
    if (const auto reserved =
            ReservedModelIdSuffix(llm.served_model_name, llm.fast_experts)) {
      std::cerr << "Error: --served-model-name must not end in '" << *reserved
                << "'\n";
      return 2;
    }
    // Kev first: the text model sizes its KV pool from the memory left.
    if (!systemone_options.model.empty()) {
      systemone = LoadSystemOne(systemone_options, max_connections);
      if (systemone == nullptr)
        return 1;
    }
    std::string err;
    ModelLoadLog load_log("text", llm.model);
    backend = std::make_shared<server::InferenceBackend>();
    if (!backend->load(
            llm.model, &err, llm.max_context, session_count,
            server::TextPrefillPolicy{
                .decode_active_tokens = llm.prefill_chunk_tokens,
                .max_itl = std::chrono::milliseconds(llm.max_itl_ms),
            },
            server::TextSchedulerPolicy{
                .max_pending_requests = llm.max_pending_requests,
                .max_pending_requests_per_client =
                    llm.max_pending_requests_per_client,
                .flex_sessions = llm.flex_sessions,
                .refuse_when_full = llm.admission == "refuse",
                .max_output_bytes_per_request = llm.max_output_bytes,
                .max_buffered_output_bytes_per_request =
                    llm.max_buffered_output_bytes,
                .max_buffered_output_bytes_total =
                    llm.max_buffered_output_bytes_total,
                .request_timeout =
                    std::chrono::milliseconds{
                        static_cast<std::chrono::milliseconds::rep>(
                            llm.request_timeout_ms)},
                .log_progress = llm.log_progress,
            },
            speculative_config,
            server::TextDiskCacheConfig{
                .directory = llm.cache_disk_directory,
                .capacity_bytes = llm.cache_disk_bytes,
                .staging_capacity_bytes = llm.cache_disk_staging_bytes,
                .pin_capacity_bytes = llm.cache_pin_bytes,
                .pin_min_conversations = llm.cache_pin_conversations,
                .model_artifact_fingerprint = {},
            },
            llm.vision_model_path, llm.moe_routing, llm.kv_cache, llm.adapters,
            llm.dense_prefill, llm.gdn_state, llm.routed_prefill_a4,
            llm.prefill_capacity, llm.routed_down_w4, llm.routed_w4,
            llm.prefill_chunk, llm.dense_prefill_weights, llm.experts_used,
            llm.router_norm_k, llm.prefill_residual, llm.gguf_decode_override,
            llm.decode_twin_rows, llm.fast_experts)) {
      std::cerr << "Error loading model '" << llm.model << "': " << err << "\n";
      return 1;
    }
    backend->set_model_id(llm.served_model_name);
    // Without --served-model-name the id comes from the model file.
    if (const auto reserved =
            ReservedModelIdSuffix(backend->model_id(), llm.fast_experts)) {
      std::cerr << "Error: model id '" << backend->model_id() << "' ends in '"
                << *reserved << "'; choose another with --served-model-name\n";
      return 2;
    }
    if (systemone != nullptr)
      systemone->SetTextBusy([text = backend] {
        const auto m = text->metrics_snapshot();
        return m.sessions_active > 0 || m.prefilling_default > 0;
      });
    backend->set_sampling_defaults(
        llm.max_tokens < 0 ? 0 : static_cast<std::size_t>(llm.max_tokens),
        llm.sampling_config);
    backend->set_reasoning_defaults(*reasoning_defaults);
    const char* speculation =
        speculative_config.backend == server::TextSpeculativeBackend::kDFlash
            ? "dflash2"
        : speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? "dspark"
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? "mtp"
            : "off";
    std::string adapter_ids;
    for (const auto& id : backend->model_ids())
      if (id != backend->model_id())
        adapter_ids += (adapter_ids.empty() ? "" : ",") + id;
    load_log.Complete(
        "model=" + backend->model_id() +
        (adapter_ids.empty() ? "" : " adapters=" + adapter_ids) +
        " sessions=" + std::to_string(session_count) + " context_tokens=" +
        std::to_string(backend->max_context()) + " speculative=" + speculation +
        " draft_limit=" + std::to_string(speculative_config.max_draft_tokens) +
        " disk_cache=" +
        (llm.cache_disk_directory.empty() ? "off" : "enabled"));
  }

  std::optional<server::HttpServer> metrics_server;
  std::jthread metrics_thread;
  if (!metrics_address.empty()) {
    if (backend == nullptr) {
      std::cerr << "--metrics-address requires a loaded text model\n";
      return 1;
    }
    const auto colon = metrics_address.rfind(':');
    int metrics_port = 0;
    bool valid =
        colon != std::string::npos && colon + 1 < metrics_address.size();
    if (valid) {
      const std::string_view port_text(metrics_address.data() + colon + 1,
                                       metrics_address.size() - colon - 1);
      const auto [ptr, ec] = std::from_chars(
          port_text.data(), port_text.data() + port_text.size(), metrics_port);
      valid = ec == std::errc{} && ptr == port_text.data() + port_text.size();
    }
    if (!valid) {
      std::cerr << "--metrics-address must be HOST:PORT\n";
      return 1;
    }
    metrics_server.emplace(metrics_address.substr(0, colon), metrics_port,
                           backend, nullptr, nullptr, nullptr,
                           server::HttpServerOptions{.metrics_only = true},
                           nullptr, systemone);
    std::string metrics_error;
    if (!metrics_server->start(&metrics_error)) {
      std::cerr << "Error starting metrics listener: " << metrics_error << "\n";
      return 1;
    }
    metrics_thread = std::jthread(
        [&metrics_server] { metrics_server->run(/*handle_signals=*/false); });
    std::cerr << "Metrics listener (unauthenticated): " << metrics_address
              << "\n";
  }

  server::HttpServer server(
      host, port, backend, video_jobs, tts, asr,
      server::HttpServerOptions{
          .max_request_body_bytes = max_request_body_bytes,
          .max_connections = max_connections,
          .api_key = std::move(api_key),
          .allowed_origins = std::move(allowed_origins),
          .admin_key = std::move(admin_key),
      },
      images, systemone);
  std::string err;
  if (!server.start(&err)) {
    std::cerr << "Error starting HTTP server: " << err << "\n";
    return 1;
  }
  server.run(/*handle_signals=*/true);
  if (metrics_server) {
    metrics_server->stop();
  }
  return 0;
}

}  // namespace gufo::cli
