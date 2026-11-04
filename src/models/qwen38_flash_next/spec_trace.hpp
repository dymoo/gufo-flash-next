#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_TRACE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_TRACE_HPP_

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>

namespace gufo::models::qwen38_flash_next {

/// Diagnostics (GUFO_SPEC_TRACE=/path): one line per speculative decode
/// cycle with the draft source, widths, acceptance, per-depth draft
/// confidence and the phase wall times. Off unless the variable is set;
/// tracing requests compact draft candidates, so it never captures the
/// production graphs' timing exactly (a top-64 pass per draft round).
class SpecTrace {
public:
  static SpecTrace& Get() {
    static SpecTrace trace;
    return trace;
  }
  [[nodiscard]] bool enabled() const noexcept { return file_ != nullptr; }
  /// GUFO_SPEC_CONF=1 also traces greedy draft confidence and the rejected
  /// target token's candidate rank (adds a top-64 pass per draft round).
  [[nodiscard]] bool confidence() const noexcept { return confidence_; }
  [[nodiscard]] double Now() const noexcept {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - origin_)
        .count();
  }
  void Line(const char* format, ...) __attribute__((format(printf, 2, 3))) {
    if (file_ == nullptr)
      return;
    const std::lock_guard lock(mutex_);
    std::va_list args;
    va_start(args, format);
    std::vfprintf(file_, format, args);
    va_end(args);
    std::fputc('\n', file_);
    std::fflush(file_);
  }

private:
  SpecTrace() {
    if (const char* path = std::getenv("GUFO_SPEC_TRACE")) {
      file_ = std::fopen(path, "a");
      if (file_ != nullptr) {
        // Line times are milliseconds after this instant.
        std::fprintf(file_, "# origin_unix=%.3f\n",
                     std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());
      }
    }
    confidence_ = file_ != nullptr && std::getenv("GUFO_SPEC_CONF") != nullptr;
  }
  std::FILE* file_{nullptr};
  bool confidence_{false};
  std::mutex mutex_;
  std::chrono::steady_clock::time_point origin_{
      std::chrono::steady_clock::now()};
};

/// Benchmarks: every greedy cycle drafts exactly N MTP tokens (budget
/// permitting), bypassing the length controllers, to measure cycle cost per
/// width and the acceptance of every depth. GUFO_MTP_FORCE=N fixes N
/// (0..7); GUFO_MTP_FORCE_FILE=path rereads N from that file (a missing
/// file or "-" restores the controllers) so one server can sweep widths.
/// Output is unchanged: verification decides every draft.
[[nodiscard]] inline std::optional<std::uint32_t> ForcedMtpDrafts() noexcept {
  const auto parse = [](const char* text) -> std::optional<std::uint32_t> {
    if (text == nullptr || *text < '0' || *text > '9')
      return std::nullopt;
    const long n = std::strtol(text, nullptr, 10);
    return static_cast<std::uint32_t>(n > 7 ? 7 : n);
  };
  static const std::optional<std::uint32_t> fixed =
      parse(std::getenv("GUFO_MTP_FORCE"));
  static const char* const path = std::getenv("GUFO_MTP_FORCE_FILE");
  if (path == nullptr)
    return fixed;
  // Called from the scheduler thread only.
  static std::optional<std::uint32_t> cached;
  static auto checked = std::chrono::steady_clock::time_point{};
  const auto now = std::chrono::steady_clock::now();
  if (now - checked > std::chrono::milliseconds(200)) {
    checked = now;
    cached = std::nullopt;
    if (std::FILE* file = std::fopen(path, "r")) {
      char text[16] = {};
      if (std::fgets(text, sizeof(text), file) != nullptr)
        cached = parse(text);
      (void)std::fclose(file);
    }
  }
  return cached;
}

/// Experimental (GUFO_MTP_GATE=s, 0 < s < 1): greedy MTP chains draft up
/// to the budget and stop once the running product of the drafts' top-1
/// confidences falls below s. 0 (unset) keeps the length controllers.
[[nodiscard]] inline float MtpConfidenceGate() noexcept {
  static const float value = [] {
    const char* text = std::getenv("GUFO_MTP_GATE");
    const float s = text != nullptr ? std::strtof(text, nullptr) : 0.0F;
    return s > 0.0F && s < 1.0F ? s : 0.0F;
  }();
  return value;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SPEC_TRACE_HPP_
