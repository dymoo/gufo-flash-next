#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Prompt-lookup drafts: the tokens that followed an earlier occurrence of
/// the current 3-token suffix (last two committed tokens plus the sampled
/// anchor) in the session's prompt and generated text. Among the latest
/// kCandidates occurrences, the one whose match extends furthest back wins
/// (ties: the latest), so copied code and tool output resume at the right
/// place rather than at any repeated trigram.
///
/// The index is a hash table of the latest continuation position per slot
/// plus a chain to each position's predecessor in its slot (LZ77 style),
/// extended incrementally with each committed token (~12 bytes per context
/// token). Entries are never trusted: every candidate is re-checked against
/// the tokens, so collisions, rollbacks and replaced contexts can only cost
/// a draft, never produce a wrong one. Verification stays exact either way.
class PromptLookup {
public:
  static constexpr std::size_t kSuffix = 3;
  /// Verified occurrences compared per lookup.
  static constexpr std::size_t kCandidates = 32;
  /// Longest backward match measured (tokens, including the suffix).
  static constexpr std::size_t kMaxMatch = 64;

  /// Writes up to drafts.size() continuation tokens; returns their count.
  std::size_t Find(std::span<const std::int32_t> tokens, std::int32_t anchor,
                   std::span<std::int32_t> drafts) {
    match_ = 0;
    if (drafts.empty() || tokens.size() < kSuffix - 1)
      return 0;
    Index(tokens);
    const auto size = tokens.size();
    // The virtual sequence tokens + anchor + drafts lets a match overlap
    // its own continuation (runs and short periods copy themselves).
    const auto at = [&](std::size_t i) {
      return i < size ? tokens[i] : i == size ? anchor : drafts[i - size - 1];
    };
    const std::int32_t key[kSuffix] = {tokens[size - 2], tokens[size - 1],
                                       anchor};
    std::uint32_t best = 0;
    std::size_t best_length = 0;
    std::size_t verified = 0;
    // Collisions share chains; bound the walk, not only the verified count.
    std::uint32_t end = table_[Slot(key)];
    for (std::size_t step = 0;
         step < 4 * kCandidates && verified < kCandidates && end >= kSuffix &&
         end <= size;
         ++step, end = prev_[end]) {
      bool gram = true;
      for (std::size_t i = 0; i < kSuffix && gram; ++i)
        gram = at(end - kSuffix + i) == key[i];
      if (gram) {
        ++verified;
        // Match length backwards from the suffix: position size (the
        // anchor) aligns with end - 1.
        std::size_t length = kSuffix;
        while (length < kMaxMatch && length <= end - 1 &&
               at(end - 1 - length) == at(size - length))
          ++length;
        if (length > best_length) {
          best_length = length;
          best = end;
        }
      }
      if (prev_[end] >= end)
        break;  // chains only point backwards
    }
    if (best_length == 0)
      return 0;
    match_ = static_cast<std::uint32_t>(best_length);
    std::size_t n = 0;
    while (n < drafts.size()) {
      drafts[n] = at(best + n);
      ++n;
    }
    return n;
  }

  /// Backward match length (tokens) of the last successful Find, else 0.
  [[nodiscard]] std::uint32_t MatchLength() const noexcept { return match_; }

  void Reset() {
    table_.clear();
    prev_.clear();
    indexed_ = 0;
  }

private:
  static constexpr std::size_t kMinSlots = 4096;

  [[nodiscard]] std::size_t Slot(const std::int32_t* gram) const noexcept {
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (std::size_t i = 0; i < kSuffix; ++i)
      h = (h ^ static_cast<std::uint32_t>(gram[i])) * 0xBF58476D1CE4E5B9ULL;
    return static_cast<std::size_t>(h >> 32) & (table_.size() - 1);
  }

  // Entry for the gram ending at position e (exclusive) stores e, the
  // position of its continuation; e == tokens.size() continues into the
  // anchor. prev_[e] is the previous entry of e's slot.
  void Index(std::span<const std::int32_t> tokens) {
    if (indexed_ > tokens.size())
      Reset();  // Context shrank: rebuild rather than trust old positions.
    if (table_.size() < 4 * tokens.size() + 4) {
      std::size_t slots = kMinSlots;
      while (slots < 4 * tokens.size() + 4)
        slots *= 2;
      table_.assign(slots, 0);
      indexed_ = 0;
    }
    prev_.resize(tokens.size() + 1, 0);
    for (auto e = std::max(indexed_ + 1, kSuffix); e <= tokens.size(); ++e) {
      auto& head = table_[Slot(tokens.data() + e - kSuffix)];
      prev_[e] = head;
      head = static_cast<std::uint32_t>(e);
    }
    indexed_ = tokens.size();
  }

  std::vector<std::uint32_t> table_;
  std::vector<std::uint32_t> prev_;
  std::size_t indexed_{0};
  std::uint32_t match_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
