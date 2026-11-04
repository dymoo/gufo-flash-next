#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Brute force: among the latest kCandidates occurrences of the suffix, the
// continuation of the one with the longest backward match (ties: latest).
std::vector<std::int32_t> Reference(const std::vector<std::int32_t>& tokens,
                                    std::int32_t anchor, std::size_t want) {
  std::vector<std::int32_t> seq = tokens;
  seq.push_back(anchor);
  const auto size = tokens.size();
  if (size < 2)
    return {};
  std::size_t best = 0, best_length = 0, seen = 0;
  for (std::size_t end = size;
       end >= 3 && seen < qfn::PromptLookup::kCandidates; --end) {
    if (seq[end - 3] == seq[size - 2] && seq[end - 2] == seq[size - 1] &&
        seq[end - 1] == anchor) {
      ++seen;
      std::size_t length = 3;
      while (length < qfn::PromptLookup::kMaxMatch && length <= end - 1 &&
             seq[end - 1 - length] == seq[size - length])
        ++length;
      if (length > best_length) {
        best_length = length;
        best = end;
      }
    }
  }
  if (best_length == 0)
    return {};
  std::vector<std::int32_t> out;
  for (std::size_t n = 0; n < want; ++n) {
    out.push_back(seq[best + n]);
    seq.push_back(out.back());
  }
  return out;
}

std::vector<std::int32_t> Find(qfn::PromptLookup& lookup,
                               const std::vector<std::int32_t>& tokens,
                               std::int32_t anchor, std::size_t want) {
  std::array<std::int32_t, 7> drafts{};
  const auto n = lookup.Find(tokens, anchor, std::span(drafts).first(want));
  return {drafts.begin(), drafts.begin() + n};
}

}  // namespace

int main() {
  try {
    qfn::PromptLookup lookup;
    Require(Find(lookup, {1, 2, 3, 4, 5, 6, 1, 2}, 3, 7) ==
                std::vector<std::int32_t>{4, 5, 6, 1, 2, 3, 4},
            "continuation did not copy through the anchor");
    lookup.Reset();
    Require(Find(lookup, {1, 2, 3, 9, 1, 2, 3, 7, 8, 1, 2}, 3, 2) ==
                std::vector<std::int32_t>{7, 8},
            "latest occurrence was not preferred among equal matches");
    lookup.Reset();
    Require(Find(lookup, {7, 1, 2, 3, 4, 8, 1, 2, 3, 5, 6, 7, 1, 2}, 3, 2) ==
                    std::vector<std::int32_t>{4, 8} &&
                lookup.MatchLength() == 4,
            "longest backward match was not preferred");
    lookup.Reset();
    Require(
        Find(lookup, {5, 5, 5}, 5, 4) == std::vector<std::int32_t>{5, 5, 5, 5},
        "a run did not copy itself");
    Require(Find(lookup, {5, 5, 5}, 6, 4).empty(), "phantom match");
    Require(Find(lookup, {5}, 6, 4).empty(), "short context matched");

    // Incremental growth, rollback and replacement against brute force.
    std::uint64_t state = 1;
    const auto next = [&] {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<std::int32_t>((state >> 33) % 6);
    };
    std::vector<std::int32_t> tokens;
    std::size_t hits = 0, checks = 0;
    for (unsigned step = 0; step < 20000; ++step) {
      if (step % 4000 == 3999) {
        tokens.resize(tokens.size() / 3);  // rollback without Reset
        for (auto& t : tokens)
          t = next();  // and replaced content
      }
      const auto anchor = next();
      const auto want = 1 + step % 7;
      const auto got = Find(lookup, tokens, anchor, want);
      const auto ref = Reference(tokens, anchor, want);
      Require(got.empty() || got == ref,
              "lookup returned something other than the latest match");
      checks += !ref.empty();
      hits += !got.empty();
      tokens.push_back(anchor);
    }
    Require(hits * 10 >= checks * 9, "lookup missed too many matches");
    std::cout << "prompt lookup matches brute force (" << hits << "/" << checks
              << " hits)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
