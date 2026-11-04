// Builds a frequency-ranked draft vocabulary for the MTP draft head
// (GUFO_MTP_VOCAB) from text corpora, and reports token coverage.
//
//   qwen38_flash_next_spec_vocab MODEL.gguf OUT.txt FILE... [--eval FILE...]
//
// Each training file contributes its normalized token frequencies with equal
// weight, so a large corpus does not drown a small one. Chat-template control
// tokens come first. `--eval` files are only measured: the share of their
// tokens inside the top N ranked ids, for several N (held-out coverage).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf OUT.txt FILE... [--eval FILE...]\n",
                 argv[0]);
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 1;
  }
  const std::size_t vocab = tokenizer->GetVocabSize();
  std::vector<std::string> train, eval;
  bool evaluating = false;
  // --by-id ranks tokens by id (BPE merge order, i.e. the tokenizer's own
  // training frequency) instead of the corpus counts.
  bool by_id = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--eval") {
      evaluating = true;
      continue;
    }
    if (arg == "--by-id") {
      by_id = true;
      continue;
    }
    (evaluating ? eval : train).push_back(arg);
  }
  std::vector<double> score(vocab, 0.0);
  for (const auto& path : train) {
    const auto ids = tokenizer->Encode(ReadFile(path));
    if (ids.empty())
      continue;
    for (const auto id : ids)
      if (id < vocab)
        score[id] += 1.0 / static_cast<double>(ids.size());
    std::fprintf(stderr, "train %s: %zu tokens\n", path.c_str(), ids.size());
  }
  // Control tokens of the chat template rank first.
  const auto control = tokenizer->Encode(
      "<|im_start|><|im_end|><|endoftext|><think></think><tool_call>"
      "</tool_call><tool_response></tool_response>");
  for (const auto id : control)
    if (id < vocab)
      score[id] = 1e9;
  if (by_id)
    for (std::uint32_t id = 0; id < vocab; ++id)
      if (score[id] < 1e9)
        score[id] = 1.0 / (1.0 + id);
  std::vector<std::uint32_t> order;
  for (std::uint32_t id = 0; id < vocab; ++id)
    if (score[id] > 0)
      order.push_back(id);
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return score[a] > score[b]; });
  std::FILE* out = std::fopen(argv[2], "w");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }
  for (const auto id : order)
    std::fprintf(out, "%u\n", id);
  std::fclose(out);
  std::fprintf(stderr, "ranked %zu of %zu tokens -> %s\n", order.size(), vocab,
               argv[2]);
  std::vector<std::uint32_t> rank(vocab, static_cast<std::uint32_t>(vocab));
  for (std::size_t r = 0; r < order.size(); ++r)
    rank[order[r]] = static_cast<std::uint32_t>(r);
  const std::size_t sizes[] = {4096, 8192, 16384, 24576, 32768, 49152, 65536};
  for (const auto& path : eval) {
    const auto ids = tokenizer->Encode(ReadFile(path));
    std::printf("eval %s tokens=%zu", path.c_str(), ids.size());
    for (const auto n : sizes) {
      std::size_t inside = 0;
      for (const auto id : ids)
        inside += id < vocab && rank[id] < n ? 1 : 0;
      std::printf(" top%zu=%.4f", n,
                  ids.empty() ? 0.0
                              : static_cast<double>(inside) /
                                    static_cast<double>(ids.size()));
    }
    std::printf("\n");
  }
  return 0;
}
