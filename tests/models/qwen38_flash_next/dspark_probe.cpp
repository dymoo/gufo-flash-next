// DSpark acceptance probe: greedy target continuation of a prompt file with
// one drafter block per position and block size. Prints one TSV row per
// (position, block size): the anchor, the drafts and their confidences;
// acceptance is scored offline against the continuation itself.
//
//   qwen38_flash_next_dspark_probe <target.gguf> <dspark.gguf> <prompt.txt>
//       <tokens> [blocks=4,8,15]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/session_mode.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace q = gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s target.gguf dspark.gguf prompt.txt tokens "
                 "[blocks]\n",
                 argv[0]);
    return 2;
  }
  std::vector<std::uint32_t> blocks{4, 8, 15};
  if (argc > 5) {
    blocks.clear();
    std::stringstream list(argv[5]);
    for (std::string item; std::getline(list, item, ',');)
      blocks.push_back(static_cast<std::uint32_t>(std::stoul(item)));
  }
  std::ifstream file(argv[3]);
  const std::string text((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  const auto count = static_cast<std::size_t>(std::stoul(argv[4]));
  std::string error;
  q::ModelOptions options;
  options.max_context = 16384;
  options.dspark_model_path = argv[2];
  const auto model = q::Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                      options.max_context, &error);
  const auto prompt = model->Tokenize(text);
  if (!session || !session->Sync(prompt, &error)) {
    std::fprintf(stderr, "prompt: %s\n", error.c_str());
    return 1;
  }
  std::fprintf(stderr, "prompt tokens %zu\n", prompt.size());
  std::vector<std::int32_t> drafts(16);
  std::vector<float> confidence(16);
  for (std::size_t step = 0; step < count; ++step) {
    const auto logits = session->Logits();
    const auto anchor = static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    for (const auto rows : blocks) {
      const auto start = std::chrono::steady_clock::now();
      if (!session->DsparkDraft(anchor, std::span(drafts).first(rows),
                                std::span(confidence).first(rows), &error)) {
        std::fprintf(stderr, "draft: %s\n", error.c_str());
        return 1;
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      std::printf("%zu\t%d\t%u\t%.3f\t", step, anchor, rows, ms);
      for (std::uint32_t i = 0; i < rows; ++i)
        std::printf("%s%d", i ? "," : "", drafts[i]);
      std::printf("\t");
      for (std::uint32_t i = 0; i < rows; ++i)
        std::printf("%s%.4f", i ? "," : "", confidence[i]);
      std::printf("\n");
    }
    if (model->IsStopToken(anchor))
      break;
    if (!session->Evaluate(anchor, &error)) {
      std::fprintf(stderr, "evaluate: %s\n", error.c_str());
      return 1;
    }
  }
  std::fprintf(
      stderr, "text: %s\n",
      model->Decode(std::span(session->Tokens()).subspan(prompt.size()))
          .c_str());
  return 0;
}
