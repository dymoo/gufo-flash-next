// Kev packed-pass timing on synthetic support tickets (tools/kev/load.py's
// shape: a short new text and six questions).
//   KEV_GGUF=kev-4b-q8_0.gguf kev_bench [batch...]
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/kev/model.hpp"
#include "src/models/kev/systemone.hpp"

namespace kev = gufo::models::kev;

int main(int argc, char** argv) {
  const char* gguf = std::getenv("KEV_GGUF");
  if (gguf == nullptr) {
    std::cout << "SKIP kev_bench: set KEV_GGUF\n";
    return 77;
  }
  std::string error;
  auto model = kev::Model::Load(gguf, {}, &error);
  if (model == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  const char* words[] = {"order",   "refund",  "invoice", "charged",  "twice",
                         "parcel",  "damaged", "login",   "password", "reset",
                         "account", "urgent",  "manager", "delay",    "angry"};
  std::mt19937 rng(7);
  const auto request = [&](int n) {
    std::string state;
    for (int i = 0; i < n; ++i)
      state += std::string(i ? " " : "") + words[rng() % 15];
    return R"({"state": ")" + state +
           R"(", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments, invoices, refunds", "shipping": "Delivery problems", "access": "Login and account access", "other": null}}, "urgent": {"type": "noul", "instructions": "Does the customer need help today?"}, "tone": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Annoyed", "Frustrated", "Angry"]}, "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}, "language": {"type": "choice", "instructions": "Which language is the message in?", "criteria": {"en": null, "fr": null, "de": null, "es": null}}, "churn": {"type": "noul", "instructions": "Is the customer threatening to leave?"}}})";
  };
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i)
    sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty())
    sizes = {1, 8, 32};
  for (const int batch : sizes) {
    for (int round = 0; round < 3; ++round) {
      std::vector<kev::Encoding> encodings;
      std::size_t tokens = 0;
      for (int i = 0; i < batch; ++i) {
        auto parsed = kev::ParseRequest(request(60), &error);
        encodings.push_back(
            *model->encoder().Encode(*parsed, model->limits(), &error));
        tokens += encodings.back().Tokens();
      }
      std::vector<const kev::Encoding*> pointers;
      for (const auto& e : encodings)
        pointers.push_back(&e);
      const auto before = model->stats();
      const auto start = std::chrono::steady_clock::now();
      (void)model->ScoreBatch(pointers);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      const auto after = model->stats();
      std::cout << "batch " << batch << " tokens " << tokens << " total_ms "
                << ms << " gpu_ms " << after.gpu_ms - before.gpu_ms
                << " head_ms " << after.head_ms - before.head_ms << " passes "
                << after.passes - before.passes << " tok/s "
                << tokens / (ms / 1000.0) << '\n';
    }
  }
  return 0;
}
