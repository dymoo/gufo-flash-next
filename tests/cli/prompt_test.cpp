#include "src/cli/prompt/prompt.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/chat_protocol.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->prompt_text == "Hello world");
  assert(opt->max_tokens == 128);
  assert(opt->sampling.temperature == 0.0F);
  assert(opt->use_chat_template);
  assert(opt->system_prompt.empty());
  assert(opt->reasoning_mode == "auto");
  assert(opt->reasoning_effort == "auto");
  assert(opt->preserve_thinking == "auto");
  assert(!opt->verbose);
  assert(opt->draft_tokens == 7);
  assert(opt->min_draft_tokens == 1);
}

void TestExplicitFlags() {
  const std::array<const char*, 10> args = {
      "--model", "model.gguf", "-n", "256",  "--temperature",
      "0.7",     "--raw",      "-v", "Test", "prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->max_tokens == 256);
  assert(opt->sampling.temperature > 0.69F &&
         opt->sampling.temperature < 0.71F);
  assert(!opt->use_chat_template);
  assert(opt->verbose);
  assert(opt->prompt_text == "Test prompt");
}

void TestFlashMtpFlags() {
  const char* args[] = {"--speculative",  "mtp", "--mtp-model",   "mtp.gguf",
                        "--draft-tokens", "3",   "--temperature", "0.7",
                        "--seed",         "1",   "Prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->speculative_backend == "mtp");
  assert(opt->draft_tokens == 3);
  assert(opt->sampling.temperature == 0.7F && opt->sampling.seed == 1);
}

void TestImageFlags() {
  const char* args[] = {"--image", "a.png",           "--image",
                        "b.jpg",   "--add-vision-id", "Compare"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt && opt->add_vision_id);
  assert((opt->image_paths == std::vector<std::string>{"a.png", "b.jpg"}));
  assert(!gufo::cli::ParsePromptOptions(std::array{"Compare"})->add_vision_id);
}

void TestInvalidFlags() {
  std::string err;
  const std::array<const char*, 1> args1 = {"--model"};
  assert(!gufo::cli::ParsePromptOptions(args1, &err).has_value());
  assert(!err.empty());

  const std::array<const char*, 2> args2 = {"-n", "not_a_number"};
  assert(!gufo::cli::ParsePromptOptions(args2, &err).has_value());

  const std::array<const char*, 2> args3 = {"-t", "invalid_float"};
  assert(!gufo::cli::ParsePromptOptions(args3, &err).has_value());

  const std::array<const char*, 4> args5 = {"--draft-tokens", "3",
                                            "--min-draft-tokens", "4"};
  assert(!gufo::cli::ParsePromptOptions(args5, &err).has_value());

  const std::array<const char*, 2> args6 = {"--top-p", "0"};
  assert(!gufo::cli::ParsePromptOptions(args6, &err).has_value());

  const std::array<const char*, 6> unsupported_floor = {
      "--speculative",      "dflash2", "--dflash-model", "draft.gguf",
      "--min-draft-tokens", "2"};
  assert(!gufo::cli::ParsePromptOptions(unsupported_floor, &err).has_value());
  assert(err.find("min-draft-tokens") != std::string::npos);
  for (const char* policy : {"fixed", "adaptive", "unknown"}) {
    const std::array<const char*, 6> args = {"--speculative",  "dflash2",
                                             "--dflash-model", "draft.gguf",
                                             "--draft-policy", policy};
    const auto parsed = gufo::cli::ParsePromptOptions(args, &err);
    assert(parsed.has_value() == (std::string_view(policy) != "unknown"));
    if (parsed)
      assert(parsed->draft_policy == policy);
  }
  const std::array<const char*, 2> policy_without_backend = {"--draft-policy",
                                                             "adaptive"};
  assert(!gufo::cli::ParsePromptOptions(policy_without_backend, &err));

  const std::array<const char*, 2> args8 = {"--chat-template", "qwen"};
  assert(!gufo::cli::ParsePromptOptions(args8, &err).has_value());

  const std::array<const char*, 2> args9 = {"--reasoning-budget", "1024"};
  assert(!gufo::cli::ParsePromptOptions(args9, &err).has_value());

  for (const auto* backend : {"dflash2", "mtp"}) {
    const std::array<const char*, 2> missing_path = {"--speculative", backend};
    assert(!gufo::cli::ParsePromptOptions(missing_path, &err).has_value());
    assert(err.find("requires --") != std::string::npos);
  }
  const std::array<const char*, 5> cpu_spec = {
      "--cpu", "--speculative", "dflash2", "--dflash-model", "draft.gguf"};
  assert(!gufo::cli::ParsePromptOptions(cpu_spec, &err).has_value());
  assert(err.find("ROCm") != std::string::npos);
  const std::array<const char*, 3> cpu_ar = {"--cpu", "--speculative", "off"};
  assert(gufo::cli::ParsePromptOptions(cpu_ar, &err).has_value());
}

void TestSamplingAndReasoningFlags() {
  const std::array<const char*, 29> args = {"--model",
                                            "model.gguf",
                                            "--top-p",
                                            "0.9",
                                            "--top-k",
                                            "50",
                                            "--min-p",
                                            "0.05",
                                            "--min-keep",
                                            "3",
                                            "-s",
                                            "42",
                                            "--repeat-penalty",
                                            "1.15",
                                            "--repeat-last-n",
                                            "128",
                                            "--frequency-penalty",
                                            "0.25",
                                            "--presence-penalty",
                                            "0.5",
                                            "--think",
                                            "on",
                                            "--reasoning-effort",
                                            "high",
                                            "--preserve-thinking",
                                            "off",
                                            "--no-display-prompt",
                                            "-p",
                                            "Explicit prompt text"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->sampling.top_p > 0.89F && opt->sampling.top_p < 0.91F);
  assert(opt->sampling.top_k == 50);
  assert(opt->sampling.min_p > 0.04F && opt->sampling.min_p < 0.06F);
  assert(opt->sampling.min_keep == 3);
  assert(opt->sampling.seed == 42);
  assert(opt->sampling.repeat_penalty > 1.14F &&
         opt->sampling.repeat_penalty < 1.16F);
  assert(opt->sampling.repeat_last_n == 128);
  assert(opt->sampling.frequency_penalty > 0.24F &&
         opt->sampling.frequency_penalty < 0.26F);
  assert(opt->sampling.presence_penalty > 0.49F &&
         opt->sampling.presence_penalty < 0.51F);
  assert(opt->reasoning_mode == "on");
  assert(opt->reasoning_effort == "high");
  assert(opt->preserve_thinking == "off");
  assert(!opt->display_prompt);
  assert(opt->prompt_text == "Explicit prompt text");
}

/// The request `gufo prompt` renders for these flags.
gufo::server::ChatRequest PromptRequest(const std::vector<const char*>& args) {
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  auto request = gufo::cli::PromptChatRequest(*opt);
  request.messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                opt->prompt_text);
  return request;
}

void TestChatProtocolRendering() {
  namespace ds4 = gufo::models::deepseek_v4_flash;
  using gufo::core::TemplateFamily;
  using gufo::server::ProtocolFor;
  const auto flagged = PromptRequest(
      {"--system", "Be brief.", "--think", "off", "--add-vision-id", "Hi"});
  assert(flagged.messages.size() == 2 &&
         flagged.messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         flagged.messages[0].content == "Be brief." &&
         flagged.messages[1].content == "Hi" &&
         flagged.reasoning.enabled == false && flagged.add_vision_id);
  // Qwen3.8 and DeepSeek V4 prompts match what the commands rendered before
  // they adopted the chat protocol.
  for (const auto& flags : std::vector<std::vector<const char*>>{
           {"Hi"},
           {"--system", "Be brief.", "--think", "off", "Hi"},
           {"--reasoning-effort", "high", "--preserve-thinking", "on", "Hi"},
           {"--think", "on", "--add-vision-id", "Hi"}}) {
    const auto request = PromptRequest(flags);
    const auto qwen = gufo::tokenization::QwenChatTemplate::Render(
        request.messages, gufo::tokenization::ResolveQwenChatOptions(
                              request.reasoning, request.add_vision_id));
    assert(qwen &&
           ProtocolFor(TemplateFamily::kQwen38).RenderText(request) == *qwen);
    std::vector<ds4::ChatMessage> messages;
    for (const auto& message : request.messages)
      messages.push_back({.role = std::string(ToString(message.role)),
                          .content = message.content});
    const auto& reasoning = request.reasoning;
    assert(ProtocolFor(TemplateFamily::kDeepSeekV4).RenderText(request) ==
           ds4::RenderChat(
               messages, {.enable_thinking = reasoning.enabled.value_or(false),
                          .reasoning_effort = reasoning.effort.value_or(
                              gufo::ReasoningEffort::kLow),
                          .preserve_thinking =
                              reasoning.preserve_thinking.value_or(false)}));
  }
  // LFM2.5 renders with its own template, not Qwen ChatML.
  assert(ProtocolFor(TemplateFamily::kLfm2)
             .RenderText(PromptRequest({"--system", "Be brief.", "Hi"})) ==
         "<|startoftext|><|im_start|>system\nBe brief.<|im_end|>\n"
         "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>");
}

int main() {
  TestDefaultOptions();
  TestExplicitFlags();
  TestSamplingAndReasoningFlags();
  TestFlashMtpFlags();
  TestImageFlags();
  TestInvalidFlags();
  TestChatProtocolRendering();
  std::cout << "All prompt CLI tests passed.\n";
  return 0;
}
