#include "src/core/model_admission.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

using gufo::core::Engine;
using gufo::core::LoadPlan;
using gufo::core::LoadRequest;
using gufo::core::Surface;
using gufo::core::TemplateFamily;
using KvCacheFormat = gufo::models::qwen38_flash_next::KvCacheFormat;
using Metadata =
    std::map<std::string, std::variant<std::uint32_t, std::string>>;

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

/// Metadata-only GGUF: admission reads no tensors.
std::vector<std::uint8_t> Gguf(const Metadata& fields) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    if (const auto* number = std::get_if<std::uint32_t>(&value)) {
      pod(std::uint32_t{4});
      pod(*number);
    } else {
      pod(std::uint32_t{8});
      text(std::get<std::string>(value));
    }
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  return bytes;
}

Metadata Model(const std::string& architecture) {
  return {{"general.architecture", architecture},
          {architecture + ".context_length", std::uint32_t{262144}},
          {architecture + ".attention.key_length", std::uint32_t{256}},
          {architecture + ".embedding_length", std::uint32_t{5120}}};
}

std::optional<LoadPlan> Admit(const Metadata& fields,
                              const LoadRequest& request, std::string* error) {
  const auto bytes = Gguf(fields);
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  if (reader == nullptr) {
    std::cerr << "FAIL: synthetic GGUF: " << *error << '\n';
    std::exit(1);
  }
  error->clear();
  return gufo::core::Admit(*reader, request, error);
}

void ExpectAdmit(const std::string& architecture, const LoadRequest& request,
                 Engine engine, const std::string& what) {
  std::string error;
  const auto plan = Admit(Model(architecture), request, &error);
  Expect(plan.has_value(), what + " admitted: " + error);
  if (plan)
    Expect(plan->engine == engine, what + " routes to its engine");
}

void ExpectReject(const Metadata& fields, const LoadRequest& request,
                  const std::string& message) {
  std::string error;
  const auto plan = Admit(fields, request, &error);
  Expect(!plan.has_value() && error == message,
         "expected \"" + message + "\", got \"" + error + "\"");
}

void ExpectReject(const std::string& architecture, const LoadRequest& request,
                  const std::string& message) {
  ExpectReject(Model(architecture), request, message);
}

constexpr LoadRequest kCli{.surface = Surface::kCli};

void Routing() {
  for (const auto surface : {Surface::kServe, Surface::kCli}) {
    const LoadRequest request{.surface = surface};
    ExpectAdmit("qwen4exp", request, Engine::kFlashNext, "qwen4exp");
    ExpectAdmit("qwen35moe", request, Engine::kFlashNext, "qwen35moe");
    ExpectAdmit("lfm2moe", request, Engine::kFlashNext, "lfm2moe");
    ExpectAdmit("deepseek4", request, Engine::kDeepSeek, "deepseek4");
    ExpectAdmit("qwen35", request, Engine::kQwen, "qwen35");
    ExpectAdmit("qwen3", request, Engine::kQwen, "other architecture");
  }
  ExpectAdmit("mimo2", {}, Engine::kMimo, "mimo2 under serve");
  ExpectReject("mimo2", kCli,
               "MiMo-V2.6-Flash (mimo2) runs only under gufo serve");

  std::string error;
  const auto plan = Admit(Model("lfm2moe"), kCli, &error);
  Expect(plan && plan->family == "LFM2.5 (lfm2moe)", "family names the GGUF");
}

void TemplateFamilies() {
  const std::pair<std::string, TemplateFamily> rows[] = {
      {"qwen4exp", TemplateFamily::kQwen38},
      {"qwen35moe", TemplateFamily::kQwen35},
      {"lfm2moe", TemplateFamily::kLfm2},
      {"deepseek4", TemplateFamily::kDeepSeekV4},
      {"mimo2", TemplateFamily::kMimo},
      {"qwen35", TemplateFamily::kQwen38},
      {"qwen3", TemplateFamily::kQwen38}};
  for (const auto& [architecture, chat] : rows) {
    std::string error;
    const auto plan = Admit(Model(architecture), {}, &error);
    Expect(plan && plan->template_family == chat,
           architecture + " renders with its template family: " + error);
  }
}

void Context() {
  std::string error;
  auto plan = Admit(Model("qwen35moe"), {}, &error);
  Expect(plan && plan->max_context == 262144, "0 selects native context");
  plan = Admit(Model("qwen35moe"), {.max_context = 4096}, &error);
  Expect(plan && plan->max_context == 4096, "explicit context passes");
  ExpectReject({{"general.architecture", std::string{"qwen35"}}}, {},
               "GGUF has no valid native context length; specify --context");
}

void KvCache() {
  for (const auto* format : {"f16", "tq8", "tq4", "tq3"})
    ExpectAdmit("lfm2moe", {.kv_cache = format}, Engine::kFlashNext,
                std::string("lfm2moe --kv-cache ") + format);
  for (const auto* architecture : {"qwen4exp", "qwen35moe"}) {
    ExpectAdmit(architecture, {.surface = Surface::kCli, .kv_cache = "tq4"},
                Engine::kFlashNext, std::string(architecture) + " tq4");
    ExpectAdmit(architecture, {.kv_cache = "tq3"}, Engine::kFlashNext,
                std::string(architecture) + " tq3");
  }
  std::string error;
  const auto plan = Admit(Model("lfm2moe"), {.kv_cache = "tq8"}, &error);
  Expect(plan && plan->kv_cache == KvCacheFormat::kTq8, "tq8 reaches plan");

  ExpectReject("qwen4exp", {.kv_cache = "tq8"},
               "--kv-cache tq8: Qwen3.8-Flash-Next (qwen4exp) supports f16, "
               "tq4 or tq3");
  ExpectReject("qwen35moe", {.surface = Surface::kCli, .kv_cache = "tq8"},
               "--kv-cache tq8: MiniCPM-V 4.7 / Qwen3.6 (qwen35moe) supports "
               "f16, tq4 or tq3");
  ExpectReject("deepseek4", {.kv_cache = "tq4"},
               "--kv-cache tq4: DeepSeek V4 Flash (deepseek4) supports only "
               "f16");
  ExpectReject("qwen35", {.surface = Surface::kCli, .kv_cache = "tq3"},
               "--kv-cache tq3: Qwen3.8 (qwen35) supports only f16");
  ExpectReject("mimo2", {.kv_cache = "tq8"},
               "--kv-cache tq8: MiMo-V2.6-Flash (mimo2) supports only f16");
  ExpectReject("lfm2moe", {.kv_cache = "q8"},
               "--kv-cache must be f16, tq8, tq4 or tq3");

  auto narrow = Model("qwen35moe");
  narrow["qwen35moe.attention.key_length"] = std::uint32_t{128};
  ExpectReject(narrow, {.kv_cache = "tq4"},
               "--kv-cache tq4 needs 256-wide attention heads; this MiniCPM-V "
               "4.7 / Qwen3.6 (qwen35moe) GGUF has 128");
}

void Speculative() {
  ExpectAdmit("qwen4exp",
              {.speculative = "mtp", .draft_model_path = "mtp.gguf"},
              Engine::kFlashNext, "Flash-Next MTP");
  ExpectAdmit("deepseek4",
              {.surface = Surface::kCli,
               .speculative = "dspark",
               .draft_model_path = "dspark.gguf"},
              Engine::kDeepSeek, "DeepSeek DSpark");
  ExpectAdmit("qwen35",
              {.speculative = "dflash2", .draft_model_path = "dflash.gguf"},
              Engine::kQwen, "Qwen DFlash2 under serve");
  ExpectAdmit("qwen35",
              {.surface = Surface::kCli,
               .speculative = "mtp",
               .draft_model_path = "mtp.gguf",
               .min_draft_tokens = 3},
              Engine::kQwen, "Qwen MTP adapts its draft floor");
  ExpectAdmit("qwen35moe",
              {.speculative = "dspark", .draft_model_path = "dspark.gguf"},
              Engine::kFlashNext, "MiniCPM-V 4.7 DSpark under serve");
  ExpectAdmit("lfm2moe", {.speculative = "off"}, Engine::kFlashNext,
              "explicit off");

  ExpectReject("lfm2moe", {.speculative = "mtp", .draft_model_path = "m"},
               "--speculative mtp: gufo serve runs LFM2.5 (lfm2moe) without "
               "speculative decoding");
  ExpectReject(
      "qwen35moe",
      {.surface = Surface::kCli, .speculative = "mtp", .draft_model_path = "m"},
      "--speculative mtp: gufo prompt/chat/bench runs MiniCPM-V 4.7 / "
      "Qwen3.6 (qwen35moe) without speculative decoding");
  ExpectReject("qwen35", {.speculative = "mtp", .draft_model_path = "m"},
               "--speculative mtp: gufo serve runs Qwen3.8 (qwen35) with off "
               "or dflash2");
  ExpectReject("qwen4exp", {.speculative = "dflash2", .draft_model_path = "d"},
               "--speculative dflash2: gufo serve runs Qwen3.8-Flash-Next "
               "(qwen4exp) with off or mtp");
  ExpectReject("deepseek4", {.speculative = "mtp", .draft_model_path = "m"},
               "--speculative mtp: gufo serve runs DeepSeek V4 Flash "
               "(deepseek4) with off or dspark");
  ExpectReject("mimo2", {.speculative = "dspark", .draft_model_path = "d"},
               "--speculative dspark: gufo serve runs MiMo-V2.6-Flash (mimo2) "
               "without speculative decoding");
  ExpectReject("qwen3", {.surface = Surface::kCli, .speculative = "dspark"},
               "--speculative dspark: gufo prompt/chat/bench runs Qwen (qwen3) "
               "with off, dflash2 or mtp");
  ExpectReject("deepseek4", {.speculative = "dspark"},
               "--speculative dspark requires --dspark-model");
  ExpectReject("qwen4exp", {.surface = Surface::kCli, .speculative = "mtp"},
               "--speculative mtp requires --mtp-model");
  ExpectReject("qwen35", {.speculative = "dflash2"},
               "--speculative dflash2 requires --dflash-model");
  ExpectReject(
      "deepseek4",
      {.speculative = "dspark", .draft_model_path = "d", .min_draft_tokens = 2},
      "--speculative dspark requires --min-draft-tokens 1");
  ExpectReject(
      "qwen4exp",
      {.speculative = "mtp", .draft_model_path = "m", .min_draft_tokens = 2},
      "--speculative mtp requires --min-draft-tokens 1");
  ExpectReject("qwen35",
               {.speculative = "dflash2",
                .draft_model_path = "d",
                .min_draft_tokens = 2},
               "--speculative dflash2 requires --min-draft-tokens 1");
  ExpectReject("qwen35", {.speculative = "eagle"},
               "--speculative must be dflash2, dspark, mtp or off");
}

void Vision() {
  std::string error;
  auto plan = Admit(Model("qwen35"), {}, &error);
  Expect(plan && plan->vision, "5120-wide qwen35 opens its vision encoder");
  auto narrow = Model("qwen35");
  narrow["qwen35.embedding_length"] = std::uint32_t{4096};
  plan = Admit(narrow, {}, &error);
  Expect(plan && !plan->vision, "other qwen35 widths are text only");
  ExpectReject(narrow, {.vision_model_path = "mmproj.gguf"},
               "--mmproj: Qwen3.8 (qwen35) takes image input only at "
               "embedding width 5120; this GGUF has 4096");
  ExpectAdmit("qwen35", {.surface = Surface::kCli, .image_input = true},
              Engine::kQwen, "Qwen3.8-27B image input");
  ExpectAdmit("qwen4exp", {.surface = Surface::kCli, .image_input = true},
              Engine::kFlashNext, "Flash-Next image input");
  ExpectAdmit("qwen35moe", {.vision_model_path = "mmproj.gguf"},
              Engine::kFlashNext, "MiniCPM-V vision under serve");

  ExpectReject("qwen35moe", {.surface = Surface::kCli, .image_input = true},
               "--image: gufo prompt/chat cannot feed MiniCPM-V 4.7 / Qwen3.6 "
               "(qwen35moe) images; use gufo serve");
  ExpectReject("lfm2moe", {.vision_model_path = "mmproj.gguf"},
               "--mmproj: LFM2.5 (lfm2moe) has no image input");
  ExpectReject("deepseek4", {.surface = Surface::kCli, .image_input = true},
               "--image: DeepSeek V4 Flash (deepseek4) has no image input");
  ExpectReject("mimo2", {.vision_model_path = "mmproj.gguf"},
               "--mmproj: MiMo-V2.6-Flash (mimo2) has no image input");
  ExpectReject("qwen3", {.vision_model_path = "mmproj.gguf"},
               "--mmproj: Qwen (qwen3) has no image input");
}

void RoutingPolicy() {
  for (const auto* architecture : {"qwen4exp", "qwen35moe", "lfm2moe"})
    ExpectAdmit(architecture, {.moe_routing = "affinity:2:0.5"},
                Engine::kFlashNext, std::string(architecture) + " routing");
  ExpectReject("deepseek4", {.moe_routing = "mass:2:0.9"},
               "--moe-routing mass:2:0.9: DeepSeek V4 Flash (deepseek4) "
               "supports only exact");
  ExpectReject("qwen35", {.moe_routing = "budget:2:64"},
               "--moe-routing budget:2:64: Qwen3.8 (qwen35) supports only "
               "exact");
}

void DensePrefill() {
  std::string error;
  const auto plan = Admit(Model("qwen4exp"), {.dense_prefill = "w4a4"}, &error);
  Expect(plan.has_value() && plan->dense_prefill_w4a4,
         "qwen4exp admits --dense-prefill w4a4: " + error);
  const auto exact = Admit(Model("qwen4exp"), {}, &error);
  Expect(exact.has_value() && !exact->dense_prefill_w4a4,
         "--dense-prefill defaults to f16");
  ExpectReject("qwen4exp", {.dense_prefill = "w8"},
               "--dense-prefill must be f16 or w4a4");
  ExpectReject("qwen35moe", {.dense_prefill = "w4a4"},
               "--dense-prefill w4a4: MiniCPM-V 4.7 / Qwen3.6 (qwen35moe) "
               "supports only f16");
}

void Adapters() {
  const std::vector<std::string> two{"calm=calm.json", "terse=t.json"};
  std::string error;
  const auto plan = Admit(Model("qwen4exp"), {.adapters = two}, &error);
  Expect(plan && plan->adapters.size() == 2 &&
             plan->adapters[1].first == "terse" &&
             plan->adapters[1].second == "t.json",
         "adapters parse NAME=PATH");

  const std::vector<std::string> one{"calm=calm.json"};
  ExpectReject("qwen35moe", {.adapters = one},
               "--adapter: MiniCPM-V 4.7 / Qwen3.6 (qwen35moe) has no "
               "steering adapters");
  ExpectReject("lfm2moe", {.adapters = one},
               "--adapter: LFM2.5 (lfm2moe) has no steering adapters");
  ExpectReject("deepseek4", {.adapters = one},
               "--adapter: DeepSeek V4 Flash (deepseek4) has no steering "
               "adapters");
  for (const auto& bad :
       {std::vector<std::string>{"calm"}, std::vector<std::string>{"=a.json"},
        std::vector<std::string>{"calm="},
        std::vector<std::string>{"calm=a.json", "calm=b.json"}})
    ExpectReject("qwen4exp", {.adapters = bad},
                 "--adapter needs distinct NAME=PATH.json values");
}

}  // namespace

int main() {
  Routing();
  TemplateFamilies();
  Context();
  KvCache();
  Speculative();
  Vision();
  RoutingPolicy();
  DensePrefill();
  Adapters();
  if (failures != 0) {
    std::cerr << failures << " model admission checks failed\n";
    return 1;
  }
  std::cout << "Model admission checks passed.\n";
  return 0;
}
