#include "src/models/qwen38_flash_next/mtmd_vision.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#if defined(GUFO_WITH_MTMD)
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#endif
#include "src/core/crypto/sha256.hpp"

namespace gufo::models::qwen38_flash_next {
#if defined(GUFO_WITH_MTMD)
namespace {

/// Image rows carry this id; their embeddings replace the token's.
constexpr std::int32_t kImagePlaceholder = 0;

struct Chunks {
  mtmd_input_chunks* p{mtmd_input_chunks_init()};
  ~Chunks() { mtmd_input_chunks_free(p); }
};

struct Bitmaps {
  std::vector<mtmd_bitmap*> p;
  ~Bitmaps() {
    for (auto* b : p)
      mtmd_bitmap_free(b);
  }
};

}  // namespace

std::unique_ptr<MtmdVision> MtmdVision::Open(const std::string& text_model,
                                             const std::string& mmproj,
                                             std::uint32_t hidden_size,
                                             std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return nullptr;
  };
  static std::once_flag backend;
  std::call_once(backend, [] { llama_backend_init(); });
  std::unique_ptr<MtmdVision> v(new MtmdVision());
  auto model_params = llama_model_default_params();
  model_params.vocab_only = true;
  model_params.n_gpu_layers = 0;
  v->model_ = llama_model_load_from_file(text_model.c_str(), model_params);
  if (v->model_ == nullptr)
    return fail("mtmd: cannot load the vocabulary of " + text_model);
  // mtmd checks its projector against the text model; DeviceInput checks
  // every embedding's size against the trunk.
  auto params = mtmd_context_params_default();
  params.use_gpu = true;
  params.print_timings = false;
  params.n_threads = 8;
  params.media_marker = kMarker.data();
  params.warmup = true;
  v->ctx_ = mtmd_init_from_file(mmproj.c_str(), v->model_, params);
  if (v->ctx_ == nullptr || !mtmd_support_vision(v->ctx_))
    return fail("mtmd: cannot load the vision projector " + mmproj);
  if (!mtmd_decode_use_mrope(v->ctx_))
    return fail("mtmd: the projector does not use M-RoPE decoder positions");
  v->hidden_size_ = hidden_size;
  v->identity_ = "mtmd:" + mmproj;
  return v;
}

MtmdVision::~MtmdVision() {
  if (ctx_ != nullptr)
    mtmd_free(ctx_);
  if (model_ != nullptr)
    llama_model_free(model_);
}

MtmdVision::Plan MtmdVision::Prepare(
    std::string_view prompt,
    std::span<const std::shared_ptr<const std::vector<std::uint8_t>>> images)
    const {
  const std::lock_guard lock(mutex_);
  Bitmaps bitmaps;
  crypto::Sha256Hasher identity;
  identity.Update({reinterpret_cast<const std::uint8_t*>(identity_.data()),
                   identity_.size()});
  for (const auto& bytes : images) {
    if (!bytes || bytes->empty())
      throw std::invalid_argument("empty image input");
    auto wrapper =
        mtmd_helper_bitmap_init_from_buf(ctx_, bytes->data(), bytes->size(),
                                         false, mtmd_helper_init_opt_default());
    if (wrapper.bitmap == nullptr)
      throw std::invalid_argument("image could not be decoded");
    bitmaps.p.push_back(wrapper.bitmap);
    // The bitmap id is the SHA-256 of the file.
    const std::string id = mtmd_bitmap_get_id(wrapper.bitmap);
    identity.Update(
        {reinterpret_cast<const std::uint8_t*>(id.data()), id.size()});
  }
  const std::string text(prompt);
  const mtmd_input_text input{text.c_str(), text.size(), false, true};
  Chunks chunks;
  const auto rc =
      mtmd_tokenize(ctx_, chunks.p, &input, bitmaps.p.data(), bitmaps.p.size());
  if (rc == 1)
    throw std::invalid_argument("image count does not match the prompt");
  if (rc != 0)
    throw std::invalid_argument("image preprocessing failed");

  Plan plan;
  std::vector<std::array<std::int32_t, 3>> positions;
  std::int32_t pos = 0;  // llama.cpp position (KV slot t)
  std::size_t prefix = 0;
  std::int32_t prefix_pos = 0;  // position right after the last image
  for (std::size_t c = 0; c < mtmd_input_chunks_size(chunks.p); ++c) {
    const auto* chunk = mtmd_input_chunks_get(chunks.p, c);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      std::size_t n = 0;
      const auto* ids = mtmd_input_chunk_get_tokens_text(chunk, &n);
      for (std::size_t i = 0; i < n; ++i) {
        plan.tokens.push_back(ids[i]);
        positions.push_back({pos, pos, pos});
        ++pos;
      }
      continue;
    }
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE)
      throw std::invalid_argument("only image media is supported");
    const auto n = mtmd_input_chunk_get_n_tokens(chunk);
    const auto* image = mtmd_input_chunk_get_tokens_image(chunk);
    std::vector<mtmd_decoder_pos> dec(n);
    mtmd_helper_image_get_decoder_pos(image, pos, dec.data());
    if (mtmd_encode_chunk(ctx_, chunk) != 0)
      throw std::runtime_error("image encoding failed");
    const float* embd = mtmd_get_output_embd(ctx_);
    auto data = std::make_shared<std::vector<float>>(
        embd, embd + n * std::size_t{hidden_size_});
    plan.external.push_back({static_cast<std::uint32_t>(plan.tokens.size()),
                             static_cast<std::uint32_t>(n), std::move(data)});
    for (std::size_t i = 0; i < n; ++i) {
      plan.tokens.push_back(kImagePlaceholder);
      // Section order zyxt: the first RoPE section reads z, then y, then x.
      positions.push_back({static_cast<std::int32_t>(dec[i].z),
                           static_cast<std::int32_t>(dec[i].y),
                           static_cast<std::int32_t>(dec[i].x)});
    }
    pos += mtmd_input_chunk_get_n_pos(chunk);
    prefix = plan.tokens.size();
    prefix_pos = pos;
  }
  // Explicit positions stop after the last image so a later turn with the
  // same images keeps an identical layout (its snapshots stay restorable).
  positions.resize(prefix);
  plan.rope.explicit_positions = std::move(positions);
  // Text after the last image continues at prefix_pos, prefix_pos + 1, ...
  plan.rope.explicit_delta = prefix_pos - static_cast<std::int32_t>(prefix);
  // A raw SHA-256, the identity size session snapshots accept.
  const auto digest = identity.Finish();
  plan.identity.assign(digest.begin(), digest.end());
  return plan;
}

#else  // Built without llama.cpp: MiniCPM-V serves text only.

std::unique_ptr<MtmdVision> MtmdVision::Open(const std::string&,
                                             const std::string&, std::uint32_t,
                                             std::string* error) {
  if (error != nullptr)
    *error = "MiniCPM-V images need a build with GUFO_LLAMA_CPP_DIR (mtmd)";
  return nullptr;
}

MtmdVision::~MtmdVision() = default;

MtmdVision::Plan MtmdVision::Prepare(
    std::string_view,
    std::span<const std::shared_ptr<const std::vector<std::uint8_t>>>) const {
  throw std::invalid_argument("this build has no MiniCPM-V image support");
}

#endif

}  // namespace gufo::models::qwen38_flash_next
