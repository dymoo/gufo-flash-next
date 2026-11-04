#include "src/models/kev/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"
#else
namespace gufo::hip {
class QwenGpuExecutor {};
class QwenGpuModel {};
}  // namespace gufo::hip
#endif

namespace gufo::models::kev {
namespace {

bool CopyF32(const core::GgufReader& reader, std::string_view name,
             std::size_t elements, std::vector<float>* out,
             std::string* error) {
  const core::GgufTensorInfo* t = reader.FindTensor(name);
  if (t == nullptr || t->type != core::GgmlType::kF32 ||
      t->ElementCount() != elements) {
    if (error != nullptr)
      *error = "Kev GGUF lacks an F32 " + std::string(name) + " of " +
               std::to_string(elements) + " elements";
    return false;
  }
  const auto* data = static_cast<const float*>(t->data);
  out->assign(data, data + elements);
  return true;
}

std::vector<double> NormedRow(std::span<const float> x,
                              const std::vector<float>& weight, float eps) {
  double sum = 0.0;
  for (const float v : x)
    sum += static_cast<double>(v) * v;
  const double inv = 1.0 / std::sqrt(sum / static_cast<double>(x.size()) + eps);
  std::vector<double> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = static_cast<double>(x[i]) * inv * weight[i];
  return out;
}

std::vector<double> Project(const std::vector<double>& h,
                            const std::vector<float>& weight,
                            const std::vector<float>& bias, std::size_t dim) {
  std::vector<double> out(dim);
  for (std::size_t r = 0; r < dim; ++r) {
    const float* row = weight.data() + r * h.size();
    double acc = bias[r];
    for (std::size_t c = 0; c < h.size(); ++c)
      acc += row[c] * h[c];
    out[r] = acc;
  }
  return out;
}

}  // namespace

std::vector<double> PointerHead::Probabilities(
    std::span<const float> decide,
    std::span<const std::span<const float>> options) const {
  const auto q =
      Project(NormedRow(decide, final_norm, eps), q_weight, q_bias, dim);
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim)) / temperature;
  std::vector<double> z(options.size());
  for (std::size_t i = 0; i < options.size(); ++i) {
    const auto k =
        Project(NormedRow(options[i], final_norm, eps), k_weight, k_bias, dim);
    double dot = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
      dot += k[d] * q[d];
    z[i] = dot * scale;
  }
  const double top = *std::max_element(z.begin(), z.end());
  double total = 0.0;
  for (double& v : z) {
    v = std::exp(v - top);
    total += v;
  }
  for (double& v : z)
    v /= total;
  return z;
}

bool LoadPointerHead(const core::GgufReader& reader, PointerHead* head,
                     std::string* error) {
  const auto hidden = reader.GetMetadataUint32("qwen35.embedding_length");
  const auto dim = reader.GetMetadataUint32("kev.pointer_dim");
  const auto temperature = reader.GetMetadataFloat32("kev.temperature");
  if (!hidden || !dim || !temperature || *temperature <= 0.0F) {
    if (error != nullptr)
      *error = "not a Kev GGUF (kev.pointer_dim / kev.temperature missing)";
    return false;
  }
  head->hidden = *hidden;
  head->dim = *dim;
  head->temperature = *temperature;
  if (auto eps =
          reader.GetMetadataFloat32("qwen35.attention.layer_norm_rms_epsilon"))
    head->eps = *eps;
  const std::size_t matrix = head->hidden * head->dim;
  return CopyF32(reader, "kev.pointer.q.weight", matrix, &head->q_weight,
                 error) &&
         CopyF32(reader, "kev.pointer.q.bias", head->dim, &head->q_bias,
                 error) &&
         CopyF32(reader, "kev.pointer.k.weight", matrix, &head->k_weight,
                 error) &&
         CopyF32(reader, "kev.pointer.k.bias", head->dim, &head->k_bias,
                 error) &&
         CopyF32(reader, "output_norm.weight", head->hidden, &head->final_norm,
                 error);
}

Model::~Model() = default;

#if defined(ENGINE_ENABLE_HIP)

std::unique_ptr<Model> Model::Load(const std::filesystem::path& path,
                                   const ModelOptions& options,
                                   std::string* error) {
  auto model = std::unique_ptr<Model>(new Model());
  model->limits_ = options.limits;
  model->limits_.max_branch = std::min<std::size_t>(model->limits_.max_branch,
                                                    options.packed.max_tokens);
  model->limits_.max_state =
      std::min<std::size_t>(model->limits_.max_state, options.packed.kv_slots);
  model->baseline_context_ = options.baseline_context;
  std::shared_ptr<const core::GgufReader> reader =
      core::GgufReader::OpenFile(path, error);
  if (reader == nullptr)
    return nullptr;
  model->reader_ = reader;
  if (!LoadPointerHead(*reader, &model->head_, error))
    return nullptr;
  model->encoder_ = Encoder::Create(*reader, error);
  if (model->encoder_ == nullptr)
    return nullptr;
  model->gpu_model_ = ::gufo::hip::QwenGpuModel::CreateFromGguf(reader, error);
  if (model->gpu_model_ == nullptr)
    return nullptr;
  model->packed_ = PackedExecutor::Create(model->gpu_model_, model->head_,
                                          options.packed, error);
  if (model->packed_ == nullptr)
    return nullptr;
  return model;
}

std::vector<std::vector<double>> Model::Score(const Encoding& encoding) {
  const Encoding* one[] = {&encoding};
  return std::move(packed_->Score(one)[0]);
}

std::vector<std::vector<std::vector<double>>> Model::ScoreBatch(
    std::span<const Encoding* const> batch) {
  return packed_->Score(batch);
}

std::vector<std::vector<double>> Model::ScoreBaseline(
    const Encoding& encoding) {
  if (baseline_ == nullptr) {
    std::string error;
    baseline_ = ::gufo::hip::QwenGpuExecutor::Create(gpu_model_, &error,
                                                     baseline_context_);
    if (baseline_ == nullptr)
      throw std::runtime_error("Kev baseline executor: " + error);
  }
  const std::size_t hidden = head_.hidden;
  const auto state_length = static_cast<std::uint32_t>(encoding.state.size());
  baseline_->Reset();
  baseline_->SetPromptHiddenCapture(false);
  (void)baseline_->ForwardPromptBatch(encoding.state, 0, false);
  baseline_->SaveState(state_length);
  std::vector<std::vector<double>> probabilities;
  for (const Row& row : encoding.rows) {
    baseline_->RestoreState();
    baseline_->SetPromptHiddenCapture(true);
    (void)baseline_->ForwardPromptBatch(row.ids, state_length, false);
    const std::span<const float> rows = baseline_->GetPromptHiddenStates();
    if (rows.size() != row.ids.size() * hidden)
      throw std::runtime_error("Kev hidden capture has the wrong shape");
    std::vector<std::span<const float>> options;
    for (const std::uint32_t end : row.option_ends)
      options.push_back(rows.subspan(end * hidden, hidden));
    probabilities.push_back(head_.Probabilities(
        rows.subspan(row.decide * hidden, hidden), options));
  }
  baseline_->SetPromptHiddenCapture(false);
  return probabilities;
}

#else

std::unique_ptr<Model> Model::Load(const std::filesystem::path&,
                                   const ModelOptions&, std::string* error) {
  if (error != nullptr)
    *error = "Kev needs a HIP build";
  return nullptr;
}

std::vector<std::vector<double>> Model::Score(const Encoding&) {
  throw std::logic_error("Kev needs a HIP build");
}
std::vector<std::vector<std::vector<double>>> Model::ScoreBatch(
    std::span<const Encoding* const>) {
  throw std::logic_error("Kev needs a HIP build");
}
std::vector<std::vector<double>> Model::ScoreBaseline(const Encoding&) {
  throw std::logic_error("Kev needs a HIP build");
}

#endif

}  // namespace gufo::models::kev
