#include "src/cli/serve/systemone_api.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/http_server.hpp"
#include "src/models/kev/model.hpp"
#include "src/models/kev/systemone.hpp"

namespace gufo::server {
namespace {

namespace kev = models::kev;

std::string RandomHex32() {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  char buffer[33];
  (void)std::snprintf(buffer, sizeof(buffer), "%016llx%016llx",
                      static_cast<unsigned long long>(rng()),
                      static_cast<unsigned long long>(rng()));
  return buffer;
}

const char* Reason(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 405:
      return "Method Not Allowed";
    case 422:
      return "Unprocessable Entity";
    case 429:
      return "Too Many Requests";
    case 503:
      return "Service Unavailable";
    default:
      return "Internal Server Error";
  }
}

}  // namespace

HttpResponse SystemOneError(int status, std::string detail) {
  json::Value body = json::Value::object();
  body["detail"] = std::move(detail);
  return {.status = status, .reason = Reason(status), .body = body.dump()};
}

SystemOneService::SystemOneService(std::unique_ptr<kev::Model> model,
                                   Options options)
    : model_(std::move(model)), options_(std::move(options)) {
  if (model_ == nullptr)
    throw std::invalid_argument("SystemOneService needs a model");
  encoder_ = &model_->encoder();
  limits_ = model_->limits();
  scorer_ = [this](std::span<const kev::Encoding* const> batch) {
    return model_->ScoreBatch(batch);
  };
  // Four layers of a large pass take tens of milliseconds: short enough that
  // a paused Kev leaves the text model's steps mostly alone.
  model_->SetSliceHook([this](std::uint32_t layer) { Pace(layer); }, 4);
  worker_ = std::thread([this] { Work(); });
}

void SystemOneService::SetTextBusy(std::function<bool()> busy) {
  const std::lock_guard<std::mutex> lock(mutex_);
  text_busy_ = std::move(busy);
}

void SystemOneService::Pace(std::uint32_t layer) {
  const auto now = std::chrono::steady_clock::now();
  if (layer != 0 && options_.gpu_share > 0.0 && options_.gpu_share < 1.0) {
    std::function<bool()> busy;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      busy = text_busy_;
    }
    if (busy && busy()) {
      // Kev ran for `ran`; give the text model the rest of its share.
      const auto ran = now - slice_start_;
      const auto pause = std::min<std::chrono::steady_clock::duration>(
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              ran * ((1.0 - options_.gpu_share) / options_.gpu_share)),
          std::chrono::seconds(1));
      std::this_thread::sleep_for(pause);
      paced_ms_.fetch_add(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(pause)
              .count()));
    }
  }
  slice_start_ = std::chrono::steady_clock::now();
}

SystemOneService::SystemOneService(const kev::Encoder* encoder,
                                   kev::Limits limits, Scorer scorer,
                                   Options options)
    : encoder_(encoder),
      limits_(limits),
      scorer_(std::move(scorer)),
      options_(std::move(options)) {
  worker_ = std::thread([this] { Work(); });
}

struct SystemOneService::Job {
  const kev::Encoding* encoding;
  std::vector<std::vector<double>> probabilities;
  double model_ms{0.0};
  std::exception_ptr error;
  std::promise<void> done;
};

SystemOneService::~SystemOneService() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void SystemOneService::Work() {
  while (true) {
    std::vector<Job*> batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty())
        return;
      std::size_t tokens = 0;
      while (!queue_.empty() &&
             (batch.empty() || tokens + queue_.front()->encoding->Tokens() <=
                                   options_.max_batch_tokens)) {
        tokens += queue_.front()->encoding->Tokens();
        batch.push_back(queue_.front());
        queue_.pop_front();
      }
    }
    std::vector<const kev::Encoding*> encodings;
    for (const Job* job : batch)
      encodings.push_back(job->encoding);
    const auto start = std::chrono::steady_clock::now();
    try {
      auto results = scorer_(encodings);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      for (std::size_t i = 0; i < batch.size(); ++i) {
        batch[i]->probabilities = std::move(results[i]);
        batch[i]->model_ms = ms;
      }
      if (model_ != nullptr) {
        const std::lock_guard<std::mutex> lock(mutex_);
        bank_ = model_->stats();
      }
    } catch (...) {
      for (Job* job : batch)
        job->error = std::current_exception();
    }
    batches_.fetch_add(1);
    for (Job* job : batch)
      job->done.set_value();
  }
}

HttpResponse SystemOneService::Handle(const HttpRequest& request) {
  // Echo the client's id only when it is a plain token; anything else could
  // inject response headers.
  std::string request_id = request.header("x-typesafe-request-id");
  const bool plain =
      !request_id.empty() && request_id.size() <= 128 &&
      std::ranges::all_of(request_id, [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' ||
               c == '_' || c == '.' || c == ':';
      });
  if (!plain)
    request_id = RandomHex32();
  HttpResponse response = [&]() -> HttpResponse {
    if (request.method != "POST")
      return SystemOneError(405, "Method Not Allowed");
    std::string error;
    auto parsed = kev::ParseRequest(request.body, &error);
    if (!parsed)
      return SystemOneError(422, error);
    auto encoding = encoder_->Encode(*parsed, limits_, &error);
    if (!encoding)
      return SystemOneError(422, error);
    if (pending_.fetch_add(1) >= options_.max_pending) {
      pending_.fetch_sub(1);
      rejected_.fetch_add(1);
      auto busy = SystemOneError(429, "the System One queue is full");
      busy.headers.emplace_back("Retry-After", "1");
      return busy;
    }
    Job job{.encoding = &*encoding};
    auto done = job.done.get_future();
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(&job);
    }
    wake_.notify_one();
    done.wait();
    if (job.error) {
      pending_.fetch_sub(1);
      std::rethrow_exception(job.error);
    }
    const auto& probabilities = job.probabilities;
    const double model_ms = job.model_ms;
    pending_.fetch_sub(1);
    requests_.fetch_add(1);
    tokens_.fetch_add(encoding->Tokens());
    questions_.fetch_add(encoding->rows.size());
    const std::string output =
        kev::AnswersJson(*parsed, probabilities, kev::JsonStyle::kPythonDumps);
    const std::size_t output_tokens =
        encoder_->tokenizer().Encode(output).size();
    std::string body = "{\"model\":";
    body += kev::PythonJsonString(parsed->model, kev::JsonStyle::kFastApi);
    body += ",\"answers\":";
    body += kev::AnswersJson(*parsed, probabilities, kev::JsonStyle::kFastApi);
    body +=
        ",\"usage\":{\"input_tokens\":" + std::to_string(encoding->Tokens()) +
        ",\"output_tokens\":" + std::to_string(output_tokens) +
        "},\"latency_ms\":" +
        kev::PythonFloatRepr(std::round(model_ms * 10.0) / 10.0) + "}";
    return {.status = 200, .reason = "OK", .body = std::move(body)};
  }();
  response.headers.emplace_back("x-typesafe-request-id", request_id);
  return response;
}

json::Value SystemOneService::Cards() const {
  json::Value cards = json::Value::array();
  std::vector<std::string> names = {"kev-latest", "jev-latest"};
  if (options_.model_id != names[0] && options_.model_id != names[1])
    names.push_back(options_.model_id);
  for (const std::string& name : names) {
    json::Value card = json::Value::object();
    card["name"] = name;
    card["description"] =
        "Kev pointer head on Qwen3.5-4B-Base, served by Gufo (" +
        options_.model_id + ")";
    card["release_date"] = options_.release_date;
    cards.push_back(std::move(card));
  }
  return cards;
}

std::string SystemOneService::PrometheusText() const {
  models::kev::PackedStats bank;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    bank = bank_;
  }
  std::ostringstream out;
  const auto series = [&](const char* name, const char* type, const char* help,
                          double value) {
    out << "# HELP gufo_systemone_" << name << ' ' << help << '\n'
        << "# TYPE gufo_systemone_" << name << ' ' << type << '\n'
        << "gufo_systemone_" << name << ' ' << value << '\n';
  };
  series("requests_total", "counter", "System One requests answered",
         static_cast<double>(requests_.load()));
  series("rejected_total", "counter",
         "System One requests refused because the queue was full",
         static_cast<double>(rejected_.load()));
  series("questions_total", "counter", "Questions answered",
         static_cast<double>(questions_.load()));
  series("tokens_total", "counter", "Input tokens encoded",
         static_cast<double>(tokens_.load()));
  series("pending", "gauge", "Requests queued or running",
         static_cast<double>(pending_.load()));
  series("batches_total", "counter", "Batches the worker scored",
         static_cast<double>(batches_.load()));
  series("passes_total", "counter", "Packed GPU passes",
         static_cast<double>(bank.passes));
  series("pass_tokens_total", "counter", "Tokens run in packed passes",
         static_cast<double>(bank.pass_tokens));
  series("gpu_seconds_total", "counter", "Wall time of packed passes",
         bank.gpu_ms / 1000.0);
  series("paced_seconds_total", "counter",
         "Time Kev yielded the GPU to the text model",
         static_cast<double>(paced_ms_.load()) / 1000.0);
  series("state_cache_hits_total", "counter", "States served from the bank",
         static_cast<double>(bank.cache_hits));
  series("state_cache_misses_total", "counter", "States computed",
         static_cast<double>(bank.cache_misses));
  return out.str();
}

json::Value SystemOneService::RuntimeJson() const {
  json::Value out = json::Value::object();
  out["model"] = options_.model_id;
  out["pending"] = pending_.load();
  out["max_pending"] = options_.max_pending;
  out["requests"] = static_cast<double>(requests_.load());
  out["rejected"] = static_cast<double>(rejected_.load());
  out["tokens"] = static_cast<double>(tokens_.load());
  out["questions"] = static_cast<double>(questions_.load());
  out["batches"] = static_cast<double>(batches_.load());
  models::kev::PackedStats bank;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    bank = bank_;
  }
  json::Value passes = json::Value::object();
  passes["count"] = static_cast<double>(bank.passes);
  passes["tokens"] = static_cast<double>(bank.pass_tokens);
  passes["gpu_ms"] = bank.gpu_ms;
  out["passes"] = std::move(passes);
  json::Value cache = json::Value::object();
  cache["hits"] = static_cast<double>(bank.cache_hits);
  cache["misses"] = static_cast<double>(bank.cache_misses);
  cache["states"] = static_cast<double>(bank.cached_states);
  out["state_cache"] = std::move(cache);
  out["gpu_share"] = options_.gpu_share;
  out["paced_ms"] = static_cast<double>(paced_ms_.load());
  return out;
}

}  // namespace gufo::server
