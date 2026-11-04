#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cli/serve/chat_protocol.hpp"
#include "src/cli/serve/logging.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::TextGenerationBackend;

class FakeBackend final : public TextGenerationBackend {
public:
  struct Call {
    std::string prompt;
    gufo::server::ChatRequest chat;
    std::size_t max_tokens = 0;
    gufo::sampling::SamplingConfig sampling;
    std::string client_id;
    std::vector<std::string> stop_sequences;
  };
  Call LastCall() {
    const std::lock_guard lock(mutex_);
    return last_;
  }
  void SetOutput(std::string text) {
    const std::lock_guard lock(mutex_);
    output_ = std::move(text);
  }
  std::string model_id() const override { return "test"; }
  std::vector<std::string> model_ids() const override {
    std::vector<std::string> ids{model_id()};
    ids.insert(ids.end(), adapters.begin(), adapters.end());
    return ids;
  }
  bool ready() const override { return true; }
  gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning;
  }
  /// MiMo detects a leading <think> and otherwise starts in content.
  const gufo::server::ChatProtocol& chat_protocol() const override {
    return gufo::server::ProtocolFor(gufo::server::TemplateFamily::kMimo);
  }
  std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  Result complete(
      std::string_view prompt, std::size_t limit,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancel, const TokenCallback& token,
      std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {}) override {
    ++calls;
    if (failure == 1)
      throw std::length_error("context exceeded");
    if (failure == 2)
      throw std::invalid_argument("invalid prompt");
    Result result;
    if (wait_for_disconnect) {
      entered.release();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!(disconnected = cancel && cancel()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished.release();
      result.cancelled = disconnected;
      return result;
    }
    {
      const std::lock_guard lock(mutex_);
      last_ = {.prompt = std::string(prompt),
               .max_tokens = limit,
               .sampling = sampling,
               .client_id = std::string(client_id),
               .stop_sequences = stop_sequences};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_tokens = 8;
    result.prefill_tokens = 2;
    result.prefill_ms = 4;
    result.completion_tokens = 1;
    result.decode_ms = 2;
    result.finish_reason =
        limit == 1 ? FinishReason::kLength : FinishReason::kStop;
    if (!forced_stop_sequence.empty()) {
      result.finish_reason = FinishReason::kStopSequence;
      result.stop_sequence = forced_stop_sequence;
    }
    if (token)
      (void)token(result.text);
    return result;
  }
  Result chat(const gufo::server::ChatRequest& request, std::size_t limit,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancel,
              const TokenCallback& token) override {
    if (failure == 3) {
      {
        const std::lock_guard lock(mutex_);
        last_.chat = request;
      }
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kCapacityExhausted,
          "every session is busy");
    }
    auto result = complete("", limit, sampling, cancel, token,
                           request.client_id, request.stop_sequences);
    {
      const std::lock_guard lock(mutex_);
      last_.chat = request;
    }
    return result;
  }
  std::atomic<int> calls{0};
  std::atomic<int> failure{0};
  std::string forced_stop_sequence;
  gufo::ReasoningOptions reasoning;
  /// Steering adapter ids served beside "test"; set before any request.
  std::vector<std::string> adapters;
  bool wait_for_disconnect{false};
  std::atomic<bool> disconnected{false};
  std::binary_semaphore entered{0};
  std::binary_semaphore finished{0};

private:
  std::mutex mutex_;
  Call last_;
  std::string output_{"ok"};
};

class RunningServer {
public:
  explicit RunningServer(gufo::server::HttpServerOptions options = {},
                         bool handle_signals = false)
      : backend(std::make_shared<FakeBackend>()),
        server("127.0.0.1", 0, backend, nullptr, nullptr, nullptr,
               std::move(options)) {
    server.add("POST", "/echo", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.body};
    });
    server.add("POST", "/stream-error", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .streaming_body = [](const auto& write) {
            (void)write("first chunk");
            throw std::runtime_error("injected stream failure");
          }};
    });
    server.add("POST", "/stream", [](const auto& request, auto&) {
      auto log = std::make_shared<gufo::server::HttpResponse::StreamLog>();
      return gufo::server::HttpResponse{
          .streaming_body =
              [log, fail = !request.body.empty(),
               reported = request.body == "reported"](const auto& write) {
                (void)write(std::string_view("a\0b", 3));
                (void)write("");
                (void)write("end");
                if (fail)
                  log->error_code = "injected";
                log->error_event_sent = reported;
              },
          .stream_log = log,
      };
    });
    std::string error;
    assert(server.start(&error));
    worker = std::jthread([this, handle_signals] {
      server.run(handle_signals);
      run_finished.release();
    });
  }
  ~RunningServer() {
    server.stop();
    worker.join();
  }

  int Connect() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    const timeval timeout{3, 0};
    assert(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                        sizeof(timeout)) == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                     sizeof(address)) == 0);
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false) {
    const int fd = Connect();
    while (!request.empty()) {
      const auto count =
          ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      ::shutdown(fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = ::read(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    ::close(fd);
    return response;
  }

  std::string Post(std::string_view path, std::string_view body) {
    return Send("POST " + std::string(path) + " HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  }

  std::shared_ptr<FakeBackend> backend;
  HttpServer server;
  std::binary_semaphore run_finished{0};
  std::jthread worker;
};

void ExpectStatus(const std::string& response, int status) {
  if (!response.starts_with("HTTP/1.1 " + std::to_string(status) + " ")) {
    std::cerr << response << '\n';
    std::abort();
  }
}

void TestAuthorization() {
  RunningServer secured({.api_key = "test-secret"});
  for (const std::string path :
       {"/health", "/ready", "/v1/models", "/v1/chat/completions",
        "/v1/responses", "/v1/messages", "/v1/audio/speech",
        "/v1/audio/transcriptions", "/v1/video/generations", "/echo"}) {
    ExpectStatus(secured.Send("POST " + path + " HTTP/1.1\r\n\r\n"), 401);
  }
  for (const std::string header :
       {"", "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer "
        "test-secret\r\n"}) {
    const auto response =
        secured.Send("GET /v1/models HTTP/1.1\r\n" + header + "\r\n");
    ExpectStatus(response, 401);
    assert(response.find("WWW-Authenticate: Bearer") != std::string::npos);
    assert(response.find("test-secret") == std::string::npos);
  }
  assert(secured.backend->calls == 0);
  ExpectStatus(secured.Send("GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  "
                            "test-secret\r\n\r\n"),
               200);
  // A request without the key is answered when its headers arrive, without
  // waiting for the body it announces.
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nContent-Length: "
                            "100\r\n\r\n"),
               401);
  // Without an Origin, OPTIONS is not a browser preflight.
  ExpectStatus(secured.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"),
               401);
  const std::string body = R"({"prompt":"hi","max_tokens":1})";
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nAuthorization: "
                            "Bearer test-secret\r\n"
                            "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body),
               200);
  assert(secured.backend->calls == 1);
}

void TestCrossOriginRequests() {
  const std::string upgrade =
      "GET /v1/realtime?intent=transcription HTTP/1.1\r\nUpgrade: "
      "websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: "
      "13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
  {
    // CORS is opt-in: by default any Origin is refused before routing,
    // preflights and WebSocket upgrades included, and nothing carries CORS
    // headers.
    RunningServer server;
    const auto plain = server.Send("GET /health HTTP/1.1\r\n\r\n");
    ExpectStatus(plain, 200);
    assert(plain.find("Access-Control-") == std::string::npos);
    for (const std::string request :
         {std::string("GET /health HTTP/1.1\r\n"),
          std::string("OPTIONS /v1/chat/completions HTTP/1.1\r\n"),
          std::string("POST /echo HTTP/1.1\r\nContent-Length: 2\r\n"),
          upgrade}) {
      const auto refused =
          server.Send(request + "Origin: https://app.example\r\n\r\n");
      ExpectStatus(refused, 403);
      assert(refused.find("origin_not_allowed") != std::string::npos);
      assert(refused.find("Access-Control-") == std::string::npos);
    }
    // Without an Origin the upgrade reaches its route (no ASR is loaded).
    ExpectStatus(server.Send(upgrade + "\r\n"), 503);
  }
  {
    RunningServer server(
        {.api_key = "test-secret", .allowed_origins = {"https://app.example"}});
    // An allowed preflight needs no credentials.
    const auto preflight = server.Send(
        "OPTIONS /v1/chat/completions HTTP/1.1\r\nOrigin: "
        "https://app.example\r\n\r\n");
    ExpectStatus(preflight, 204);
    assert(preflight.find(
               "Access-Control-Allow-Origin: https://app.example\r\n") !=
           std::string::npos);
    assert(preflight.find("Vary: Origin\r\n") != std::string::npos);
    assert(preflight.find("Access-Control-Allow-Headers: Content-Type, "
                          "Authorization, Range\r\n") != std::string::npos);
    // Other requests still need the key, and the browser can read the 401.
    const auto denied = server.Send(
        "GET /v1/models HTTP/1.1\r\nOrigin: https://app.example\r\n\r\n");
    ExpectStatus(denied, 401);
    assert(denied.find("Access-Control-Allow-Origin: https://app.example") !=
           std::string::npos);
    ExpectStatus(server.Send("GET /v1/models HTTP/1.1\r\nOrigin: "
                             "https://app.example\r\nAuthorization: Bearer "
                             "test-secret\r\n\r\n"),
                 200);
    for (const std::string origin :
         {"Origin: https://app.example.evil\r\n",
          "Origin: https://app.example/\r\n", "Origin: http://app.example\r\n",
          "Origin: \r\n",
          "Origin: https://app.example\r\nOrigin: https://other.example\r\n"}) {
      ExpectStatus(server.Send("GET /v1/models HTTP/1.1\r\n" + origin +
                               "Authorization: Bearer test-secret\r\n\r\n"),
                   403);
    }
  }
  {
    RunningServer server({.allowed_origins = {"*"}});
    const auto any = server.Send(
        "GET /health HTTP/1.1\r\nOrigin: https://any.example\r\n\r\n");
    ExpectStatus(any, 200);
    assert(any.find("Access-Control-Allow-Origin: *\r\n") != std::string::npos);
    assert(any.find("Vary: Origin") == std::string::npos);
  }
}

void TestRequestLogging() {
  std::ostringstream output;
  auto* previous = std::clog.rdbuf(output.rdbuf());
  std::string request_id;
  {
    RunningServer server;
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
    const auto response = server.Post(
        "/v1/chat/completions?private-query",
        R"({"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true})");
    ExpectStatus(response, 200);
    const auto header = response.find("X-Request-ID: ");
    assert(header != std::string::npos);
    const auto begin = header + std::string("X-Request-ID: ").size();
    request_id = response.substr(begin, response.find("\r\n", begin) - begin);

    const auto failed = server.Post("/stream-error", "");
    ExpectStatus(failed, 200);
    assert(failed.find("first chunk") != std::string::npos);
    assert(failed.find("HTTP/1.1", 1) == std::string::npos);
  }
  gufo::server::Logger::Info("test", "escaped\n\x1b[31m");
  std::clog.rdbuf(previous);
  const auto log = output.str();
  assert(log.find("request=" + request_id + " event=received") !=
         std::string::npos);
  assert(log.find("request=" + request_id + " event=completed") !=
         std::string::npos);
  assert(log.find("cache=memory") != std::string::npos);
  assert(log.find("acceptance_pct=50.0") != std::string::npos);
  assert(log.find("rss_mib=") != std::string::npos);
  assert(log.find("error_code=server_exception") != std::string::npos);
  assert(log.find("path=/health") == std::string::npos);
  assert(log.find("private-query") == std::string::npos);
  assert(log.find("private-prompt") == std::string::npos);
  assert(log.find("escaped\\x0a\\x1b[31m") != std::string::npos);
}

void TestFramingAndMetrics() {
  RunningServer server({.max_request_body_bytes = 8192});
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  for (const std::string header :
       {"Content-Length: nope", "Content-Length: -1", "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3", "Transfer-Encoding: chunked",
        "broken-header"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 400);
  }
  ExpectStatus(server.Send("POST /echo\r\n\r\n"), 400);
  ExpectStatus(server.Send("POST /echo HTTP/1.1 extra\r\n\r\n"), 400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
      400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 8193\r\n\r\n"), 413);
  {
    // A body that trickles in past its deadline is refused, however long
    // the client keeps the connection open.
    RunningServer slow({.body_timeout = std::chrono::milliseconds(200)});
    const auto start = std::chrono::steady_clock::now();
    const auto timed_out =
        slow.Send("POST /echo HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc");
    ExpectStatus(timed_out, 408);
    assert(timed_out.find("request_timeout") != std::string::npos);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(2));
  }
  const std::string payload(5000, 'x');
  const auto echo = server.Send(
      "POST /echo HTTP/1.1\r\nContent-Length: 5000\r\nContent-Length: "
      "5000\r\n\r\n" +
      payload);
  ExpectStatus(echo, 200);
  assert(echo.substr(echo.find("\r\n\r\n") + 4) == payload);

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  const auto response =
      server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body);
  ExpectStatus(response, 200);
  const auto parsed =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  const auto* timings = parsed.find("timings");
  assert(timings != nullptr);
  assert(timings->member_double("prompt_n") == 2);
  assert(timings->member_double("cache_n") == 8);
  assert(timings->member_double("prompt_per_second") == 500);
  assert(timings->member_double("prompt_per_token_ms") == 2);
}

void TestMetricsListener() {
  // The unauthenticated --metrics-address listener answers liveness and
  // metrics and nothing else, not even the routes the harness adds.
  RunningServer server({.metrics_only = true});
  for (const char* path : {"/health", "/metrics", "/v1/metrics"})
    ExpectStatus(server.Send("GET " + std::string(path) + " HTTP/1.1\r\n\r\n"),
                 200);
  for (const char* path : {"/ready", "/v1/models", "/v1/runtime", "/props"})
    ExpectStatus(server.Send("GET " + std::string(path) + " HTTP/1.1\r\n\r\n"),
                 404);
  ExpectStatus(server.Post("/echo", "hi"), 404);
  ExpectStatus(server.Post("/v1/runtime/drain", ""), 404);
  ExpectStatus(server.Post("/v1/chat/completions",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               404);
  assert(server.backend->calls == 0);
}

std::string ExtractMetric(const std::string& exposition,
                          const std::string& sample_prefix) {
  const auto pos = exposition.find(sample_prefix);
  if (pos == std::string::npos)
    return "";
  const auto line_end = exposition.find('\n', pos);
  return exposition.substr(pos, line_end - pos);
}

void TestMetricsEndpoint() {
  RunningServer server;
  const auto metrics_response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
  ExpectStatus(metrics_response, 200);
  assert(metrics_response.find("Content-Type: text/plain; version=0.0.4") !=
         std::string::npos);
  const std::string before =
      metrics_response.substr(metrics_response.find("\r\n\r\n") + 4);

  // Every declared metric name carries HELP + TYPE before any sample
  // (Prometheus text exposition 0.0.4), and gauges/labels degrade to "no
  // line" rather than a fabricated zero when the backend has nothing to
  // report (FakeBackend uses the base MetricsSnapshot{}).
  assert(before.find("# HELP gufo_requests_total") != std::string::npos);
  assert(before.find("# TYPE gufo_time_to_first_token_seconds histogram") !=
         std::string::npos);
  assert(before.find(R"(gufo_build_info{revision=")") != std::string::npos);
  assert(before.find(R"(model="test")") != std::string::npos);
  assert(before.find("llamacpp:kv_cache_usage_ratio 0") != std::string::npos);
  const std::string before_completed = ExtractMetric(
      before,
      R"(gufo_requests_total{endpoint="completion",outcome="completed"})");
  const auto before_count =
      std::stoull(before_completed.substr(before_completed.rfind(' ') + 1));

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  ExpectStatus(server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                           std::to_string(body.size()) + "\r\n\r\n" + body),
               200);

  const auto after_response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
  const std::string after =
      after_response.substr(after_response.find("\r\n\r\n") + 4);
  const std::string after_completed = ExtractMetric(
      after,
      R"(gufo_requests_total{endpoint="completion",outcome="completed"})");
  const auto after_count =
      std::stoull(after_completed.substr(after_completed.rfind(' ') + 1));
  assert(after_count == before_count + 1);
  assert(after.find("gufo_generated_tokens_total") != std::string::npos);

  // A refuse-mode refusal: 503 with Retry-After, counted as its own outcome.
  server.backend->failure = 3;
  const std::string input = R"({"input":"hi"})";
  const auto refused = server.Send(
      "POST /v1/responses HTTP/1.1\r\nX-Gufo-Queue: 1\r\nContent-Length: " +
      std::to_string(input.size()) + "\r\n\r\n" + input);
  ExpectStatus(refused, 503);
  assert(refused.find("Retry-After: 1\r\n") != std::string::npos);
  assert(refused.find(R"("code":"capacity_exhausted")") != std::string::npos);
  assert(server.backend->LastCall().chat.queue &&
         !server.backend->LastCall().chat.no_queue);
  const auto counted = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
  assert(counted.find(R"(gufo_requests_total{endpoint="responses",)"
                      R"(outcome="rejected_capacity"} 1)") !=
         std::string::npos);
  server.backend->failure = 0;
}

void TestCompatibilityRequests() {
  RunningServer server;
  using gufo::json::parse;
  const auto response_body = [](const std::string& response) {
    ExpectStatus(response, 200);
    return parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  struct Endpoint {
    const char *path, *body, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi"})", "max_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi"})", "max_output_tokens"},
           Endpoint{"/v1/messages",
                    R"({"messages":[{"role":"user","content":"hi"}]})",
                    "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi"})", "n_predict"},
       }) {
    auto body = parse(endpoint.body);
    response_body(server.Post(endpoint.path, body.dump()));
    assert(server.backend->LastCall().max_tokens == 0);
    body["model"] = "test";
    body[endpoint.limit] = 1;
    body["temperature"] = 0.6;
    body["top_k"] = 40;
    body["top_p"] = 0.9;
    body["seed"] = 123;
    body["repeat_penalty"] = 1.1;
    const auto output = response_body(server.Post(endpoint.path, body.dump()));
    const auto last = server.backend->LastCall();
    for (int failure : {1, 2}) {
      server.backend->failure = failure;
      const auto rejected = server.Post(endpoint.path, body.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find(failure == 1
                               ? "context_length_exceeded"
                               : "invalid_prompt") != std::string::npos);
    }
    server.backend->failure = 0;
    assert(last.client_id == "127.0.0.1");
    assert(last.max_tokens == 1 && last.sampling.temperature == 0.6F &&
           last.sampling.top_k == 40 && last.sampling.top_p == 0.9F &&
           last.sampling.seed == 123 && last.sampling.repeat_penalty == 1.1F);
    if (std::string_view(endpoint.path) == "/v1/responses") {
      assert(output.member_str("status") == "incomplete");
      assert(output.find("incomplete_details")->member_str("reason") ==
             "max_output_tokens");
      assert(output.find("usage")->contains("input_tokens_details"));
    } else if (std::string_view(endpoint.path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "max_tokens");
    } else if (std::string_view(endpoint.path) == "/completion") {
      assert(output.find("stopped_length")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "length");
    }
    const int calls = server.backend->calls;
    for (const auto value : {"0", "-1", "1.5", "1e100", "\"1\"", "null"}) {
      auto invalid = body;
      invalid[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto field :
         {"stream", "echo", "store", "background", "tools", "stop", "reasoning",
          "output_config", "logit_bias"}) {
      if (std::string_view(endpoint.path) == "/v1/responses" &&
          std::string_view(field) == "stream")
        continue;
      auto invalid = body;
      invalid[field] = true;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    auto invalid = body;
    invalid["n"] = 1.4;
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    invalid = body;
    invalid["model"] = "wrong";
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 404);
    ExpectStatus(server.Post(endpoint.path, "[]"), 400);
    ExpectStatus(server.Post(endpoint.path, "{"), 400);
    assert(server.backend->calls == calls);
  }
  const int calls = server.backend->calls;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":["one","two"]})"),
               400);
  ExpectStatus(server.Post("/v1/responses",
                           R"({"input":[{"role":"user","content":[
                           {"type":"input_text","text":"describe"},
                           {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"tool","content":"hi"}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages", R"({"messages":[]})"), 400);
  ExpectStatus(
      server.Post("/infill", R"({"input_prefix":"one","input_suffix":"two"})"),
      501);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               501);
  assert(server.backend->calls == calls);

  const auto response = response_body(server.Post(
      "/v1/responses",
      R"({"instructions":"Be concise.","input":[{"role":"user","content":[
          {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false})"));
  assert(response.member_str("status") == "completed");
  const auto messages = server.backend->LastCall().chat.messages;
  assert(messages.size() == 2);
  assert(messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         messages[0].content == "Be concise." && messages[1].content == "hi");
  for (const auto limit : {1, 2}) {
    const auto streaming = server.Post(
        "/v1/responses",
        std::string(R"({"input":"hi","stream":true,"max_output_tokens":)") +
            std::to_string(limit) + "}");
    ExpectStatus(streaming, 200);
    assert(streaming.find("text/event-stream") != std::string::npos);
    assert(streaming.find("event: response.created") != std::string::npos);
    assert(streaming.find("event: response.output_text.delta") !=
           std::string::npos);
    assert(streaming.find(limit == 1 ? "event: response.incomplete"
                                     : "event: response.completed") !=
           std::string::npos);
  }
  server.backend->failure = 1;
  const auto failed_stream =
      server.Post("/v1/responses", R"({"input":"hi","stream":true})");
  ExpectStatus(failed_stream, 200);  // Fake backend fails after headers.
  assert(failed_stream.find("event: response.failed") != std::string::npos);
  assert(failed_stream.find("event: response.completed") == std::string::npos);
  const auto failed_data = failed_stream.find(
      "data: ", failed_stream.find("event: response.failed"));
  assert(failed_data != std::string::npos);
  const auto failed_event = gufo::json::parse(
      std::string_view(failed_stream)
          .substr(failed_data + 6,
                  failed_stream.find("\n\n", failed_data) - failed_data - 6));
  assert(failed_event.find("response")->find("error")->member_str("code") ==
         "server_error");
  assert(failed_stream.ends_with("0\r\n\r\n"));
  server.backend->failure = 0;
  const auto continued = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}]},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]},
        {"role":"user","content":"Next question"}]})"));
  assert(continued.member_str("status") == "completed");
  const auto replay_messages = server.backend->LastCall().chat.messages;
  assert(replay_messages.size() == 3 &&
         replay_messages[1].thought == "Thoughts" &&
         replay_messages[1].content == "Answer");

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");
}

void TestInvalidBindSettings() {
  for (const int port : {-1, 65536}) {
    HttpServer server("127.0.0.1", port, nullptr);
    std::string error;
    assert(!server.start(&error));
    assert(!error.empty());
  }
  HttpServer invalid("bad.address", 0, nullptr);
  std::string error;
  assert(!invalid.start(&error));
  assert(!error.empty());
}

void TestCompatibilityUtf8() {
  RunningServer server;
  server.backend->SetOutput("é中😀\xE2\x94!\xF0\x9F");
  const std::string expected = "é中😀\xEF\xBF\xBD!\xEF\xBF\xBD";
  for (const auto& [path, input] : {
           std::pair{"/v1/completions", R"({"prompt":"hi"})"},
           std::pair{"/v1/responses", R"({"input":"hi"})"},
           std::pair{"/v1/messages",
                     R"({"messages":[{"role":"user","content":"hi"}]})"},
           std::pair{"/completion", R"({"prompt":"hi"})"},
       }) {
    const auto response = server.Post(path, input);
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    std::string text;
    if (std::string_view(path) == "/v1/completions")
      text = output.find("choices")->items()[0].member_str("text");
    else if (std::string_view(path) == "/v1/responses")
      text = output.find("output")
                 ->items()[0]
                 .find("content")
                 ->items()[0]
                 .member_str("text");
    else if (std::string_view(path) == "/v1/messages")
      text = output.find("content")->items()[0].member_str("text");
    else
      text = output.member_str("content");
    assert(text == expected);
  }
}

void TestQueryParameters() {
  gufo::server::HttpRequest request;
  request.query = "notafter=wrong&note=after=wrong&after=right+value%26x";
  assert(request.query_param("after") == "right value&x");
  request.query = "notafter=wrong&note=after=wrong";
  assert(request.query_param("after").empty());
  request.query = "%61fter=encoded&broken=%xz&empty";
  assert(request.query_param("after") == "encoded");
  assert(request.query_param("broken") == "%xz");
  assert(request.query_param("empty").empty());
}

void TestPeerDisconnect() {
  RunningServer server;
  server.backend->wait_for_disconnect = true;
  const int fd = server.Connect();
  const std::string body = R"({"prompt":"hi","max_tokens":128})";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
  ::close(fd);
  assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
  assert(server.backend->disconnected);
}

void TestCompatibilityStopSequences() {
  RunningServer server;
  server.backend->forced_stop_sequence = "END";
  for (
      const auto& [path, body] : {
          std::pair{"/v1/completions", R"({"prompt":"hello","stop":"END"})"},
          std::pair{"/completion", R"({"prompt":"hello","stop":["END"]})"},
          std::pair{
              "/v1/messages",
              R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"]})"},
      }) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    assert(server.backend->LastCall().stop_sequences ==
           std::vector<std::string>{"END"});
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    if (std::string_view(path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "stop_sequence");
      assert(output.member_str("stop_sequence") == "END");
    } else if (std::string_view(path) == "/completion") {
      assert(output.find("stopped_word")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
      assert(output.member_str("stopping_word") == "END");
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "stop");
    }
  }
  const auto before = server.backend->calls.load();
  for (const auto* stop : {"null", "\"END\"", "[null]", "[\"\"]", "true"}) {
    auto body = gufo::json::parse(
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})");
    body["stop_sequences"] = gufo::json::parse(stop);
    ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  }
  ExpectStatus(server.Post("/v1/responses", R"({"input":"hi","stop":"END"})"),
               400);
  assert(server.backend->calls == before);
  auto body = gufo::json::parse(
      R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":[]})");
  for (int i = 0; i < 65; ++i)
    body["stop_sequences"].push_back(std::to_string(i));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  body["stop_sequences"] = gufo::json::Value::array();
  for (int i = 0; i < 5; ++i)
    body["stop_sequences"].push_back(std::string(4096, 'x'));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  server.backend->forced_stop_sequence.clear();
  for (const int limit : {1, 32}) {
    body["stop_sequences"] = gufo::json::Value::array();
    body["max_tokens"] = limit;
    const auto response = server.Post("/v1/messages", body.dump());
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(output.member_str("stop_reason") ==
           (limit == 1 ? "max_tokens" : "end_turn"));
    assert(output.find("stop_sequence")->is_null());
  }
}

void TestCompatibilityThinkingDefaults() {
  RunningServer server;
  for (const bool enabled : {false, true}) {
    server.backend->reasoning = {
        .enabled = enabled,
        .effort = gufo::ReasoningEffort::kHigh,
        .preserve_thinking = true,
    };
    for (
        const auto& [path, body] : {
            std::pair{"/v1/responses", R"({"input":"hello"})"},
            std::pair{
                "/v1/messages",
                R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})"},
        }) {
      ExpectStatus(server.Post(path, body), 200);
      const auto reasoning = server.backend->LastCall().chat.reasoning;
      assert(reasoning.enabled == enabled);
      assert(reasoning.effort == gufo::ReasoningEffort::kHigh);
      assert(reasoning.preserve_thinking == true);
    }
  }
}

void TestStreamingFraming() {
  RunningServer server;
  const std::string chunks = std::string("3\r\na\0b\r\n", 8) + "3\r\nend\r\n";
  for (const std::string body : {"", "fail", "reported"}) {
    const auto response = server.Post("/stream", body);
    ExpectStatus(response, 200);
    assert(response.find("Transfer-Encoding: chunked\r\n") !=
           std::string::npos);
    assert(response.substr(response.find("\r\n\r\n") + 4) ==
           chunks + (body == "fail" ? "" : "0\r\n\r\n"));
  }
  const auto thrown = server.Post("/stream-error", "");
  assert(thrown.substr(thrown.find("\r\n\r\n") + 4) == "b\r\nfirst chunk\r\n");
  // Existing HTTP/1.0 clients retain close-delimited framing.
  const auto legacy = server.Send("POST /stream HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.substr(legacy.find("\r\n\r\n") + 4) ==
         std::string("a\0bend", 6));
}

void TestSignalShutdown() {
  // Process signals must never terminate the test runner itself. Prove that
  // both idle listeners and active generation return through normal cleanup.
  for (const int signal : {SIGINT, SIGTERM}) {
    for (const bool active : {false, true}) {
      const pid_t child = ::fork();
      assert(child >= 0);
      if (child == 0) {
        ::alarm(5);
        {
          RunningServer server({}, true);
          // Accepting a request proves run() installed its handlers.
          ExpectStatus(server.Post("/echo", "ready"), 200);
          int fd = -1;
          if (active) {
            server.backend->wait_for_disconnect = true;
            fd = server.Connect();
            const std::string body = R"({"prompt":"hello"})";
            const std::string request =
                "POST /v1/completions HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body;
            assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(request.size()));
            assert(server.backend->entered.try_acquire_for(
                std::chrono::seconds(2)));
          }
          assert(::kill(::getpid(), signal) == 0);
          assert(server.run_finished.try_acquire_for(std::chrono::seconds(2)));
          if (active) {
            assert(server.backend->disconnected);
            ::close(fd);
          }
        }
        ::_exit(0);
      }
      int status = 0;
      assert(::waitpid(child, &status, 0) == child);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
  }
}

}  // namespace

void TestRuntimeEndpointAndRequestIds() {
  RunningServer server;
  const auto runtime = server.Send("GET /v1/runtime HTTP/1.1\r\n\r\n");
  ExpectStatus(runtime, 200);
  assert(runtime.find(R"("object":"gufo.runtime")") != std::string::npos);
  assert(runtime.find(R"("contract_version":1)") != std::string::npos);
  // The fake backend has no sessions, so it can never take flex work.
  assert(runtime.find(R"("flex":false)") != std::string::npos);
  // A backend without a scheduler cannot drain.
  ExpectStatus(server.Send("POST /v1/runtime/drain HTTP/1.1\r\n"
                           "Content-Length: 0\r\n\r\n"),
               503);

  const auto named = server.Send(
      "GET /v1/runtime HTTP/1.1\r\nX-Request-ID: router-42.a\r\n\r\n");
  assert(named.find("X-Request-ID: router-42.a\r\n") != std::string::npos);
  const auto unsafe =
      server.Send("GET /v1/runtime HTTP/1.1\r\nX-Request-ID: bad id\r\n\r\n");
  assert(unsafe.find("X-Request-ID: bad id") == std::string::npos &&
         unsafe.find("X-Request-ID: r") != std::string::npos);
}

void TestDrainAuthorization() {
  // Without --admin-key-file, drain is for loopback peers with the API key.
  assert(gufo::server::IsLoopbackAddress("127.0.0.1") &&
         gufo::server::IsLoopbackAddress("127.255.0.9"));
  for (const auto* peer :
       {"10.0.0.1", "128.0.0.1", "0.0.0.0", "anonymous", "", "127.0.0.1.5"})
    assert(!gufo::server::IsLoopbackAddress(peer));
  const std::string drain =
      "POST /v1/runtime/drain HTTP/1.1\r\nContent-Length: 0\r\n";
  {
    RunningServer server({.api_key = "test-secret"});
    ExpectStatus(server.Send(drain + "\r\n"), 401);
    // Reaches the backend, which has no scheduler to drain.
    ExpectStatus(
        server.Send(drain + "Authorization: Bearer test-secret\r\n\r\n"), 503);
  }
  {
    // With an admin key only that key drains, from loopback too, and it
    // opens nothing else.
    RunningServer server({.api_key = "test-secret", .admin_key = "test-admin"});
    ExpectStatus(
        server.Send(drain + "Authorization: Bearer test-secret\r\n\r\n"), 401);
    ExpectStatus(
        server.Send(drain + "Authorization: Bearer test-admin\r\n\r\n"), 503);
    ExpectStatus(server.Send("GET /v1/runtime HTTP/1.1\r\nAuthorization: "
                             "Bearer test-admin\r\n\r\n"),
                 401);
  }
}

void TestFlexModelAlias() {
  RunningServer server;
  using gufo::json::parse;
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  ExpectStatus(models, 200);
  assert(models.find(R"("id":"test-flex")") != std::string::npos &&
         models.find(R"("parent":"test")") != std::string::npos);

  // Responses: the alias runs at flex and is echoed for gateway logs.
  const auto response =
      server.Post("/v1/responses", R"({"model":"test-flex","input":"hi"})");
  ExpectStatus(response, 200);
  const auto body = parse(response.substr(response.find("\r\n\r\n") + 4));
  assert(body.member_str("model") == "test-flex" &&
         body.member_str("service_tier") == "flex");
  assert(server.backend->LastCall().chat.service_tier ==
         gufo::server::ServiceTier::kFlex);
  ExpectStatus(server.Post("/v1/responses", R"({"model":"test","input":"hi"})"),
               200);
  assert(server.backend->LastCall().chat.service_tier ==
         gufo::server::ServiceTier::kDefault);
  const auto conflict = server.Post(
      "/v1/responses",
      R"({"model":"test-flex","input":"hi","service_tier":"default"})");
  ExpectStatus(conflict, 400);
  assert(conflict.find("invalid_service_tier") != std::string::npos);

  // Surfaces without service tiers refuse the alias.
  ExpectStatus(server.Post("/v1/messages",
                           R"({"model":"test-flex","max_tokens":1,)"
                           R"("messages":[{"role":"user","content":"hi"}]})"),
               400);
  ExpectStatus(
      server.Post("/v1/completions", R"({"model":"test-flex","prompt":"hi"})"),
      404);
}

void TestAdapterFlexAlias() {
  RunningServer server;
  server.backend->adapters = {"test-abliterated"};
  using gufo::json::parse;
  // Every served variant gets its own flex alias, parented on that variant.
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  ExpectStatus(models, 200);
  const auto listed = parse(models.substr(models.find("\r\n\r\n") + 4));
  bool base_flex = false;
  bool adapter_flex = false;
  for (const auto& model : listed.find("data")->items()) {
    const auto id = model.member_str("id");
    base_flex |= id == "test-flex" && model.member_str("parent") == "test";
    adapter_flex |= id == "test-abliterated-flex" &&
                    model.member_str("parent") == "test-abliterated";
  }
  assert(base_flex && adapter_flex);

  // The adapter's alias runs at flex and keeps its public id for the engine
  // (which resolves the adapter) and for the echoed response.
  const auto response = server.Post(
      "/v1/responses", R"({"model":"test-abliterated-flex","input":"hi"})");
  ExpectStatus(response, 200);
  const auto body = parse(response.substr(response.find("\r\n\r\n") + 4));
  assert(body.member_str("model") == "test-abliterated-flex" &&
         body.member_str("service_tier") == "flex");
  const auto call = server.backend->LastCall().chat;
  assert(call.service_tier == gufo::server::ServiceTier::kFlex &&
         call.model == "test-abliterated-flex");
  ExpectStatus(server.Post("/v1/chat/completions",
                           R"({"model":"test-abliterated-flex",)"
                           R"("messages":[{"role":"user","content":"hi"}]})"),
               200);
  assert(server.backend->LastCall().chat.service_tier ==
         gufo::server::ServiceTier::kFlex);
  // The adapter itself stays on the default tier.
  ExpectStatus(server.Post("/v1/responses",
                           R"({"model":"test-abliterated","input":"hi"})"),
               200);
  assert(server.backend->LastCall().chat.service_tier ==
         gufo::server::ServiceTier::kDefault);
  // A suffix on an id that is not served is not an alias.
  ExpectStatus(
      server.Post("/v1/responses", R"({"model":"other-flex","input":"hi"})"),
      404);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"model":"test-abliterated-flex","max_tokens":1,)"
                           R"("messages":[{"role":"user","content":"hi"}]})"),
               400);
}

void TestFastModelIds() {
  using Backend = gufo::server::TextGenerationBackend;
  // `-flex` schedules; `-fast` is a different model, and they combine.
  auto id = Backend::SplitModelId("workhorse-abliterated-fast-flex", true);
  assert(id.variant == "workhorse-abliterated" && id.fast && id.flex);
  id = Backend::SplitModelId("workhorse-fast", true);
  assert(id.variant == "workhorse" && id.fast && !id.flex);
  id = Backend::SplitModelId("workhorse-flex", true);
  assert(id.variant == "workhorse" && !id.fast && id.flex);
  // Without fast models, -fast is just part of a name.
  id = Backend::SplitModelId("workhorse-fast", false);
  assert(id.variant == "workhorse-fast" && !id.fast);

  // A backend serving fast models lists them, and each gets a flex alias.
  RunningServer server;
  server.backend->adapters = {"test-abliterated", "test-fast",
                              "test-abliterated-fast"};
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  ExpectStatus(models, 200);
  for (const char* listed :
       {R"("id":"test-fast")", R"("id":"test-abliterated-fast-flex")",
        R"("parent":"test-abliterated-fast")"})
    assert(models.find(listed) != std::string::npos);
  const auto response =
      server.Post("/v1/responses",
                  R"({"model":"test-abliterated-fast-flex","input":"hi"})");
  ExpectStatus(response, 200);
  assert(server.backend->LastCall().chat.service_tier ==
             gufo::server::ServiceTier::kFlex &&
         server.backend->LastCall().chat.model == "test-abliterated-fast-flex");
}

int main() {
  TestRequestLogging();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestCrossOriginRequests();
  TestFramingAndMetrics();
  TestMetricsEndpoint();
  TestMetricsListener();
  TestRuntimeEndpointAndRequestIds();
  TestDrainAuthorization();
  TestCompatibilityRequests();
  TestFlexModelAlias();
  TestAdapterFlexAlias();
  TestFastModelIds();
  TestCompatibilityStopSequences();
  TestCompatibilityThinkingDefaults();
  TestCompatibilityUtf8();
  TestPeerDisconnect();
  TestStreamingFraming();
  TestSignalShutdown();
  std::cout << "HTTP transport checks passed.\n";
}
