#!/usr/bin/env python3
"""Concurrent System One load, modelled on Kev's published benchmark: N clients
each sending six questions about a new short text (or a repeated one).

    python3 load.py --url http://127.0.0.1:18093 --clients 64 --seconds 60

Standard library only, so it runs on the serving host. Reports requests and
questions per second, input tokens per second, client latency percentiles and
the model time the server reports.
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

WORDS = ("order refund invoice charged twice parcel tracking warehouse damaged login password reset account "
         "subscription renewal discount expired payment failed card declined shipping label customer support "
         "urgent manager delay weeks broken replacement angry calm polite follow up ticket escalation").split()

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments, invoices, refunds", "shipping": "Delivery problems",
                          "access": "Login and account access", "other": None}},
    "urgent": {"type": "noul", "instructions": "Does the customer need help today?"},
    "tone": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["Calm", "Annoyed", "Frustrated", "Angry"]},
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "language": {"type": "choice", "instructions": "Which language is the message in?",
                 "criteria": {"en": None, "fr": None, "de": None, "es": None}},
    "churn": {"type": "noul", "instructions": "Is the customer threatening to leave?"},
}


def text(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words)).capitalize() + "."


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18093")
    ap.add_argument("--clients", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--words", type=int, default=60, help="state length in words (~1.3 tokens each)")
    ap.add_argument("--repeat", type=int, default=0, help="cycle this many fixed states (0 = every state new)")
    ap.add_argument("--api-key-file")
    a = ap.parse_args()
    headers = {"content-type": "application/json"}
    if a.api_key_file:
        with open(a.api_key_file) as f:
            headers["authorization"] = "Bearer " + f.read().strip()
    fixed = [text(random.Random(i), a.words) for i in range(a.repeat)]
    lock = threading.Lock()
    latencies, model_ms, tokens, errors = [], [], [0], [0]
    stop = time.monotonic() + a.seconds

    def client(k):
        rng = random.Random(1000 + k)
        while time.monotonic() < stop:
            state = rng.choice(fixed) if fixed else text(rng, a.words)
            body = json.dumps({"model": "kev-latest", "state": state, "questions": QUESTIONS}).encode()
            req = urllib.request.Request(a.url + "/v1/systemone", data=body, headers=headers)
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
            except (urllib.error.URLError, OSError, ValueError):
                with lock:
                    errors[0] += 1
                continue
            with lock:
                latencies.append((time.perf_counter() - start) * 1000)
                model_ms.append(out.get("latency_ms", 0.0))
                tokens[0] += out["usage"]["input_tokens"]

    started = time.monotonic()
    threads = [threading.Thread(target=client, args=(k,)) for k in range(a.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - started
    n = len(latencies)
    print(json.dumps({
        "clients": a.clients, "seconds": round(wall, 1), "requests": n, "errors": errors[0],
        "requests_per_s": round(n / wall, 2), "questions_per_s": round(n * len(QUESTIONS) / wall, 1),
        "input_tokens_per_s": round(tokens[0] / wall, 1),
        "tokens_per_request": round(tokens[0] / n, 1) if n else 0,
        "latency_ms_p50": round(percentile(latencies, 50), 1), "latency_ms_p90": round(percentile(latencies, 90), 1),
        "latency_ms_p99": round(percentile(latencies, 99), 1),
        "model_ms_p50": round(statistics.median(model_ms), 1) if model_ms else 0,
    }))


if __name__ == "__main__":
    main()
