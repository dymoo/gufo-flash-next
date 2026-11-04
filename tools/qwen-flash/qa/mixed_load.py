#!/usr/bin/env python3
"""Mixed agent load: N streaming decoders + periodic long/short prompt appends.

Measures decoder inter-token latency (p50/p99), aggregate decode tok/s and the
TTFT of the appended prompts. Stdlib only (the test host has no numpy).
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

WORDS = ("alpha beta gamma delta kernel memory tensor layer cache token stream "
         "prefill decode expert router batch session agent tool output value").split()


def text(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def stream(url, prompt, max_tokens, on_token):
    body = json.dumps({"model": "qa", "stream": True, "max_tokens": max_tokens,
                       "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            now = time.monotonic()
            first = first if first is not None else now - t0
            on_token(now)
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=768)
    ap.add_argument("--long-words", type=int, default=3000, help="~4K-token appends")
    ap.add_argument("--long-every", type=float, default=10.0)
    ap.add_argument("--short-words", type=int, default=220, help="~300-token appends")
    ap.add_argument("--short-every", type=float, default=3.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    gaps, lock, stop = [], threading.Lock(), threading.Event()
    decoded = [0]
    ttft = {"long": [], "short": []}

    def decoder(i):
        last = [None]

        def tok(now):
            with lock:
                decoded[0] += 1
                if last[0] is not None:
                    gaps.append(now - last[0])
            last[0] = now
        stream(a.base_url, "Write a long story, number every sentence. " + text(40, i), a.decode_tokens, tok)

    def appender(kind, words, every):
        n = 0
        while not stop.is_set():
            prompt = f"Summarise this tool output in one line:\n{text(words, 1000 + n)}"
            n += 1
            try:
                t = stream(a.base_url, prompt, 8, lambda _now: None)
                ttft[kind].append(t)
            except Exception as e:  # server overload is a result, not a crash
                print(kind, "append failed:", e)
            stop.wait(every)

    dec = [threading.Thread(target=decoder, args=(i,)) for i in range(a.decoders)]
    app = [threading.Thread(target=appender, args=("long", a.long_words, a.long_every), daemon=True),
           threading.Thread(target=appender, args=("short", a.short_words, a.short_every), daemon=True)]
    t0 = time.monotonic()
    for t in dec + app:
        t.start()
    for t in dec:
        t.join()
    wall = time.monotonic() - t0
    stop.set()

    def pct(xs, p):
        xs = sorted(xs)
        return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None

    res = {"wall_s": round(wall, 1), "decode_tok_s": round(decoded[0] / wall, 1),
           "itl_ms_p50": pct(gaps, 0.5), "itl_ms_p99": pct(gaps, 0.99), "itl_ms_max": pct(gaps, 1.0),
           "ttft_long_ms_p50": pct(ttft["long"], 0.5), "ttft_short_ms_p50": pct(ttft["short"], 0.5),
           "appends": {k: len(v) for k, v in ttft.items()}}
    print(json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
