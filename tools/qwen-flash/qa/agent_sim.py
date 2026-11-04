#!/usr/bin/env python3
"""Realistic async coding-agent load for a Gufo OpenAI-compatible server.

Agents arrive at random (Poisson), share one long system prompt, and run
multi-turn loops: generate a reply (streamed), "run a tool" for a random
think time, append a random-size tool output, repeat until their context
budget or turn limit is reached. Prefill of new turns therefore overlaps
other agents' decoding, and bursts, idle gaps and long contexts all occur.

Reports per-turn TTFT (first generated token), inter-token latency, and
aggregate prefill/decode throughput over the whole run. Stdlib only.
"""
import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CHARS_PER_TOKEN = 3.8


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.ttft = []            # seconds, per turn
        self.ttft_first = []      # seconds, first turn of each agent (shared system prompt)
        self.itl = []             # seconds between generated tokens
        self.gen_tokens = 0
        self.prompt_tokens = 0    # new (uncached) prompt tokens reported by the server
        self.cached_tokens = 0
        self.turns = 0
        self.hit = []             # per-turn cached / prompt tokens (turn > 1)
        self.errors = []
        self.flex_rejected = 0    # 429 resource_unavailable, retried
        self.active = 0
        self.max_active = 0
        self.timeline = []        # (t, active)


def chunk_text(corpus, rng, n_tokens):
    n = int(n_tokens * CHARS_PER_TOKEN)
    start = rng.randrange(0, max(1, len(corpus) - n))
    return corpus[start:start + n]


def stream_turn(base, api_key, messages, max_tokens, stats, service_tier=None):
    body = {"model": "qa", "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True}, "messages": messages}
    if service_tier:
        body["service_tier"] = service_tier
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.monotonic()
    first = last = None
    text, n_tok, usage = [], 0, None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if d.get("tool_calls"):
                    piece += json.dumps(d["tool_calls"])
                if not piece:
                    continue
                now = time.monotonic()
                if first is None:
                    first = now - t0
                elif last is not None:
                    with stats.lock:
                        stats.itl.append(now - last)
                last = now
                n_tok += 1
                text.append(d.get("content") or "")
    gen = (usage or {}).get("completion_tokens", n_tok)
    prompt = (usage or {}).get("prompt_tokens")
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return first, gen, prompt, cached, "".join(text)


def agent(i, args, corpus, system, stats, deadline):
    rng = random.Random(args.seed * 1000 + i)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Task {i}: refactor module {rng.randrange(1000)}. "
                                            + chunk_text(corpus, rng, rng.randint(200, 1500))}]
    ctx_tokens = (len(system) + len(messages[1]["content"])) / CHARS_PER_TOKEN
    budget = rng.randint(args.min_context, args.max_context)
    with stats.lock:
        stats.active += 1
        stats.max_active = max(stats.max_active, stats.active)
    try:
        for _ in range(args.max_turns):
            if time.monotonic() > deadline or ctx_tokens > budget:
                break
            max_tokens = rng.randint(args.min_gen, args.max_gen)
            try:
                while True:
                    try:
                        first, gen, prompt, cached, reply = stream_turn(
                            args.base_url, args.api_key, messages, max_tokens, stats, args.service_tier)
                        break
                    except urllib.error.HTTPError as e:
                        # Flex: no spare capacity now; back off and retry, like an OpenAI flex client.
                        if e.code != 429 or b"resource_unavailable" not in e.read() or \
                                time.monotonic() > deadline:
                            raise
                        with stats.lock:
                            stats.flex_rejected += 1
                        time.sleep(float(e.headers.get("Retry-After") or 1) * rng.uniform(1, 2))
            except Exception as e:  # overload/429/context errors are results
                with stats.lock:
                    stats.errors.append(f"{type(e).__name__}: {str(e)[:120]}")
                break
            with stats.lock:
                stats.turns += 1
                stats.gen_tokens += gen
                if first is not None:
                    stats.ttft.append(first)
                    if len(messages) == 2:
                        stats.ttft_first.append(first)
                if prompt is not None:
                    stats.prompt_tokens += max(0, prompt - cached)
                    stats.cached_tokens += cached
                    if len(messages) > 2 and prompt:
                        stats.hit.append(cached / prompt)
            messages.append({"role": "assistant", "content": reply or "ok"})
            time.sleep(rng.expovariate(1.0 / args.think_mean))      # tool running
            tool = chunk_text(corpus, rng, int(rng.lognormvariate(args.tool_log_mean, 0.9)))
            messages.append({"role": "user", "content": "Tool output:\n" + tool})
            ctx_tokens += (len(reply) + len(tool)) / CHARS_PER_TOKEN
    finally:
        with stats.lock:
            stats.active -= 1


def pct(xs, p):
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--api-key-file")
    ap.add_argument("--corpus", nargs="+", required=True, help="text files to draw prompts/tool outputs from")
    ap.add_argument("--duration", type=float, default=1200, help="seconds of new work")
    ap.add_argument("--arrival-rate", type=float, default=0.1, help="new agents per second (Poisson)")
    ap.add_argument("--max-agents", type=int, default=64)
    ap.add_argument("--burst", type=int, default=8, help="agents started at t=0")
    ap.add_argument("--system-tokens", type=int, default=20000)
    ap.add_argument("--agent-types", type=int, default=1,
                    help="distinct system prompts (agent kinds); agent i uses type i %% N")
    ap.add_argument("--min-context", type=int, default=30000)
    ap.add_argument("--max-context", type=int, default=100000)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--min-gen", type=int, default=64)
    ap.add_argument("--max-gen", type=int, default=600)
    ap.add_argument("--think-mean", type=float, default=6.0, help="mean tool time, s")
    ap.add_argument("--tool-log-mean", type=float, default=6.9, help="ln(tokens): 6.9 ≈ 1000 median")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--service-tier", choices=["flex"], help="send service_tier; 429s are retried")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    corpus = "\n".join(Path(p).read_text(errors="ignore") for p in args.corpus)
    systems = [f"You are coding agent type {k}. Follow the repository rules below.\n" +
               chunk_text(corpus, random.Random(k), args.system_tokens) for k in range(max(1, args.agent_types))]

    stats = Stats()
    t0 = time.monotonic()
    deadline = t0 + args.duration
    threads = []
    rng = random.Random(args.seed)

    def spawn():
        i = len(threads)
        th = threading.Thread(target=agent, args=(i, args, corpus, systems[i % len(systems)], stats, deadline),
                              daemon=True)
        threads.append(th)
        th.start()

    for _ in range(args.burst):
        spawn()
    while time.monotonic() < deadline and len(threads) < args.max_agents:
        time.sleep(rng.expovariate(args.arrival_rate))
        spawn()
        with stats.lock:
            stats.timeline.append((round(time.monotonic() - t0), stats.active))
    for th in threads:
        th.join()
    wall = time.monotonic() - t0
    res = {"wall_s": round(wall), "agents": len(threads), "max_concurrent": stats.max_active,
           "turns": stats.turns, "decode_tok_s": round(stats.gen_tokens / wall, 1),
           "prefill_tok_s_uncached": round(stats.prompt_tokens / wall, 1),
           "cached_prompt_tokens": stats.cached_tokens, "gen_tokens": stats.gen_tokens,
           "ttft_ms_p50": pct(stats.ttft, 0.5), "ttft_ms_p90": pct(stats.ttft, 0.9),
           "ttft_ms_p99": pct(stats.ttft, 0.99), "ttft_first_ms_p50": pct(stats.ttft_first, 0.5),
           "ttft_first_ms_p90": pct(stats.ttft_first, 0.9), "itl_ms_p50": pct(stats.itl, 0.5),
           "itl_ms_p90": pct(stats.itl, 0.9), "itl_ms_p99": pct(stats.itl, 0.99),
           "cache_hit_p10_turn2plus": round(sorted(stats.hit)[len(stats.hit) // 10], 4) if stats.hit else None,
           "cache_hit_p50_turn2plus": round(sorted(stats.hit)[len(stats.hit) // 2], 4) if stats.hit else None,
           "flex_rejected": stats.flex_rejected, "errors": len(stats.errors), "first_errors": stats.errors[:5], "timeline": stats.timeline}
    print(json.dumps({k: v for k, v in res.items() if k != "timeline"}))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
