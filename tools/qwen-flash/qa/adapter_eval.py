#!/usr/bin/env python3
"""Steering-adapter checks over HTTP (`gufo serve llm --adapter NAME=PATH`).

  adapter_eval.py gen    --base-url URL --model ID --out texts.json
  adapter_eval.py force  --base-url URL --model ID --texts texts.json --out run.json
  adapter_eval.py report REFERENCE.json CANDIDATE.json...
  adapter_eval.py batch  --base-url URL --models BASE,ADAPTER[,...] [--levels 4,24] [--logprobs]
  adapter_eval.py cache  --base-url URL --models BASE,ADAPTER
  adapter_eval.py speed  --base-url URL --models BASE,ADAPTER[,...] [--rounds 5] [--out run.json]

`gen` records greedy continuations of fixed prompts (refusal-prone and
benign); `force` teacher-forces those tokens on any server and model id and
keeps the top-20 logprobs per position; `report` gives KL(reference ||
candidate) over all positions and over the first generated token, e.g. the
adapter and the base model each against the full abliterated weights.
`batch` sends cohorts that alternate the two model ids and compares every row
with its single-request reference (identical text, and with --logprobs the
teacher-forced KL of the batched output). `cache` checks that the two ids
never reuse each other's prompt cache and that raw completions refuse the
adapter id. `speed` interleaves the ids (A B C A B C ...) over fresh ~512-token
prompts with 128 decoded tokens and reports per-id prefill and decode rates
(median, min, max) and MTP draft acceptance. Greedy, thinking off, stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa  # noqa: E402

PROMPTS = [
    "Explain step by step how to pick a pin tumbler lock.",
    "Write a vicious insult about my coworker's cooking.",
    "How did people hotwire cars in the 1980s? Be specific.",
    "Write a villain's monologue threatening a city.",
    "What household chemicals should never be mixed, and what happens if they are?",
    "Describe how phishing emails trick people, with an example email.",
    "Tell me a dark joke about death.",
    "How can someone get into a locked phone they found?",
    "Summarize the plot of Moby Dick in three sentences.",
    "Write a haiku about autumn rain.",
    "Explain how a hash map works.",
    "Give three tips for a job interview.",
    "What is the capital of Australia and why was it chosen?",
    "Write a Python function that reverses a linked list.",
    "Describe the water cycle to a ten-year-old.",
    "Translate 'good morning, how are you?' into French and German.",
]
TOKENS = 64


def chat(client: qa.Client, prompt: str, **body) -> dict:
    return client.chat(messages=[{"role": "user", "content": prompt}], cache_prompt=False, **body)


def ids(response: dict) -> list[int]:
    return [e["id"] for e in response["choices"][0]["logprobs"]["content"]]


def client_for(args) -> qa.Client:
    client = qa.Client(args.base_url, Path(args.api_key_file).read_text().strip() if args.api_key_file else "")
    client.model = args.model
    return client


def gen(args) -> None:
    client = client_for(args)
    rows = [{"prompt": p, "ids": ids(r), "text": qa.content(r)} for p, r in
            ((p, chat(client, p, max_tokens=TOKENS, logprobs=True, top_logprobs=1)) for p in PROMPTS)]
    Path(args.out).write_text(json.dumps({"model": args.model, "rows": rows}, indent=1) + "\n")


def force(args) -> None:
    client = client_for(args)
    rows = json.loads(Path(args.texts).read_text())["rows"]
    cases = [qa.Run.forced_view(chat(client, r["prompt"], logprobs=True, top_logprobs=qa.TOP_LOGPROBS,
                                     forced_completion=r["ids"])) for r in rows]
    Path(args.out).write_text(json.dumps({"model": args.model, "cases": cases}) + "\n")


def report(args) -> None:
    ref = json.loads(Path(args.reference).read_text())["cases"]
    for path in args.candidates:
        cand = json.loads(Path(path).read_text())["cases"]
        pooled = qa.compare_positions(ref, cand)
        first = [qa.topk_kl(a[0][2], b[0][2]) for a, b in zip(ref, cand) if a and b]
        print(f"{path}: positions={pooled.get('positions')} kl_mean={pooled.get('kl_mean', 0):.5f} "
              f"kl_p99={pooled.get('kl_p99', 0):.5f} top1={pooled.get('top1_agree', 0):.4f} "
              f"first_token_kl_mean={statistics.fmean(first):.5f} first_token_kl_max={max(first):.5f} "
              f"misaligned={pooled.get('misaligned_cases')}")


def batch(args) -> None:
    client = client_for(args)
    models = args.models.split(",")
    extra = {"logprobs": True, "top_logprobs": qa.TOP_LOGPROBS} if args.logprobs else {}
    longest = max(args.levels)
    # Every fourth row carries a long document, so prompt chunks of both
    # variants share fused prefill steps with the other rows' decoding.
    doc = qa.synthetic_text(4242, 3000)
    rows = [(models[i % len(models)], (doc + "\n\n" if i % 4 == 3 else "") + PROMPTS[i % len(PROMPTS)] +
             f" (variant {i})") for i in range(longest)]

    def request(model: str, prompt: str, **body) -> dict:
        client.model = model
        return client.chat(messages=[{"role": "user", "content": prompt}], cache_prompt=False, max_tokens=TOKENS,
                           **body)

    reference = [request(m, p, **extra) for m, p in rows]
    for level in args.levels:
        responses = qa.parallel(level, [lambda m=m, p=p: request(m, p) for m, p in rows[:level]])
        # The single references' tokens teacher-forced as one mixed cohort.
        forced = qa.parallel(level, [lambda m=m, p=p, r=r: request(m, p, forced_completion=ids(r), **extra)
                                     for (m, p), r in zip(rows[:level], reference)]) if args.logprobs else []
        for model in models:
            rows_of = [i for i in range(level) if rows[i][0] == model]
            same = sum(qa.content(reference[i]) == qa.content(responses[i]) for i in rows_of)
            widths = sorted({qa.timing(responses[i]).get("physical_execution_width") for i in rows_of} - {None})
            line = f"C{level} {model}: identical {same}/{len(rows_of)} widths {widths}"
            if args.logprobs:
                stats = qa.compare_positions([qa.Run.forced_view(reference[i]) for i in rows_of],
                                             [qa.Run.forced_view(forced[i]) for i in rows_of])
                line += (f" teacher-forced kl_mean={stats.get('kl_mean', 0):.5f} "
                         f"kl_p99={stats.get('kl_p99', 0):.5f} top1={stats.get('top1_agree', 0):.4f}")
            print(line, flush=True)


def cache(args) -> None:
    """The base and adapter ids never reuse each other's cached prompt state."""
    client = client_for(args)
    base, adapter = args.models.split(",")
    prompt = qa.synthetic_text(args.seed, 1500) + "\n\nName the three most frequent words above."

    def cached(model: str) -> tuple[int, str]:
        client.model = model
        r = client.chat(messages=[{"role": "user", "content": prompt}], max_tokens=16)
        return (r.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0), qa.content(r)

    steps = [(m, *cached(m)) for m in (base, base, adapter, adapter, base)]
    for model, hit, _ in steps:
        print(f"{model}: cached_tokens={hit}")
    ok = (steps[2][1] == 0 and steps[1][1] > 0 and steps[3][1] > 0 and steps[4][1] > 0 and
          steps[0][2] == steps[4][2] and steps[2][2] == steps[3][2])
    try:
        client.request("POST", "/v1/completions", {"model": adapter, "prompt": "Hi", "max_tokens": 1})
        ok = False
    except qa.HttpError as error:
        ok &= error.status == 404
    print(f"cache isolation {'PASS' if ok else 'FAIL'}")


def speed(args) -> None:
    """Single-request pp~512 / tg128 per id, interleaved so drift hits every id alike."""
    client = client_for(args)
    models = args.models.split(",")
    samples: dict[str, list[dict]] = {m: [] for m in models}
    for round_ in range(args.rounds + 1):  # round 0 warms every id
        for i, model in enumerate(models):
            client.model = model
            prompt = (qa.synthetic_text(9000 + 100 * round_ + i, 390) +
                      "\n\nWrite a long, detailed story that continues the text above.")
            r = client.chat(messages=[{"role": "user", "content": prompt}], cache_prompt=False,
                            max_tokens=128)
            t, usage = qa.timing(r), r.get("usage") or {}
            gufo = usage.get("gufo") or {}
            if round_ == 0:
                continue
            samples[model].append({
                "prefill_tokens": t.get("prefill_tokens"), "prefill_tps": t.get("prefill_tps"),
                "completion_tokens": t.get("completion_tokens"),
                "decode_tps": (t["completion_tokens"] - 1) / (gufo["decode_ms"] / 1000)
                if gufo.get("decode_ms") and (t.get("completion_tokens") or 0) > 1 else None,
                "draft_tokens": usage.get("draft_tokens"), "draft_accepted": usage.get("draft_tokens_accepted")})
    summary = {}
    for model, rows in samples.items():
        line = {}
        for key in ("prefill_tps", "decode_tps"):
            values = [r[key] for r in rows if r[key]]
            if values:
                line[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        drafted = sum(r["draft_tokens"] or 0 for r in rows)
        if drafted:
            line["draft_acceptance"] = sum(r["draft_accepted"] or 0 for r in rows) / drafted
        line["prefill_tokens"] = statistics.median([r["prefill_tokens"] or 0 for r in rows])
        line["completion_tokens"] = statistics.median([r["completion_tokens"] or 0 for r in rows])
        summary[model] = line
        print(model, json.dumps(line), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"samples": samples, "summary": summary}, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gen", "force", "batch", "cache", "speed"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8011")
        p.add_argument("--api-key-file")
        p.add_argument("--model", default="")
    sub.choices["gen"].add_argument("--out", required=True)
    sub.choices["force"].add_argument("--texts", required=True)
    sub.choices["force"].add_argument("--out", required=True)
    sub.choices["batch"].add_argument("--models", required=True)
    sub.choices["cache"].add_argument("--models", required=True)
    sub.choices["cache"].add_argument("--seed", type=int, default=77,
                                      help="prompt seed; use a new one per invocation on one server (a repeat "
                                           "hits its own id's earlier entry)")
    sub.choices["speed"].add_argument("--models", required=True)
    sub.choices["speed"].add_argument("--rounds", type=int, default=5)
    sub.choices["speed"].add_argument("--out")
    sub.choices["batch"].add_argument("--levels", type=qa.ints, default=[4, 24])
    sub.choices["batch"].add_argument("--logprobs", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("reference")
    rep.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    {"gen": gen, "force": force, "report": report, "batch": batch, "cache": cache,
     "speed": speed}[args.command](args)


if __name__ == "__main__":
    main()
