#!/usr/bin/env python3
"""MTP draft acceptance per steering id over the same prompts (greedy, concurrency 1).

  mtp_accept.py --base-url URL --models base,huihui,orca,orca-nomtp --out FILE [--tokens 192]
                [--max-seconds 600]

Every id answers every prompt (prompt order outermost, ids interleaved inside, so server
state and thermal drift hit all ids alike). Acceptance = draft_tokens_accepted / draft_tokens
from the usage block; prompts are identical across ids, unlike adapter_eval speed (a different
synthetic text per id and round). Prints pooled acceptance per id, the per-prompt paired
difference of `--pair A,B` (default orca,orca-nomtp: same adapter with and without the MTP
writers) with a bootstrap 95% interval over prompts, decode tok/s medians, and writes all rows.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa"))
import adapter_eval  # noqa: E402
import qa  # noqa: E402

EXTRA = [
    "Write a Python function that parses an ISO-8601 date string without any library, with tests.",
    "Explain the difference between a mutex and a semaphore with a short C example.",
    "Draft a polite email asking a landlord to fix a broken heater, then summarize it in one line.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the steps to deploy a containerized web app to a small Kubernetes cluster.",
    "Explain how TLS handshakes work, including certificate verification.",
    "Describe how a bill becomes law in the United States.",
    "Write a SQL query that finds the second highest salary per department, and explain it.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", default="base,huihui,orca,orca-nomtp")
    ap.add_argument("--pair", default="orca,orca-nomtp")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key-file")
    args = ap.parse_args()
    client = qa.Client(args.base_url, Path(args.api_key_file).read_text().strip() if args.api_key_file else "")
    models = args.models.split(",")
    prompts = adapter_eval.PROMPTS + EXTRA
    rows: list[dict] = []
    start = time.monotonic()
    for p_index, prompt in enumerate(prompts):
        if time.monotonic() - start > args.max_seconds:
            print(f"time cap after {p_index} prompts", flush=True)
            break
        for model in models:
            client.model = model
            r = client.chat(messages=[{"role": "user", "content": prompt}], cache_prompt=False,
                            max_tokens=args.tokens)
            usage = r.get("usage") or {}
            gufo = usage.get("gufo") or {}
            n = usage.get("completion_tokens") or 0
            rows.append({"prompt": p_index, "model": model, "completion_tokens": n,
                         "draft_tokens": usage.get("draft_tokens") or 0,
                         "draft_accepted": usage.get("draft_tokens_accepted") or 0,
                         "decode_tps": (n - 1) / (gufo["decode_ms"] / 1000)
                         if gufo.get("decode_ms") and n > 1 else None})
        print(f"prompt {p_index + 1}/{len(prompts)} " + " ".join(
            f"{r['model']}={r['draft_accepted']}/{r['draft_tokens']}" for r in rows[-len(models):]), flush=True)
    done = sorted({r["prompt"] for r in rows if sum(1 for q in rows if q["prompt"] == r["prompt"]) == len(models)})
    summary = {}
    for model in models:
        mine = [r for r in rows if r["model"] == model and r["prompt"] in done]
        drafted = sum(r["draft_tokens"] for r in mine)
        accepted = sum(r["draft_accepted"] for r in mine)
        tps = [r["decode_tps"] for r in mine if r["decode_tps"]]
        summary[model] = {"prompts": len(mine), "drafted": drafted, "accepted": accepted,
                          "acceptance": accepted / drafted if drafted else None,
                          "decode_tps_median": statistics.median(tps) if tps else None,
                          "completion_tokens": sum(r["completion_tokens"] for r in mine)}
        print(model, json.dumps(summary[model]), flush=True)
    a, b = args.pair.split(",")
    if a in models and b in models and done:
        def rate(model: str, prompt: int) -> float:
            row = next(r for r in rows if r["model"] == model and r["prompt"] == prompt)
            return row["draft_accepted"] / row["draft_tokens"] if row["draft_tokens"] else float("nan")
        diffs = [rate(a, p) - rate(b, p) for p in done]
        diffs = [d for d in diffs if d == d]
        rng = random.Random(1)
        boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
        pair = {"pair": f"{a} - {b}", "prompts": len(diffs), "mean_diff": statistics.fmean(diffs),
                "median_diff": statistics.median(diffs), "ci95": [boots[100], boots[3899]],
                "a_better": sum(d > 0 for d in diffs), "b_better": sum(d < 0 for d in diffs)}
        summary["pair"] = pair
        print("paired", json.dumps(pair), flush=True)
    Path(args.out).write_text(json.dumps({"tokens": args.tokens, "rows": rows, "summary": summary}, indent=1) + "\n")


if __name__ == "__main__":
    main()
