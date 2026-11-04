#!/usr/bin/env python3
"""Compare result files on the queue indices every one of them graded (a stall-proof, same-queue comparison).

    compare.py --set 0-47 RESULT.json [RESULT.json ...] [--common] [--md]

Without --common each file is summarized over its own graded indices inside --set; with it, only over the
indices all files graded, which is the fair comparison when runs stopped at different points. Token costs are
per correct answer; "ideal prefill" is the server-independent prefill a perfect prefix cache would do: the shared
system prompt is removed and each turn counts only what it adds to the conversation so far.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def wilson(k: int, n: int) -> tuple[int | None, int | None]:
    if not n:
        return None, None
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * (centre - half)), round(100 * (centre + half))


def ideal_prefill(result: dict, indices: list[int]) -> int:
    by_task = collections.defaultdict(list)
    for q in result["requests"]:
        by_task[q[0]].append(q)
    family = {t["g"]: t["family"] for t in result["tasks"]}
    first = collections.defaultdict(list)
    for g, requests in by_task.items():
        requests.sort(key=lambda q: q[1])
        first[family.get(g)].append(requests[0][5])
    system = {f: min(v) - 200 for f, v in first.items() if v}  # smallest first prompt minus the task text
    total = 0
    for g in indices:
        requests = sorted(by_task.get(g, []), key=lambda q: q[1])
        if requests:
            total += requests[0][5] - system[family[g]]
            total += sum(max(0, b[5] - a[5] - a[7]) for a, b in zip(requests, requests[1:]))
    return total


def summarize(label: str, result: dict, indices: list[int]) -> dict:
    tasks = {t["g"]: t for t in result["tasks"]}
    picked = [tasks[g] for g in indices]
    n, k = len(picked), sum(t["outcome"] == "success" for t in picked)
    total = lambda key: sum(t.get(key, 0) for t in picked)  # noqa: E731
    cfg = result["config"]
    return {
        "label": label, "setting": cfg["thinking"] + (f"/{cfg['effort']}" if cfg.get("effort") else ""),
        "n": n, "correct": k, "rate": round(100 * k / n) if n else None, "ci": wilson(k, n),
        "compl": round(total("completion_tokens") / k) if k else None,
        "think": round(total("reasoning_tokens") / k) if k else None,
        "ideal": round(ideal_prefill(result, indices) / k) if k else None,
        "turns": round(total("turns") / n, 2) if n else None,
        "calls": round(total("tool_calls") / n, 1) if n else None,
        "outcomes": dict(collections.Counter(t["outcome"] for t in picked)),
        "by_family": {f: f"{sum(t['outcome'] == 'success' for t in picked if t['family'] == f)}/"
                         f"{sum(1 for t in picked if t['family'] == f)}" for f in sorted({t["family"] for t in picked})},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--set", required=True, help="queue index range, e.g. 0-47")
    ap.add_argument("--common", action="store_true", help="only indices every file graded")
    ap.add_argument("--md", action="store_true", help="print a Markdown table")
    args = ap.parse_args()
    lo, hi = map(int, args.set.split("-"))
    want = set(range(lo, hi + 1))
    runs = [json.loads(Path(p).read_text()) for p in args.results]
    graded = [{t["g"] for t in r["tasks"] if t["outcome"] != "unfinished"} & want for r in runs]
    common = set.intersection(*graded) if args.common else None
    if common is not None:
        print(f"common indices: {len(common)} of {len(want)}; not graded by every file: {sorted(want - common)}")
    rows = [summarize(Path(p).stem, r, sorted(common if common is not None else g))
            for p, r, g in zip(args.results, runs, graded)]
    if args.md:
        print("| Run | Setting | Correct | Success (95% CI) | Compl. tok/correct | Think tok/correct | "
              "Ideal prefill tok/correct | Turns/task | Tool calls/task |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            ci = f"{r['rate']}% ({r['ci'][0]}-{r['ci'][1]})" if r["n"] else "-"
            fmt = lambda v: "-" if v is None else f"{v:,}"  # noqa: E731
            print(f"| {r['label']} | {r['setting']} | {r['correct']}/{r['n']} | {ci} | {fmt(r['compl'])} | "
                  f"{fmt(r['think'])} | {fmt(r['ideal'])} | {r['turns']} | {r['calls']} |")
    else:
        for r in rows:
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
