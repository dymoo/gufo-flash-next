#!/usr/bin/env python3
"""Summarise agent-benchmark runs (tools/mimo harness) side by side.

For each run directory: the official SWE-bench report (resolved tasks), the
proxy's per-request log (tokens, cache hits, TTFT, decode speed, Gufo's own
prefill/decode timings), trajectories (task mapping, steps) and speed.jsonl
(cold prefill, single decode, aggregate decode). Stdlib only.

    report.py runs/qwen runs/mimo > report.md
"""
import glob
import hashlib
import json
import os
import statistics as st
import sys
from datetime import datetime


def task_key(messages):
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return hashlib.sha1((c or "")[:4000].encode()).hexdigest()[:16]
    return "none"


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def load(run):
    out = {"name": os.path.basename(run.rstrip("/"))}
    reports = glob.glob(os.path.join(run, "*.json"))
    report = next((json.load(open(p)) for p in reports
                   if "resolved_ids" in open(p).read(200000)), {})
    out["resolved"] = set(report.get("resolved_ids", []))
    out["submitted"] = report.get("submitted_instances")
    out["errors"] = len(report.get("error_ids", []))
    out["empty"] = len(report.get("empty_patch_ids", []))
    keys = {}
    steps = {}
    for traj in glob.glob(os.path.join(run, "*", "*.traj.json")):
        t = json.load(open(traj))
        iid = os.path.basename(traj)[: -len(".traj.json")]
        msgs = t.get("messages", [])
        keys[task_key(msgs)] = iid
        steps[iid] = sum(1 for m in msgs if m.get("role") == "assistant")
    out["steps"] = steps
    reqs = [json.loads(l) for l in open(os.path.join(run, "requests.jsonl"))]
    for r in reqs:
        r["iid"] = keys.get(r["task"])
    out["requests"] = reqs
    try:
        start = datetime.fromisoformat(open(os.path.join(run, "started")).read().strip())
        end = datetime.fromisoformat(open(os.path.join(run, "finished")).read().strip())
        out["wall_s"] = (end - start).total_seconds()
    except (OSError, ValueError):
        out["wall_s"] = None
    speed = os.path.join(run, "speed.jsonl")
    out["speed"] = [json.loads(l) for l in open(speed)] if os.path.exists(speed) else []
    return out


def summarise(r):
    ok = [q for q in r["requests"] if q.get("status") == 200]
    by_task = {}
    for q in ok:
        by_task.setdefault(q["iid"], []).append(q)
    tasks = {k: v for k, v in by_task.items() if k}
    tot = lambda f, qs: sum((q.get(f) or 0) for q in qs)
    per_task_out = [tot("completion_tokens", v) for v in tasks.values()]
    per_task_in = [tot("prompt_tokens", v) for v in tasks.values()]
    per_task_new = [tot("prompt_tokens", v) - tot("cached_tokens", v) for v in tasks.values()]
    per_task_wall = [max(q["t_end"] for q in v) - min(q["t0"] for q in v) for v in tasks.values()]
    resolved = r["resolved"]
    gufo = [q.get("gufo") or {} for q in ok]
    pre = [g["prefill_tokens"] / (g["prefill_ms"] / 1000) for g in gufo
           if g.get("prefill_tokens", 0) >= 512 and g.get("prefill_ms")]
    s = {
        "resolved": len(resolved), "tasks": len(tasks), "errors": r["errors"],
        "empty": r["empty"], "requests": len(ok),
        "failed_requests": len(r["requests"]) - len(ok),
        "wall_h": r["wall_s"] / 3600 if r["wall_s"] else None,
        "task_wall_min_mean": st.mean(per_task_wall) / 60 if per_task_wall else None,
        "task_wall_min_p90": pct(per_task_wall, 0.9) / 60 if per_task_wall else None,
        "steps_mean": st.mean(r["steps"].values()) if r["steps"] else None,
        "out_tokens_per_task": st.mean(per_task_out) if per_task_out else None,
        "in_tokens_per_task": st.mean(per_task_in) if per_task_in else None,
        "new_prompt_tokens_per_task": st.mean(per_task_new) if per_task_new else None,
        "out_tokens_per_resolved": sum(per_task_out) / max(1, len(resolved)),
        "cache_hit": tot("cached_tokens", ok) / max(1, tot("prompt_tokens", ok)),
        "ttft_p50": pct([q["ttft_s"] for q in ok], 0.5),
        "ttft_p90": pct([q["ttft_s"] for q in ok], 0.9),
        "decode_tps_mean": st.mean([q["decode_tps"] for q in ok if q.get("decode_tps")] or [0]),
        "output_speed": tot("completion_tokens", ok) / max(1e-9, sum(q["total_s"] for q in ok)),
        "prefill_tps_p50": pct(pre, 0.5),
        "aggregate_out_tps": (tot("completion_tokens", ok) / r["wall_s"]) if r["wall_s"] else None,
    }
    sp = r["speed"]
    for n in (1024, 4096, 16384, 32768):
        v = [x["server_prompt_tps"] or (x["prompt_tokens"] / x["ttft"]) for x in sp
             if x.get("kind") == "prefill" and x.get("target") == n]
        s[f"cold_prefill_{n}"] = st.median(v) if v else None
    d = [x["client_decode_tps"] for x in sp if x.get("kind") == "decode" and x.get("client_decode_tps")]
    s["decode_single"] = st.median(d) if d else None
    for x in sp:
        if x.get("kind") == "concurrency":
            s[f"agg_c{x['c']}"] = x.get("aggregate_tps_steady") or x.get("aggregate_tps_wall")
    return s


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 1 else f"{v:,.1f}"
    return f"{v:,}"


def main():
    runs = [load(p) for p in sys.argv[1:]]
    sums = [summarise(r) for r in runs]
    keys = list(sums[0].keys())
    print("| metric | " + " | ".join(r["name"] for r in runs) + " |")
    print("|---|" + "---:|" * len(runs))
    for k in keys:
        print(f"| {k} | " + " | ".join(fmt(s.get(k)) for s in sums) + " |")
    if len(runs) == 2:
        a, b = runs[0]["resolved"], runs[1]["resolved"]
        print(f"\nResolved by both: {len(a & b)}; only {runs[0]['name']}: "
              f"{sorted(a - b)}; only {runs[1]['name']}: {sorted(b - a)}")


if __name__ == "__main__":
    main()
