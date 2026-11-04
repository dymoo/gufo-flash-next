#!/usr/bin/env python3
"""Scan result files for empty replies: a request that finished with stop, no tool call and no visible content.

A request row is [g, turn, sent, ttft, done, prompt, cached, completion, reasoning, chunks, finish, ...]. finish=stop
means no tool call, and completion - reasoning <= 2 means (almost) no content tokens (completion includes the thinking
tokens). Usage: empty_scan.py DIR_WITH_RESULT_JSON_FILES. Merged files and their parts both count, so pass a directory
of disjoint files for per-setting rates."""

import collections, glob, json, os, sys

rows = []
for path in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(path))
    cfg = r["config"]
    setting = f"{cfg.get('thinking')}/{cfg.get('effort') or '-'}"
    tasks = {t["g"]: t for t in r["tasks"]}
    last_turn = collections.defaultdict(int)
    by_task = collections.defaultdict(list)
    for q in r["requests"]:
        by_task[q[0]].append(q)
        last_turn[q[0]] = max(last_turn[q[0]], q[1])
    n_req = n_stop = n_empty = 0
    empties = []
    for q in r["requests"]:
        g, turn, finish, completion, reasoning = q[0], q[1], q[10], q[7], q[8]
        n_req += 1
        if finish != "stop":
            continue
        n_stop += 1
        if completion - reasoning <= 2:
            n_empty += 1
            t = tasks.get(g, {})
            empties.append({"g": g, "id": t.get("id"), "turn": turn, "completion": completion, "reasoning": reasoning,
                            "final_turn": turn == last_turn[g], "outcome": t.get("outcome"), "family": t.get("family"),
                            "sent": q[2]})
    rows.append((os.path.basename(path), setting, cfg.get("model"), r.get("suite"), r.get("started_utc"), n_req, n_stop, n_empty, empties,
                 sum(1 for t in r["tasks"] if t["outcome"] != "unfinished")))
print(f"{'file':44s} {'setting':10s} {'req':>5s} {'stop':>5s} {'empty':>5s} {'rate/req':>8s} {'rate/stop':>9s}  ended-task-empty / failed")
tot = collections.defaultdict(lambda: [0, 0, 0])
for name, setting, model, suite, started, n_req, n_stop, n_empty, empties, graded in rows:
    ended = [e for e in empties if e["final_turn"]]
    failed = [e for e in ended if e["outcome"] not in ("success", "unfinished")]
    key = f"{model}:{setting}"
    tot[key][0] += n_req; tot[key][1] += n_stop; tot[key][2] += n_empty
    print(f"{name:44s} {setting:10s} {n_req:5d} {n_stop:5d} {n_empty:5d} {100*n_empty/max(n_req,1):7.2f}% {100*n_empty/max(n_stop,1):8.2f}%  {len(ended)} / {len(failed)}")
print()
for key, (a, b, c) in sorted(tot.items()):
    print(f"{key:32s} requests {a:6d}  stop-turns {b:6d}  empty {c:4d}  ({100*c/max(a,1):.2f}% of requests, {100*c/max(b,1):.2f}% of stop turns)")
print()
for name, setting, model, suite, started, n_req, n_stop, n_empty, empties, graded in rows:
    for e in empties:
        print(f"{name:40s} {setting:9s} g{e['g']:<3d} {str(e['id']):14s} turn {e['turn']:>2d} completion {e['completion']:>5d} thinking {e['reasoning']:>5d} "
              f"final={e['final_turn']!s:5s} outcome={e['outcome']}")
