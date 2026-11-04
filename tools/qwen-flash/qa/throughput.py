#!/usr/bin/env python3
"""Per-second prefill/decode throughput from gufo serve request logs.

Spreads each completed request's newly prefilled tokens over [start+queue, start+ttft]
and its generated tokens over [start+ttft, end] (1 s log resolution), then reports
run-average, busy-average and 10 s / 60 s peak aggregate rates. Stdlib only.

Usage: throughput.py SERVER.log [...]
"""
import re
import sys
from datetime import datetime
def load(path):
    reqs = []
    for line in open(path, errors="ignore"):
        if "event=completed method=POST" not in line or "status=200" not in line: continue
        f = dict(re.findall(r"(\w+)=([^\s]+)", line))
        end = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp()
        dur = float(f["duration_ms"]) / 1000; q = float(f.get("queue_ms", 0)) / 1000; ttft = float(f.get("ttft_ms", 0)) / 1000
        start = end - dur
        pt = int(f.get("prefill_tokens", 0)); gt = int(f.get("generated_tokens", 0))
        reqs.append((start + q, start + ttft, pt, start + ttft, end, gt))
    return reqs
def series(reqs, which):
    b = {}
    for r in reqs:
        a, e, n = (r[0], r[1], r[2]) if which == "p" else (r[3], r[4], r[5])
        if n <= 0: continue
        e = max(e, a + 1e-3); rate = n / (e - a); t = a
        while t < e:
            s = int(t); nt = min(e, s + 1); b[s] = b.get(s, 0) + rate * (nt - t); t = nt
    return b
def roll(b, w):
    if not b: return 0
    lo, hi = min(b), max(b); best = 0; acc = 0; win = []
    vals = [b.get(s, 0) for s in range(lo, hi + 1)]
    vals += [0] * (w - len(vals))  # runs shorter than the window
    for i, v in enumerate(vals):
        acc += v
        if i >= w: acc -= vals[i - w]
        if i >= w - 1: best = max(best, acc / w)
    return best
if len(sys.argv) < 2:
    sys.exit(__doc__)
for path in sys.argv[1:]:
    reqs = load(path)
    if not reqs: print(path, "no data"); continue
    t0 = min(r[0] for r in reqs); t1 = max(r[4] for r in reqs); span = t1 - t0
    out = [path.split("/")[-1].replace(".server.log", ""), f"span={span:.0f}s reqs={len(reqs)}"]
    for k, name in (("d", "decode"), ("p", "prefill")):
        b = series(reqs, k); tot = sum(b.values()); busy = [v for v in b.values() if v > 0]
        out.append(f"{name}: avg={tot/span:.0f} busy_avg={tot/max(1,len(busy)):.0f} peak10s={roll(b,10):.0f} peak60s={roll(b,60):.0f} busy_frac={len(busy)/span:.0%}")
    print(" | ".join(out))
