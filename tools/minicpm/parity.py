#!/usr/bin/env python3
"""Runtime parity on one GGUF: llama.cpp greedy text + top-20 logprobs is the reference; Gufo is
teacher-forced on the same text (forced_completion) and also free-runs greedily. Stdlib only.

  parity.py ref  --base-url http://127.0.0.1:8811 --out ref.json   (llama-server)
  parity.py cand --base-url http://127.0.0.1:8811 --ref ref.json --out cand.json   (gufo serve)
"""
import argparse, json, math, statistics, urllib.request
from pathlib import Path

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list. Include a docstring.",
    "Return only JSON: an object describing a fictional SaaS customer with fields name, plan, seats, mrr_usd, churn_risk (low/medium/high).",
    "A shop sells pens at $1.20 each and notebooks at $3.50 each. Maria buys 7 pens and 4 notebooks and pays with a $50 note. How much change does she get? Explain briefly.",
    "Summarise the main causes of the French Revolution in five bullet points.",
    "Explain the difference between a mutex and a semaphore to a junior engineer.",
    "Write a SQL query that returns the top 5 customers by total order value in 2025 from tables customers(id, name) and orders(id, customer_id, total, created_at).",
    "用三句话介绍长城的历史。",
    "Translate into French: 'Our quarterly revenue grew by twelve percent, driven mostly by enterprise renewals.'",
    "Write a TypeScript React component that renders a paginated table from a props.rows array.",
    "List the planets of the solar system with one interesting fact each.",
    "Classify the sentiment of this review as positive, negative or mixed, then justify in one sentence: 'Setup was painful but support fixed everything within an hour and it has run flawlessly since.'",
    "Write a bash script that finds the ten largest files under a directory given as the first argument.",
]


def post(base, body):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def positions(resp):
    out = []
    for e in resp["choices"][0]["logprobs"]["content"]:
        key = "id" if "id" in e else "token"
        out.append([e[key], e["logprob"], [[t["id" if "id" in t else "token"], t["logprob"]] for t in e["top_logprobs"]]])
    return out


def body(model, prompt, **extra):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 128,
            "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}, **extra}


def topk_kl(ref, cand):
    q = dict(map(tuple, cand))
    floor = min(q.values()) if q else -30.0
    kl = p_mass = q_mass = 0.0
    for token, lp in ref:
        lq = q.get(token, floor)
        p = math.exp(lp)
        kl += p * (lp - lq)
        p_mass += p
        q_mass += math.exp(lq)
    p_tail, q_tail = max(0.0, 1.0 - p_mass), max(1e-9, 1.0 - q_mass)
    if p_tail > 1e-12:
        kl += p_tail * (math.log(p_tail) - math.log(q_tail))
    return max(0.0, kl)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["ref", "cand"])
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--ref")
    p.add_argument("--corpus", help="adds two long prompts (~1.5K and ~3K tokens) that reach the wide prefill kernels")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.corpus:
        text = Path(a.corpus).read_text()
        for start, chars in ((120000, 5600), (260000, 11500)):
            PROMPTS.append(text[start:start + chars] + "\n\nSummarise the passage above in five sentences.")
    if a.mode == "ref":
        cases = []
        for prompt in PROMPTS:
            r = post(a.base_url, body(a.model, prompt))
            cases.append({"prompt": prompt, "text": r["choices"][0]["message"]["content"], "positions": positions(r),
                          "prompt_tokens": r["usage"]["prompt_tokens"]})
        Path(a.out).write_text(json.dumps({"cases": cases}))
        print(f"ref: {len(cases)} cases, prompt tokens {[c['prompt_tokens'] for c in cases]}")
        return
    ref = json.loads(Path(a.ref).read_text())["cases"]
    kls, top1, dlp, nll_r, nll_c, rows = [], [], [], [], [], []
    for c in ref:
        forced = post(a.base_url, body(a.model, c["prompt"], forced_completion=c["text"]))
        free = post(a.base_url, body(a.model, c["prompt"]))
        fp, cp = c["positions"], positions(forced)
        aligned = 0
        for (tr, lr, topr), (tc, lc, topc) in zip(fp, cp):
            if tr != tc:
                break
            aligned += 1
            kls.append(topk_kl(topr, topc))
            top1.append(max(topr, key=lambda t: t[1])[0] == max(topc, key=lambda t: t[1])[0])
            dlp.append(abs(lr - lc)); nll_r.append(-lr); nll_c.append(-lc)
        gt = free["choices"][0]["message"]["content"]
        same = next((i for i, (x, y) in enumerate(zip(gt, c["text"])) if x != y), min(len(gt), len(c["text"])))
        rows.append({"prompt": c["prompt"][:50], "ref_prompt_tokens": c["prompt_tokens"],
                     "cand_prompt_tokens": forced["usage"]["prompt_tokens"], "aligned_positions": aligned,
                     "ref_positions": len(fp), "greedy_identical_chars": same, "ref_chars": len(c["text"]),
                     "cand_text": gt[:300], "ref_text": c["text"][:300]})
    res = {"positions": len(kls), "kl_mean": statistics.fmean(kls) if kls else None,
           "kl_p99": sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else None,
           "top1_agree": statistics.fmean(top1) if top1 else None, "mean_dlogprob": statistics.fmean(dlp) if dlp else None,
           "ppl_ref": math.exp(statistics.fmean(nll_r)) if nll_r else None,
           "ppl_cand": math.exp(statistics.fmean(nll_c)) if nll_c else None,
           "greedy_identical_cases": sum(r["greedy_identical_chars"] == r["ref_chars"] for r in rows), "cases": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))
    for r in rows:
        print(f"  {r['prompt']!r}: prompt {r['ref_prompt_tokens']}/{r['cand_prompt_tokens']} aligned {r['aligned_positions']}/{r['ref_positions']} greedy-same {r['greedy_identical_chars']}/{r['ref_chars']}")


if __name__ == "__main__":
    main()
