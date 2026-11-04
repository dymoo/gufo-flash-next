#!/usr/bin/env python3
"""Vision parity: llama.cpp (mtmd reference) greedy answers with top-20 logprobs for image prompts;
Gufo is teacher-forced on the same answers and also answers greedily. Stdlib only.

  vparity.py ref  --base-url URL --images a.png b.jpg ... --out ref.json
  vparity.py cand --base-url URL --ref ref.json --out cand.json
"""
import argparse, base64, json, math, statistics, urllib.request
from pathlib import Path

QUESTIONS = ["Describe this image in detail.", "What text appears in this image? Transcribe it exactly."]


def post(base, body):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def body(model, image_b64, mime, question, **extra):
    return {"model": model, "temperature": 0, "max_tokens": 96, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                                                      {"type": "text", "text": question}]}], **extra}


def positions(resp):
    return [[e.get("id", e.get("token")), e["logprob"], [[t.get("id", t.get("token")), t["logprob"]] for t in e["top_logprobs"]]]
            for e in resp["choices"][0]["logprobs"]["content"]]


def topk_kl(ref, cand):
    q = dict(map(tuple, cand)); floor = min(q.values()) if q else -30.0
    kl = pm = qm = 0.0
    for t, lp in ref:
        lq = q.get(t, floor); p = math.exp(lp); kl += p * (lp - lq); pm += p; qm += math.exp(lq)
    pt, qt = max(0.0, 1 - pm), max(1e-9, 1 - qm)
    return max(0.0, kl + (pt * (math.log(pt) - math.log(qt)) if pt > 1e-12 else 0))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode", choices=["ref", "cand"]); a.add_argument("--base-url", required=True); a.add_argument("--model", default="")
    a.add_argument("--images", nargs="*", default=[]); a.add_argument("--ref"); a.add_argument("--out", required=True)
    a = a.parse_args()
    if a.mode == "ref":
        cases = []
        for path in a.images:
            raw = Path(path).read_bytes(); mime = "image/png" if path.endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(raw).decode()
            for q in QUESTIONS:
                r = post(a.base_url, body(a.model, b64, mime, q))
                cases.append({"image": Path(path).name, "b64": b64, "mime": mime, "question": q, "text": r["choices"][0]["message"]["content"],
                              "positions": positions(r), "prompt_tokens": r["usage"]["prompt_tokens"]})
                print(f"ref {Path(path).name} | {q[:30]} | prompt {r['usage']['prompt_tokens']} | {cases[-1]['text'][:100]!r}")
        Path(a.out).write_text(json.dumps({"cases": cases}))
        return
    ref = json.loads(Path(a.ref).read_text())["cases"]
    kls, top1, rows = [], [], []
    for c in ref:
        forced = post(a.base_url, body(a.model, c["b64"], c["mime"], c["question"], forced_completion=c["text"]))
        free = post(a.base_url, body(a.model, c["b64"], c["mime"], c["question"]))
        aligned = 0
        for (tr, lr, topr), (tc, lc, topc) in zip(c["positions"], positions(forced)):
            if tr != tc:
                break
            aligned += 1; kls.append(topk_kl(topr, topc)); top1.append(max(topr, key=lambda t: t[1])[0] == max(topc, key=lambda t: t[1])[0])
        gt = free["choices"][0]["message"]["content"]
        same = next((i for i, (x, y) in enumerate(zip(gt, c["text"])) if x != y), min(len(gt), len(c["text"])))
        rows.append({"image": c["image"], "question": c["question"][:30], "ref_prompt_tokens": c["prompt_tokens"],
                     "cand_prompt_tokens": forced["usage"]["prompt_tokens"], "aligned": aligned, "ref_positions": len(c["positions"]),
                     "greedy_same_chars": same, "ref_chars": len(c["text"]), "ref_text": c["text"][:200], "cand_text": gt[:200]})
    res = {"positions": len(kls), "kl_mean": statistics.fmean(kls) if kls else None, "kl_p99": sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else None,
           "top1_agree": statistics.fmean(top1) if top1 else None, "cases": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}))
    for r in rows:
        print(f"  {r['image']} {r['question']!r}: prompt {r['ref_prompt_tokens']}/{r['cand_prompt_tokens']} aligned {r['aligned']}/{r['ref_positions']} greedy-same {r['greedy_same_chars']}/{r['ref_chars']}\n    ref:  {r['ref_text'][:120]!r}\n    gufo: {r['cand_text'][:120]!r}")


if __name__ == "__main__":
    main()
