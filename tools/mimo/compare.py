#!/usr/bin/env python3
"""Compare two OpenAI chat endpoints serving the same model (Gufo MiMo vs the
llama.cpp reference): greedy decoding with top-20 logprobs, thinking off.

While both histories agree, each position contributes KL(ref || test) over
the reference's top-20 (renormalised, test floored at its own 20th logprob)
and top-1 agreement. Reports the shared-prefix length per prompt; a first
divergence at a near-tie is expected, one at a confident token is a bug.
Stdlib only.
"""
import argparse
import json
import math
import random
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain why the sky is blue in three sentences.",
    "What is 17 * 23? Answer with the number only.",
    "Translate 'The cat sleeps on the warm mat' into French.",
    "List five prime numbers greater than 100.",
    "Write a haiku about a lighthouse.",
    "def quicksort(xs):\n    \"\"\"Sort a list.\"\"\"\n",
    "Summarise the plot of Moby Dick in two sentences.",
]


def chat(url, key, model, text, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True,
            "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["choices"][0]["logprobs"]["content"]


def kl(ref, test):
    ref_top = {t["token"]: t["logprob"] for t in ref["top_logprobs"]}
    test_top = {t["token"]: t["logprob"] for t in test["top_logprobs"]}
    floor = min(test_top.values())
    z_ref = math.log(sum(math.exp(v) for v in ref_top.values()))
    z_test = math.log(sum(math.exp(test_top.get(k, floor)) for k in ref_top))
    return sum(math.exp(v - z_ref) * ((v - z_ref) - (test_top.get(k, floor) - z_test))
               for k, v in ref_top.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference URL or recorded file")
    ap.add_argument("--test")
    ap.add_argument("--record", help="save the reference outputs here and stop")
    ap.add_argument("--ref-key-file")
    ap.add_argument("--test-key-file")
    ap.add_argument("--model", default="mimo")
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--long", default="2048,8192,32768")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()
    key = lambda p: open(p).read().strip() if p else "none"
    prompts = list(PROMPTS)
    words = " ".join(open(p, errors="ignore").read() for p in a.corpus).split()
    rng = random.Random(3)
    for n in (map(int, a.long.split(",")) if words else []):
        start = rng.randrange(0, len(words) - n)
        prompts.append(" ".join(words[start:start + int(n * 0.75)]) +
                       "\n\nContinue the passage for a few sentences.")
    # The two runtimes cannot share the GPU: --record saves the reference
    # outputs, a later run compares a test server against that file.
    if a.record:
        refs = [chat(a.ref, key(a.ref_key_file), a.model, t, a.max_tokens)
                for t in prompts]
        json.dump({"prompts": prompts, "outputs": refs}, open(a.record, "w"))
        print(f"recorded {len(refs)} reference outputs to {a.record}")
        return
    saved = json.load(open(a.ref)) if not a.ref.startswith("http") else None
    if saved:
        prompts = saved["prompts"]
    rows = []
    for i, text in enumerate(prompts):
        ref = (saved["outputs"][i] if saved else
               chat(a.ref, key(a.ref_key_file), a.model, text, a.max_tokens))
        test = chat(a.test, key(a.test_key_file), a.model, text, a.max_tokens)
        kls, agree, shared = [], 0, 0
        for r, t in zip(ref, test):
            kls.append(kl(r, t))
            agree += r["top_logprobs"][0]["token"] == t["top_logprobs"][0]["token"]
            shared += 1
            if r["token"] != t["token"]:
                break
        row = {"prompt_chars": len(text), "shared": shared, "ref_len": len(ref),
               "mean_kl": sum(kls) / max(1, len(kls)), "max_kl": max(kls or [0]),
               "top1": agree / max(1, shared)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"prompts": len(rows),
               "mean_kl": sum(r["mean_kl"] for r in rows) / len(rows),
               "worst_kl": max(r["max_kl"] for r in rows),
               "top1": sum(r["top1"] * r["shared"] for r in rows) / sum(r["shared"] for r in rows),
               "mean_shared": sum(r["shared"] for r in rows) / len(rows)}
    print(json.dumps(summary))
    if a.out:
        json.dump({"rows": rows, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
