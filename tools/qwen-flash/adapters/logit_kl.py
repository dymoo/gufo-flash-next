#!/usr/bin/env python3
"""Full-vocabulary KL between GPU and float32-oracle logit dumps.

  logit_kl.py --gpu gpu.bin --runs base,adapter --vocab 248320 [--prompt-tokens L] ORACLE=ref.bin...

gpu.bin is gpu_probe --dump (each run's last ROWS positions back to back);
ref.bin is reference_probe --dump (every prompt position, written as it goes, so a
file still being written holds only the first P positions). Positions are aligned
by prompt position: GPU row i is position L-ROWS+i (L = --prompt-tokens, default
ROWS), oracle row j is position j; the comparison covers the shared positions.
Prints, for every oracle and GPU run, KL(oracle || gpu) mean/max and the argmax
agreement, and the same for the last shared position.
"""
import argparse
import json

import numpy as np


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--runs", default="base,adapter")
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--prompt-tokens", type=int, default=0, help="prompt length L (default: GPU rows)")
    ap.add_argument("--json")
    ap.add_argument("oracles", nargs="+", help="LABEL=PATH")
    a = ap.parse_args()
    runs = a.runs.split(",")
    gpu = np.fromfile(a.gpu, dtype="<f4").reshape(len(runs), -1, a.vocab)
    rows = gpu.shape[1]
    length = a.prompt_tokens or rows
    first = length - rows  # prompt position of GPU row 0
    out = {}
    for spec in a.oracles:
        label, path = spec.split("=", 1)
        ref = np.fromfile(path, dtype="<f4")
        ref = ref[: ref.size // a.vocab * a.vocab].reshape(-1, a.vocab)
        lo, hi = first, min(length, ref.shape[0])  # shared prompt positions [lo, hi)
        n = hi - lo
        lr = log_softmax(ref[lo:hi])
        for run, g in zip(runs, gpu):
            g = g[: n]
            lg = log_softmax(g)
            kl = (np.exp(lr) * (lr - lg)).sum(axis=-1)
            agree = int((lr.argmax(-1) == lg.argmax(-1)).sum())
            key = f"{label} vs gpu-{run}"
            out[key] = {"positions": n, "of_prompt": length, "kl_mean": float(kl.mean()),
                        "kl_max": float(kl.max()), "kl_last": float(kl[-1]),
                        "argmax_agree": agree, "argmax_last": bool(lr[-1].argmax() == lg[-1].argmax())}
            print(f"{key:34s} positions {n}/{length} KL mean {kl.mean():.5f} max {kl.max():.5f} "
                  f"last {kl[-1]:.5f} argmax {agree}/{n} (last {'same' if lr[-1].argmax() == lg[-1].argmax() else 'DIFF'})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
