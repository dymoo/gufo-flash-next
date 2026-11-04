#!/usr/bin/env python3
"""Builds the `-abliterated` steering adapter from public weights.

OrcaRouter's Qwen3.8-Flash-Next-Uncensored is the base model with one rank-1
edit W' = W - alpha r (r^T W) on every matrix that writes the residual streams.
This recovers r as the top left singular vector of W - W' on one writer,
checks it on other writers (same direction, strength alpha, rank 1) and writes
a `gufo steering adapter v2`: a JSON file and the little-endian float32
direction it names. Gufo then serves the edit from the unmodified weights:

  HF_TOKEN=hf_... orca_adapter.py --out DIR [--cache DIR]
  gufo serve llm ... --adapter workhorse-abliterated=DIR/orca-abliteration.json

The Orca repository is gated (granted on request on its model page); HF_TOKEN
reads it. Only the compared tensors are fetched, by cached HTTP range
requests (about 0.3 GB). The embedding, PLE and MTP writers were checked
against the same direction in artifacts/orca-direction-check.json; this tool
re-checks block writers only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abliteration_check as check  # noqa: E402

ORCA = ("https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored/resolve/"
        "da29941aee972c7eeab497e05a292c569da1f1d6/")
L = "model.language_model."
# Recovered from the first; the others must agree.
WRITERS = (
    L + "layers.3.self_attn.o_proj.weight",
    L + "layers.47.self_attn.o_proj.weight",
    L + "layers.0.linear_attn.out_proj.weight",
    L + "layers.30.mlp.shared_expert.down_proj.weight",
)
MIN_COS = 0.99          # |cos| of each writer's edit direction with r
MAX_RANK1 = 0.05        # share of each edit outside span(r)
FULL_ABLATION = 0.01    # |alpha - 1| below which the edit is written as alpha 1


def recover_direction(w, wo):
    """Unit r spanning the edit W - W' of one writer ([out = hidden, in]).

    The sign is fixed so the largest component is positive; the projection
    r r^T does not depend on it."""
    u, _, _ = np.linalg.svd((w - wo).astype(np.float64), full_matrices=False)
    r = u[:, 0]
    if r[np.argmax(np.abs(r))] < 0:
        r = -r
    return (r / np.linalg.norm(r)).astype(np.float32)


def measure(w, wo, r):
    """Strength, rank-1 residual and direction agreement of one writer's edit."""
    res = check.column_space(w, wo, r)
    return {k: res[k] for k in ("strength", "rank1_residual", "cos_top_singular")}


def write_adapter(out, r, alpha, name="orca-abliteration"):
    """`gufo steering adapter v2` JSON plus its float32 direction in `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    direction = out / f"{name}.f32"
    direction.write_bytes(np.asarray(r, dtype="<f4").tobytes())
    meta = {
        "format": "gufo steering adapter v2",
        "hidden_size": int(r.shape[0]),
        "alpha": float(alpha),
        "direction_file": direction.name,
        "writers": ["blocks", "embedding", "ple", "mtp"],
    }
    path = out / f"{name}.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for the adapter files")
    ap.add_argument("--cache", default=check.CACHE)
    a = ap.parse_args()
    check.CACHE = a.cache
    if not os.environ.get("HF_TOKEN"):
        sys.exit("set HF_TOKEN to a token that can read " + ORCA.split("/resolve/")[0])

    base = check.Safetensors(check.QWEN)
    orca = check.Safetensors(ORCA)

    def pair(name):
        for st in (base, orca):
            if name not in st.index:
                sys.exit(f"{name} is missing from {st.base}")
        n = base.shape(name)[0]
        return base.rows(name, 0, n), orca.rows(name, 0, n)

    w, wo = pair(WRITERS[0])
    r = recover_direction(w, wo)
    report = {WRITERS[0]: measure(w, wo, r)}
    for name in WRITERS[1:]:
        report[name] = measure(*pair(name), r)
    for name, m in report.items():
        print(f"{name.removeprefix(L):48s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    bad = [n for n, m in report.items()
           if m["cos_top_singular"] < MIN_COS or m["rank1_residual"] > MAX_RANK1]
    if bad:
        sys.exit("not one rank-1 edit along r: " + ", ".join(b.removeprefix(L) for b in bad))
    alpha = float(np.median([m["strength"] for m in report.values()]))
    if abs(alpha - 1.0) < FULL_ABLATION:
        alpha = 1.0
    path = write_adapter(a.out, r, alpha)
    print(f"wrote {path} (alpha {alpha:.4f})")


if __name__ == "__main__":
    main()
