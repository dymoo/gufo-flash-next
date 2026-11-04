#!/usr/bin/env python3
"""Quality ladder for the int4 prefill spike: prefill-logit and
post-prefill decode distributions of a candidate server against a baseline.

  ladder.py run --base-url http://127.0.0.1:8011 --label base --out base.json [--data-dir DIR]
  ladder.py report base.json rung1.json rung2.json ... [--md out.md]

Method (teacher-forced, the same requests for every server):
  Every case is a chat request whose prompt is a slice of a pinned public-domain
  text and whose forced continuation is the text that follows. The server returns
  the top-20 logprobs for every forced position (qa.py's logits suite protocol,
  imported here). Position 0 of a case is the distribution produced by the BULK
  PREFILL of the whole prompt (its last row); positions 1.. are real single-token
  DECODE steps that read the KV cache and linear-attention state the prefill
  wrote. KL, top-1 agreement and perplexity are therefore reported separately
  for position 0 ("prefill") and positions 1.. ("decode after prefill").

  Cases:
    deep   moby at 2048 / 16384 / 65536 tokens, 256 forced tokens each
    scan   N shorter documents (1.5K-3K tokens) from different offsets of both
           books with 16 forced tokens: many prefill-position samples cheaply

The server must be started with caching off (cache_prompt=false is sent) and
no speculative decoding. KL is qa.py's top-20 KL(ref||cand) with a tail bucket,
so it slightly underestimates the full-vocabulary KL.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

QA_DIR = Path(__file__).resolve().parent / "qa"
sys.path.insert(0, str(QA_DIR))
import qa  # noqa: E402


def run(args) -> int:
    client = qa.Client(args.base_url, Path(args.api_key_file).read_text().strip() if args.api_key_file else "")
    if not client.healthy(wait=120):
        print("server not healthy", file=sys.stderr)
        return 2
    models = client.request("GET", "/v1/models")
    client.model = models["data"][0]["id"]
    ns = argparse.Namespace(quick=False, sessions=1, data_dir=args.data_dir)
    r = qa.Run(ns, client)
    r.calibrate()
    if not r.logprobs:
        print("server lacks logprobs/forced_completion", file=sys.stderr)
        return 3
    moby, pride = r.book("moby"), r.book("pride")
    cases = []
    started = time.time()

    def add(case_id: str, kind: str, depth: int, text: str, start: int, forced_tokens: int, book: str):
        doc, end = qa.cut(text, start, r.chars(depth, book))
        cont, _ = qa.cut(text, end, r.chars(forced_tokens, book))
        case = r.forced(case_id, [{"role": "user", "content": qa.LOGIT_PROMPT + doc}], cont)
        case.update({"kind": kind, "depth": depth})
        cases.append(case)

    for depth in args.deep_depths:
        qa.log(f"deep {depth}")
        add(f"deep-{depth}", "deep", depth, moby, 3000, args.deep_forced, "moby")
    depths = [1536, 2048, 3072]
    for i in range(args.scan_docs):
        book, text = ("moby", moby) if i % 2 == 0 else ("pride", pride)
        start = 20000 + (i // 2) * 9000
        depth = depths[i % len(depths)]
        qa.log(f"scan {i + 1}/{args.scan_docs} {book}@{start} depth {depth}")
        add(f"scan-{i:02d}", "scan", depth, text, start, 16, book)
    out = {"label": args.label, "note": args.note, "seconds": round(time.time() - started, 1),
           "cases": cases}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(cases)} cases in {out['seconds']} s")
    return 0


def summarize(ref_cases: list, cand_cases: list, pick) -> dict:
    """Pools the chosen positions of aligned cases through qa.compare_positions."""
    ref = [pick(c["positions"]) for c in ref_cases]
    cand = [pick(c["positions"]) for c in cand_cases]
    return qa.compare_positions(ref, cand)


def fmt(m: dict) -> str:
    if not m.get("positions"):
        return "n/a"
    return (f"KL {m['kl_mean']:.4f} (p99 {m['kl_p99']:.3f})  top-1 {100 * m['top1_agree']:.1f}%  "
            f"n={m['positions']}  ppl {m['ppl_delta_pct']:+.2f}%")


def report(args) -> int:
    base = json.loads(Path(args.files[0]).read_text())
    index = {c["id"]: c for c in base["cases"]}
    lines = [f"Baseline: {base['label']} ({len(base['cases'])} cases). Gates (qa.py): KL <= 0.005 pass / 0.02 warn; "
             "top-1 >= 98% pass / 95% warn; ppl delta <= 0.5% / 2%.", ""]
    rows = []
    for path in args.files[1:]:
        cand = json.loads(Path(path).read_text())
        cmap = {c["id"]: c for c in cand["cases"]}
        ids = [i for i in index if i in cmap]
        ref_cases = [index[i] for i in ids]
        cand_cases = [cmap[i] for i in ids]
        pre_all = summarize(ref_cases, cand_cases, lambda p: p[:1])
        dec_all = summarize(ref_cases, cand_cases, lambda p: p[1:])
        deep_ref = [c for c in ref_cases if c["kind"] == "deep"]
        deep_cand = [cmap[c["id"]] for c in deep_ref]
        pre_deep = summarize(deep_ref, deep_cand, lambda p: p[:1])
        dec_deep = summarize(deep_ref, deep_cand, lambda p: p[1:])
        pre_scan = summarize([c for c in ref_cases if c["kind"] == "scan"],
                             [cmap[c["id"]] for c in ref_cases if c["kind"] == "scan"], lambda p: p[:1])
        rows.append((cand["label"], pre_all, pre_scan, dec_all, dec_deep, pre_deep))
        lines.append(f"### {cand['label']}  ({cand.get('note', '')})")
        lines.append(f"- prefill logits, all {len(ids)} cases: {fmt(pre_all)}")
        lines.append(f"- prefill logits, scan only: {fmt(pre_scan)}")
        lines.append(f"- prefill logits, deep only: {fmt(pre_deep)}")
        lines.append(f"- decode after prefill, all: {fmt(dec_all)}")
        lines.append(f"- decode after prefill, deep (256 tokens): {fmt(dec_deep)}")
        for c in deep_ref:
            d = summarize([c], [cmap[c["id"]]], lambda p: p[1:])
            p0 = summarize([c], [cmap[c["id"]]], lambda p: p[:1])
            lines.append(f"  - {c['id']}: prefill KL {p0.get('kl_mean', float('nan')):.4f}, "
                         f"decode KL {d.get('kl_mean', float('nan')):.4f} top-1 {100 * d.get('top1_agree', 0):.1f}%")
        lines.append("")
    lines.append("| rung | prefill KL | prefill top-1 | decode KL | decode p99 | decode top-1 | ppl delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, pre_all, _pre_scan, dec_all, _dd, _pd in rows:
        lines.append(f"| {label} | {pre_all.get('kl_mean', float('nan')):.4f} | "
                     f"{100 * pre_all.get('top1_agree', 0):.1f}% | {dec_all.get('kl_mean', float('nan')):.4f} | "
                     f"{dec_all.get('kl_p99', float('nan')):.3f} | {100 * dec_all.get('top1_agree', 0):.1f}% | "
                     f"{dec_all.get('ppl_delta_pct', float('nan')):+.2f}% |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", default="http://127.0.0.1:8011")
    r.add_argument("--api-key-file")
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--data-dir", default="artifacts/qa/corpus")
    r.add_argument("--deep-depths", type=lambda s: [int(x) for x in s.split(",")], default=[2048, 16384, 65536])
    r.add_argument("--deep-forced", type=int, default=256)
    r.add_argument("--scan-docs", type=int, default=24)
    r.set_defaults(func=run)
    rep = sub.add_parser("report")
    rep.add_argument("files", nargs="+", help="baseline first, then candidates")
    rep.add_argument("--md")
    rep.set_defaults(func=report)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
