#!/usr/bin/env python3
"""Teacher-forced coding-text ladder: ladder.py's protocol on held-out agent
transcripts, source files, prose and chat (build_corpus.py eval.jsonl) instead of
the two pinned books, because the pruned experts are chosen from coding traffic.

  code_ladder.py run --base-url URL --label NAME --corpus eval.jsonl --out NAME.json
  (report with tools/qwen-flash/ladder.py report BASE.json CAND.json ...)

Each document becomes one case: the flattened text up to ~--prompt-tokens is the
user prompt (qa.LOGIT_PROMPT + text), the next ~--forced-tokens of the same text
are forced. Position 0 is the bulk-prefill distribution, positions 1.. are decode
steps (ladder.py "prefill" / "decode after prefill"); every case has kind "deep"
so ladder.py report pools them as decode, and kind_detail carries the document kind.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

QA_DIR = Path(__file__).resolve().parent / "qa"
sys.path.insert(0, str(QA_DIR))
import qa  # noqa: E402

CHARS_PER_TOKEN = 3.3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", default="http://127.0.0.1:8011")
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompt-tokens", type=int, default=3000)
    r.add_argument("--forced-tokens", type=int, default=128)
    r.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()
    client = qa.Client(args.base_url)
    if not client.healthy(wait=120):
        print("server not healthy", file=sys.stderr)
        return 2
    client.model = client.request("GET", "/v1/models")["data"][0]["id"]
    run = qa.Run(argparse.Namespace(quick=False, sessions=1, data_dir="."), client)
    docs = [json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    if args.max_docs:
        docs = docs[:args.max_docs]
    started = time.time()
    cases = []
    for i, d in enumerate(docs):
        text = d["text"]
        forced_chars = int(args.forced_tokens * CHARS_PER_TOKEN)
        prompt_chars = min(int(args.prompt_tokens * CHARS_PER_TOKEN), len(text) - forced_chars - 200)
        if prompt_chars < 2000:
            continue
        doc, end = qa.cut(text, 0, prompt_chars)
        cont, _ = qa.cut(text, end, forced_chars)
        qa.log(f"case {i + 1}/{len(docs)} {d['id']} ({d['kind']})")
        try:
            case = run.forced(d["id"], [{"role": "user", "content": qa.LOGIT_PROMPT + doc}], cont)
        except qa.HttpError as e:
            qa.log(f"case {d['id']} failed: {e}")
            continue
        case.update({"kind": "deep", "kind_detail": d["kind"], "depth": prompt_chars})
        cases.append(case)
    out = {"label": args.label, "note": args.note, "seconds": round(time.time() - started, 1), "cases": cases}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(cases)} cases in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
