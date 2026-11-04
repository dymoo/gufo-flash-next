#!/usr/bin/env python3
"""Speculative-decoding workloads against a running gufo server.

Prepared decode cohorts (as the QA `decode` suite: every session prefills,
then one synchronized decode), greedy, thinking off. Workloads cover the
content types speculation depends on: generic prose (the single-user
"mixed" benchmark task), passage copying ("repetitive"), code generation,
agent-style file edits and tool-output extraction (copy-heavy), and prose
after a 32K-token cached document. One JSON line per cohort is appended to
--out, with summed decode tok/s, draft counts and completion hashes, so
interleaved A/B runs can be compared and greedy outputs checked for
identity across servers.

  specbench.py --base-url http://127.0.0.1:8011 --label base-r1 \\
      --workloads prose,repetition,code,edit,tool,long --levels 1,2,4,8 \\
      --out results.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS))
from gufo import serving_bench  # noqa: E402
from gufo.model_bench.llm import TASKS, synthetic_text  # noqa: E402

CODE_TASK = ("\n\nWrite a complete Python module that parses the passage above into sentences, counts word "
             "frequencies, and prints a report, with docstrings and unit tests. Write at least 300 lines of code.")
EDIT_TASK = ("\n\nRewrite the complete file above with one change: add a short comment line directly above "
             "every function definition that says what the function does. Output only the full updated "
             "file in a single code block, keeping every other line exactly as it is.")
TOOL_TASK = ("\n\nThe text above is the output of a CI tool. Quote every line that contains ERROR or WARN "
             "exactly as it appears, one per line, then explain the most likely root cause.")
LONG_TASK = ("Summarize the reference document's plot so far in detail, then write a short continuation "
             "in the same style. Write at least 500 words.")


def ci_log(seed: int, lines: int) -> str:
    """Deterministic build/test log with a few warnings and errors."""
    rng = random.Random(seed)
    modules = ["engine", "scheduler", "kv_pool", "router", "tokenizer", "snapshot", "rollback", "sampler"]
    out = []
    t = 1729850000.0
    for i in range(lines):
        t += rng.random() * 0.8
        module = rng.choice(modules)
        r = rng.random()
        if r < 0.05:
            level, text = "ERROR", f"{module}: assertion failed: rows <= kDecodeRows ({rng.randint(9, 64)} > 8)"
        elif r < 0.12:
            level, text = "WARN", f"{module}: slow step {rng.randint(120, 900)} ms (budget 100 ms)"
        else:
            level = "INFO"
            text = (f"{module}: test_{rng.choice(['decode', 'prefill', 'batch', 'cancel'])}_"
                    f"{rng.randint(1, 40)} passed in {rng.random() * 3:.3f}s")
        stamp = time.strftime("%H:%M:%S", time.gmtime(t))
        out.append(f"[{stamp}.{int((t % 1) * 1000):03d}] {level:5s} {text}")
    return "\n".join(out)


def workload_messages(name: str, repo: Path, corpus: Path) -> list[dict[str, str]]:
    if name in ("prose", "repetition"):
        return [{"role": "user", "content": synthetic_text(100_000, 1500) + TASKS[name]}]
    if name == "code":
        return [{"role": "user", "content": synthetic_text(4242, 790) + CODE_TASK}]
    if name == "edit":
        source = (repo / "tools/qwen-flash/qa/agent_sim.py").read_text(encoding="utf-8")
        return [{"role": "user", "content": "```python\n" + source + "```" + EDIT_TASK}]
    if name == "tool":
        return [{"role": "user", "content": ci_log(7, 260) + TOOL_TASK}]
    if name == "long":
        text = (corpus / "moby.txt").read_text(encoding="utf-8")
        start = text.find("CHAPTER 1")
        doc = text[start:start + 32768 * 4]
        return [{"role": "system", "content": "Reference document:\n\n" + doc},
                {"role": "user", "content": LONG_TASK}]
    raise ValueError(f"unknown workload {name}")


def cohort(base_url: str, model: str, messages, level: int, max_tokens: int, case: str,
           temperature: float = 0.0, seed: int | None = None, top_p: float | None = None) -> dict:
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    if top_p is not None:
        body["top_p"] = top_p

    def request(index: int, tokens: int, gate=None):
        extra = dict(body)
        if seed is not None:
            extra["seed"] = seed + index
        return serving_bench.run_request(
            base_url=base_url, model=model, prompt="", max_tokens=tokens, temperature=temperature,
            timeout_seconds=900, client_id=f"spec-{case}-{index}", concurrency=level, repetition=0,
            request_index=index, start_gate=gate, messages=messages, cache_prompt=True,
            extra_body=extra)

    request(0, 1)  # the first session prefills; the others can restore its snapshot
    if level > 1:
        with concurrent.futures.ThreadPoolExecutor(level) as pool:
            list(pool.map(lambda i: request(i, 1), range(1, level)))
    gate = threading.Barrier(level) if level > 1 else None
    with concurrent.futures.ThreadPoolExecutor(level) as pool:
        samples = list(pool.map(lambda i: request(i, max_tokens, gate), range(level)))
    span = max(s.finished_at for s in samples) - min(s.started_at for s in samples)
    return {
        "case": case, "concurrency": level,
        "decode_tps_sum": round(sum(s.decode_tokens_per_second or 0 for s in samples), 3),
        "decode_tps": [s.decode_tokens_per_second for s in samples],
        "wall_tps": round(sum(s.completion_tokens for s in samples) / span, 3) if span > 0 else None,
        "completion_tokens": [s.completion_tokens for s in samples],
        "drafted": [s.draft_tokens for s in samples],
        "accepted": [s.draft_accepted_tokens for s in samples],
        "prompt_tokens": samples[0].prompt_tokens,
        "hashes": [s.completion_sha256[:16] for s in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--label", required=True)
    parser.add_argument("--workloads", default="prose,repetition,code,edit,tool,long")
    parser.add_argument("--levels", default="1")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repo", default=str(TOOLS.parent))
    parser.add_argument("--corpus", required=True,
                        help="directory of QA corpus prompts")
    parser.add_argument("--force-file", default="", help="GUFO_MTP_FORCE_FILE of the server")
    parser.add_argument("--force", default="", help="comma list of forced widths to sweep (needs --force-file)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="request seed (+ session index)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    forces = args.force.split(",") if args.force else [""]
    with open(args.out, "a", encoding="utf-8") as out:
        for force in forces:
            if args.force_file:
                Path(args.force_file).write_text(force if force else "-", encoding="utf-8")
                time.sleep(0.5)
            for name in args.workloads.split(","):
                messages = workload_messages(name, Path(args.repo), Path(args.corpus))
                for level in levels:
                    t0 = time.time()
                    row = cohort(args.base_url, args.model, messages, level, args.max_tokens,
                                 f"{name}-c{level}", args.temperature, args.seed, args.top_p)
                    row.update({"label": args.label, "workload": name, "force": force,
                                "max_tokens": args.max_tokens, "t": round(t0, 1)})
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"{args.label} {name} C{level} force={force or '-'}: "
                          f"{row['decode_tps_sum']} tok/s acc={sum(row['accepted'])}/{sum(row['drafted'])} "
                          f"{row['hashes'][0]}", flush=True)
        if args.force_file:
            Path(args.force_file).write_text("-", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
