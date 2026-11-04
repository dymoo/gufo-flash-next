#!/usr/bin/env python3
"""HumanEval (164, pass@1 greedy) and GSM8K (test split) over an OpenAI-compatible server. Stdlib only.
Generated code runs in a subprocess with CPU/memory/time limits. Datasets are cached with their sha256."""
import argparse, concurrent.futures, gzip, hashlib, json, re, resource, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

SOURCES = {
    "humaneval": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}


def load(name, data_dir):
    if name == "ds4":  # gufo's pinned 75-question GPQA Diamond / SuperGPQA / AIME 2025 fixture
        raw = (Path(data_dir) / "antirez-ds4.json").read_bytes()
        d = json.loads(raw)
        return [dict(c, system=d["system_prompt"]) for c in d["cases"]], hashlib.sha256(raw).hexdigest()
    path = Path(data_dir) / Path(SOURCES[name]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(SOURCES[name], timeout=120).read())
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    return [json.loads(l) for l in text.splitlines() if l.strip()], hashlib.sha256(raw).hexdigest()


def chat(a, key, content, max_tokens, think=False, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0}
    if think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    elif a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {}).get("completion_tokens", 0)


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_python(program):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        try:
            return subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=15, preexec_fn=limits).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def humaneval(a, key, case):
    prompt = ("Complete this Python function. Reply with the complete function (including the signature and any imports) "
              "in one ```python code block.\n\n```python\n" + case["prompt"] + "```")
    text, n = chat(a, key, prompt, 1024)
    code = code_block(text)
    imports = "\n".join(l for l in case["prompt"].splitlines() if l.startswith(("import ", "from ")))
    program = (imports + "\n" + code) if f"def {case['entry_point']}" in code else (case["prompt"] + code)
    program += "\n\n" + case["test"] + f"\ncheck({case['entry_point']})\n"
    return {"id": case["task_id"], "passed": run_python(program), "out_tokens": n}


def number(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k(a, key, i, case):
    text, n = chat(a, key, case["question"] + "\n\nSolve step by step. End with a final line of the form '#### <number>'.", 768)
    gold = number(case["answer"].split("####")[-1])
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    got = number(m[-1]) if m else None
    return {"id": i, "passed": got is not None and gold is not None and abs(got - gold) < 1e-6, "out_tokens": n}


def ds4(a, key, case):
    q = case["question"]
    if case["kind"] == "mcq":
        q += "\n\n" + "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(case["choices"]))
        q += "\n\nEnd your reply with a final line 'Answer: <letter>'."
    else:
        q += "\n\nEnd your reply with a final line 'Answer: <integer>'."
    text, n = chat(a, key, q, a.think_tokens, think=True, system=case["system"])
    m = re.findall(r"Answer:\s*\**\s*\(?([A-Za-z]|-?\d+)", text)
    got = m[-1].upper() if m else None
    return {"id": case["id"], "source": case["source"], "passed": got == case["answer"].upper(), "got": got, "out_tokens": n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--api-key-file")
    p.add_argument("--suites", default="humaneval,gsm8k")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--think-tokens", type=int, default=16384)
    p.add_argument("--data-dir", default="evaldata")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--only-failed-from", help="rerun only the cases that errored in this earlier result JSON")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = Path(a.api_key_file).read_text().strip() if a.api_key_file else ""
    res = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "suites": {}}
    for suite in a.suites.split(","):
        cases, sha = load(suite, a.data_dir)
        cases = cases[: a.limit] if a.limit else cases
        if a.only_failed_from:
            prev = json.loads(Path(a.only_failed_from).read_text())["suites"][suite]["cases"]
            failed = {i for i, c in enumerate(prev) if "error" in c}
            cases = [c for i, c in enumerate(cases) if i in failed]
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
            fn = {"humaneval": lambda i, c: humaneval(a, key, c), "gsm8k": lambda i, c: gsm8k(a, key, i, c),
                  "ds4": lambda i, c: ds4(a, key, c)}[suite]
            jobs = [pool.submit(fn, i, c) for i, c in enumerate(cases)]
            index = {j: i for i, j in enumerate(jobs)}
            rows = [{"passed": False, "error": "unfinished"} for _ in jobs]

            def save():  # after every case, so a killed run keeps what finished (unfinished stay errors)
                passed = sum(r["passed"] for r in rows)
                res["suites"][suite] = {"source": SOURCES.get(suite, "gufo tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"), "sha256": sha, "total": len(rows), "passed": passed,
                                        "score": round(passed / max(1, len(rows)), 4), "errors": sum("error" in r for r in rows),
                                        "wall_s": round(time.monotonic() - t0, 1), "out_tokens": sum(r.get("out_tokens", 0) for r in rows), "cases": rows}
                tmp = Path(a.out + ".tmp")
                tmp.write_text(json.dumps(res, indent=1))
                tmp.replace(a.out)

            for j in concurrent.futures.as_completed(jobs):
                try:
                    rows[index[j]] = j.result()
                except Exception as e:  # noqa: BLE001 - a failed request is a failed case
                    rows[index[j]] = {"passed": False, "error": str(e)[:200]}
                save()
        wall = time.monotonic() - t0
        passed = sum(r["passed"] for r in rows)
        print(f"[{a.label}] {suite}: {passed}/{len(rows)} = {passed / max(1, len(rows)):.3f} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
