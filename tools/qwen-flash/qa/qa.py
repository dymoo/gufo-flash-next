#!/usr/bin/env python3
"""Flash-Next serving QA: quality, correctness, performance and capacity over HTTP.

  qa.py run --base-url URL --label NAME [--suites LIST] [--quick]   -> artifacts/qa/NAME.json
  qa.py report BASELINE.json CANDIDATE.json [--out report.md]       -> quality deltas next to speed deltas
  qa.py corpus                                                       -> fetch/verify the pinned texts

Greedy, thinking off, fixed texts and seeds; stdlib only (the test host has no numpy).
Suites (--suites takes names or the groups quality, correctness, performance, all):
  quality      logits needle tasks multiturn
  correctness  determinism batch cancel stops overload
  performance  decode prefill
  capacity     capacity (not in "all": fills every session to --capacity-depth)
`logits` needs the server's opt-in `logprobs`/`forced_completion` request fields
and is skipped (not passed) on builds without them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # tools/, for gufo.serving_bench and gufo.model_bench
import tasks  # noqa: E402
from gufo import serving_bench  # noqa: E402
from gufo.model_bench.llm import TASKS as BENCH_TASKS, synthetic_text  # noqa: E402

SCHEMA = "gufo.qwen-flash-qa.v1"
GROUPS = {
    "quality": ["logits", "needle", "tasks", "multiturn"],
    "correctness": ["determinism", "batch", "cancel", "stops", "overload"],
    "performance": ["decode", "prefill"],
    "capacity": ["capacity"],
}
GROUPS["all"] = GROUPS["quality"] + GROUPS["correctness"] + GROUPS["performance"]

# Public-domain texts, pinned by SHA-256 of the body between the Gutenberg markers.
BOOKS = {
    "moby": ("https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
             "4ee728ea3b3520e321146b81172cdbacd06ee7f4255748cf92568b6f257f2c29"),
    "pride": ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
              "ec5006ab5edb21cf0ee304712665ef03a8a96bc5ef2e8bb70877dca86973d79c"),
}

TOP_LOGPROBS = 20
# Proposed gates for an opt-in lossy flag versus the exact baseline, as (pass, warn):
# worse than warn fails. Lossless changes must instead match the baseline exactly.
THRESHOLDS = {
    "kl_mean": (0.005, 0.02),       # nats, pooled over teacher-forced positions
    "kl_p99": (0.05, 0.2),
    "top1_agree": (0.98, 0.95),     # fraction of positions with the same argmax
    "ppl_delta_pct": (0.5, 2.0),    # relative increase of forced-text perplexity
    "needle_lost": (0, 0),          # needles the baseline found and the candidate missed
    "task_drop_pp": (2.0, 5.0),     # task pass-rate drop, percentage points
}
LOGIT_PROMPT = "Read the following text, then continue it exactly where it stops.\n\n"
NEEDLES = (("the lighthouse", 0.1), ("the observatory", 0.5), ("the northern archive", 0.9))


def log(message: str) -> None:
    print(f"[qa {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- HTTP ----

class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


class Client:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 3600.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = ""

    def request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise HttpError(error.code, error.read().decode(errors="replace")) from None
        return json.loads(raw) if raw.strip() else {}

    def chat(self, **body) -> dict:
        payload = {"model": self.model, "temperature": 0, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}, **body}
        return self.request("POST", "/v1/chat/completions", payload)

    def healthy(self, wait: float = 0.0) -> bool:
        deadline = time.monotonic() + wait
        while True:
            try:
                self.request("GET", "/health", timeout=5)
                return True
            except (OSError, HttpError, ValueError):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.5)

    def stream_abort(self, body: dict, after_deltas: int = 8, after_seconds: float | None = None) -> int:
        """Stream a chat request and drop the connection after `after_deltas` content deltas,
        or `after_seconds` after sending it (before reading anything: aborts during prefill)."""
        target = urllib.parse.urlsplit(self.base)
        connection = http.client.HTTPConnection(target.hostname, target.port, timeout=self.timeout)
        payload = {"model": self.model, "temperature": 0, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}, **body}
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), self.headers)
        sock = connection.sock  # getresponse() may hand the socket to the response object
        deltas = 0
        if after_seconds is not None:
            time.sleep(after_seconds)
        else:
            response = connection.getresponse()
            if response.status != 200:
                raise HttpError(response.status, response.read().decode(errors="replace"))
            for line in response:
                if line.startswith(b"data: {"):
                    choices = json.loads(line[6:]).get("choices") or [{}]
                    deltas += bool(choices[0].get("delta", {}).get("content"))
                    if deltas >= after_deltas:
                        break
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        connection.close()
        return deltas

    def stream(self, body: dict) -> dict:
        """Stream a chat request to completion; returns assembled content/tool calls."""
        target = urllib.parse.urlsplit(self.base)
        connection = http.client.HTTPConnection(target.hostname, target.port, timeout=self.timeout)
        payload = {"model": self.model, "temperature": 0, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}, **body}
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), self.headers)
        response = connection.getresponse()
        if response.status != 200:
            raise HttpError(response.status, response.read().decode(errors="replace"))
        text, calls, finish = [], {}, None
        for line in response:
            if not line.startswith(b"data: {"):
                continue
            for choice in json.loads(line[6:]).get("choices") or []:
                delta = choice.get("delta") or {}
                text.append(delta.get("content") or "")
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        connection.close()
        return {"content": "".join(text), "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish}


def content(response: dict) -> str:
    return response["choices"][0]["message"].get("content") or ""


def timing(response: dict) -> dict:
    usage = response.get("usage") or {}
    g = usage.get("gufo") or {}
    out = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "cached_tokens": usage.get("cached_tokens")}
    for key in ("prefill_tokens", "prefill_ms", "decode_ms", "ttft_ms", "queue_ms", "cache_restore_ms",
                "cache_restore_bytes", "physical_execution_width", "execution_plan"):
        if key in g:
            out[key] = g[key]
    if g.get("prefill_ms") and g.get("prefill_tokens"):
        out["prefill_tps"] = g["prefill_tokens"] / (g["prefill_ms"] / 1000)
    return out


def parallel(count: int, jobs: list) -> list:
    """Run callables together (released by one barrier) and return results in order."""
    barrier = threading.Barrier(len(jobs)) if len(jobs) > 1 else None

    def gated(job):
        if barrier is not None:
            barrier.wait()
        return job()

    with concurrent.futures.ThreadPoolExecutor(max(1, count)) as pool:
        if count >= len(jobs):
            return list(pool.map(gated, jobs))
        return list(pool.map(lambda job: job(), jobs))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# -------------------------------------------------------------- corpus ----

def book_text(data_dir: Path, name: str) -> str:
    url, pinned = BOOKS[name]
    path = data_dir / f"{name}.txt"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        log(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read().decode("utf-8-sig")
        text = raw.replace("\r\n", "\n")
        start = text.index("\n", text.index("*** START OF")) + 1
        text = text[start:text.index("*** END OF")].strip() + "\n"
        path.write_text(text)
    text = path.read_text()
    actual = hashlib.sha256(text.encode()).hexdigest()
    if pinned and actual != pinned:
        raise SystemExit(f"corpus {name}: sha256 {actual} != pinned {pinned}; references are no longer comparable")
    return text


def cut(text: str, start: int, chars: int) -> tuple[str, int]:
    """Whitespace-aligned text[start:start+chars]; returns (slice, end offset)."""
    begin = text.index(" ", start) + 1 if start else 0
    end = min(len(text), begin + chars)
    if end < len(text):
        end = text.rindex(" ", begin, end)
    return text[begin:end], end


# ------------------------------------------------------------ telemetry ----

def _read(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class Telemetry(threading.Thread):
    """GTT (device-global), container MemAvailable/cgroup and optional server RSS, every 0.5 s."""

    def __init__(self, server_pid: int | None):
        super().__init__(daemon=True)
        self.pid, self.lock, self.stop_event = server_pid, threading.Lock(), threading.Event()
        self.current, self.peaks = "setup", {}

    def sample(self) -> dict:
        rss = None
        if self.pid:
            try:
                rss = next(int(l.split()[1]) * 1024 for l in Path(f"/proc/{self.pid}/status").read_text().splitlines()
                           if l.startswith("VmRSS:"))
            except (OSError, StopIteration):
                pass
        return {"gtt_used": _read("/sys/class/drm/renderD128/device/mem_info_gtt_used"),
                "gtt_total": _read("/sys/class/drm/renderD128/device/mem_info_gtt_total"),
                "mem_available": mem_available(), "cgroup_current": _read("/sys/fs/cgroup/memory.current"),
                "server_rss": rss}

    def run(self) -> None:
        while not self.stop_event.wait(0.5):
            values = self.sample()
            with self.lock:
                peak = self.peaks.setdefault(self.current, {})
                for key, value in values.items():
                    if value is None:
                        continue
                    lowest = key == "mem_available"
                    old = peak.get(key)
                    peak[key] = value if old is None else (min(old, value) if lowest else max(old, value))

    def mark(self, suite: str) -> None:
        with self.lock:
            self.current = suite

    def summary(self) -> dict:
        with self.lock:
            gib = lambda v: round(v / 2**30, 2)  # noqa: E731
            return {suite: {("min_" if k == "mem_available" else "peak_") + k + "_gib": gib(v) for k, v in p.items()}
                    for suite, p in self.peaks.items()}


# --------------------------------------------------------------- suites ----

class Run:
    def __init__(self, args, client: Client):
        self.args, self.client = args, client
        self.quick = args.quick
        self.sessions = args.sessions
        self.ratios = {}  # characters per token for each book, calibrated at start
        self.overhead = 0
        self.logprobs = False
        self.data_dir = Path(args.data_dir)

    def book(self, name: str) -> str:
        return book_text(self.data_dir, name)

    def chars(self, tokens: int, book: str = "moby") -> int:
        return int(tokens * self.ratios[book])

    def calibrate(self) -> dict:
        self.overhead = timing(self.client.chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=1))["prompt_tokens"] - 1
        for name in BOOKS:
            sample, _ = cut(self.book(name), 60000, 16000)
            used = timing(self.client.chat(messages=[{"role": "user", "content": "Hi " + sample}], max_tokens=1))["prompt_tokens"]
            self.ratios[name] = 0.98 * len(sample) / max(1, used - self.overhead - 1)  # 2% margin: stay inside the context
        try:
            probe = self.client.chat(messages=[{"role": "user", "content": "Say yes."}], logprobs=True,
                                     top_logprobs=2, forced_completion="Yes.")
            entries = probe["choices"][0]["logprobs"]["content"]
            self.logprobs = bool(entries) and "id" in entries[0]
        except (HttpError, KeyError, TypeError, IndexError):
            self.logprobs = False
        return {"chars_per_token": {k: round(v, 4) for k, v in self.ratios.items()}, "template_tokens": self.overhead,
                "logprobs": self.logprobs}

    # -- quality ------------------------------------------------------------

    def forced(self, name: str, messages: list, forced, extra: dict | None = None) -> dict:
        response = self.client.chat(messages=messages, logprobs=True, top_logprobs=TOP_LOGPROBS,
                                    forced_completion=forced, cache_prompt=False, **(extra or {}))
        positions = self.forced_view(response)
        nll = -statistics.fmean(p[1] for p in positions) if positions else None
        return {"id": name, "positions": positions, "nll": nll, "ppl": math.exp(nll) if nll is not None else None,
                **timing(response)}

    def logits(self) -> dict:
        if not self.logprobs:
            return {"status": "unsupported", "reason": "server lacks logprobs/forced_completion"}
        moby, pride = self.book("moby"), self.book("pride")
        forced_tokens = 64 if self.quick else 256
        depths = [2048] if self.quick else self.args.logit_depths
        cases = []
        for depth in depths:
            doc, end = cut(moby, 3000, self.chars(depth))
            cont, _ = cut(moby, end, self.chars(forced_tokens))
            log(f"logits depth {depth}")
            cases.append(self.forced(f"moby-{depth}", [{"role": "user", "content": LOGIT_PROMPT + doc}], cont))
        width = min(8, self.sessions)
        jobs = []
        for index in range(width if width > 1 else 0):
            doc, end = cut(pride, 40000 + index * 60000, self.chars(2048, "pride"))
            cont, _ = cut(pride, end, self.chars(128, "pride"))
            messages = [{"role": "user", "content": LOGIT_PROMPT + doc}]
            jobs.append(lambda n=f"pride-c{width}-{index}", m=messages, c=cont: self.forced(n, m, c))
        if jobs:
            log(f"logits batched C{width}")
            cases += parallel(width, jobs)
        return {"status": "ok", "cases": cases}

    def needle(self) -> dict:
        pride = self.book("pride")
        sizes = [4096] if self.quick else self.args.needle_sizes
        rng = random.Random(1234)
        cases = []
        for size in sizes:
            hay, _ = cut(pride, 1000, self.chars(size, "pride"))
            codes = []
            for name, fraction in sorted(NEEDLES, key=lambda n: -n[1]):
                at = hay.find(". ", int(len(hay) * fraction)) + 2
                code = rng.randint(100000, 999999)
                hay = hay[:at] + f"The secret vault code for {name} is {code}. " + hay[at:]
                codes.append((name, fraction, code))
            # Multi-key retrieval: one prefill per haystack answers all three needles.
            question = ("List the secret vault codes for " + ", ".join(n for n, _ in NEEDLES) +
                        ", one per line as '<place>: <code>'.")
            prompt = "Read the document and answer the question after it.\n\n" + hay + "\n\nQuestion: " + question
            log(f"needle {size}")
            response = self.client.chat(messages=[{"role": "user", "content": prompt}], max_tokens=64)
            answer = content(response)
            lines = answer.lower().splitlines()
            for name, fraction, code in sorted(codes, key=lambda n: n[1]):
                found = any(name.split()[-1] in line and str(code) in line for line in lines)
                cases.append({"id": f"needle-{size}-{int(fraction * 100)}", "found": found,
                              "answer": answer[:160], **timing(response)})
        return {"status": "ok", "found": sum(c["found"] for c in cases), "total": len(cases), "cases": cases}

    def tasks(self) -> dict:
        selected = tasks.requests()
        if self.quick:
            selected = [t for i, t in enumerate(selected) if i % 10 == 0]
        width = min(self.args.task_concurrency, self.sessions)

        def one(task):
            try:
                response = self.client.chat(**task["body"])
                message = response["choices"][0]["message"]
                return {"id": task["id"], "kind": task["kind"], "passed": bool(task["grade"](response)),
                        "text": (message.get("content") or "") + json.dumps(message.get("tool_calls") or ""),
                        "finish_reason": response["choices"][0].get("finish_reason"), **timing(response)}
            except (HttpError, OSError, KeyError, ValueError) as error:
                return {"id": task["id"], "kind": task["kind"], "passed": False, "error": str(error)[:200]}

        log(f"tasks: {len(selected)} at C{width}")
        with concurrent.futures.ThreadPoolExecutor(width) as pool:
            cases = list(pool.map(one, selected))
        kinds = {}
        for case in cases:
            kind = kinds.setdefault(case["kind"], {"passed": 0, "total": 0})
            kind["passed"] += case["passed"]
            kind["total"] += 1
        return {"status": "ok", "concurrency": width, "passed": sum(c["passed"] for c in cases), "total": len(cases),
                "kinds": kinds, "cases": cases}

    def multiturn(self) -> dict:
        reference, _ = cut(self.book("pride"), 90000, self.chars(1500, "pride"))
        history = [{"role": "system", "content": "You are a concise assistant. Reference material:\n\n" + reference}]
        turns = ["Summarize the reference material in three sentences.",
                 "Which person is named first in the reference material? Answer in one sentence.",
                 "Write a two-sentence continuation of the reference material in the same style."]
        records, extra = [], ({"logprobs": True, "top_logprobs": 1} if self.logprobs else {})
        for index, user in enumerate(turns):
            history.append({"role": "user", "content": user})
            warm = self.client.chat(messages=history, max_tokens=96, **extra)
            again = self.client.chat(messages=history, max_tokens=96, **extra)
            cold = self.client.chat(messages=history, max_tokens=96, cache_prompt=False, **extra)
            record = {"turn": index, "text": content(warm), "cached_tokens": timing(warm)["cached_tokens"],
                      "repeat_equal": content(warm) == content(again), "cold_equal": content(warm) == content(cold)}
            if self.logprobs:
                pairs = zip(warm["choices"][0]["logprobs"]["content"], cold["choices"][0]["logprobs"]["content"])
                deltas = [abs(a["logprob"] - b["logprob"]) for a, b in pairs if a["id"] == b["id"]]
                record["cold_max_dlogprob"] = max(deltas, default=0.0)
            records.append(record)
            history.append({"role": "assistant", "content": content(warm)})
        ok = all(r["repeat_equal"] for r in records) and all(r["cached_tokens"] for r in records[1:])
        return {"status": "ok", "pass": ok, "turns": records,
                "note": "repeat_equal (restored snapshot) gates; cold full-prefill equality is informational"}

    # -- correctness ---------------------------------------------------------

    def prompts(self, count: int) -> list[str]:
        topics = ["a lighthouse keeper", "a failing bakery", "a chess tournament", "a river flood", "a lost robot",
                  "a mountain rescue", "a village election", "a desert caravan"]
        kinds = ["Write a short story about {t}.", "Explain in detail how {t} could be modelled as a Python class.",
                 "List ten facts a novelist should research before writing about {t}.",
                 "Describe {t} from three different points of view."]
        return [kinds[i % len(kinds)].format(t=topics[(i // len(kinds) + i) % len(topics)]) + f" (variant {i})"
                for i in range(count)]

    def determinism(self) -> dict:
        prompts = self.prompts(2 if self.quick else 4)
        cases = []
        for index, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            first = content(self.client.chat(messages=messages, max_tokens=96, cache_prompt=False))
            second = content(self.client.chat(messages=messages, max_tokens=96, cache_prompt=False))
            cases.append({"id": f"det-{index}", "equal": first == second, "text": first})
        return {"status": "ok", "pass": all(c["equal"] for c in cases), "cases": cases}

    def batch(self) -> dict:
        levels = sorted({c for c in ([2] if self.quick else self.args.batch_levels) if 1 < c <= self.sessions})
        if not levels:
            return {"status": "skipped", "reason": "needs --sessions >= 2"}
        prompts = self.prompts(max(levels))
        tokens = 32 if self.quick else 64
        extra = {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if self.logprobs else {}
        log(f"batch: {len(prompts)} single-session references")
        reference = [self.client.chat(messages=[{"role": "user", "content": p}], max_tokens=tokens,
                                      cache_prompt=False, **extra) for p in prompts]
        result = {"status": "ok", "levels": {}}
        for level in levels:
            log(f"batch: C{level}")
            responses = parallel(level, [lambda p=p: self.client.chat(
                messages=[{"role": "user", "content": p}], max_tokens=tokens, cache_prompt=False) for p in prompts[:level]])
            equal = [content(a) == content(b) for a, b in zip(reference, responses)]
            entry = {"identical": sum(equal), "total": level,
                     "widths": sorted({timing(r).get("physical_execution_width") for r in responses} - {None})}
            if self.logprobs:
                pairs = list(zip(prompts[:level], reference[:level]))
                forced = parallel(level, [lambda p=p, r=r, i=i: self.forced(
                    f"batch-c{level}-{i}", [{"role": "user", "content": p}],
                    [e["id"] for e in r["choices"][0]["logprobs"]["content"]]) for i, (p, r) in enumerate(pairs)])
                entry["logits"] = compare_positions([self.forced_view(r) for _, r in pairs],
                                                    [f["positions"] for f in forced])
            result["levels"][str(level)] = entry
        result["pass"] = all(e["identical"] == e["total"] for e in result["levels"].values())
        return result

    @staticmethod
    def forced_view(response: dict) -> list:
        """Positions as [id, logprob, [[id, logprob] ...]], rounded identically for exact comparisons."""
        return [[e["id"], round(e["logprob"], 5), [[t["id"], round(t["logprob"], 5)] for t in e["top_logprobs"]]]
                for e in response["choices"][0]["logprobs"]["content"]]

    def cancel(self) -> dict:
        messages = [{"role": "user", "content": self.prompts(1)[0]}]
        before = content(self.client.chat(messages=messages, max_tokens=64, cache_prompt=False))
        deltas = self.client.stream_abort({"messages": messages, "max_tokens": 1024}, after_deltas=8)
        long_doc, _ = cut(self.book("moby"), 50000, self.chars(4096 if self.quick else 16384))
        self.client.stream_abort({"messages": [{"role": "user", "content": long_doc + "\n\nSummarize."}],
                                  "max_tokens": 64}, after_seconds=0.5)
        healthy = self.client.healthy(wait=30)
        after = content(self.client.chat(messages=messages, max_tokens=64, cache_prompt=False))
        resumed = content(self.client.chat(messages=messages, max_tokens=64))
        checks = {"aborted_after_deltas": deltas >= 8, "healthy": healthy, "same_output_after_abort": before == after,
                  "cached_replay_equal": before == resumed}
        return {"status": "ok", "pass": all(checks.values()), "checks": checks}

    def stops(self) -> dict:
        messages = [{"role": "user", "content": "Count from 1 to 30 in digits, separated by commas and spaces. No other text."}]
        try:
            plain = self.client.chat(messages=messages, max_tokens=128, stop=["17"])
        except HttpError as error:  # older builds reject `stop`: report it, still check tool calls
            plain = {"choices": [{"message": {"content": f"stop rejected: HTTP {error.status}"}, "finish_reason": None}]}
        text = content(plain)
        streamed = self.client.stream({"messages": messages, "max_tokens": 128, "stop": ["17"]}) \
            if plain["choices"][0]["finish_reason"] else {"content": None}
        tool = tasks.requests()[len(tasks.CODE) + len(tasks.MATH)]  # first tool case
        tool_plain = self.client.chat(**tool["body"])
        tool_stream = self.client.stream(tool["body"])
        calls = tool_plain["choices"][0]["message"].get("tool_calls") or [{"function": {}}]

        def parsed(arguments):
            try:
                return json.loads(arguments or "null")
            except ValueError:
                return ("invalid", arguments)

        checks = {
            "stop_excluded": "17" not in text and "16" in text,
            "stop_finish_reason": plain["choices"][0].get("finish_reason") == "stop",
            "stream_matches": streamed["content"] == text,
            "tool_call_valid": bool(tool["grade"](tool_plain)),
            "stream_tool_matches": [(c["name"], parsed(c["arguments"])) for c in tool_stream["tool_calls"]]
            == [(c["function"].get("name"), parsed(c["function"].get("arguments"))) for c in calls],
        }
        return {"status": "ok", "pass": all(checks.values()), "checks": checks}

    def overload(self) -> dict:
        count = self.sessions + 2
        results = parallel(count, [lambda p=p: self._try(lambda: self.client.chat(
            messages=[{"role": "user", "content": p}], max_tokens=16, cache_prompt=False)) for p in self.prompts(count)])
        ok = [not isinstance(r, Exception) and content(r) for r in results]
        overflow_status = None
        try:
            text, _ = cut(self.book("moby"), 0, self.chars(self.args.context + 4096))
            self.client.chat(messages=[{"role": "user", "content": text}], max_tokens=8)
        except HttpError as error:
            overflow_status = error.status
        checks = {"queued_requests_complete": all(ok), "context_overflow_rejected": overflow_status == 400,
                  "healthy_after": self.client.healthy(wait=30)}
        return {"status": "ok", "pass": all(checks.values()), "checks": checks, "overflow_status": overflow_status}

    @staticmethod
    def _try(job):
        try:
            return job()
        except Exception as error:  # noqa: BLE001 - recorded as a failed request
            return error

    # -- performance ---------------------------------------------------------

    def cohort(self, messages: list, level: int, max_tokens: int, case: str) -> dict:
        """Prepare `level` sessions with the same prompt, then time one synchronized decode cohort."""
        def request(index: int, tokens: int, gate=None):
            return serving_bench.run_request(
                base_url=self.client.base, model=self.client.model, prompt="", max_tokens=tokens, temperature=0.0,
                timeout_seconds=self.client.timeout, client_id=f"qa-{case}-{index}", concurrency=level, repetition=0,
                request_index=index, start_gate=gate, messages=messages, cache_prompt=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})

        prep = [request(0, 1)]  # the first session prefills; the others can restore its snapshot
        if level > 1:
            prep += parallel(level, [lambda i=i: request(i, 1) for i in range(1, level)])
        gate = threading.Barrier(level)
        with concurrent.futures.ThreadPoolExecutor(level) as pool:
            samples = list(pool.map(lambda i: request(i, max_tokens, gate if level > 1 else None), range(level)))
        span = max(s.finished_at for s in samples) - min(s.started_at for s in samples)
        pct = lambda values, q: round(serving_bench.percentile([v for v in values if v is not None], q), 2) \
            if any(v is not None for v in values) else None  # noqa: E731
        return {
            "case": case, "concurrency": level,
            "decode_tps_sum": round(sum(s.decode_tokens_per_second or 0 for s in samples), 2),
            "wall_tps": round(sum(s.completion_tokens for s in samples) / span, 2) if span > 0 else None,
            "completion_tokens": [s.completion_tokens for s in samples],
            "ttft_ms_p50": pct([s.client_ttft_ms for s in samples], 0.5),
            "ttft_ms_p95": pct([s.client_ttft_ms for s in samples], 0.95),
            "itl_ms_p50": pct([s.server_inter_token_ms for s in samples], 0.5),
            "itl_ms_max": max((s.server_max_inter_token_ms or 0 for s in samples), default=None),
            "draft_acceptance": pct([s.draft_acceptance for s in samples], 0.5),
            "prep_prefill_tokens": [s.prefill_tokens for s in prep],
            "prep_prefill_tps_p50": pct([s.prefill_tokens_per_second for s in prep], 0.5),
            "prep_cached_tokens": [s.cached_prompt_tokens for s in prep],
            "widths": sorted({s.physical_execution_width for s in samples} - {None}),
            "hashes": [s.completion_sha256[:16] for s in samples],
        }

    def decode(self) -> dict:
        levels = [c for c in ([1, 2] if self.quick else self.args.concurrency) if c <= self.sessions]
        workloads = ["prose"] if self.quick else self.args.workloads
        depths = [0] if self.quick else self.args.perf_depths
        max_tokens = 32 if self.quick else 128
        moby = self.book("moby")
        rows = []
        for depth in depths:
            prefix, _ = cut(moby, 200000, self.chars(depth)) if depth else ("", 0)
            for workload in workloads:
                body = synthetic_text(4242, int(1024 / 1.3)) + CODE_TASK if workload == "code" else \
                    synthetic_text(4242, int(1024 / 1.3)) + BENCH_TASKS[workload]
                messages = ([{"role": "system", "content": "Reference document:\n\n" + prefix}] if prefix else []) + \
                    [{"role": "user", "content": body}]
                for level in levels:
                    log(f"decode {workload} d{depth} C{level}")
                    rows.append({"workload": workload, "depth": depth,
                                 **self.cohort(messages, level, max_tokens, f"{workload}-d{depth}-c{level}")})
        return {"status": "ok", "max_tokens": max_tokens, "rows": rows}

    def prefill(self) -> dict:
        moby = self.book("moby")
        depths = [2048] if self.quick else self.args.prefill_depths
        rows = []
        for depth in depths:
            doc, _ = cut(moby, 100000, self.chars(depth))
            log(f"prefill {depth}")
            cold = self.client.chat(messages=[{"role": "system", "content": doc},
                                              {"role": "user", "content": "Name one ship."}], max_tokens=1, cache_prompt=False)
            warm = self.client.chat(messages=[{"role": "system", "content": doc},
                                              {"role": "user", "content": "Name one ship."}], max_tokens=1)
            rows.append({"depth": depth, "cold": timing(cold), "warm": timing(warm)})
        return {"status": "ok", "rows": rows}

    def capacity(self) -> dict:
        depth = self.args.capacity_depth or max(1024, self.args.context - 8192)
        doc, _ = cut(self.book("moby"), 0, self.chars(depth))
        messages = [{"role": "system", "content": doc}, {"role": "user", "content": "Continue the story in your own words."}]
        log(f"capacity: {self.sessions} sessions at ~{depth} tokens")
        try:
            row = self.cohort(messages, self.sessions, 32, f"capacity-{depth}")
        except Exception as error:  # noqa: BLE001
            return {"status": "ok", "pass": False, "sessions": self.sessions, "depth": depth, "error": str(error)[:300]}
        return {"status": "ok", "pass": min(row["completion_tokens"]) > 0, "sessions": self.sessions, "depth": depth,
                "healthy_after": self.client.healthy(wait=30), **row}


CODE_TASK = ("\n\nWrite a complete Python module that parses the passage above into sentences, counts word "
             "frequencies, and prints a report, with docstrings and unit tests. Write at least 300 lines of code.")


# -------------------------------------------------------------- metrics ----

def topk_kl(ref: list, cand: list) -> float:
    """KL(ref || cand) over ref's top-k tokens plus one tail bucket. A token missing from the
    candidate's top-k is given the candidate's k-th logprob (an upper bound), so this
    underestimates the true KL slightly; both logprob lists come from full-vocabulary softmaxes."""
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


def compare_positions(ref_cases: list, cand_cases: list) -> dict:
    """Pooled per-position metrics for aligned teacher-forced cases (lists of [id, lp, top])."""
    kls, top1, dlp, nll_ref, nll_cand, misaligned = [], [], [], [], [], 0
    for ref, cand in zip(ref_cases, cand_cases):
        if len(ref) != len(cand) or any(a[0] != b[0] for a, b in zip(ref, cand)):
            misaligned += 1
            continue
        for (token, lp_ref, top_ref), (_, lp_cand, top_cand) in zip(ref, cand):
            kls.append(topk_kl(top_ref, top_cand))
            top1.append(max(top_ref, key=lambda t: t[1])[0] == max(top_cand, key=lambda t: t[1])[0])
            dlp.append(abs(lp_ref - lp_cand))
            nll_ref.append(-lp_ref)
            nll_cand.append(-lp_cand)
    if not kls:
        return {"positions": 0, "misaligned_cases": misaligned}
    ppl_ref, ppl_cand = math.exp(statistics.fmean(nll_ref)), math.exp(statistics.fmean(nll_cand))
    return {"positions": len(kls), "misaligned_cases": misaligned, "kl_mean": statistics.fmean(kls),
            "kl_p99": sorted(kls)[min(len(kls) - 1, int(0.99 * len(kls)))], "top1_agree": statistics.fmean(top1),
            "max_dlogprob": max(dlp), "mean_dlogprob": statistics.fmean(dlp), "ppl_ref": ppl_ref, "ppl_cand": ppl_cand,
            "ppl_delta_pct": 100 * (ppl_cand / ppl_ref - 1)}


# --------------------------------------------------------------- report ----

def verdict(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    good, warn = THRESHOLDS[metric]
    higher_better = metric == "top1_agree"
    if (value >= good) if higher_better else (value <= good):
        return "PASS"
    if (value >= warn) if higher_better else (value <= warn):
        return "WARN"
    return "FAIL"


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1 else f"{value:.{max(1, digits - 2)}f}"
    return str(value)


def pct(new, old) -> str:
    return f"{100 * (new / old - 1):+.1f}%" if new and old else "–"


def texts_by_id(results: dict) -> dict:
    """Greedy outputs keyed by case, for cross-run identity (MTP on/off, lossless changes)."""
    out = {}
    suites = results.get("suites", {})
    for case in suites.get("tasks", {}).get("cases", []):
        out[case["id"]] = case.get("text")
    for case in suites.get("determinism", {}).get("cases", []):
        out[case["id"]] = case["text"]
    for turn in suites.get("multiturn", {}).get("turns", []):
        out[f"multiturn-{turn['turn']}"] = turn["text"]
    for case in suites.get("needle", {}).get("cases", []):
        out[case["id"]] = case["answer"]
    for row in suites.get("decode", {}).get("rows", []):
        for index, value in enumerate(row.get("hashes", [])):
            out[f"decode-{row['case']}-{index}"] = value
    return out


def report(base: dict, cand: dict) -> str:
    b, c = base.get("suites", {}), cand.get("suites", {})
    lines = [f"# Flash-Next QA: `{cand.get('label')}` vs `{base.get('label')}`", "",
             f"Baseline: {base.get('server_note') or '–'}  ", f"Candidate: {cand.get('server_note') or '–'}", "",
             "## Quality (candidate vs baseline)", "", "| Check | Baseline | Candidate | Δ | Gate (pass / warn) | Verdict |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    verdicts = []

    def row(name, old, new, delta, metric=None, value=None):
        v = verdict(metric, value) if metric else ""
        if v:
            verdicts.append(v)
        gate = " / ".join(fmt(x) for x in THRESHOLDS[metric]) if metric else ""
        lines.append(f"| {name} | {old} | {new} | {delta} | {gate} | {v} |")

    bl, cl = b.get("logits", {}), c.get("logits", {})
    if bl.get("status") == "ok" and cl.get("status") == "ok":
        ref = {x["id"]: x for x in bl["cases"]}
        groups = {}
        for case in cl["cases"]:
            if case["id"] in ref:
                group = case["id"].rsplit("-", 1)[0] if case["id"].startswith("pride-") else case["id"]
                groups.setdefault(group, ([], []))
                groups[group][0].append(ref[case["id"]]["positions"])
                groups[group][1].append(case["positions"])
        for group, (refs, cands) in groups.items():
            m = compare_positions(refs, cands)
            if not m.get("positions"):
                row(f"logits {group}", "–", "–", f"misaligned {m['misaligned_cases']}")
                continue
            row(f"KL mean, {group} ({m['positions']} pos)", "0", fmt(m["kl_mean"]), "", "kl_mean", m["kl_mean"])
            row(f"KL p99, {group}", "0", fmt(m["kl_p99"]), "", "kl_p99", m["kl_p99"])
            row(f"top-1 agreement, {group}", "1", fmt(m["top1_agree"]), "", "top1_agree", m["top1_agree"])
            row(f"max \\|Δ logprob\\| forced token, {group}", "0", fmt(m["max_dlogprob"]), "")
            row(f"PPL forced text, {group}", fmt(m["ppl_ref"]), fmt(m["ppl_cand"]), f"{m['ppl_delta_pct']:+.2f}%",
                "ppl_delta_pct", m["ppl_delta_pct"])
    else:
        row("teacher-forced logits", bl.get("status", "–"), cl.get("status", "–"), "not compared")
    # Batch-dependent drift (e.g. batch-aware routing): each run's CN rows vs its own C1 rows.
    for level, entry in c.get("batch", {}).get("levels", {}).items():
        new, old = entry.get("logits", {}), b.get("batch", {}).get("levels", {}).get(level, {}).get("logits", {})
        if new.get("positions"):
            row(f"KL mean, C{level} rows vs C1 (same run)", fmt(old.get("kl_mean")), fmt(new["kl_mean"]), "",
                "kl_mean", new["kl_mean"])
            row(f"top-1 agreement, C{level} vs C1", fmt(old.get("top1_agree")), fmt(new["top1_agree"]), "",
                "top1_agree", new["top1_agree"])
    bn, cn = b.get("needle", {}), c.get("needle", {})
    if bn.get("cases") and cn.get("cases"):
        found = {x["id"]: x["found"] for x in cn["cases"]}
        lost = sum(1 for x in bn["cases"] if x["found"] and not found.get(x["id"], False))
        row("needle found", f"{bn['found']}/{bn['total']}", f"{cn['found']}/{cn['total']}", f"lost {lost}", "needle_lost", lost)
    bt, ct = b.get("tasks", {}), c.get("tasks", {})
    if bt.get("total") and ct.get("total"):
        for kind in sorted(set(bt["kinds"]) | set(ct["kinds"])):
            old, new = bt["kinds"].get(kind, {}), ct["kinds"].get(kind, {})
            row(f"tasks {kind}", f"{old.get('passed', 0)}/{old.get('total', 0)}", f"{new.get('passed', 0)}/{new.get('total', 0)}", "")
        drop = 100 * (bt["passed"] / bt["total"] - ct["passed"] / ct["total"])
        row("tasks total", f"{bt['passed']}/{bt['total']}", f"{ct['passed']}/{ct['total']}", f"{-drop:+.1f} pp", "task_drop_pp", drop)
    old_texts, new_texts = texts_by_id(base), texts_by_id(cand)
    shared = sorted(set(old_texts) & set(new_texts))
    if shared:
        same = sum(old_texts[k] == new_texts[k] for k in shared)
        row("greedy outputs identical to baseline", "", f"{same}/{len(shared)}", f"{100 * same / len(shared):.1f}%")
    lines += ["", "Lossless changes (same math, re-layout, fusion, batching) must show KL 0, identical greedy "
              "outputs and unchanged needle/task results; the gates apply to opt-in lossy flags.", "",
              "## Correctness (candidate)", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
    for name in GROUPS["correctness"] + ["multiturn", "capacity"]:
        suite = c.get(name)
        if not suite:
            continue
        status = suite.get("status")
        result = "PASS" if suite.get("pass") else ("FAIL" if "pass" in suite else status)
        if result == "FAIL":
            verdicts.append("FAIL")
        detail = suite.get("checks") or {k: f"{v['identical']}/{v['total']}" for k, v in suite.get("levels", {}).items()} \
            or suite.get("reason") or suite.get("error") or ""
        lines.append(f"| {name} | {result} | {json.dumps(detail)[:200] if not isinstance(detail, str) else detail} |")
    lines += ["", "## Performance (candidate vs baseline)", ""]
    if c.get("decode", {}).get("rows"):
        lines += ["| Workload | Depth | C | Base decode tok/s (sum) | Cand | Δ | Base wall tok/s | Cand | Δ | TTFT p50 ms (base → cand) | ITL p50 ms (base → cand) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    base_rows = {r["case"]: r for r in b.get("decode", {}).get("rows", [])}
    for r in c.get("decode", {}).get("rows", []):
        o = base_rows.get(r["case"], {})
        lines.append(f"| {r['workload']} | {r['depth']} | {r['concurrency']} | {fmt(o.get('decode_tps_sum'))} | "
                     f"{fmt(r['decode_tps_sum'])} | {pct(r['decode_tps_sum'], o.get('decode_tps_sum'))} | "
                     f"{fmt(o.get('wall_tps'))} | {fmt(r['wall_tps'])} | {pct(r['wall_tps'], o.get('wall_tps'))} | "
                     f"{fmt(o.get('ttft_ms_p50'))} → {fmt(r['ttft_ms_p50'])} | {fmt(o.get('itl_ms_p50'))} → {fmt(r['itl_ms_p50'])} |")
    base_prefill = {r["depth"]: r for r in b.get("prefill", {}).get("rows", [])}
    if c.get("prefill", {}).get("rows"):
        lines += ["", "| Prefill depth | Base tok/s | Cand tok/s | Δ | Warm restore ms (base → cand) | Warm cached tokens |",
                  "| ---: | ---: | ---: | ---: | --- | ---: |"]
        for r in c["prefill"]["rows"]:
            o = base_prefill.get(r["depth"], {"cold": {}, "warm": {}})
            lines.append(f"| {r['depth']} | {fmt(o['cold'].get('prefill_tps'))} | {fmt(r['cold'].get('prefill_tps'))} | "
                         f"{pct(r['cold'].get('prefill_tps'), o['cold'].get('prefill_tps'))} | "
                         f"{fmt(o['warm'].get('cache_restore_ms'))} → {fmt(r['warm'].get('cache_restore_ms'))} | "
                         f"{r['warm'].get('cached_tokens')} |")
    memory = cand.get("memory", {})
    if memory:
        lines += ["", "Candidate memory peaks: " + "; ".join(
            f"{suite}: GTT {m.get('peak_gtt_used_gib', '–')} GiB, MemAvailable min {m.get('min_mem_available_gib', '–')} GiB"
            for suite, m in memory.items())]
    overall = "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS" if verdicts else "n/a"
    lines[1:1] = ["", f"**Quality/correctness verdict: {overall}**"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI ----

def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def run(args) -> int:
    suites = []
    for name in args.suites.split(","):
        suites += GROUPS.get(name, [name])
    unknown = [s for s in suites if not any(s in g for g in GROUPS.values())]
    if unknown:
        raise SystemExit(f"unknown suites: {unknown}")
    api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key  # serving_bench reads it; never printed or stored
    client = Client(args.base_url, api_key)
    models = client.request("GET", "/v1/models")["data"]
    client.model = args.model or next(m["id"] for m in models if m.get("capability") in (None, "text"))
    out = Path(args.out or f"artifacts/qa/{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    qa = Run(args, client)
    telemetry = Telemetry(args.server_pid)
    telemetry.start()
    results = {"schema": SCHEMA, "label": args.label, "server_note": args.note, "model": client.model,
               "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "tool_revision": _revision(), "options": {k: v for k, v in vars(args).items() if k not in ("api_key_file",)},
               "suites": {}}
    results["calibration"] = qa.calibrate()
    log(f"model {client.model}; calibration {results['calibration']}")
    for suite in dict.fromkeys(suites):
        telemetry.mark(suite)
        started = time.monotonic()
        try:
            results["suites"][suite] = getattr(qa, suite)()
        except Exception as error:  # noqa: BLE001 - one broken suite must not hide the others
            results["suites"][suite] = {"status": "error", "error": f"{type(error).__name__}: {error}"[:500]}
        results["suites"][suite]["seconds"] = round(time.monotonic() - started, 1)
        results["memory"] = telemetry.summary()
        out.write_text(json.dumps(results, indent=1) + "\n")
        state = results["suites"][suite]
        log(f"{suite}: {state.get('status')} pass={state.get('pass', '-')} ({state['seconds']} s)")
    telemetry.stop_event.set()
    log(f"wrote {out}")
    return 0


def _revision() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GUFO_QA_REVISION", "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run suites against one server")
    r.add_argument("--base-url", default="http://127.0.0.1:8011")
    r.add_argument("--api-key-file", help="bearer key file (read in-process, never printed)")
    r.add_argument("--model", help="served model id (default: discovered)")
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="", help="server flags/build, stored in the results")
    r.add_argument("--out")
    r.add_argument("--suites", default="all")
    r.add_argument("--quick", action="store_true", help="tiny sizes for smoke tests")
    r.add_argument("--sessions", type=int, default=8, help="server --sessions (bounds concurrency)")
    r.add_argument("--context", type=int, default=131072, help="server --context per session")
    r.add_argument("--server-pid", type=int, help="sample this process's RSS")
    r.add_argument("--data-dir", default="artifacts/qa/corpus")
    r.add_argument("--logit-depths", type=ints, default=ints("2048,16384,65536,122880"))
    r.add_argument("--needle-sizes", type=ints, default=ints("8192,32768,65536,122880"))
    r.add_argument("--task-concurrency", type=int, default=8)
    r.add_argument("--batch-levels", type=ints, default=ints("8,32,64"))
    r.add_argument("--concurrency", type=ints, default=ints("1,2,4,8,16,32,64"))
    r.add_argument("--workloads", type=lambda s: s.split(","), default=["prose", "repetition", "code"])
    r.add_argument("--perf-depths", type=ints, default=ints("0,32768"))
    r.add_argument("--prefill-depths", type=ints, default=ints("2048,16384,32768,65536,122880"))
    r.add_argument("--capacity-depth", type=int, default=0, help="tokens per session (default: context - 8192)")
    rep = sub.add_parser("report", help="baseline vs candidate Markdown")
    rep.add_argument("baseline")
    rep.add_argument("candidate")
    rep.add_argument("--out")
    corpus = sub.add_parser("corpus", help="download and hash the pinned texts")
    corpus.add_argument("--data-dir", default="artifacts/qa/corpus")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "report":
        text = report(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()))
        if args.out:
            Path(args.out).write_text(text)
        print(text)
        return 0
    for name in BOOKS:
        text = book_text(Path(args.data_dir), name)
        print(name, len(text), hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
