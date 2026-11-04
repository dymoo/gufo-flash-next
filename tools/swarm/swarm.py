#!/usr/bin/env python3
"""Swarm goodput: verified agent tasks per wall-clock hour under concurrent agent load.

  swarm.py run --base-url URL --agents N --budget SECONDS --label NAME [...]   -> artifacts/swarm/NAME.json
  swarm.py report RESULT.json [...] [--out report.md]                          -> ranked comparison
  swarm.py tasks [--count N] [--seed S] [--show ID]                              -> the pinned task queue

A swarm of N agent slots arrives like tools/qwen-flash/qa/agent_sim.py (a burst,
then Poisson arrivals) and works through one fixed, ordered task queue: each
slot takes the next task, runs a real multi-turn tool loop against the server
(every reply and tool call comes from the model under test), and takes the next
task when the checker has graded it. The wall-clock budget ends the run; tasks
still running then are reported as unfinished. Stdlib only (the test host has no numpy).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib
import http.client
import json
import math
import re
import signal
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import families  # noqa: E402
import toolcalls  # noqa: E402
from families import digest, rng_for, synth  # noqa: E402

# suite name -> (task package, suite version). v2 is the pinned corpus (its version string predates the
# name); v3 is the hard tier. A suite's queue digest covers only its own task sources.
SUITES = {"v2": ("families", "swarm-v1"), "v3": ("families_v3", "swarm-v3")}


def suite_module(name: str):
    return importlib.import_module(SUITES[name][0])

SCHEMA = "gufo.swarm-goodput.v1"
CAPACITY = (429, 503)


def log(message: str) -> None:
    print(f"[swarm {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def pct(values: list, p: float, scale: float = 1.0):
    if not values:
        return None
    ordered = sorted(values)
    return round(scale * ordered[min(len(ordered) - 1, int(p * len(ordered)))], 3)


def task_digest(task) -> str:
    """Identity of a task: its prompt and its reference solution."""
    return digest(task.prompt + json.dumps([[c.name, c.arguments, s.content] for s in task.oracle() for c in s.calls],
                                           sort_keys=True, default=str))


def graded_in(paths: list, seed: int, suite: str = "v2") -> set:
    """Queue indices already graded (any outcome but unfinished) in earlier results of the same queue."""
    done = set()
    for path in paths or []:
        r = json.loads(Path(path).read_text())
        if r["queue_digest"] != queue_digest(seed, suite):
            raise SystemExit(f"{path}: different task queue ({r['queue_digest']})")
        done |= {t["g"] for t in r["tasks"] if t["outcome"] != "unfinished"}
    return done


def merge_cmd(args) -> int:
    """Merge results of one setting run in parts: per queue index the graded record wins."""
    results = [json.loads(Path(p).read_text()) for p in args.results]
    if len({r["queue_digest"] for r in results}) != 1:
        raise SystemExit("results come from different task queues")
    keys = ("model", "thinking", "effort", "temperature", "top_p", "extra", "system_tokens", "max_turns", "max_tokens")
    if len({json.dumps([r["config"].get(k) for k in keys], sort_keys=True) for r in results}) != 1:
        raise SystemExit("results differ in model or settings; refusing to merge")
    chosen: dict[int, tuple[int, dict]] = {}
    for i, r in enumerate(results):
        for t in r["tasks"]:
            if t["g"] not in chosen or (chosen[t["g"]][1]["outcome"] == "unfinished" and t["outcome"] != "unfinished"):
                chosen[t["g"]] = (i, t)
    merged = dict(results[0])
    merged.update(label=args.label, partial=any(r.get("partial") for r in results),
                  aborted="; ".join(r["aborted"] for r in results if r.get("aborted")),
                  merged_from=[{"path": p, "started_utc": r.get("started_utc"), "ended_s": r["ended_s"]}
                               for p, r in zip(args.results, results)],
                  tasks=[t for _, (_, t) in sorted(chosen.items())],
                  requests=[q for i, r in enumerate(results) for q in r["requests"] if chosen.get(q[0], (None,))[0] == i],
                  samples=[s for r in results for s in r.get("samples", [])],
                  ended_s=round(sum(r["ended_s"] for r in results), 1))
    merged["config"] = dict(results[0]["config"], budget=sum(r["config"]["budget"] for r in results))
    merged["summary"] = summarize(merged)
    Path(args.out).write_text(json.dumps(merged, indent=1))
    log(f"merged {len(results)} parts into {args.out}: {len(merged['tasks'])} tasks")
    return 0


def queue_digest(seed: int, suite: str = "v2") -> str:
    """Identity of the whole queue: suite version, seed and the task generators' source."""
    package, version = SUITES[suite]
    if package == "families":
        files = sorted((HERE / package).glob("[!.]*.py"))
    else:  # the hard tier: every file of its package (the shopkit app too) and the v2 code it builds on
        files = sorted(p for p in (HERE / package).rglob("[!.]*") if p.is_file() and "__pycache__" not in p.parts)
        files += [HERE / "families" / name for name in ("__init__.py", "synth.py", "code.py", "ops.py")]
    sources = b"".join(p.read_bytes() for p in files)
    return digest(f"{version}:{seed}:" + hashlib.sha256(sources).hexdigest())


# ---------------------------------------------------------------- client ----

class Interrupted(Exception):
    """The run budget ended while a request was in flight."""


class HttpError(Exception):
    def __init__(self, status: int, body: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body, self.retry_after = status, body, retry_after


class Run:
    """Shared state of one swarm run."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.next_task = args.first_task
        self.skip = graded_in(args.skip_graded, args.seed, args.suite)
        self.suite = suite_module(args.suite)
        self.only = {f for f in (getattr(args, "families", "") or "").split(",") if f} or None
        self.inflight: dict[int, tuple[int, float]] = {}  # slot -> (task, start) while a task runs
        self.server_errors = 0
        self.trace_seq = 0
        self.aborted = ""
        self.tasks: list[dict] = []
        self.requests: list[list] = []
        self.itl: list[float] = []      # inter-chunk gaps of requests sent inside the measured span, seconds
        self.samples: list[dict] = []
        self.live: dict[int, http.client.HTTPConnection] = {}
        self.t0 = 0.0
        target = urllib.parse.urlsplit(args.base_url.rstrip("/"))
        self.scheme, self.host, self.port = target.scheme, target.hostname, target.port or (443 if target.scheme == "https" else 80)
        self.path = (target.path or "") + "/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if args.api_key:
            self.headers["Authorization"] = "Bearer " + args.api_key
        kinds = getattr(self.suite, "KIND", {})
        self.systems = {f: synth.system_prompt(kinds.get(f, f), args.system_tokens, rng_for("system", f, args.seed))
                        for f in self.suite.FAMILIES}

    def now(self) -> float:
        return time.monotonic() - self.t0

    def family_of(self, g: int) -> str:
        """The family of queue index g without building the task (matches suite.make(g, seed).family)."""
        block, position = divmod(g, self.suite.BLOCK)
        order = list(range(self.suite.BLOCK))
        rng_for("v3-block" if self.args.suite == "v3" else "block", self.args.seed, block).shuffle(order)
        return self.suite.FAMILIES[order[position] % len(self.suite.FAMILIES)]

    def take(self) -> int | None:
        with self.lock:
            while self.next_task in self.skip or (
                    self.only is not None and self.family_of(self.next_task) not in self.only):
                self.next_task += 1
            if self.stop.is_set() or self.next_task >= self.args.first_task + self.args.max_tasks:
                return None
            g = self.next_task
            self.next_task += 1
            return g

    def connection(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.args.read_timeout)

    def body(self, task, messages: list, turn: int) -> dict:
        a = self.args
        body = {"model": a.model, "messages": messages, "tools": task.tools, "tool_choice": "auto", "stream": True,
                "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens,
                "seed": int(digest(f"{a.seed}:{task.id}:{turn}"), 16) % (2 ** 31)}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        if a.top_p is not None:
            body["top_p"] = a.top_p
        if a.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif a.thinking == "on":
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if a.effort:
            body["reasoning_effort"] = a.effort
        body.update(a.extra)
        return body

    @staticmethod
    def is_empty(resp: dict) -> bool:
        """finish=stop with no visible content and no tool call: an agent loop that stops on a reply without
        tool calls ends the task here."""
        return resp["finish"] == "stop" and not "".join(resp["content"]).strip() and not resp["calls"]

    def dump_trace(self, task, turn: int, body: dict, resp: dict, tag: str = "empty") -> str:
        """Save the request shape and the raw stream of one reply. No credentials are recorded."""
        directory = Path(self.args.trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        reasoning, content = "".join(resp["reasoning"]), "".join(resp["content"])
        usage = resp["usage"] or {}
        record = {
            "tag": tag, "label": getattr(self.args, "label", ""), "task": task.id, "turn": turn,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "request": {**{k: v for k, v in body.items() if k not in ("messages", "tools")},
                        "tools": [t["function"]["name"] for t in body.get("tools") or []],
                        "body_bytes": resp.get("body_bytes"),
                        "messages": [{"role": m["role"], "chars": len(str(m.get("content") or ""))}
                                     for m in body["messages"]],
                        "last_user_message": next((m["content"] for m in reversed(body["messages"])
                                                   if m["role"] == "user"), "")[:2000]},
            "response": {"headers": resp.get("headers"), "event_id": resp.get("event_id"), "finish": resp["finish"],
                         "usage": usage, "reasoning_chars": len(reasoning), "content": content,
                         "reasoning_text": reasoning, "tool_calls": list(resp["calls"].values()),
                         "ends_with_think_close": reasoning.rstrip().endswith("</think>"),
                         "think_close_in_reasoning": "</think>" in reasoning, "think_close_in_content": "</think>" in content,
                         "tool_markup_in_reasoning": bool(toolcalls.MARKUP.search(reasoning)),
                         "tool_markup_in_content": bool(toolcalls.MARKUP.search(content))},
            "raw_events": resp.get("raw", [])[:600],
        }
        with self.lock:
            self.trace_seq += 1
            seq = self.trace_seq
        name = f"{record['label'] or 'trace'}-{task.id}-t{turn}-{tag}-{int(time.time() * 1000) % 10**8}-{seq}.json"
        path = directory / name
        path.write_text(json.dumps(record, indent=1))
        return str(path)

    def stream(self, slot: int, body: dict) -> dict:
        """One streamed chat completion. Raises HttpError, Interrupted or OSError."""
        conn = self.connection()
        with self.lock:
            if self.stop.is_set():
                raise Interrupted()
            self.live[slot] = conn
        out = {"content": [], "reasoning": [], "calls": {}, "finish": None, "usage": None, "ttft": None,
               "itl": [], "chunks": 0, "reasoning_chunks": 0, "timings": None}
        trace = bool(self.args.trace_dir)
        if trace:  # keep the raw stream so an empty reply can be saved for the server owner
            out.update(raw=[], headers={}, event_id=None, body_bytes=len(json.dumps(body)))
        sent = time.monotonic()
        try:
            conn.request("POST", self.path, json.dumps(body), self.headers)
            resp = conn.getresponse()
            if trace:
                out["headers"] = dict(resp.getheaders())
            if resp.status != 200:
                retry = resp.getheader("Retry-After")
                raise HttpError(resp.status, resp.read().decode(errors="replace"),
                                float(retry) if retry and retry.replace(".", "", 1).isdigit() else None)
            last, complete = None, False
            for raw in resp:
                line = raw.strip()
                if trace and line:
                    out["raw"].append([round(time.monotonic() - sent, 4), line.decode(errors="replace")])
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    complete = True
                    break
                event = json.loads(data)
                if trace and out["event_id"] is None:
                    out["event_id"] = event.get("id")
                if event.get("error"):
                    error = event["error"]
                    raise HttpError(int(error.get("status") or 500) if str(error.get("status", "")).isdigit() else 500,
                                    json.dumps(error))
                if event.get("usage"):
                    out["usage"] = event["usage"]
                if event.get("timings"):
                    out["timings"] = event["timings"]
                token = text = False
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        out["content"].append(delta["content"])
                        token = text = True
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        out["reasoning"].append(reasoning)
                        out["reasoning_chunks"] += 1
                        token = text = True
                    for call in delta.get("tool_calls") or []:
                        slot_call = out["calls"].setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        slot_call["id"] = call.get("id") or slot_call["id"]
                        fn = call.get("function") or {}
                        slot_call["name"] += fn.get("name") or ""
                        slot_call["arguments"] += fn.get("arguments") or ""
                        token = True
                    if choice.get("finish_reason"):
                        out["finish"] = choice["finish_reason"]
                if token:  # first generated output of any kind; ITL only between streamed text chunks
                    now = time.monotonic()
                    if out["ttft"] is None:
                        out["ttft"] = now - sent
                    if text:
                        if last is not None:
                            out["itl"].append(now - last)
                        last = now
                    out["chunks"] += 1
            if not complete and out["finish"] is None:  # cut off: never grade a truncated reply
                raise OSError("stream ended before a finish_reason")
        except (OSError, http.client.HTTPException, ValueError) as error:
            if self.stop.is_set():
                raise Interrupted() from None
            raise OSError(f"{type(error).__name__}: {error}") from None
        finally:
            with self.lock:
                self.live.pop(slot, None)
            conn.close()
        if self.stop.is_set() and out["finish"] is None:
            raise Interrupted()
        out["sent"], out["done"] = sent - self.t0, time.monotonic() - self.t0
        return out


# ----------------------------------------------------------------- agent ----

def context_error(error: HttpError) -> bool:
    text = error.body.lower()
    return error.status not in CAPACITY and any(k in text for k in ("context", "too long", "prompt is too large"))


def run_task(run: Run, slot: int, g: int) -> dict:
    a = run.args
    task = run.suite.make(g, a.seed)
    rec = {"g": g, "id": task.id, "digest": task_digest(task), "family": task.family, "tier": task.tier,
           "slot": slot, "start": run.now(),
           "end": None, "outcome": None, "detail": "", "turns": 0, "tool_calls": 0, "invalid_calls": 0, "nudges": 0,
           "rejections": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
           "reasoning_from_usage": True, "context_tokens": 0, "final_finish": None, "model_malformed": 0,
           "server_parse_failure": 0}
    messages = [{"role": "system", "content": run.systems[task.family]}, {"role": "user", "content": task.prompt}]
    rng = rng_for("latency", a.seed, task.id)
    bad_turns = []  # attribution of consecutive turns without a valid call
    last_content = ""

    def finish(outcome: str, detail: str = "") -> dict:
        rec.update(outcome=outcome, detail=detail[:300], end=run.now())
        if getattr(task, "requirements", None) is not None:  # v3: which of the task's requirements it met
            rec["requirements"] = dict(task.requirements)
        if outcome != "success":
            rec["last_content"] = last_content[:400]
        return rec

    for turn in range(a.max_turns):
        if run.stop.is_set():
            return finish("unfinished")
        body = run.body(task, messages, turn)
        refusals, down_since = 0, None
        while True:  # capacity refusals and brief outages are retried; the wait is part of the task's time
            try:
                resp = run.stream(slot, body)
                break
            except Interrupted:
                return finish("unfinished")
            except HttpError as error:
                if error.status in CAPACITY:  # Retry-After, then OpenAI-SDK-style backoff capped at 8 s
                    rec["rejections"] += 1
                    wait = min(8.0, max(0.5, error.retry_after or 1.0) * 1.5 ** refusals) * rng.uniform(0.8, 1.2)
                    refusals += 1
                    if run.stop.wait(wait):
                        return finish("unfinished")
                    continue
                return finish("context_overflow" if context_error(error) else "server_error", str(error))
            except OSError as error:  # connection refused/reset: the server may be restarting
                down_since = down_since or run.now()
                if run.now() - down_since > a.outage_grace or run.stop.wait(min(8.0, 1.0 + run.now() - down_since)):
                    return finish("unfinished" if run.stop.is_set() else "server_error", str(error))
                continue
        rec["turns"] += 1
        usage = resp["usage"] or {}
        details = usage.get("completion_tokens_details") or {}
        prompt = usage.get("prompt_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get("cached_tokens") or 0
        completion = usage.get("completion_tokens") or resp["chunks"]
        if "reasoning_tokens" in details:
            reasoning = details["reasoning_tokens"] or 0
        else:
            reasoning = resp["reasoning_chunks"]
            rec["reasoning_from_usage"] = False
        rec["prompt_tokens"] += prompt
        rec["cached_tokens"] += cached
        rec["completion_tokens"] += completion
        rec["reasoning_tokens"] += reasoning
        rec["context_tokens"] = prompt + completion
        rec["final_finish"] = resp["finish"]
        gufo, timings = usage.get("gufo") or {}, resp["timings"] or {}
        server_ttft = gufo.get("ttft_ms")
        if server_ttft is None and timings.get("prompt_ms") is not None:
            server_ttft = timings["prompt_ms"] + (timings.get("queue_ms") or 0)
        server_itl = gufo.get("mean_inter_token_ms") or timings.get("predicted_per_token_ms")
        queue = gufo.get("queue_ms")
        with run.lock:
            run.requests.append([g, turn, round(resp["sent"], 3), None if resp["ttft"] is None else round(resp["ttft"], 4),
                                 round(resp["done"], 3), prompt, cached, completion, reasoning, resp["chunks"],
                                 resp["finish"], rec["rejections"],
                                 None if server_ttft is None else round(server_ttft / 1000, 4),
                                 None if server_itl is None else round(server_itl, 2),
                                 None if queue is None else round(queue / 1000, 3)])
            if resp["sent"] >= a.warmup:
                run.itl.extend(resp["itl"])
        content = "".join(resp["content"])
        last_content = content
        if a.trace_dir and run.is_empty(resp):
            log(f"empty reply: {task.id} turn {turn} -> {run.dump_trace(task, turn, body, resp)}")
        calls = [resp["calls"][k] for k in sorted(resp["calls"])]
        message = {"role": "assistant", "content": content or (None if calls else "")}
        if resp["reasoning"] and a.history_reasoning == "keep":
            message["reasoning_content"] = "".join(resp["reasoning"])
        if calls:
            for i, call in enumerate(calls):
                call["id"] = call["id"] or f"call_{turn}_{i}"
            message["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        messages.append(message)
        if calls:
            invalid, waits, causes = 0, [], []
            for call in calls:
                rec["tool_calls"] += 1
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                    problem = task.invalid(call["name"], arguments)
                    cause = toolcalls.attribute_invalid(task, call["name"], arguments) if problem else None
                except ValueError as error:
                    problem, cause = f"arguments are not valid JSON ({error})", "model_malformed"
                if problem:
                    invalid += 1
                    rec[cause] += 1
                    causes.append(cause)
                    output = f"Error: invalid tool call: {problem}"
                else:
                    try:
                        output = task.call(call["name"], arguments)
                    except families.ToolError as error:
                        output = f"Error: {error}"
                    mean = task.latency.get(call["name"], 0.0) * a.tool_time_scale
                    waits.append(rng.expovariate(1.0 / mean) if mean > 0 else 0.0)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            rec["invalid_calls"] += invalid
            bad_turns = bad_turns + [max(causes, key=causes.count)] if invalid == len(calls) else []
            if len(bad_turns) >= 3:
                return finish(max(bad_turns, key=bad_turns.count), "three consecutive turns of invalid tool calls")
            if task.submitted is not None:
                ok, detail = task.check(content)
                return finish("success" if ok else "wrong_answer", detail)
            if run.stop.wait(min(max(waits, default=0.0), 60.0)):  # parallel calls run concurrently
                return finish("unfinished")
            continue
        if toolcalls.MARKUP.search(content):  # tool markup the server left in the text
            cause = toolcalls.attribute_markup(task, content)
            rec["invalid_calls"] += 1
            rec[cause] += 1
            bad_turns.append(cause)
            if len(bad_turns) >= 3:
                return finish(max(bad_turns, key=bad_turns.count), "tool call markup in content: " + content[-200:])
            messages.append({"role": "user", "content": "Your last message contained a tool call that could not be "
                             "parsed. Call tools with the provided function-calling interface."})
            continue
        if task.submit_tool and task.submitted is None and rec["nudges"] == 0:
            rec["nudges"] += 1
            messages.append({"role": "user", "content": f"You have not called {task.submit_tool} yet. Call it to "
                             "complete the ticket."})
            continue
        ok, detail = task.check(content)
        if ok:
            return finish("success", detail)
        if resp["finish"] == "length":
            limit = a.context or 0
            overflow = limit and prompt + completion >= limit - 16
            return finish("context_overflow" if overflow else "output_cap", detail)
        if not content.strip() and not task.submit_tool and resp["finish"] == "stop":
            return finish("empty_reply", "the model stopped with no reply and no tool call; " + detail)
        return finish("no_submission" if task.submit_tool and task.submitted is None else "wrong_answer", detail)
    return finish("turn_limit", f"{a.max_turns} turns")


def agent(run: Run, slot: int, start: float) -> None:
    if run.stop.wait(max(0.0, start - run.now())):
        return
    while not run.stop.is_set():
        g = run.take()
        if g is None:
            return
        with run.lock:
            run.inflight[slot] = (g, run.now())
        try:
            rec = run_task(run, slot, g)
        except Exception as error:  # noqa: BLE001 - a harness bug must not stall the swarm; it is recorded
            rec = {"g": g, "id": f"g{g}", "family": "?", "tier": "?", "slot": slot, "start": run.now(),
                   "end": run.now(), "outcome": "harness_error", "detail": f"{type(error).__name__}: {error}"}
            log(f"harness error on task {g}: {rec['detail']}")
        with run.lock:
            run.tasks.append(rec)
            run.inflight.pop(slot, None)
            run.server_errors = run.server_errors + 1 if rec["outcome"] == "server_error" else 0
            if run.server_errors >= 3 * run.args.agents and not run.aborted:
                run.aborted = f"{run.server_errors} consecutive tasks ended in server errors"
                log(f"aborting: {run.aborted}")
                run.stop.set()
        if run.stop.wait(run.args.task_gap):
            return


def sampler(run: Run) -> None:
    """Server-side state every 2 s: Gufo /v1/runtime, Prometheus /metrics gauges, local GTT."""
    a = run.args
    gtt = Path("/sys/class/drm/renderD128/device/mem_info_gtt_used")
    while not run.stop.wait(2.0):
        sample = {"t": round(run.now(), 1)}
        if a.gtt and gtt.exists():
            try:
                sample["gtt_bytes"] = int(gtt.read_text())
            except (OSError, ValueError):
                pass
        if a.runtime:
            try:
                req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/runtime", headers=run.headers)
                rt = json.loads(urllib.request.urlopen(req, timeout=5).read())
                sessions = rt.get("sessions") or {}
                sample.update(resident=sessions.get("resident"), decoding=sessions.get("decoding"),
                              queued=(rt.get("queued") or {}).get("default"),
                              kv_used=(rt.get("kv_pool_bytes") or {}).get("used"))
            except (OSError, ValueError):
                pass
        if a.metrics_url:
            try:
                text = urllib.request.urlopen(a.metrics_url, timeout=5).read().decode()
                for name in ("gufo_sessions_active", "gufo_sessions_resident", "gufo_sessions_queued",
                             "gufo_sessions_paged_out", "gufo_kv_pool_bytes_used", "gufo_kv_pool_bytes_limit",
                             "gufo_gpu_gtt_bytes_used", "gufo_host_snapshot_bytes", "gufo_session_pageouts_total",
                             "gufo_session_pageins_total", "gufo_generated_tokens_total",
                             "gufo_draft_tokens_proposed_total", "gufo_draft_tokens_accepted_total"):
                    values = re.findall(rf"^{name}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)
                    if values:  # summed over label sets
                        sample[name] = sum(float(v) for v in values)
            except (OSError, ValueError):
                pass
        with run.lock:
            run.samples.append(sample)


def reasoning_probe(run: Run) -> dict:
    """Does the server's chat template keep earlier reasoning inside a tool loop? Renders the
    same tool-loop history with and without reasoning_content and compares prompt tokens."""
    a = run.args
    task = run.suite.make(0, a.seed)
    reasoning = "I should look at the repository layout first, then read the failing module. " * 6
    call = {"id": "call_probe", "type": "function", "function": {"name": task.tools[0]["function"]["name"],
                                                                  "arguments": "{}"}}
    counts = []
    for keep in (True, False):
        assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
        if keep:
            assistant["reasoning_content"] = reasoning
        messages = [{"role": "system", "content": "You are a test agent."}, {"role": "user", "content": "Start."},
                    assistant, {"role": "tool", "tool_call_id": "call_probe", "content": "README.md"}]
        body = run.body(task, messages, 0)
        body.update(max_tokens=1, stream=False)
        body.pop("stream_options", None)
        try:
            req = urllib.request.Request(run.args.base_url.rstrip("/") + "/v1/chat/completions",
                                         json.dumps(body).encode(), run.headers)
            counts.append((json.loads(urllib.request.urlopen(req, timeout=120).read()).get("usage") or {})
                          .get("prompt_tokens"))
        except (OSError, ValueError) as error:
            return {"error": str(error)[:200]}
    if None in counts:
        return {"error": "no prompt_tokens in usage"}
    return {"with": counts[0], "without": counts[1], "kept": counts[0] > counts[1] + 20}


def server_info(run: Run) -> dict:
    info = {}
    for path in ("/v1/models", "/v1/runtime"):
        try:
            req = urllib.request.Request(run.args.base_url.rstrip("/") + path, headers=run.headers)
            data = json.loads(urllib.request.urlopen(req, timeout=10).read())
            info[path] = data
        except (OSError, ValueError):
            info[path] = None
    return info


def schedule(args) -> list[float]:
    """Slot arrival times: a burst at t=0, then Poisson arrivals spread over the ramp."""
    rng = rng_for("arrivals", args.seed)
    burst = min(args.burst, args.agents)
    times, t = [0.0] * burst, 0.0
    rate = (args.agents - burst) / max(1e-9, args.ramp)
    for _ in range(args.agents - burst):
        t += rng.expovariate(rate)
        times.append(round(min(t, args.ramp), 3))
    return times


def run_cmd(args) -> int:
    args.api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else ""
    args.extra = json.loads(args.extra)
    run = Run(args)
    info = server_info(run)
    if args.model is None:
        models = (info.get("/v1/models") or {}).get("data") or [{}]
        args.model = models[0].get("id", "")
    runtime = info.get("/v1/runtime") or {}
    if not args.context:
        args.context = ((runtime.get("sessions") or {}).get("context_tokens")) or 0
    probe = reasoning_probe(run)
    log(f"reasoning history probe: {probe}")
    arrivals = schedule(args)
    log(f"{args.label}: {args.agents} agents, budget {args.budget:.0f} s, model {args.model!r}, thinking "
        f"{args.thinking}{' effort ' + args.effort if args.effort else ''}, suite {args.suite}, "
        f"queue {queue_digest(args.seed, args.suite)}")
    # SIGTERM/SIGINT end the run early but still write the result, marked aborted
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, _frame: setattr(run, "aborted", f"signal {signum}"))
    run.t0 = time.monotonic()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    threads = [threading.Thread(target=agent, args=(run, i, t), daemon=True) for i, t in enumerate(arrivals)]
    threads.append(threading.Thread(target=sampler, args=(run,), daemon=True))
    for th in threads:
        th.start()
    next_log, next_checkpoint = 60.0, args.checkpoint or float("inf")
    while run.now() < args.budget and not run.aborted:
        time.sleep(max(0.0, min(0.5, args.budget - run.now())))
        if not any(th.is_alive() for th in threads[:-1]):  # a finite queue (--max-tasks) is exhausted
            break
        if run.now() >= next_log:
            with run.lock:
                done = len(run.tasks)
                ok = sum(t["outcome"] == "success" for t in run.tasks)
            log(f"{run.now():.0f} s: {done} tasks graded, {ok} correct, {run.next_task} started")
            next_log += 60.0
        if run.now() >= next_checkpoint:
            write_result(run, args, info, runtime, probe, started, run.now(), partial=True)
            next_checkpoint += args.checkpoint
    with run.lock:
        run.stop.set()
        for conn in list(run.live.values()):
            try:
                conn.sock and conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    for th in threads:
        th.join(timeout=60)
    ended = run.now()
    with run.lock:  # a task whose thread did not finish in time is still recorded, as unfinished
        recorded = {t["g"] for t in run.tasks}
        for slot, (g, start) in sorted(run.inflight.items()):
            if g not in recorded:
                task = run.suite.make(g, args.seed)
                run.tasks.append({"g": g, "id": task.id, "digest": task_digest(task), "family": task.family,
                                  "tier": task.tier, "slot": slot, "start": start, "end": ended, "outcome": "unfinished",
                                  "detail": "still running when the run stopped", "turns": 0, "tool_calls": 0,
                                  "invalid_calls": 0, "nudges": 0, "rejections": 0, "prompt_tokens": 0,
                                  "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                                  "reasoning_from_usage": True, "context_tokens": 0, "final_finish": None,
                                  "model_malformed": 0, "server_parse_failure": 0})
    out = write_result(run, args, info, runtime, probe, started, ended)
    log(f"wrote {out}")
    print(json.dumps(json.loads(out.read_text())["summary"]))
    return 0


def write_result(run: Run, args, info: dict, runtime: dict, probe: dict, started: str, ended: float,
                 partial: bool = False) -> Path:
    """Write the run's result (atomically); a partial result is a checkpoint of finished tasks so far."""
    with run.lock:
        tasks = sorted(run.tasks, key=lambda r: r["g"])
        requests, samples, itl = list(run.requests), list(run.samples), list(run.itl)
    result = {"schema": SCHEMA, "suite": SUITES[args.suite][1], "label": args.label, "note": args.note,
              "started_utc": started, "queue_digest": queue_digest(args.seed, args.suite), "reasoning_probe": probe,
              "aborted": run.aborted,
              "partial": partial,
              "config": {k: getattr(args, k) for k in (
                  "model", "agents", "budget", "warmup", "seed", "suite", "first_task", "max_tasks", "burst", "ramp",
                  "system_tokens", "max_turns",
                  "max_tokens", "temperature", "top_p", "thinking", "effort", "history_reasoning", "tool_time_scale",
                  "task_gap", "extra", "context", "server", "families")},
              "server_info": {"models": [m.get("id") for m in ((info.get("/v1/models") or {}).get("data") or [])],
                              "runtime": {k: runtime.get(k) for k in ("model", "revision", "kv_cache", "speculative",
                                                                       "sessions", "kv_pool_bytes") if k in runtime}},
              "ended_s": round(ended, 1), "tasks": tasks, "requests": requests, "samples": samples,
              "itl_ms": {"p50": pct(itl, 0.5, 1000), "p90": pct(itl, 0.9, 1000), "p99": pct(itl, 0.99, 1000),
                         "n": len(itl)}}
    result["summary"] = summarize(result)
    out = Path(args.out or f"artifacts/swarm/{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out)
    return out


# --------------------------------------------------------------- summary ----

def wilson(successes: int, n: int) -> list:
    if not n:
        return [None, None]
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def summarize(result: dict) -> dict:
    cfg = result["config"]
    budget, warmup = cfg["budget"], cfg["warmup"]
    tasks = result["tasks"]
    graded = [t for t in tasks if t["outcome"] != "unfinished"]
    wins = [t for t in tasks if t["outcome"] == "success"]
    span = budget - warmup
    measured = [t for t in tasks if t["outcome"] == "success" and t["end"] >= warmup]
    reqs = [r for r in result["requests"] if r[2] >= warmup]
    ttft = [r[3] for r in reqs if r[3] is not None]
    server_ttft = [r[12] for r in reqs if len(r) > 12 and r[12] is not None]
    server_itl = [r[13] for r in reqs if len(r) > 13 and r[13] and r[7] > 1]
    queue = [r[14] for r in reqs if len(r) > 14 and r[14] is not None]
    prompt = sum(r[5] for r in reqs)
    cached = sum(r[6] for r in reqs)
    hits = [r[6] / r[5] for r in reqs if r[1] > 0 and r[5]]
    outcomes: dict = {}
    for t in tasks:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    by = {}
    for key in ("family", "tier"):
        by[key] = {}
        for t in graded:
            row = by[key].setdefault(t[key], [0, 0])
            row[0] += t["outcome"] == "success"
            row[1] += 1
    requirements: dict = {}  # "family.name" -> [missed, checked] over graded tasks
    for t in graded:
        for name, met in (t.get("requirements") or {}).items():
            row = requirements.setdefault(f"{t['family']}.{name}", [0, 0])
            if isinstance(met, list):  # [met, total]: one check per item (ticket, invoice, table row)
                row[0] += met[1] - met[0]
                row[1] += met[1]
            else:
                row[0] += not met
                row[1] += 1
    total = lambda field: sum(t.get(field, 0) for t in graded)  # noqa: E731
    calls = total("tool_calls")
    samples = result.get("samples") or []
    peak = lambda key: max((s[key] for s in samples if s.get(key) is not None), default=None)  # noqa: E731
    n_win = len(wins)
    return {
        "budget_binding": bool(len(tasks) - len(graded)) or bool(result.get("aborted")),  # False: the queue ended first
        "correct_per_hour": round(n_win * 3600 / budget, 2),
        "correct_per_hour_ci95": [round(max(0, n_win - 1.96 * math.sqrt(n_win)) * 3600 / budget, 2),
                                  round((n_win + 1.96 * math.sqrt(n_win)) * 3600 / budget, 2)],
        "correct_per_hour_steady": round(len(measured) * 3600 / span, 2) if warmup and span > 0 else None,
        "steady_window_s": [warmup, budget],
        "tasks_started": len(tasks), "tasks_graded": len(graded), "tasks_correct": n_win,
        "tasks_unfinished": len(tasks) - len(graded),
        "success_rate": round(n_win / len(graded), 4) if graded else None,
        "success_rate_ci95": wilson(n_win, len(graded)),
        "completion_tokens_per_correct": round(total("completion_tokens") / n_win) if n_win else None,
        "prefill_tokens_per_correct": round((total("prompt_tokens") - total("cached_tokens")) / n_win) if n_win else None,
        "turns_per_task": round(sum(t["turns"] for t in graded) / len(graded), 2) if graded else None,
        "thinking_tokens_per_task": round(sum(t["reasoning_tokens"] for t in graded) / len(graded)) if graded else None,
        "seconds_per_solved_task": round(sum(t["end"] - t["start"] for t in wins) / n_win, 1) if n_win else None,
        "seconds_per_solved_task_p90": pct([t["end"] - t["start"] for t in wins], 0.9),
        "turns_per_hour": round(len(reqs) * 3600 / span, 1) if span > 0 else None,
        "output_tokens_per_s": round(sum(r[7] for r in reqs) / span, 1) if span > 0 else None,
        "prefill_tokens_per_s": round((prompt - cached) / span, 1) if span > 0 else None,
        "prefix_cache_hit": round(cached / prompt, 4) if prompt else None,
        "prefix_cache_hit_turn2_p50": pct(hits, 0.5),
        "ttft_s": {"p50": pct(ttft, 0.5), "p90": pct(ttft, 0.9), "p99": pct(ttft, 0.99)},
        "server_ttft_s": {"p50": pct(server_ttft, 0.5), "p90": pct(server_ttft, 0.9), "p99": pct(server_ttft, 0.99)},
        "itl_ms": result.get("itl_ms"),
        "server_itl_ms": {"p50": pct(server_itl, 0.5), "p99": pct(server_itl, 0.99)},
        "server_queue_s": {"p50": pct(queue, 0.5), "p90": pct(queue, 0.9), "p99": pct(queue, 0.99)},
        "requests_over_600s": sum(1 for r in reqs if r[4] - r[2] > 600),
        "capacity_rejections": total("rejections"),
        "tool_calls": calls, "invalid_tool_call_rate": round(total("invalid_calls") / calls, 4) if calls else None,
        "model_malformed_rate": round(total("model_malformed") / calls, 4) if calls else None,
        "server_parse_failures": total("server_parse_failure"),
        "template_keeps_reasoning": (result.get("reasoning_probe") or {}).get("kept"),
        "outcomes": outcomes, "by_family": by["family"], "by_tier": by["tier"],
        "requirements": {k: requirements[k] for k in sorted(requirements)},
        "context_tokens_p50": pct([t.get("context_tokens", 0) for t in graded], 0.5),
        "context_tokens_max": max((t.get("context_tokens", 0) for t in tasks), default=None),
        "session_pageouts": (peak("gufo_session_pageouts_total") or 0) - min(
            (x["gufo_session_pageouts_total"] for x in samples if "gufo_session_pageouts_total" in x), default=0),
        "peak": {"gtt_bytes": peak("gtt_bytes"), "kv_pool_bytes": peak("kv_used") or peak("gufo_kv_pool_bytes_used"),
                 "kv_pool_limit": peak("gufo_kv_pool_bytes_limit"), "paged_out": peak("gufo_sessions_paged_out"),
                 "sessions_resident": peak("resident") or peak("gufo_sessions_resident"),
                 "sessions_decoding": peak("decoding") or peak("gufo_sessions_active"),
                 "queued": peak("queued") or peak("gufo_sessions_queued")},
    }


# ---------------------------------------------------------------- report ----

def fmt(value, digits: int = 1, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "–"
    value = value * scale if scale != 1.0 else value
    return (f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}") + suffix


def project(results: list, rates: dict) -> list[str]:
    """Correct answers per hour each run's token costs would reach on Gufo.

    Box seconds per correct answer = kappa * (P / R_p + C / R_d): P and C are the run's newly
    prefilled and generated tokens per correct answer, R_p and R_d Gufo's prefill and decode
    rates for a model of that shape at that swarm width (from `rates`), and kappa calibrates
    this serial cost model on a run measured on Gufo (kappa = measured / modelled box time)."""
    by_label = {r["label"]: r for r in results}
    ref = by_label.get(rates.get("calibrate", ""))
    kappa = None
    if ref and ref["label"] in rates.get("models", {}):
        rr, s = rates["models"][ref["label"]], ref["summary"]
        modelled = s["prefill_tokens_per_correct"] / rr["prefill_tok_s"] + s["completion_tokens_per_correct"] / rr["decode_tok_s"]
        kappa = (3600 / s["correct_per_hour"]) / modelled if s["correct_per_hour"] else None
    lines = ["", f"Projection on Gufo (not a measurement; kappa {fmt(kappa, 2)} from {rates.get('calibrate', '–')}):", "",
             "| Run | Measured correct/h | Prefill tok/correct | Compl. tok/correct | Gufo prefill tok/s | "
             "Gufo decode tok/s | Projected correct/h | Rate source |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:
        rr, s = rates.get("models", {}).get(r["label"]), r["summary"]
        if not rr or not s["tasks_correct"] or kappa is None:
            continue
        seconds = kappa * (s["prefill_tokens_per_correct"] / rr["prefill_tok_s"]
                           + s["completion_tokens_per_correct"] / rr["decode_tok_s"])
        lines.append(f"| {r['label']} | {fmt(s['correct_per_hour'])} | {fmt(s['prefill_tokens_per_correct'])} | "
                     f"{fmt(s['completion_tokens_per_correct'])} | {fmt(rr['prefill_tok_s'])} | {fmt(rr['decode_tok_s'])} | "
                     f"**{fmt(3600 / seconds)}** | {rr.get('source', '')} |")
    return lines


def report_cmd(args) -> int:
    results = [json.loads(Path(p).read_text()) for p in args.results]
    lines = []
    if len({r["queue_digest"] for r in results}) > 1:
        lines.append("**Warning:** results use different task queues; not comparable.\n")
    if len({(r["config"]["budget"], r["config"]["seed"], r["config"]["system_tokens"]) for r in results}) > 1:
        lines.append("**Warning:** budgets, seeds or system prompts differ between results.\n")
    results.sort(key=lambda r: -r["summary"]["correct_per_hour"])
    setting = lambda c: c["thinking"] + (f"/{c['effort']}" if c.get("effort") else "")  # noqa: E731
    lines += ["| Run | Agents | Setting | Correct/h | Steady correct/h | Success (finished) | Unfinished | "
              "Compl. tok/correct | Prefill tok/correct | Turns/task | Think tok/task | s/solved |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        s, c = r["summary"], r["config"]
        ci = s["success_rate_ci95"]
        lines.append(
            f"| {r['label']} | {c['agents']} | {setting(c)} | **{fmt(s['correct_per_hour'])}**"
            f"{'' if s.get('budget_binding', True) else ' †'} "
            f"({fmt(s['correct_per_hour_ci95'][0], 0)}–{fmt(s['correct_per_hour_ci95'][1], 0)}) | "
            f"{fmt(s.get('correct_per_hour_steady'))} | {fmt(s['success_rate'], 1, 100, '%')} "
            f"({fmt(ci[0], 0, 100)}–{fmt(ci[1], 0, 100)}) | {s['tasks_unfinished']} | "
            f"{fmt(s['completion_tokens_per_correct'])} | {fmt(s['prefill_tokens_per_correct'])} | "
            f"{fmt(s['turns_per_task'])} | {fmt(s['thinking_tokens_per_task'])} | {fmt(s['seconds_per_solved_task'])} |")
    lines += ["", "| Run | TTFT p50/p99 s | Server TTFT p50/p99 s | ITL p50/p99 ms | Queue p90 s | Refusals | "
              "Cache hit | Output tok/s | Prefill tok/s | Peak decoding | Peak KV GiB | Peak GTT GiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        s = r["summary"]
        itl, peak = s["itl_ms"] or {}, s["peak"]
        lines.append(
            f"| {r['label']} | {fmt(s['ttft_s']['p50'])}/{fmt(s['ttft_s']['p99'])} | "
            f"{fmt(s['server_ttft_s']['p50'])}/{fmt(s['server_ttft_s']['p99'])} | "
            f"{fmt(itl.get('p50'), 0)}/{fmt(itl.get('p99'), 0)} | {fmt(s['server_queue_s']['p90'])} | "
            f"{s['capacity_rejections']} | {fmt(s['prefix_cache_hit'], 1, 100, '%')} | {fmt(s['output_tokens_per_s'])} | "
            f"{fmt(s['prefill_tokens_per_s'])} | {fmt(peak.get('sessions_decoding'), 0)} | "
            f"{fmt(peak.get('kv_pool_bytes'), 1, 2 ** -30)} | {fmt(peak.get('gtt_bytes'), 1, 2 ** -30)} |")
    fams = sorted({f for r in results for f in r["summary"]["by_family"]})
    lines += ["", "| Run | " + " | ".join(fams) + " | easy | medium | hard | Model malformed | "
              "Server parse failures | Template keeps reasoning | Outcomes |",
              "| --- |" + " ---: |" * (len(fams) + 5) + " --- | --- |"]
    for r in results:
        s = r["summary"]
        cell = lambda row: "–" if not row else f"{row[0]}/{row[1]}"  # noqa: E731
        lines.append(f"| {r['label']} | " + " | ".join(cell(s["by_family"].get(f)) for f in fams) + " | "
                     + " | ".join(cell(s["by_tier"].get(t)) for t in ("easy", "medium", "hard"))
                     + f" | {fmt(s.get('model_malformed_rate'), 2, 100, '%')} | {s.get('server_parse_failures', 0)} | "
                     f"{s.get('template_keeps_reasoning')} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items(), key=lambda kv: -kv[1])) + " |")
    if any(not r["summary"].get("budget_binding", True) for r in results):
        lines += ["", "† The queue ended before the budget: correct/h divides by the configured budget, not by elapsed "
                  "time, so it is not a rate. Compare token costs per correct answer instead."]
    names = sorted({k for r in results for k in r["summary"].get("requirements") or {}})
    if names:  # which requirements discriminate: miss rate per requirement over the tasks that have it
        lines += ["", "Requirement miss rates (missed/checked):", "",
                  "| Requirement | " + " | ".join(r["label"] for r in results) + " |",
                  "| --- |" + " ---: |" * len(results)]
        for name in names:
            cells = []
            for r in results:
                row = (r["summary"].get("requirements") or {}).get(name)
                cells.append("–" if not row else f"{row[0]}/{row[1]} ({100 * row[0] / row[1]:.0f}%)")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if args.rates:
        lines += project(results, json.loads(Path(args.rates).read_text()))
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def trace_cmd(args) -> int:
    """Replay one task's first request many times and save the raw stream of every empty reply (finish=stop,
    no content, no tool call). Each repeat varies the sampling seed unless --same-seed."""
    args.api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else ""
    args.extra = json.loads(args.extra)
    args.first_task, args.skip_graded, args.label = 0, [], getattr(args, "label", "") or "trace"
    args.trace_dir = args.trace_dir or "traces"
    run = Run(args)
    family, index = args.task.rsplit("-", 1)
    task = run.suite.CLASSES[family](int(index), args.seed)
    messages = [{"role": "system", "content": run.systems[task.family]}, {"role": "user", "content": task.prompt}]
    base = run.body(task, messages, 0)
    run.t0 = time.monotonic()

    def once(i: int) -> dict:
        body = dict(base)
        if not args.same_seed:
            body["seed"] = (base["seed"] + i * 7919) % (2 ** 31)
        try:
            resp = run.stream(i, body)
        except (HttpError, OSError, Interrupted) as error:
            return {"i": i, "kind": "error", "detail": str(error)[:200]}
        reasoning = "".join(resp["reasoning"])
        kind = "empty" if run.is_empty(resp) else "tool_call" if resp["calls"] else resp["finish"]
        row = {"i": i, "kind": kind, "finish": resp["finish"], "seed": body["seed"], "event_id": resp.get("event_id"),
               "completion_tokens": (resp["usage"] or {}).get("completion_tokens"), "reasoning_chars": len(reasoning),
               "body_bytes": resp.get("body_bytes"), "sent_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "think_close_in_reasoning": "</think>" in reasoning,
               "tool_markup_in_reasoning": bool(toolcalls.MARKUP.search(reasoning)),
               "content_chars": len("".join(resp["content"]))}
        if kind == "empty":
            row["trace"] = run.dump_trace(task, 0, body, resp, "empty")
        elif i < args.keep_normal:
            row["trace"] = run.dump_trace(task, 0, body, resp, kind)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        rows = list(pool.map(once, range(args.repeat)))
    kinds: dict = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    summary = {"task": task.id, "repeat": args.repeat, "kinds": kinds, "rows": rows}
    log(f"trace {task.id}: {kinds}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def tasks_cmd(args) -> int:
    suite = suite_module(args.suite)
    if args.show:
        family, index = args.show.rsplit("-", 1)
        task = suite.CLASSES[family](int(index), args.seed)
        print(json.dumps({"id": task.id, "tier": task.tier, "prompt": task.prompt,
                          "tools": [t["function"]["name"] for t in task.tools],
                          "oracle": [[{"name": c.name, "arguments": c.arguments} for c in s.calls] or s.content
                                     for s in task.oracle()]}, indent=1, default=str))
        return 0
    for g in range(args.count):
        task = suite.make(g, args.seed)
        print(f"{g:5d} {task.id:18s} {task.tier:6s} {task_digest(task)}")
    print(f"queue {queue_digest(args.seed, args.suite)} (seed {args.seed}, {SUITES[args.suite][1]})")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    shared = argparse.ArgumentParser(add_help=False)  # options that shape a request: run and trace
    shared.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:8011")
    shared.add_argument("--api-key-file")
    shared.add_argument("--model", help="model id to send (default: the first /v1/models entry)")
    shared.add_argument("--seed", type=int, default=1)
    shared.add_argument("--suite", choices=sorted(SUITES), default="v2", help="task corpus: v2, or v3 (the hard tier)")
    shared.add_argument("--system-tokens", type=int, default=20000, help="shared system prompt size per agent type")
    shared.add_argument("--max-tokens", type=int, default=32768, help="per-request output cap")
    shared.add_argument("--temperature", type=float)
    shared.add_argument("--top-p", type=float)
    shared.add_argument("--thinking", choices=["off", "on", "default"], default="off")
    shared.add_argument("--effort", default="", help="reasoning_effort to send (e.g. low, medium, high)")
    shared.add_argument("--history-reasoning", choices=["keep", "drop"], default="keep",
                        help="send reasoning_content of earlier turns back (interleaved thinking) or drop it")
    shared.add_argument("--extra", default="{}", help="JSON merged into every request body")
    shared.add_argument("--read-timeout", type=float, default=3600, help="socket read timeout, s")
    shared.add_argument("--trace-dir", default="", help="save the raw stream of every empty reply (finish=stop, no "
                        "content, no tool call) here")
    r = sub.add_parser("run", help="drive a swarm against an OpenAI-compatible server", parents=[shared])
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="", help="free text recorded with the result (server flags, revision)")
    r.add_argument("--server", default="", help="serving configuration recorded with the result")
    r.add_argument("--out")
    r.add_argument("--agents", type=int, default=32, help="swarm size: concurrent agent slots")
    r.add_argument("--budget", type=float, default=1800, help="wall-clock budget, s; unfinished tasks fail")
    r.add_argument("--warmup", type=float, default=300,
                   help="steady-state correct/h counts completions after this many s (latencies too)")
    r.add_argument("--burst", type=int, default=8, help="slots that start at t=0")
    r.add_argument("--ramp", type=float, default=120, help="the other slots arrive (Poisson) within this many s")
    r.add_argument("--max-turns", type=int, default=None, help="model requests per task (default 40; v3: 60)")
    r.add_argument("--max-tasks", type=int, default=100000, help="queue length")
    r.add_argument("--families", default="",
                   help="comma list of task families to run (e.g. code); the other queue positions are skipped, so "
                        "the pinned order and digest are unchanged")
    r.add_argument("--first-task", type=int, default=0,
                   help="start at this queue index (split one task range over several windows)")
    r.add_argument("--skip-graded", action="append", default=[], metavar="RESULT",
                   help="skip queue indices already graded in this earlier result (repeatable); merge parts after")
    r.add_argument("--tool-time-scale", type=float, default=1.0, help="multiplies the simulated tool latencies")
    r.add_argument("--task-gap", type=float, default=1.0, help="s between a slot's tasks")
    r.add_argument("--checkpoint", type=float, default=300,
                   help="s between partial result writes (finished tasks so far, marked partial); 0 disables")
    r.add_argument("--outage-grace", type=float, default=120,
                   help="s a request keeps retrying refused connections before its task fails as server_error")
    r.add_argument("--context", type=int, default=0, help="server context per session (default: /v1/runtime)")
    r.add_argument("--runtime", action="store_true", help="sample Gufo /v1/runtime")
    r.add_argument("--metrics-url", default="", help="Prometheus /metrics URL to sample")
    r.add_argument("--gtt", action="store_true", help="sample local GPU GTT use from sysfs")
    tr = sub.add_parser("trace", help="replay a task's first request repeatedly, saving the raw stream of every "
                        "empty reply", parents=[shared])
    tr.add_argument("--label", default="trace", help="prefix of the saved trace files")
    tr.add_argument("--task", required=True, help="task id, e.g. code3-00004")
    tr.add_argument("--repeat", type=int, default=20)
    tr.add_argument("--parallel", type=int, default=2, help="concurrent requests (stay at 4 or fewer on a live server)")
    tr.add_argument("--same-seed", action="store_true", help="send the harness seed unchanged on every repeat")
    tr.add_argument("--keep-normal", type=int, default=1, help="also save this many ordinary replies for comparison")
    tr.add_argument("--out", help="write the summary JSON here")
    p = sub.add_parser("report", help="rank results")
    p.add_argument("results", nargs="+")
    p.add_argument("--out")
    p.add_argument("--rates", help="JSON for the Gufo projection: {calibrate: label, models: {label: "
                   "{prefill_tok_s, decode_tok_s, source}}}")
    m = sub.add_parser("merge", help="merge results of one setting run in parts (graded records win)")
    m.add_argument("results", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--label", required=True)
    t = sub.add_parser("tasks", help="print the pinned task queue")
    t.add_argument("--count", type=int, default=32)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--show", help="print one task, e.g. code-00003")
    t.add_argument("--suite", choices=sorted(SUITES), default="v2")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "run" and args.max_turns is None:
        args.max_turns = 60 if args.suite == "v3" else 40
    return {"run": run_cmd, "report": report_cmd, "tasks": tasks_cmd, "merge": merge_cmd,
            "trace": trace_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
