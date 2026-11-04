#!/usr/bin/env python3
"""Offline check of tools/swarm: every family's checker accepts its oracle and rejects
a do-nothing agent, the code targets' bugs are caught, and whole runs against a fake
streaming server grade, classify and summarize as specified."""

import http.server
import json
import re
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "swarm"))
import families  # noqa: E402
import families_v3  # noqa: E402
import swarm  # noqa: E402
from families import code  # noqa: E402

SEED = 7
POLICY = {"name": "oracle"}


def check_targets() -> None:
    for name, (module, src, (old, new), visible, weak, hidden, _) in code.TARGETS.items():
        src = textwrap.dedent(src).strip("\n") + "\n"
        assert src.count(old) == 1, name
        bug = src.replace(old, new)

        def passes(body: str, asserts: list) -> bool:
            files = {"pkg/__init__.py": "", "pkg/m.py": "import math\nimport re\n\n" + body,
                     "tests/test_m.py": "from pkg.m import *\n\n\ndef test_x():\n"
                                        + "".join(f"    assert {a}\n" for a in asserts)}
            return code.run_suite(files)[0]
        assert passes(src, visible + weak + hidden), f"{name}: reference fails its tests"
        assert passes(bug, weak), f"{name}: the bug fails the weak tests that report mode shows"
        assert not passes(bug, visible), f"{name}: visible tests miss the bug"
        assert not passes(bug, hidden), f"{name}: hidden tests miss the bug"
    for name, (module, src, stub, visible, hidden) in code.SPECS.items():
        src = textwrap.dedent(src).strip("\n") + "\n"
        for asserts in (visible, hidden):
            files = {"pkg/__init__.py": "", "pkg/m.py": "import math\nimport re\n\n" + src,
                     "tests/test_m.py": code.TEST_HEADER + "from pkg.m import *\n\n" + code.RAISES
                                        + "\n\ndef test_x():\n" + "".join(f"    assert {a}\n" for a in asserts)}
            assert code.run_suite(files)[0], f"{name}: reference fails its specification tests"


def check_families() -> None:
    for g in range(24):
        task = families.make(g, SEED)
        assert not task.check("Done.")[0], f"{task.id}: an agent that does nothing passes"
        twin = families.make(g, SEED)
        assert twin.prompt == task.prompt and swarm.task_digest(twin) == swarm.task_digest(task), task.id
        for step in task.oracle():
            for call in step.calls:
                assert task.invalid(call.name, call.arguments) is None, (task.id, call.name)
                task.call(call.name, call.arguments)
        ok, detail = task.check(task.oracle()[-1].content)
        assert ok, f"{task.id}: oracle fails: {detail}"
    tools = {f: json.dumps(families.CLASSES[f].tools, sort_keys=True) for f in families.FAMILIES}
    for g in range(8):  # the tool set is fixed per agent type, so the shared prompt prefix stays cacheable
        task = families.make(g, SEED + g)
        assert json.dumps(task.tools, sort_keys=True) == tools[task.family]
    block = sorted((families.make(g, SEED).family, families.make(g, SEED).tier) for g in range(families.BLOCK))
    assert block == sorted((families.CLASSES[f](i, SEED).family, families.CLASSES[f](i, SEED).tier)
                           for f in families.FAMILIES for i in range(families.VARIANTS)), "blocks are stratified"
    task = families.CLASSES["extract"](0, SEED)
    assert "required" in task.invalid("submit_invoice", {"vendor": "x"})
    assert task.invalid("nope", {}).startswith("unknown tool")
    assert task.invalid("search_documents", {"query": 3}) == "arguments.query must be a string"


def check_families_v3() -> None:
    """The hard tier: every family x variant (and every code3 combination) accepts its oracle, rejects a
    do-nothing agent and regenerates identically; blocks are stratified; v2 stays pinned."""
    from families_v3 import code as code3

    cases = [(f, i, None) for f in families_v3.FAMILIES for i in range(families_v3.VARIANTS)]
    seen: set = set()  # every scenario at least once (all 39 combinations take too long for one test run)
    for v, kind in enumerate(code3.VARIANTS):
        for combo in code3.COMBOS[kind]:
            if not set(combo) <= seen:
                seen |= set(combo)
                cases.append(("code3", v, combo))
    for family, index, combo in cases:
        make = (lambda: families_v3.CLASSES[family](index, SEED, combo=combo)) if combo else \
            (lambda: families_v3.CLASSES[family](index, SEED))
        task = make()
        assert task.tier == "hard" and not task.check("Done.")[0], f"{task.id}: an agent that does nothing passes"
        twin = make()
        assert twin.prompt == task.prompt and swarm.task_digest(twin) == swarm.task_digest(task), task.id
        for step in task.oracle():
            for call in step.calls:
                assert task.invalid(call.name, call.arguments) is None, (task.id, call.name)
                task.call(call.name, call.arguments)
        ok, detail = task.check(task.oracle()[-1].content)
        assert ok, f"{task.id}: oracle fails: {detail}"
    block = sorted(families_v3.make(g, SEED).family for g in range(families_v3.BLOCK))
    assert block == sorted(families_v3.FAMILIES * families_v3.VARIANTS), "v3 blocks are stratified"
    assert swarm.queue_digest(1) == "e5cdb992da06574e", "the v2 queue changed"
    assert swarm.queue_digest(1, "v3") != swarm.queue_digest(1)


def ticket(messages: list) -> "families.Task":
    family, index = re.search(r"SWARM-(\w+)-(\d+)\.", messages[1]["content"]).groups()
    registry = families_v3.CLASSES if family in families_v3.CLASSES else families.CLASSES
    return registry[family](int(index), SEED)


class Fake(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    rejected: set = set()

    def log_message(self, *args):
        pass

    def send(self, status: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        for k, v in {"Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, {"data": [{"id": "fake"}]} if self.path == "/v1/models" else {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        if not body.get("stream"):  # the reasoning-history probe
            text = json.dumps(messages) if POLICY["name"] == "keep" else json.dumps(
                [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages])
            return self.send(200, {"choices": [{"message": {"role": "assistant", "content": "x"}}],
                                   "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 1}})
        task = ticket(messages)
        turn = sum(m["role"] == "assistant" for m in messages)
        policy = POLICY["name"]
        if policy == "busy" and task.id not in Fake.rejected:
            Fake.rejected.add(task.id)
            return self.send(503, {"error": {"code": "capacity_exhausted"}}, {"Retry-After": "0"})
        if policy == "cut":  # the server dies mid-reply: headers and one chunk, then the connection closes
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"choices": [{"index": 0, "delta": {"content": "Do"}}]}\n\n')
            self.close_connection = True
            return
        if policy == "overflow":
            return self.send(400, {"error": {"message": "prompt exceeds the context length", "code": "context_length"}})
        content, calls, reasoning = "", [], ""
        if policy in ("oracle", "busy", "think"):
            steps = task.oracle()
            step = steps[min(turn, len(steps) - 1)]
            content = step.content
            calls = [{"name": c.name, "arguments": json.dumps(c.arguments)} for c in step.calls]
            reasoning = "Let me think about the next step. " * 3 if policy == "think" else ""
        elif policy == "empty":  # thinking only: finish=stop with no content and no tool call
            reasoning = "Let me think about this. " * 4
        elif policy == "garbage":
            calls = [{"name": "made_up_tool", "arguments": "{not json"}]
        elif policy == "parsefail":  # valid Qwen XML calls the "server" failed to parse
            step = task.oracle()[min(turn, len(task.oracle()) - 1)]
            call = step.calls[0] if step.calls else task.oracle()[0].calls[0]
            content = f"<tool_call>\n<function={call.name}>\n" + "".join(
                f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                for k, v in call.arguments.items()) + "</function>\n</tool_call>"
        elif policy == "markup":
            content = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
        else:
            content = "Done."
        prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4
        chunks = [{"role": "assistant"}]
        chunks += [{"reasoning_content": w} for w in re.findall(r"\S+\s*", reasoning)]
        chunks += [{"content": w} for w in re.findall(r"\S+\s*", content)]
        for i, call in enumerate(calls):  # split arguments to exercise delta assembly
            args = call["arguments"]
            chunks.append({"tool_calls": [{"index": i, "id": f"c{turn}_{i}", "type": "function",
                                           "function": {"name": call["name"], "arguments": args[:len(args) // 2]}}]})
            chunks.append({"tool_calls": [{"index": i, "function": {"arguments": args[len(args) // 2:]}}]})
        events = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in chunks]
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
        events.append({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": len(chunks) - 1,
                                                "prompt_tokens_details": {"cached_tokens": prompt * 3 // 4}}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


def run(server, policy: str, out: Path, budget: float = 3.0, agents: int = 4, extra: list | None = None) -> dict:
    POLICY["name"] = policy
    argv = ["run", "--base-url", f"http://127.0.0.1:{server.server_port}", "--label", policy, "--out", str(out),
            "--agents", str(agents), "--budget", str(budget), "--burst", "2", "--ramp", "1", "--seed", str(SEED),
            "--system-tokens", "300", "--tool-time-scale", "0", "--task-gap", "0", "--max-tasks", "32", "--warmup", "0", *(extra or [])]
    assert swarm.main(argv) == 0
    return json.loads(out.read_text())


def main() -> int:
    check_targets()
    check_families()
    check_families_v3()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        # The run ends when its queue does; a budget well past the ~1 s it needs
        # keeps a loaded CI host from leaving tasks unfinished.
        oracle = run(server, "oracle", out / "oracle.json", budget=30)
        s = oracle["summary"]
        graded = [t for t in oracle["tasks"] if t["outcome"] != "unfinished"]
        assert len(graded) == 32 and all(t["outcome"] == "success" for t in graded), s["outcomes"]
        assert s["success_rate"] == 1.0 and s["correct_per_hour"] > 0 and s["tasks_unfinished"] == 0
        assert oracle["reasoning_probe"]["kept"] is False and s["template_keeps_reasoning"] is False
        assert {t["family"] for t in graded} == set(families.FAMILIES)
        started = sorted(t["g"] for t in oracle["tasks"])
        assert started == list(range(len(started))), "slots must take the queue in order"
        for t in oracle["tasks"]:
            assert t["digest"] == swarm.task_digest(families.make(t["g"], SEED))
        assert abs(s["prefix_cache_hit"] - 0.75) < 0.01 and s["tool_calls"] > 0 and s["invalid_tool_call_rate"] == 0
        assert s["completion_tokens_per_correct"] > 0 and s["turns_per_task"] >= 2
        assert oracle["queue_digest"] == swarm.queue_digest(SEED)

        v3 = run(server, "oracle", out / "v3.json", budget=60, extra=["--suite", "v3", "--max-tasks", "16"])
        graded = [t for t in v3["tasks"] if t["outcome"] != "unfinished"]
        assert len(graded) == 16 and all(t["outcome"] == "success" for t in graded), v3["summary"]["outcomes"]
        assert {t["family"] for t in graded} == set(families_v3.FAMILIES) and v3["config"]["max_turns"] == 60
        assert v3["queue_digest"] == swarm.queue_digest(SEED, "v3")
        met = lambda v: v[0] == v[1] if isinstance(v, list) else v  # noqa: E731  [right, checked] or bool
        assert all(t["requirements"] and all(met(v) for v in t["requirements"].values()) for t in graded), \
            "v3 records requirements"
        assert any(isinstance(v, list) for t in graded for v in t["requirements"].values()), "counted requirements"
        reqs = v3["summary"]["requirements"]
        assert reqs and all(missed == 0 for missed, _ in reqs.values()), reqs
        assert swarm.main(["report", str(out / "v3.json"), "--out", str(out / "v3.md")]) == 0
        assert "Requirement miss rates" in (out / "v3.md").read_text()
        for t in v3["tasks"]:
            assert t["digest"] == swarm.task_digest(families_v3.make(t["g"], SEED))

        think = run(server, "think", out / "think.json", extra=["--thinking", "on", "--effort", "high"])
        assert think["summary"]["thinking_tokens_per_task"] > 0, think["summary"]
        assert think["summary"]["success_rate"] == 1.0
        POLICY["name"] = "keep"
        args = swarm.parse_args(["run", "--base-url", f"http://127.0.0.1:{server.server_port}", "--label", "p",
                                 "--system-tokens", "300"])
        args.api_key, args.extra = "", {}
        assert swarm.reasoning_probe(swarm.Run(args))["kept"] is True

        later = run(server, "oracle", out / "later.json", budget=30, extra=["--first-task", "32"])
        assert sorted(t["g"] for t in later["tasks"]) == list(range(32, 64)), "--first-task offsets the queue"

        part1 = run(server, "oracle", out / "part1.json", budget=1.5, agents=2, extra=["--max-tasks", "6"])
        rest = run(server, "oracle", out / "part2.json", budget=30, extra=["--skip-graded", str(out / "part1.json")])
        done1 = {t["g"] for t in part1["tasks"] if t["outcome"] != "unfinished"}
        assert done1 and not done1 & {t["g"] for t in rest["tasks"]}, (done1, sorted(t["g"] for t in rest["tasks"]))
        assert swarm.main(["merge", str(out / "part1.json"), str(out / "part2.json"), "--out", str(out / "m.json"),
                           "--label", "m"]) == 0
        merged = json.loads((out / "m.json").read_text())
        assert sorted(t["g"] for t in merged["tasks"]) == list(range(32)) and merged["summary"]["success_rate"] == 1.0

        busy = run(server, "busy", out / "busy.json")
        assert busy["summary"]["capacity_rejections"] > 0 and busy["summary"]["success_rate"] == 1.0

        null = run(server, "null", out / "null.json", budget=2)
        assert null["summary"]["tasks_correct"] == 0
        assert set(null["summary"]["outcomes"]) <= {"wrong_answer", "no_submission", "unfinished"}, null["summary"]
        assert any(t["nudges"] == 1 for t in null["tasks"])

        garbage = run(server, "garbage", out / "garbage.json", budget=2)
        assert set(garbage["summary"]["outcomes"]) <= {"model_malformed", "unfinished"}, garbage["summary"]
        assert garbage["summary"]["invalid_tool_call_rate"] == 1.0 and garbage["summary"]["model_malformed_rate"] == 1.0
        markup = run(server, "markup", out / "markup.json", budget=2)
        assert set(markup["summary"]["outcomes"]) <= {"model_malformed", "unfinished"}, markup["summary"]
        parsefail = run(server, "parsefail", out / "parsefail.json", budget=2)
        assert set(parsefail["summary"]["outcomes"]) <= {"server_parse_failure", "unfinished"}, parsefail["summary"]
        assert parsefail["summary"]["server_parse_failures"] > 0 and parsefail["summary"]["model_malformed_rate"] in (0, None)
        empty = run(server, "empty", out / "empty.json", budget=2, extra=["--trace-dir", str(out / "traces")])
        assert set(empty["summary"]["outcomes"]) <= {"empty_reply", "no_submission", "unfinished"}, empty["summary"]
        assert empty["summary"]["outcomes"].get("empty_reply", 0) > 0, "a code task that ends on a thinking-only reply"
        assert any(t.get("last_content") == "" for t in empty["tasks"] if t["outcome"] == "empty_reply")
        traces = sorted((out / "traces").glob("*.json"))
        assert traces, "empty replies are saved"
        saved = json.loads(traces[0].read_text())
        assert saved["response"]["finish"] == "stop" and saved["response"]["content"] == "" \
            and saved["response"]["reasoning_chars"] > 0 and saved["raw_events"] and "Authorization" not in json.dumps(saved)
        POLICY["name"] = "empty"
        assert swarm.main(["trace", "--base-url", f"http://127.0.0.1:{server.server_port}", "--model", "fake",
                           "--task", "code-00000", "--repeat", "3", "--parallel", "2", "--system-tokens", "300",
                           "--trace-dir", str(out / "traces2"), "--out", str(out / "trace.json")]) == 0
        assert json.loads((out / "trace.json").read_text())["kinds"] == {"empty": 3}
        assert len(list((out / "traces2").glob("*.json"))) == 3
        overflow = run(server, "overflow", out / "overflow.json", budget=1.5)
        assert set(overflow["summary"]["outcomes"]) <= {"context_overflow", "unfinished"}, overflow["summary"]

        slow = run(server, "oracle", out / "slow.json", budget=2, agents=2,
                   extra=["--tool-time-scale", "20", "--checkpoint", "0.6"])  # tool waits outlast the budget
        assert slow["partial"] is False
        assert slow["summary"]["tasks_unfinished"] == 2 and slow["summary"]["success_rate"] is None

        cut = run(server, "cut", out / "cut.json", budget=3, extra=["--outage-grace", "0.5"])
        assert "wrong_answer" not in cut["summary"]["outcomes"] and cut["summary"]["tasks_correct"] == 0, \
            cut["summary"]["outcomes"]

        dead = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        port = dead.server_port
        dead.server_close()  # nothing listens: every connection is refused
        POLICY["name"] = "oracle"
        assert swarm.main(["run", "--base-url", f"http://127.0.0.1:{port}", "--label", "down", "--out", str(out / "down.json"),
                           "--agents", "2", "--budget", "30", "--warmup", "0", "--system-tokens", "300",
                           "--outage-grace", "0.5", "--task-gap", "0"]) == 0
        down = json.loads((out / "down.json").read_text())
        assert down["aborted"] and down["summary"]["outcomes"]["server_error"] == 6, down["summary"]["outcomes"]

        report = out / "report.md"
        rates = out / "rates.json"
        rates.write_text(json.dumps({"calibrate": "oracle", "models": {
            "oracle": {"prefill_tok_s": 1000.0, "decode_tok_s": 100.0},
            "think": {"prefill_tok_s": 500.0, "decode_tok_s": 50.0}}}))
        assert swarm.main(["report", str(out / "oracle.json"), str(out / "null.json"), str(out / "think.json"),
                           "--rates", str(rates), "--out", str(report)]) == 0
        text = report.read_text()
        assert text.index("| oracle |") < text.index("| null |"), "report ranks by correct answers per hour"
        oracle_row = [row for row in text.split("Projection")[1].splitlines() if row.startswith("| oracle |")][0]
        assert f"**{oracle['summary']['correct_per_hour']:,.1f}**" in oracle_row, "calibration reproduces its run"
    server.shutdown()
    print("swarm_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
