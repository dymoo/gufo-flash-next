#!/usr/bin/env python3
"""Offline check of tools/qwen-flash/qa: metrics, graders, and one baseline/candidate
round trip (teacher-forced logits, needle, tasks, determinism) against a fake server."""

import http.server
import json
import math
import re
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen-flash" / "qa"))
import qa  # noqa: E402
import tasks  # noqa: E402

PERTURB = {"value": 0.0}


class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"data": [{"id": "fake"}]} if self.path == "/v1/models" else {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "".join(m.get("content") or "" for m in body["messages"])
        forced = body.get("forced_completion")
        if forced is not None:
            tokens = forced if isinstance(forced, list) else [ord(c) % 997 for c in forced]
            text = "".join(chr(97 + t % 26) for t in tokens)
        else:
            codes = "\n".join(f"{n}: {c}" for n, c in re.findall(r"code for (.+?) is (\d+)\.", prompt))
            text = f"{codes}\nAnswer: 42 ```python\ndef f():\n    return 1\n```"
            tokens = [ord(c) % 997 for c in text][: body.get("max_tokens") or 64]
        entries = []
        for position, token in enumerate(tokens):
            lp = -0.2 - (token % 5) * 0.05 + PERTURB["value"] * math.sin(position)
            top = [[token, lp], [token + 1, lp - 1.5 - PERTURB["value"]]]
            entries.append({"id": token, "token": chr(97 + token % 26), "logprob": lp, "bytes": [97],
                            "top_logprobs": [{"id": t, "token": "x", "logprob": v, "bytes": [120]}
                                             for t, v in top[: body.get("top_logprobs", 0)]]})
        choice = {"message": {"role": "assistant", "content": text}, "finish_reason": "length"}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": entries}
        self.reply(200, {"choices": [choice], "usage": {
            "prompt_tokens": len(prompt) // 4 + 12, "completion_tokens": len(tokens), "cached_tokens": 0,
            "gufo": {"prefill_tokens": len(prompt) // 4 + 12, "prefill_ms": 10.0, "decode_ms": 5.0}}})


def check_metrics():
    same = [[1, -0.1], [2, -2.5]]
    assert qa.topk_kl(same, same) < 1e-12
    shifted = qa.topk_kl(same, [[2, -0.1], [1, -2.5]])
    assert shifted > 1.0, shifted
    ref = [[[1, -0.1, same]], [[3, -0.2, [[3, -0.2], [4, -2.0]]]]]
    m = qa.compare_positions(ref, ref)
    assert m["kl_mean"] == 0 and m["top1_agree"] == 1 and m["ppl_delta_pct"] == 0
    assert qa.compare_positions(ref, [[[9, -0.1, same]], ref[1]])["misaligned_cases"] == 1
    assert qa.verdict("kl_mean", 0.001) == "PASS" and qa.verdict("kl_mean", 0.01) == "WARN"
    assert qa.verdict("top1_agree", 0.9) == "FAIL"
    assert tasks.extract_integer("so 12 + 30\nAnswer: 2,842.") == 2842
    assert tasks.extract_integer("the result is 58") == 58
    assert tasks.json_ok('```json\n{"a": 1}\n```', lambda v: v == {"a": 1})
    assert not tasks.json_ok("{broken", lambda v: True)
    call = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "get_weather", "arguments": '{"city": "Paris, FR", "unit": "celsius"}'}}]}}]}
    assert tasks.tool_ok(call, "get_weather", {"city": ["paris"], "unit": "celsius"})
    assert not tasks.tool_ok(call, "get_weather", {"city": ["rome"], "unit": "celsius"})
    assert tasks.run_python("def f():\n    return 1", ["assert f() == 1"])
    assert not tasks.run_python("def f():\n    return 2", ["assert f() == 1"])


def main():
    check_metrics()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "corpus"
        data.mkdir()
        words = " ".join(f"word{i % 97} and. then" for i in range(40000))
        for name in qa.BOOKS:
            (data / f"{name}.txt").write_text(words)
            qa.BOOKS[name] = (qa.BOOKS[name][0], None)
        common = ["run", "--base-url", url, "--quick", "--sessions", "2", "--data-dir", str(data),
                  "--suites", "logits,needle,tasks,determinism,batch"]
        assert qa.main(common + ["--label", "base", "--out", f"{tmp}/base.json"]) == 0
        PERTURB["value"] = 0.3
        assert qa.main(common + ["--label", "cand", "--out", f"{tmp}/cand.json"]) == 0
        base = json.loads(Path(f"{tmp}/base.json").read_text())
        cand = json.loads(Path(f"{tmp}/cand.json").read_text())
        assert base["calibration"]["logprobs"] is True, base["calibration"]
        suites = base["suites"]
        assert suites["logits"]["status"] == "ok" and len(suites["logits"]["cases"]) == 3, suites["logits"]
        assert suites["needle"]["found"] == suites["needle"]["total"] == 3, suites["needle"]
        assert suites["determinism"]["pass"] is True
        assert suites["batch"]["pass"] is True and suites["batch"]["levels"]["2"]["logits"]["kl_mean"] == 0, suites["batch"]
        assert qa.compare_positions([c["positions"] for c in suites["logits"]["cases"]],
                                    [c["positions"] for c in cand["suites"]["logits"]["cases"]])["kl_mean"] > 0
        text = qa.report(base, cand)
        assert "KL mean, moby-2048" in text and "needle found | 3/3 | 3/3" in text, text
        assert "FAIL" in text or "WARN" in text, text
        if "-v" in sys.argv:
            print(text)
        assert "PASS" in qa.report(base, base).split("\n")[2], qa.report(base, base)
    server.shutdown()
    print("qwen_flash_qa_test: ok")


if __name__ == "__main__":
    main()
