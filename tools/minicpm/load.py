#!/usr/bin/env python3
"""Runtime-agnostic streaming load generator for OpenAI-compatible servers (stdlib only).

closed: C users each send N requests back to back (C1..C128 sweeps).
open:   Poisson arrivals at each rate for a fixed duration (goodput under SLO).
Every prompt is unique (random corpus slice) so prefix caches cannot flatter results.
Timing is client-side from SSE: TTFT = first content delta, ITL = decode span / (tokens - 1).
"""
import argparse, http.client, json, random, statistics, threading, time, urllib.parse
from pathlib import Path

SHAPES = {  # name: (approx prompt tokens, max output tokens, instruction)
    "chat":   (1000, 300, "Summarise the passage above for a product manager in clear bullet points, then list five follow-up questions."),
    "decode": (200, 512, "Rewrite the passage above as a long, detailed modern short story. Keep going until you run out of space."),
    "rag":    (6000, 300, "Using only the document above, answer: who are the main characters, what do they want, and what happens? Cite short quotes."),
    "code":   (800, 512, "Write a well-documented Python module that indexes the passage above: tokenise it, build an inverted index, and expose a search(query) function with ranking. Include unit tests."),
    "extract": (1500, 150, "Extract every person and place named in the passage above as JSON: {\"people\": [...], \"places\": [...]}. Output only JSON."),
}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 2)


class Gen:
    def __init__(self, a):
        self.a = a
        self.url = urllib.parse.urlsplit(a.base_url)
        self.key = Path(a.api_key_file).read_text().strip() if a.api_key_file else ""
        self.text = "\n".join(Path(p).read_text(errors="replace") for p in a.corpus)
        self.chars_per_token = 4.0
        self.rng = random.Random(a.seed)
        self.lock = threading.Lock()

    def prompt(self, shape):
        tokens, max_out, instruction = SHAPES[shape]
        n = int(tokens * self.chars_per_token)
        with self.lock:
            start = self.rng.randrange(0, max(1, len(self.text) - n))
            tag = self.rng.randrange(1 << 30)
        return [{"role": "system", "content": f"You are a helpful assistant for a SaaS product. Request {tag}."},
                {"role": "user", "content": self.text[start:start + n] + "\n\n" + instruction}], max_out

    def one(self, shape):
        messages, max_out = self.prompt(shape)
        body = {"model": self.a.model, "messages": messages, "max_tokens": max_out,
                "temperature": self.a.temperature, "stream": True, "stream_options": {"include_usage": True}}
        if self.a.top_p < 1:
            body["top_p"] = self.a.top_p
        if self.a.no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = {"shape": shape, "t0": time.monotonic(), "ttft": None, "out": 0, "in": None, "deltas": 0, "error": None}
        try:
            c = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=self.a.timeout)
            c.request("POST", "/v1/chat/completions", json.dumps(body), headers)
            resp = c.getresponse()
            if resp.status != 200:
                r["error"] = f"HTTP {resp.status}: {resp.read()[:200].decode(errors='replace')}"
                return r
            for line in resp:
                if not line.startswith(b"data: {"):
                    continue
                ev = json.loads(line[6:])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        r["deltas"] += 1
                        if r["ttft"] is None:
                            r["ttft"] = time.monotonic() - r["t0"]
                        r["t_last"] = time.monotonic() - r["t0"]
                if ev.get("usage"):
                    r["out"] = ev["usage"].get("completion_tokens") or 0
                    r["in"] = ev["usage"].get("prompt_tokens")
            c.close()
        except Exception as e:  # noqa: BLE001 - a failed request is a result
            r["error"] = f"{type(e).__name__}: {e}"[:200]
        r["e2e"] = time.monotonic() - r["t0"]
        r["out"] = r["out"] or r["deltas"]
        if r["ttft"] is not None and r["out"] > 1:
            r["itl"] = (r["t_last"] - r["ttft"]) / (r["out"] - 1)
        return r

    def calibrate(self, shape):
        r = self.one(shape)
        if r.get("in"):
            self.chars_per_token = SHAPES[shape][0] * self.chars_per_token / r["in"]
        return r

    def summarise(self, rs, wall, extra):
        ok = [r for r in rs if not r["error"] and r["ttft"] is not None]
        out = sum(r["out"] for r in ok)
        prompt = sum(r["in"] or 0 for r in ok)
        itl = [r["itl"] * 1000 for r in ok if r.get("itl")]
        slo = [r for r in ok if r["ttft"] <= self.a.slo_ttft and r.get("itl", 0) * 1000 <= self.a.slo_itl_ms]
        return {**extra, "requests": len(rs), "errors": len(rs) - len(ok), "wall_s": round(wall, 2),
                "out_tok_s": round(out / wall, 1), "prompt_tok_s": round(prompt / wall, 1),
                "req_s": round(len(ok) / wall, 3), "goodput_req_s": round(len(slo) / wall, 3),
                "mean_out_tokens": round(out / max(1, len(ok)), 1), "mean_in_tokens": round(prompt / max(1, len(ok)), 1),
                "ttft_p50_s": pct([r["ttft"] for r in ok], 50), "ttft_p95_s": pct([r["ttft"] for r in ok], 95),
                "itl_p50_ms": pct(itl, 50), "itl_p95_ms": pct(itl, 95),
                "per_user_tok_s_p50": pct([1000 / x for x in itl], 50),
                "sample_errors": list({r["error"] for r in rs if r["error"]})[:3]}

    def closed(self, shape, users, per_user):
        rs, threads = [], []
        barrier = threading.Barrier(users)

        def user():
            barrier.wait()
            for _ in range(per_user):
                rs.append(self.one(shape))
        t0 = time.monotonic()
        for _ in range(users):
            threads.append(threading.Thread(target=user)); threads[-1].start()
        for t in threads:
            t.join()
        return self.summarise(rs, time.monotonic() - t0, {"mode": "closed", "shape": shape, "users": users,
                                                         "t_end": time.time(), "t_start": time.time() - (time.monotonic() - t0)})

    def open(self, shape, rate, duration):
        rs, threads = [], []
        t0 = time.monotonic()
        nxt = t0
        while nxt - t0 < duration:
            time.sleep(max(0.0, nxt - time.monotonic()))
            threads.append(threading.Thread(target=lambda: rs.append(self.one(shape)))); threads[-1].start()
            with self.lock:
                nxt += self.rng.expovariate(rate)
        for t in threads:
            t.join()
        return self.summarise(rs, time.monotonic() - t0, {"mode": "open", "shape": shape, "rate": rate,
                                                         "t_end": time.time(), "t_start": time.time() - (time.monotonic() - t0)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--api-key-file")
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--shapes", default="chat")
    p.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    p.add_argument("--per-user", type=int, default=2, help="closed: requests per user (min total 8)")
    p.add_argument("--rates", default="", help="open: comma list of req/s")
    p.add_argument("--rates-from", help="open: scale --rate-fracs by the peak closed req/s for the shape in this JSON")
    p.add_argument("--rate-fracs", default="0.25,0.5,0.75,1.0,1.25")
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--slo-ttft", type=float, default=3.0)
    p.add_argument("--slo-itl-ms", type=float, default=100.0)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy; sampled runs exercise the sampler")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    g = Gen(a)
    out = Path(a.out)
    res = {"label": a.label, "base_url": a.base_url, "model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "slo": {"ttft_s": a.slo_ttft, "itl_ms": a.slo_itl_ms}, "runs": []}
    for shape in a.shapes.split(","):
        cal = g.calibrate(shape)
        g.one(shape)  # warm
        print(f"[{a.label}] {shape} calibrated chars/token={g.chars_per_token:.2f} err={cal['error']}", flush=True)
        for c in [int(x) for x in a.concurrency.split(",") if x]:
            s = g.closed(shape, c, max(a.per_user, -(-8 // c)))
            res["runs"].append(s); out.write_text(json.dumps(res, indent=1))
            print(f"[{a.label}] {shape} C{c}: {s['out_tok_s']} out tok/s, ttft p50 {s['ttft_p50_s']}s, itl p50 {s['itl_p50_ms']}ms, err {s['errors']}", flush=True)
        rates = [float(x) for x in a.rates.split(",") if x]
        if a.rates_from:
            peak = max(r["req_s"] for r in json.loads(Path(a.rates_from).read_text())["runs"]
                       if r["mode"] == "closed" and r["shape"] == shape)
            rates = [round(peak * float(f), 3) for f in a.rate_fracs.split(",")]
        for rate in rates:
            s = g.open(shape, rate, a.duration)
            res["runs"].append(s); out.write_text(json.dumps(res, indent=1))
            print(f"[{a.label}] {shape} open {rate}/s: goodput {s['goodput_req_s']} req/s, ttft p95 {s['ttft_p95_s']}s, itl p95 {s['itl_p95_ms']}ms, err {s['errors']}", flush=True)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
