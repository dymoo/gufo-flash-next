"""Serving speed of one OpenAI endpoint: cold prefill, single decode and
aggregate decode under concurrency. Streams every request and records the
client clock plus the server's own usage timings. Thinking is off so decode
lengths are comparable."""
import argparse, json, random, time, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)        # http://host:port/v1
ap.add_argument("--key-file", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefill", default="1024,4096,16384,32768")
ap.add_argument("--concurrency", default="1,2,4,8,16,24")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
KEY = open(a.key_file).read().strip()
TEXT = " ".join(open(p, errors="ignore").read() for p in a.corpus).split()
OUT = open(a.out, "a", buffering=1)


def words(n_tokens, rng):
    # ~0.75 words per token for English prose; the server reports the truth.
    n = int(n_tokens * 0.75)
    start = rng.randrange(0, len(TEXT) - n)
    return " ".join(TEXT[start:start + n])


def call(prompt, max_tokens, nonce):
    body = {"model": a.model, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{nonce}]\n{prompt}"}],
            "max_tokens": max_tokens, "temperature": 0.0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content")):
                    first = time.time()
    end = time.time()
    comp = usage.get("completion_tokens", 0)
    return {"t0": t0, "ttft": (first or end) - t0, "end": end, "total": end - t0,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
            "client_decode_tps": (comp - 1) / (end - first) if first and comp > 1 and end > first else None,
            "server_prompt_tps": usage.get("prompt_tokens_per_second"),
            "server_decode_tps": usage.get("completion_tokens_per_second"),
            "gufo": usage.get("gufo")}


rng = random.Random(7)
call("Say OK.", 4, "warmup")
for n in map(int, a.prefill.split(",")):
    for rep in range(a.reps):
        r = call(words(n, rng) + "\n\nReply with one word.", 1, f"pp{n}-{rep}-{random.random()}")
        r.update(kind="prefill", target=n, rep=rep)
        OUT.write(json.dumps(r) + "\n")
        print("prefill", n, r["prompt_tokens"], round(r["ttft"], 2), r["server_prompt_tps"], flush=True)

LONG = "Write a detailed, continuous story of at least 2000 words about a lighthouse keeper. Do not stop early."
for rep in range(a.reps):
    r = call(LONG, 512, f"tg-{rep}-{random.random()}")
    r.update(kind="decode", rep=rep)
    OUT.write(json.dumps(r) + "\n")
    print("decode", r["completion_tokens"], r["client_decode_tps"], r["server_decode_tps"], flush=True)

for c in map(int, a.concurrency.split(",")):
    results = [None] * c
    def worker(i):
        results[i] = call(words(1024, random.Random(i * 7919 + c)) + "\n\n" + LONG, 256, f"c{c}-{i}-{random.random()}")
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] for r in results)
    # Aggregate decode: tokens over the window where every stream was decoding.
    first_all = max(r["t0"] + r["ttft"] for r in results)
    end_first = min(r["end"] for r in results)
    overlap = sum(r["completion_tokens"] * max(0.0, min(end_first, r["end"]) - first_all) / max(1e-9, r["end"] - (r["t0"] + r["ttft"])) for r in results)
    row = {"kind": "concurrency", "c": c, "wall": wall, "completion_tokens": toks,
           "aggregate_tps_wall": toks / wall,
           "aggregate_tps_steady": overlap / max(1e-9, end_first - first_all) if end_first > first_all else None,
           "mean_stream_tps": sum(r["client_decode_tps"] or 0 for r in results) / c,
           "mean_ttft": sum(r["ttft"] for r in results) / c, "streams": results}
    OUT.write(json.dumps(row) + "\n")
    print("conc", c, round(row["aggregate_tps_wall"], 1), row["aggregate_tps_steady"] and round(row["aggregate_tps_steady"], 1), round(row["mean_stream_tps"], 1), flush=True)
