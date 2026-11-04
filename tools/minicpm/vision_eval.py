#!/usr/bin/env python3
"""Document-vision eval over an OpenAI-compatible server (stdlib only): each image goes as a base64
data URL with its question; greedy, thinking off; exact-match grading (numbers numerically)."""
import argparse, base64, concurrent.futures, json, re, time, urllib.request
from pathlib import Path


def norm(s):
    s = s.strip().strip(".").strip().lower()
    s = re.sub(r"^(the )?(answer|total|value)( is)?:?\s*", "", s)
    return s.replace(",", "").replace("$", "").replace("£", "")


def correct(got, want):
    g, w = norm(got), norm(want)
    try:
        return abs(float(re.findall(r"-?\d+(?:\.\d+)?", g)[0]) - float(w)) < 1e-6 if re.fullmatch(r"-?\d+(\.\d+)?", w) else w in g
    except (IndexError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--dir", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    items = [json.loads(l) for l in (Path(a.dir) / "questions.jsonl").read_text().splitlines() if l.strip()]

    def one(item):
        data = base64.b64encode((Path(a.dir) / item["image"]).read_bytes()).decode()
        body = {"model": a.model, "temperature": 0, "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                                                          {"type": "text", "text": item["question"]}]}]}
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            got = r["choices"][0]["message"].get("content") or ""
            return {**item, "got": got, "passed": correct(got, item["answer"]), "seconds": round(time.monotonic() - t0, 2),
                    "prompt_tokens": r.get("usage", {}).get("prompt_tokens")}
        except Exception as e:  # noqa: BLE001 - a failed request is a failed item
            return {**item, "got": None, "passed": False, "error": str(e)[:200]}

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        rows = list(pool.map(one, items))
    wall = time.monotonic() - t0
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], [0, 0]); k[0] += r["passed"]; k[1] += 1
    res = {"label": a.label, "passed": sum(r["passed"] for r in rows), "total": len(rows), "wall_s": round(wall, 1),
           "images_per_s": round(len(rows) / wall, 2), "kinds": kinds, "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[{a.label}] vision {res['passed']}/{res['total']} in {wall:.0f}s ({res['images_per_s']} img/s); " +
          " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(kinds.items())) +
          f"; errors {sum('error' in r for r in rows)}" + (f" e.g. {next(r['error'] for r in rows if 'error' in r)}" if any('error' in r for r in rows) else ""))


if __name__ == "__main__":
    main()
