#!/usr/bin/env python3
"""Checks that an abliterated Flash-Next is a rank-1 edit W' = W - r (r^T W) of the base.

Reads only the tensors it compares, by HTTP range requests: the base from Qwen's BF16
safetensors and the abliterated weights from a GGUF of them (Q8_0 dense, Q5_1 routed
down projections). For every writer class it reports

  strength  lambda = <r^T (W - W'), r^T W> / |r^T W|^2      (1 for a full ablation)
  leakage   |r^T W'| / |r^T W|                              (quantization floor only)
  cos       |cos| of the top left singular vector of W - W' with r
  rank-1    |(I - r r^T)(W - W')| / |W - W'|               (quantization floor only)
  q8 match  share of Q8_0 codes equal to llama.cpp's Q8_0 of bf16(W - r r^T W)

and the same with the rows' roles swapped for the embedding (E' = E - (E r) r^T).
Untouched tensors (lm_head, router, input projections) report leakage ~1.

  abliteration_check.py --direction orca_direction.safetensors \
      --gguf URL [--mtp-gguf URL] [--cache DIR] [--json OUT]
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

QWEN = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/"
CACHE = "/tmp/abliteration-check"


def auth_headers():
    """A Hugging Face token from HF_TOKEN, for gated repositories."""
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch(url, a, b):
    """Bytes [a, b) of url, cached on disk."""
    key = hashlib.sha256(f"{url} {a} {b}".encode()).hexdigest()[:32]
    path = os.path.join(CACHE, key)
    if os.path.exists(path) and os.path.getsize(path) == b - a:
        return open(path, "rb").read()
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", **auth_headers()})
            data = urllib.request.urlopen(req, timeout=600).read()
            if len(data) == b - a:
                break
        except Exception as e:  # noqa: BLE001 - retried
            print(f"  retry {attempt}: {e}", file=sys.stderr)
    else:
        raise RuntimeError(f"range {a}-{b} of {url}")
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return data


# ---------------------------------------------------------------- safetensors
class Safetensors:
    def __init__(self, base):
        self.base = base
        self.index = json.loads(fetch_all(base + "model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def header(self, file):
        if file not in self.headers:
            n = struct.unpack("<Q", fetch(self.base + file, 0, 8))[0]
            self.headers[file] = (json.loads(fetch(self.base + file, 8, 8 + n)), 8 + n)
        return self.headers[file]

    def shape(self, name):
        h, _ = self.header(self.index[name])
        return h[name]["shape"]

    def rows(self, name, first, count):
        """Rows [first, first + count) of the tensor's leading dimension, as float32."""
        file = self.index[name]
        h, base = self.header(file)
        info = h[name]
        assert info["dtype"] == "BF16", info["dtype"]
        row = int(np.prod(info["shape"][1:])) * 2
        s, _ = info["data_offsets"]
        raw = fetch(self.base + file, base + s + first * row, base + s + (first + count) * row)
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape([count] + info["shape"][1:])


def fetch_all(url):
    req = urllib.request.Request(url, headers=auth_headers())
    return urllib.request.urlopen(req, timeout=600).read()


# ----------------------------------------------------------------------- GGUF
GGUF_SIZES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 7: (32, 24), 30: (1, 2)}  # f32 f16 q8_0 q5_1 bf16


class Gguf:
    def __init__(self, url):
        self.url = url
        self.buf = b""
        self.tensors = {}
        self._parse()

    def _need(self, end):
        while len(self.buf) < end:
            size = max(len(self.buf), 8 << 20)
            self.buf += fetch(self.url, len(self.buf), len(self.buf) + size)

    def _parse(self):
        pos = 0

        def take(n):
            nonlocal pos
            self._need(pos + n)
            pos += n
            return self.buf[pos - n:pos]

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def string():
            return take(u64()).decode("utf-8", "replace")

        scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = u32(), u64()
                if et in scalar:
                    take(scalar[et] * n)
                    return None
                return [value(et) for _ in range(n)]
            raw = take(scalar[t])
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                   10: "<Q", 11: "<q", 12: "<d"}[t]
            return struct.unpack(fmt, raw)[0]

        assert take(4) == b"GGUF"
        u32()
        n_tensors, n_kv = u64(), u64()
        self.kv = {}
        for _ in range(n_kv):
            k = string()
            v = value(u32())
            if not isinstance(v, list) or len(v) < 64:
                self.kv[k] = v
        infos = []
        for _ in range(n_tensors):
            name = string()
            dims = [u64() for _ in range(u32())]
            infos.append((name, dims, u32(), u64()))
        align = self.kv.get("general.alignment", 32)
        data = (pos + align - 1) // align * align
        for name, dims, t, off in infos:
            self.tensors[name] = (dims, t, data + off)

    def rows(self, name, first, count, slab=None):
        """float32 rows [first, first + count) of a 2D tensor, or of 3D slab `slab`."""
        dims, t, off = self.tensors[name]
        cols = dims[0]
        block, size = GGUF_SIZES[t]
        row = cols // block * size
        if slab is not None:
            off += slab * dims[1] * row
        raw = fetch(self.url, off + first * row, off + (first + count) * row)
        return dequant(raw, t, count, cols), raw


def dequant(raw, t, rows, cols):
    if t == 0:
        return np.frombuffer(raw, "<f4").reshape(rows, cols).copy()
    if t == 1:
        return np.frombuffer(raw, "<f2").astype(np.float32).reshape(rows, cols)
    if t == 30:
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
    b = np.frombuffer(raw, np.uint8)
    if t == 8:
        b = b.reshape(-1, 34)
        d = b[:, :2].copy().view("<f2").astype(np.float32)
        q = b[:, 2:].view(np.int8).astype(np.float32)
        return (d * q).reshape(rows, cols)
    if t == 7:
        b = b.reshape(-1, 24)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        m = b[:, 2:4].copy().view("<f2").astype(np.float32)
        qh = b[:, 4:8].copy().view("<u4")[:, 0]
        qs = b[:, 8:24]
        lo = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.uint32)
        bits = (qh[:, None] >> np.arange(32, dtype=np.uint32)) & 1
        q = lo | (bits << 4)
        return (d * q + m).astype(np.float32).reshape(rows, cols)
    raise ValueError(f"ggml type {t}")


def q8_0(x):
    """llama.cpp quantize_row_q8_0_ref codes of float32 rows (fp16 scale, int8)."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0).astype(np.float32)
    q = np.round(b * inv[:, None])  # roundf: ties away from zero, below
    q = np.where(np.abs(b * inv[:, None] - np.trunc(b * inv[:, None])) == 0.5,
                 np.trunc(b * inv[:, None]) + np.sign(b), q)
    return d.astype(np.float16), q.astype(np.int8)


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


# --------------------------------------------------------------------- checks
def column_space(w, wo, r, raw_q8=None):
    """W: [out=hidden, in]; the edit acts on the output (row) space."""
    rw = r @ w
    d = w - wo
    rd = r @ d
    out = {
        "strength": float(rd @ rw / (rw @ rw)),
        "leakage": float(np.linalg.norm(r @ wo) / np.linalg.norm(rw)),
        "rank1_residual": float(np.linalg.norm(d - np.outer(r, rd)) / np.linalg.norm(d)),
        "rel_edit": float(np.linalg.norm(np.outer(r, rw)) / np.linalg.norm(w)),
    }
    u, _, _ = np.linalg.svd(d.astype(np.float64), full_matrices=False)
    out["cos_top_singular"] = float(abs(u[:, 0] @ r))
    if raw_q8 is not None:
        _, q = q8_0(bf16(w - np.outer(r, rw)))
        got = np.frombuffer(raw_q8, np.uint8).reshape(-1, 34)[:, 2:].view(np.int8)
        out["q8_code_match"] = float((q.reshape(got.shape) == got).mean())
        _, q_base = q8_0(w)
        out["q8_code_match_unedited"] = float((q_base.reshape(got.shape) == got).mean())
    return out


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--direction-key", default="orca_refusal_direction")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mtp-gguf")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--json")
    a = ap.parse_args()
    CACHE = a.cache
    raw = open(a.direction, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    h = json.loads(raw[8:8 + n])
    s, e = h[a.direction_key]["data_offsets"]
    r = np.frombuffer(raw[8 + n + s:8 + n + e], "<f4").astype(np.float64)
    r = (r / np.linalg.norm(r)).astype(np.float32)

    base = Safetensors(QWEN)
    g = Gguf(a.gguf)
    m = Gguf(a.mtp_gguf) if a.mtp_gguf else None
    L = "model.language_model."
    results = {}

    def dense(label, hf, gg, gguf=g, slab=None, expert=None, head_block=None):
        if hf not in base.index or gg not in gguf.tensors:
            print(f"{label}: missing ({hf in base.index}, {gg in gguf.tensors})")
            return
        dims, t, _ = gguf.tensors[gg]
        if expert is None:
            w = base.rows(hf, 0, base.shape(hf)[0])
        else:
            w = base.rows(hf, expert, 1)[0]
        if w.shape[0] != dims[1]:
            w = w.T.copy()  # [in, out] storage
        wo, raw_t = gguf.rows(gg, 0, dims[1], slab)
        if head_block:
            # The converter reorders GDN value heads (input columns); match
            # each GGUF head block to its base block before comparing.
            heads = w.shape[1] // head_block
            wb = w.reshape(w.shape[0], heads, head_block)
            ob = wo.reshape(wo.shape[0], heads, head_block)
            score = np.einsum("rhc,rkc->hk", ob, wb)
            order = score.argmax(axis=1)
            assert len(set(order.tolist())) == heads, "head blocks do not match one to one"
            w = wb[:, order, :].reshape(w.shape).copy()
        res = column_space(w, wo, r, raw_t if t == 8 else None)
        res["ggml_type"] = t
        results[label] = res
        print(f"{label:34s} " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k != "ggml_type"),
              flush=True)

    dense("attn o_proj L3", L + "layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight")
    dense("attn o_proj L47", L + "layers.47.self_attn.o_proj.weight", "blk.47.attn_output.weight")
    dense("gdn out_proj L0", L + "layers.0.linear_attn.out_proj.weight", "blk.0.ssm_out.weight",
          head_block=128)
    dense("gdn out_proj L45", L + "layers.45.linear_attn.out_proj.weight", "blk.45.ssm_out.weight",
          head_block=128)
    dense("shared down L0", L + "layers.0.mlp.shared_expert.down_proj.weight", "blk.0.ffn_down_shexp.weight")
    dense("shared down L30", L + "layers.30.mlp.shared_expert.down_proj.weight", "blk.30.ffn_down_shexp.weight")
    for layer, expert in ((0, 0), (10, 137), (30, 511)):
        dense(f"routed down L{layer} e{expert}", L + f"layers.{layer}.mlp.experts.down_proj",
              f"blk.{layer}.ffn_down_exps.weight", slab=expert, expert=expert)
    dense("ple value_proj L1", L + "layers.1.ple.value_proj.weight", "blk.1.ple_value.weight")
    if m is not None:
        mtp = [k for k in m.tensors if "attn_output" in k or "ffn_down" in k]
        print("MTP sidecar writer tensors:", mtp)
        for gg in mtp:
            if "attn_output" in gg:
                dense("mtp o_proj", "mtp.layers.0.self_attn.o_proj.weight", gg, gguf=m)
            elif "shexp" in gg:
                dense("mtp shared down", "mtp.layers.0.mlp.shared_expert.down_proj.weight", gg, gguf=m)
            elif "exps" in gg:
                for expert in (0, 255):
                    dense(f"mtp routed down e{expert}", "mtp.layers.0.mlp.experts.down_proj", gg,
                          gguf=m, slab=expert, expert=expert)

    # Embedding rows (row space): E' = E - (E r) r^T.
    name = L + "embed_tokens.weight"
    rows = 8192
    for first in (0, 120000, 240000):
        e = base.rows(name, first, rows)
        eo, raw_t = g.rows("token_embd.weight", first, rows)
        er = e @ r
        res = {
            "strength": float(((e - eo) @ r) @ er / (er @ er)),
            "leakage": float(np.linalg.norm(eo @ r) / np.linalg.norm(er)),
            "rank1_residual": float(np.linalg.norm((e - eo) - np.outer((e - eo) @ r, r)) /
                                    np.linalg.norm(e - eo)),
        }
        _, q = q8_0(bf16(e - np.outer(er, r)))
        got = np.frombuffer(raw_t, np.uint8).reshape(-1, 34)[:, 2:].view(np.int8)
        res["q8_code_match"] = float((q.reshape(got.shape) == got).mean())
        results[f"embed rows {first}+{rows}"] = res
        print(f"{'embed rows %d+%d' % (first, rows):34s} " +
              " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)

    # Untouched controls: the edit must not appear here (lm_head, router, input projections).
    def control(label, hf, gg, rows=None):
        n = base.shape(hf)[0] if rows is None else rows
        w = base.rows(hf, 0, n)
        dims, t, _ = g.tensors[gg]
        wo, raw_t = g.rows(gg, 0, dims[1] if rows is None else rows)
        if w.shape != wo.shape:
            w = w.T.copy()
        res = {"rel_diff": float(np.linalg.norm(w - wo) / np.linalg.norm(w))}
        if t == 8:
            _, q = q8_0(w)
            got = np.frombuffer(raw_t, np.uint8).reshape(-1, 34)[:, 2:].view(np.int8)
            res["q8_code_match_unedited"] = float((q.reshape(got.shape) == got).mean())
        results[label] = res
        print(f"{label:34s} " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)

    control("control router L0 (F32)", L + "layers.0.mlp.gate.weight", "blk.0.ffn_gate_inp.weight")
    control("control shared gate_proj L0", L + "layers.0.mlp.shared_expert.gate_proj.weight",
            "blk.0.ffn_gate_shexp.weight")
    control("control lm_head rows 0+8192", "lm_head.weight", "output.weight", rows=8192)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"direction_sha256": hashlib.sha256(raw).hexdigest(), "gguf": a.gguf,
                       "mtp_gguf": a.mtp_gguf, "base": QWEN, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
