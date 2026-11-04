#!/usr/bin/env python3
"""Union size and routing distortion of Flash-Next --moe-routing policies.

Every decode step reads each expert in the batch's routing union once per
layer, so the union (not the row count) sets routed-expert bytes. This models
the policies implemented by RoutePolicy in kernels.hip.cpp over batches of R
router rows and reports, per policy and R:

  union   distinct experts per layer-step (bytes = union * ~3.13 MB/layer)
  tv      total variation between exact and policy mixture weights; with
          renormalized subsets it equals the exact routing mass dropped
  dkl     extra KL of the renormalized selection from the router softmax,
          log(sum_T p) - log(sum_S p) >= 0 (T exact top-k, S policy set)

Rows come from synthetic skewed logits, or from a router dump written by
GUFO_FLASH_NEXT_ROUTER_DUMP (records: u32 layer, rows, stride, prefill, then
rows*stride f32; the last column gates the shared expert). Dumped rows of
one layer are mixed across positions/prompts to form batches, which models
independent sessions (conservative: real co-scheduled agents share more).

  python3 tools/qwen-flash/moe_routing_eval.py            # synthetic sweep
  python3 tools/qwen-flash/moe_routing_eval.py --dump router.bin
  python3 tools/qwen-flash/moe_routing_eval.py --self-test
"""

import argparse
import sys

import numpy as np

E, K = 512, 10
REF_K = 10  # metrics always compare against the model's exact top-10 routing
EXPERT_MB = 3.134  # mean routed gate+up+down bytes per expert per layer
LAYERS = 48


def parse(spec):
    """exact | affinity:KEEP:MARGIN | mass:KEEP:TARGET | budget:KEEP:EXPERTS
    (the --moe-routing syntax)."""
    name, *args = spec.split(":")
    if name == "exact" and not args:
        return ("exact",)
    if name in ("affinity", "mass") and len(args) == 2:
        return (name, int(args[0]), float(args[1]))
    if name == "budget" and len(args) == 2:
        return (name, int(args[0]), int(args[1]))
    raise ValueError(f"bad policy {spec!r}")


def topk(scores, k):
    """Descending, lowest index first on ties (the RouterTopK contract)."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def route(logits, policy):
    """Returns (ids [R,K], weights [R,K]); dropped slots have weight 0."""
    rows = logits.shape[0]
    ids = topk(logits, K)
    take = np.ones((rows, K), bool)
    if policy[0] == "affinity":
        _, keep, margin = policy
        core = np.zeros(E, bool)
        core[ids[:, :keep]] = True
        score = logits + margin * core[None, :]
        kept = np.take_along_axis(logits, ids[:, :keep], axis=1)
        np.put_along_axis(score, ids[:, :keep], kept + 1e9, axis=1)
        ids = topk(score, K)
    elif policy[0] == "mass":
        _, kmin, target = policy
        w = softmax_at(logits, ids, take)
        before = np.cumsum(w, axis=1) - w
        take = before < target
        take[:, :kmin] = True
    elif policy[0] == "budget":
        _, kmin, budget = policy
        w = softmax_at(logits, ids, take)
        must = np.zeros(E, bool)
        must[ids[:, :kmin]] = True
        demand = np.zeros(E)
        np.add.at(demand, ids, np.floor(w * 2**16))
        demand[must] = 0
        order = np.lexsort((np.arange(E), -demand))
        extra = order[: max(0, budget - must.sum())]
        allowed = must.copy()
        allowed[extra[demand[extra] > 0]] = True
        score = np.where(allowed[None, :], logits, -np.inf)
        ids = topk(score, K)
        take = np.isfinite(np.take_along_axis(score, ids, axis=1))
    return ids, softmax_at(logits, ids, take)


def softmax_at(logits, ids, take):
    sel = np.take_along_axis(logits, ids, axis=1)
    sel = np.where(take, sel, -np.inf)
    w = np.exp(sel - sel.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def metrics(logits, ids, w):
    ref_ids = topk(logits, REF_K)
    ref_w = softmax_at(logits, ref_ids, np.ones_like(ref_ids, bool))
    rows = logits.shape[0]
    dense = np.zeros((rows, E))
    np.add.at(dense, (np.arange(rows)[:, None], ref_ids), ref_w)
    ref_dense = dense.copy()
    dense[:] = 0
    np.add.at(dense, (np.arange(rows)[:, None], ids), w)
    tv = 0.5 * np.abs(dense - ref_dense).sum(axis=1)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    cover_ref = (p * (ref_dense > 0)).sum(axis=1)
    cover = (p * (dense > 0)).sum(axis=1)
    union = len(np.unique(ids[w > 0]))
    return union, tv, np.log(cover_ref) - np.log(cover)


def synthetic(rng, rows, sigma, skew, bias):
    return bias[None, :] * skew + sigma * rng.standard_normal((rows, E))


def load_dump(path, decode_only):
    raw = np.fromfile(path, dtype=np.uint8)
    layers, at = {}, 0
    while at < raw.size:
        layer, rows, stride, prefill = np.frombuffer(raw[at : at + 16], np.uint32)
        at += 16
        n = int(rows) * int(stride) * 4
        block = np.frombuffer(raw[at : at + n], np.float32).reshape(rows, stride)
        if not (decode_only and prefill):
            layers.setdefault(int(layer), []).append(block[:, :E].astype(np.float64))
        at += n
    return {l: np.concatenate(b) for l, b in layers.items()}


def sweep(args):
    rng = np.random.default_rng(args.seed)
    policies = [parse(p) for p in args.policy]
    dump = load_dump(args.dump, args.decode_only) if args.dump else None
    print(
        f"# {'real dump ' + args.dump if dump else f'synthetic sigma={args.sigma} skew={args.skew}'}"
        f", trials={args.trials}; union in experts/layer-step, MB/step = union*{EXPERT_MB}*{LAYERS}"
    )
    print(f"# routing width k={K}; vs_exact is the union relative to exact top-{REF_K}")
    print(f"{'policy':<18} {'R':>4} {'union':>7} {'vs_exact':>8} "
          f"{'tv':>7} {'tv_p99':>7} {'dkl':>7} {'rows/exp':>8}")
    for R in args.rows:
        batches = []  # the same batches for every policy (paired)
        for _ in range(args.trials):
            if dump:
                pool = dump[rng.choice(list(dump))]
                batches.append(pool[rng.choice(len(pool), R,
                                                replace=len(pool) < R)])
            else:
                batches.append(synthetic(rng, R, args.sigma, args.skew,
                                         rng.standard_normal(E)))
        base = None
        if K != REF_K:  # the exact top-10 union is the denominator
            global_k = K
            globals()["K"] = REF_K
            base = float(np.mean([metrics(x, *route(x, ("exact",)))[0] for x in batches]))
            globals()["K"] = global_k
        for pol in [("exact",)] + [p for p in policies if p != ("exact",)]:
            u, t, k = [], [], []
            for x in batches:
                ids, w = route(x, pol)
                union, tv, dkl = metrics(x, ids, w)
                u.append(union), t.append(tv), k.append(dkl)
            mean_u = float(np.mean(u))
            base = base or mean_u
            t = np.concatenate(t)
            print(f"{':'.join(map(str, pol)):<18} {R:>4} {mean_u:>7.1f} "
                  f"{mean_u / base:>8.3f} {t.mean():>7.4f} {np.quantile(t, 0.99):>7.4f} "
                  f"{np.concatenate(k).mean():>7.4f} "
                  f"{R * K / mean_u:>8.2f}")


def self_test():
    rng = np.random.default_rng(1)
    one = rng.standard_normal((1, E)) * 1.5
    for spec in ("affinity:4:2", "budget:4:10"):
        ids, w = route(one, parse(spec))
        ref, rw = route(one, ("exact",))
        assert (ids == ref).all() and np.allclose(w, rw), spec
    x = rng.standard_normal((64, E)) * 1.5
    for spec in ("exact", "affinity:6:1", "mass:4:0.8", "budget:4:128"):
        ids, w = route(x, parse(spec))
        assert np.allclose(w.sum(axis=1), 1), spec
        union, tv, dkl = metrics(x, ids, w)
        assert (dkl >= -1e-12).all() and (tv <= 1 + 1e-12).all(), spec
    assert metrics(x, *route(x, ("budget", 2, 256)))[0] <= 256
    ids, w = route(x, ("mass", 4, 0.8))
    assert ((w > 0).sum(axis=1) >= 4).all()
    exact_union = metrics(x, *route(x, ("exact",)))[0]
    assert metrics(x, *route(x, ("affinity", 6, 1.0)))[0] < exact_union
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", nargs="*", default=[
        "affinity:8:1", "affinity:6:1", "affinity:4:2", "mass:2:0.9",
        "mass:2:0.8", "budget:2:192", "budget:2:128"])
    ap.add_argument("--rows", type=int, nargs="*",
                    default=[1, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="per-row router logit scale (synthetic)")
    ap.add_argument("--skew", type=float, default=0.5,
                    help="per-expert popularity bias scale (synthetic)")
    ap.add_argument("--dump", help="GUFO_FLASH_NEXT_ROUTER_DUMP file")
    ap.add_argument("--decode-only", action="store_true",
                    help="skip dumped prefill rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10,
                    help="routed experts per token the policies start from (GUFO_FLASH_NEXT_EXPERTS_USED); "
                         "tv/dkl stay relative to exact top-10")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    global K
    K = args.k
    if args.self_test:
        self_test()
        return 0
    sweep(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
