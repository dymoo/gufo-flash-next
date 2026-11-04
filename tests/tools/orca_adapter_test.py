#!/usr/bin/env python3
"""orca_adapter.py recovers a rank-1 ablation direction and writes an adapter
Gufo's loader accepts (format, hidden size, float32 direction), offline."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen-flash" / "adapters"))
import orca_adapter  # noqa: E402


def ablate(w, r, alpha):
    return w - alpha * np.outer(r, r @ w)


def main() -> int:
    rng = np.random.default_rng(7)
    hidden = 64
    r = rng.standard_normal(hidden)
    r /= np.linalg.norm(r)
    w = rng.standard_normal((hidden, 200)).astype(np.float32)
    # bf16-sized noise on the edited weights, as in a published checkpoint.
    wo = ablate(w, r, 1.0) + 1e-3 * rng.standard_normal(w.shape)
    got = orca_adapter.recover_direction(w, wo.astype(np.float32))
    assert abs(float(got @ r)) > 0.999, float(got @ r)
    assert got[np.argmax(np.abs(got))] > 0, "sign is canonical"

    m = orca_adapter.measure(w, wo.astype(np.float32), got)
    assert abs(m["strength"] - 1.0) < 0.01 and m["rank1_residual"] < 0.05, m
    other = rng.standard_normal((hidden, 120)).astype(np.float32)
    assert orca_adapter.measure(other, ablate(other, r, 1.0).astype(np.float32), got)[
        "cos_top_singular"] > 0.999
    # An edit along another direction is not this adapter.
    q = rng.standard_normal(hidden)
    q /= np.linalg.norm(q)
    assert orca_adapter.measure(other, ablate(other, q, 1.0).astype(np.float32), got)[
        "cos_top_singular"] < orca_adapter.MIN_COS

    with tempfile.TemporaryDirectory() as d:
        path = orca_adapter.write_adapter(d, got, 1.0)
        meta = json.loads(path.read_text())
        assert meta["format"] == "gufo steering adapter v2" and meta["hidden_size"] == hidden
        assert meta["alpha"] == 1.0 and meta["writers"] == ["blocks", "embedding", "ple", "mtp"]
        direction = Path(d) / meta["direction_file"]
        assert direction.stat().st_size == hidden * 4, "exactly hidden_size float32 values"
        assert np.allclose(np.fromfile(direction, dtype="<f4"), got)
    print("orca_adapter_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
