"""Hard-tier task families (suite v3): long-horizon agent work that separates strong models.

Same Task contract as the v2 families (whose base classes this reuses); its own suite version and
queue, so v2 results stay pinned. Every family's variants are hard; the queue is a sequence of
stratified blocks (each family x variant once, seeded shuffle).
"""

from __future__ import annotations

from families import rng_for

FAMILIES = ["code3", "extract3", "research3", "ops3"]
KIND = {"code3": "code", "extract3": "extract", "research3": "research", "ops3": "ops"}  # handbook per agent type
VARIANTS = 4
BLOCK = len(FAMILIES) * VARIANTS


def make(g: int, seed: int):
    block, position = divmod(g, BLOCK)
    order = list(range(BLOCK))
    rng_for("v3-block", seed, block).shuffle(order)
    variant, family = divmod(order[position], len(FAMILIES))
    return CLASSES[FAMILIES[family]](block * VARIANTS + variant, seed)


from .code import Code3Task  # noqa: E402
from .extract import Extract3Task  # noqa: E402
from .ops import Ops3Task  # noqa: E402
from .research import Research3Task  # noqa: E402

CLASSES = {c.family: c for c in (Code3Task, Extract3Task, Research3Task, Ops3Task)}
