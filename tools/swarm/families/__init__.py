"""Task families for the swarm goodput benchmark.

Every task is generated from (family, index, seed) alone, so a run's task set is
pinned by SUITE_VERSION and its seed. A task owns a tool environment, the fixed
tool set of its agent type, an initial user message, a mechanical checker and
an oracle (the reference action sequence the offline tests replay).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

SUITE_VERSION = "swarm-v1"


class ToolError(Exception):
    """A well-formed call that the environment refuses (missing file, unknown id).
    Reported to the model as the tool result; not an invalid tool call."""


@dataclass
class Call:
    name: str
    arguments: dict


@dataclass
class Step:
    """One oracle assistant turn: tool calls, or a final message when `calls` is empty."""
    content: str = ""
    calls: list[Call] = field(default_factory=list)


def rng_for(*parts) -> random.Random:
    """Deterministic RNG; str seeds hash with SHA-512, stable across Python versions."""
    return random.Random(":".join(str(p) for p in parts))


def digest(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


def validate(schema: dict, value, path: str = "arguments") -> str | None:
    """The JSON-schema subset the tool sets use; returns the first violation or None."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key} is required"
        for key, item in value.items():
            if key not in properties:
                if schema.get("additionalProperties", True) is False:
                    return f"{path}.{key} is not a parameter"
                continue
            error = validate(properties[key], item, f"{path}.{key}")
            if error:
                return error
        return None
    if kind == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        for i, item in enumerate(value):
            error = validate(schema.get("items", {}), item, f"{path}[{i}]")
            if error:
                return error
        return None
    checks = {"string": lambda v: isinstance(v, str),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "boolean": lambda v: isinstance(v, bool)}
    if kind in checks and not checks[kind](value):
        return f"{path} must be a {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    return None


class Task:
    """Base class. Subclasses set the class attributes and implement the hooks."""
    family = ""
    tools: list[dict] = []     # the agent type's fixed tool set (identical for every task)
    submit_tool: str | None = None  # a call to it ends the task; None: the task ends on a reply without calls
    max_turns = 30

    def __init__(self, index: int, seed: int):
        self.index, self.seed = index, seed
        self.id = f"{self.family}-{index:05d}"
        self.tier = "medium"
        self.prompt = ""
        self.submitted: dict | None = None
        self._schemas = {t["function"]["name"]: t["function"]["parameters"] for t in self.tools}

    def header(self) -> str:
        return f"Ticket SWARM-{self.id}."

    def invalid(self, name: str, arguments) -> str | None:
        """Why a call is not a valid call of the tool set, or None."""
        if name not in self._schemas:
            return f"unknown tool {name!r}; available: {', '.join(sorted(self._schemas))}"
        return validate(self._schemas[name], arguments)

    def call(self, name: str, arguments: dict) -> str:
        """Execute a valid call; raise ToolError for an environment refusal."""
        if name == self.submit_tool:
            self.submitted = arguments
            return "Submitted."
        return getattr(self, "tool_" + name)(**arguments)

    def check(self, final: str) -> tuple[bool, str]:
        raise NotImplementedError

    def oracle(self) -> list[Step]:
        raise NotImplementedError


FAMILIES = ["code", "extract", "research", "ops"]
VARIANTS = 4  # every family cycles four variants (tiers) by its own index
BLOCK = len(FAMILIES) * VARIANTS


def make(g: int, seed: int) -> Task:
    """Task g of the run's queue. The queue is a sequence of stratified blocks: each block of
    16 holds every family x variant once, in a seeded shuffle, so the family and tier mix is
    the same at every queue position however far a run gets."""
    block, position = divmod(g, BLOCK)
    order = list(range(BLOCK))
    rng_for("block", seed, block).shuffle(order)
    variant, family = divmod(order[position], len(FAMILIES))
    return CLASSES[FAMILIES[family]](block * VARIANTS + variant, seed)


from .code import CodeTask  # noqa: E402  (the subclasses import this module's base classes)
from .extract import ExtractTask  # noqa: E402
from .ops import OpsTask  # noqa: E402
from .research import ResearchTask  # noqa: E402

CLASSES = {c.family: c for c in (CodeTask, ExtractTask, ResearchTask, OpsTask)}
