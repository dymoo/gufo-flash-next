"""Attribute tool-call failures to the model or to the server's parser.

A server returns tool calls it parsed from the model's markup. When markup shows
up in `content` instead, or parsed arguments violate the schema only in their
scalar types, the server's parser (not the model) may be at fault. These
reference parsers read the common native formats; a call they recover that is
valid against the tool schema is a server parse failure, anything else is the
model's own malformed call.
"""

from __future__ import annotations

import ast
import json
import re

MARKUP = re.compile(r"<tool_call>|</?function[=>]|<\|tool_call|\[TOOL_CALLS\]|\"name\"\s*:\s*\"\w+\"\s*,\s*\"arguments\"")


def _value(text: str):
    text = text.strip("\n")
    try:
        return json.loads(text)
    except ValueError:
        return text


def recover(content: str) -> list[dict]:
    """Calls written in Hermes JSON, Qwen XML, LFM Pythonic or Mistral formats."""
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.S):
        block = block.strip()
        fn = re.match(r"<function=([^>\s]+)>(.*?)</function>\s*$", block, re.S)
        if fn:  # Qwen3.5 / Qwen3-Coder XML
            params = re.findall(r"<parameter=([^>\s]+)>(.*?)</parameter>", fn.group(2), re.S)
            calls.append({"name": fn.group(1), "arguments": {k: _value(v) for k, v in params}})
            continue
        try:  # Hermes / Qwen3 JSON
            data = json.loads(block)
            args = data.get("arguments", data.get("parameters", {}))
            calls.append({"name": data["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    for body in re.findall(r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>", content, re.S):
        try:  # LFM Pythonic: [name(a='v', b=1), ...]
            tree = ast.parse("[" + body + "]", mode="eval")
            for node in tree.body.elts:
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    calls.append({"name": node.func.id,
                                  "arguments": {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}})
        except (SyntaxError, ValueError, AttributeError):
            pass
    for body in re.findall(r"\[TOOL_CALLS\]\s*(\[.*\])", content, re.S):
        try:
            for data in json.loads(body):
                calls.append({"name": data["name"], "arguments": data.get("arguments", {})})
        except (ValueError, KeyError, TypeError):
            pass
    return calls


def coerce(schema: dict, value):
    """Convert string scalars to the schema's declared types, as a schema-aware parser would."""
    kind = schema.get("type")
    if isinstance(value, str) and kind in ("integer", "number", "boolean", "array", "object"):
        if kind == "boolean" and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return coerce(schema, parsed)
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: coerce(props.get(k, {}), v) for k, v in value.items()}
    if kind == "array" and isinstance(value, list):
        return [coerce(schema.get("items", {}), v) for v in value]
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def attribute_markup(task, content: str) -> str:
    """'server_parse_failure' when the markup in content holds a valid call, else 'model_malformed'."""
    for call in recover(content):
        if not isinstance(call.get("arguments"), dict):
            continue
        schema = task._schemas.get(call["name"])
        if schema is not None and task.invalid(call["name"], coerce(schema, call["arguments"])) is None:
            return "server_parse_failure"
    return "model_malformed"


def attribute_invalid(task, name: str, arguments) -> str:
    """For a schema-invalid parsed call: 'server_parse_failure' when only scalar types are wrong."""
    schema = task._schemas.get(name)
    if schema is not None and isinstance(arguments, dict) and task.invalid(name, coerce(schema, arguments)) is None:
        return "server_parse_failure"
    return "model_malformed"
