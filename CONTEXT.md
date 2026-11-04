# Context

Domain terms shared by Gufo's code, docs and reviews.

## Architecture

The GGUF `general.architecture` value, such as `qwen4exp`, `qwen35moe`,
`lfm2moe`, `deepseek4`, `mimo2` or `qwen35`. It names a weight layout, not a
product.

## Engine

A runtime that hosts one or more architectures: the Qwen, Flash-Next, DeepSeek
and MiMo engines. Each command (`gufo serve`, `gufo prompt` and `gufo chat`,
`gufo bench`) reaches an engine through its own adapter.

## Model family

An architecture as hosted by an engine, named for people. MiniCPM-V 4.7 and
Qwen3.6 are the `qwen35moe` family and LFM2.5 is the `lfm2moe` family, both on
the Flash-Next engine. Messages name a model by family and architecture.

## Model admission

The decision, made before any weight upload, whether a command's request (KV
format, speculative backend, steering adapters, image input, MoE routing,
context) is valid for an architecture, and which engine runs it. A rejection
names the flag and the model family.

## Load plan

What model admission returns: the engine, the model family, the template
family and the resolved settings that an engine adapter instantiates. Nothing
in a load plan has touched the GPU.

## Capability table

The single table, one row per architecture, that model admission consults:
the engine, the template family, KV formats and their head-width rules, the
speculative backends each command hosts, steering adapters, image input, MoE
routing and the native-context default.

## Template family
Models that share one chat template and output syntax: Qwen3.8, Qwen3.5,
LFM2.5, DeepSeek V4 and MiMo. Several models can share one; Qwen3.8-27B and
Qwen3.8-Flash-Next are both the Qwen3.8 template family.
## Chat protocol
The module that owns a template family's prompt rendering (messages, tools,
tool results, thinking, reasoning effort, stable cache prefix) and its output
parser: `src/cli/serve/chat_protocol.hpp`.
## Output parser
The per-request parser that splits generated text into reasoning, content and
tool calls, starting where the family's prompt leaves the model: inside
reasoning, in content, or either.
