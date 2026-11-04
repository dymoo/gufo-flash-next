# Gufo: the Strix Halo inference engine (Flash-Next fork)

> [!IMPORTANT]
> **This is a hard fork of [gufo-org/gufo](https://github.com/gufo-org/gufo),
> focused on serving Qwen3.8 Flash-Next to many concurrent agents.** Its git
> history is not shared with upstream: the tree was re-rooted from a working
> snapshot on 25 October 2026, so it is not a fork in the git sense, `git merge`
> from upstream does not work, and the two code bases will keep diverging.
> Upstream fixes are ported by hand, one reviewed change at a time
> ([what has been ported](docs/FORK.md)). Report fork issues here, not upstream.

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

**Contributions are welcome!**

## Why Flash-Next

Other models still build and keep their guides below, but this fork's serving,
speed and quality work goes into **[Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md)**:
of everything spiked on this hardware it is the best model for its cost, and
the one this fork runs in production. What the fork adds on top of upstream is
summarised in [What this fork adds](#what-this-fork-adds).

## What this fork adds

Everything below is in this tree and documented in its guide; upstream has
none of it unless noted. Exact means tokens and logprobs are unchanged; every
lossy option is off by default and changes the disk-cache identity.

### Serving many agents from one process

- **Concurrent sessions:** up to 64 batched decode sessions; decode batches
  share projections and stream each active expert once per step.
- **SplitFuse scheduling:** prompt chunks ride inside decode steps, sized to an
  inter-token-latency target (`--max-itl-ms`), and several sessions' prompt
  chunks pack into one batch.
- **Quantized KV cache (`--kv-cache tq4|tq3`):** TurboQuant-style rotated 4-bit
  and 3-bit keys and values, 3.9× and 5.2× smaller than F16 (a 131K-token slot
  drops from 3.0 GiB to 0.76 or 0.57 GiB). Lossy and opt-in;
  [quality](docs/models/qwen3.8-flash-next/QUALITY.md).
- **KV memory that follows the context:** KV and indexer pages commit lazily
  from one HIP virtual-memory pool, so a slot costs the context it holds.
- **Three-tier conversation cache:** idle sessions page out of the device to
  host RAM (`--host-memory-budget`), then to `O_DIRECT` files on NVMe
  (`--cache-disk`); shared system prompts are checkpointed and pinned
  (`--cache-pin-*`) and survive restarts; queued requests' disk checkpoints are
  read ahead and verified before their turn.
- **Router contract** ([docs/ROUTER.md](docs/ROUTER.md)): the `service_tier:
  "flex"` tier for background work, `--admission refuse` for gateways that fail
  over, `X-Gufo-No-Queue`, `GET /v1/runtime`, drain, request IDs, bearer keys
  from a file, and a separate Prometheus listener (`--metrics-address`).
- **Structured outputs:** grammar-constrained `response_format` (`json_schema`,
  `json_object`) and required or named `tool_choice`.
- **Reproducible sampling:** sampler randomness is keyed by seed and committed
  position, so a seeded request replays the same tokens regardless of batching.
- **Safe exposure defaults:** no CORS unless `--allow-origin` names an origin
  (other browser origins get 403, so a web page cannot drive a local server);
  unauthenticated requests are refused before their body is read, and bodies
  must arrive within a deadline; drain is limited to loopback clients unless
  `--admin-key-file` sets its own key; the metrics listener serves only
  `/metrics` and `/health`; serving beyond loopback without `--api-key-file`
  prints a warning ([Authentication and exposure](docs/SERVER.md)).

### Model ids: variants and the flex tier

One process serves several **model ids** from the same weights, sessions and
batches. A variant is a different model; the flex tier is the same model
scheduled differently. They are kept apart on purpose:

| Suffix | Kind | What it changes |
| --- | --- | --- |
| *(none)*, e.g. `workhorse` | model | The base model, exact. |
| `-abliterated` | model | The Orca steering adapter: refusal direction projected out of the residual writers, from the base weights ([Steering adapters](docs/models/qwen3.8-flash-next/README.md#steering-adapters)). |
| `-fast` | model (**experimental**, `--fast-experts 7`) | Lossy decode: each row keeps its top 7 routed experts; prompts stay exact. Compiled, not yet GPU-measured on this path. |
| `-flex` | service tier | Same weights and arithmetic as the id it suffixes; runs only on idle compute and is refused with 429 otherwise ([Router](docs/ROUTER.md#service-tiers)). |

`-flex` goes on the end of any served id (`workhorse-abliterated-flex`). Fast
mode is a model suffix rather than a service tier because it is not "the same
model, faster": its lossy changes give it its own outputs, quality numbers and
cache identity, and it has to combine with the flex tier
(`workhorse-abliterated-fast-flex`). Variants never share cache entries; the
flex tier shares its variant's.

### Flash-Next speed and quality

- **Exact by default:** the campaign's bit-identical wins (prefetching small
  decode GEMMs, fused hyper-connection and GatedDeltaNet epilogues, a
  register-resident tq4 KV-store epilogue) give +4% pp2048 and +6.7% tg128 on
  one session.
- **Measured opt-in options:** int4 W4A4 dense prefill with offline GPTQ
  weights, F16 GatedDeltaNet state, GPTQ decode twins, larger prefill batches
  and an exact adaptive MTP policy (compact draft head, calibrated confidence
  gate, longest-match prompt lookup). Each has its quality numbers against the
  accept envelope; see [Speed options](docs/models/qwen3.8-flash-next/README.md#speed-options).
- **Negative results are kept:** fewer routed experts for agents, batch-aware
  expert routing, routed int4 experts and others are documented with why they
  did not ship ([Experiments](docs/models/qwen3.8-flash-next/EXPERIMENTS.md)).

### Other additions

- **Kev / System One:** Kev-4B served behind TypeSafe's System One API
  (`POST /v1/systemone`), batched into packed passes and paced to a share of
  GPU time while the text model is busy ([Kev](docs/models/kev-4b/README.md)).
- **More models on the Flash-Next engine:** MiniCPM-V 4.7 (`qwen35moe`, with
  DSpark speculative decoding) and LFM2.5 (`lfm2moe`).

### Experiments not in this release

Kept on their own branches and not built here: EmbeddingGemma 2 and Kev
projections on the XDNA2 NPU, XDNA2 NPU int4 GEMM, EXL3 format and Ornith 1.5
122B, Swift 1.5 AutoRound int4 for Flash-Next, and the Flint comparison.
Upstream has since added a native GPU EmbeddingGemma 2 path that this fork has
not ported ([fork status](docs/FORK.md)).

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Q4: **656.33 tok/s pp**; up to **70.56 tok/s tg** single user and **123.00 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-27b/QUALITY.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4/IQ4_XS, images, AR, MTP, tq4/tq3 KV | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) | **1,628.52 tok/s pp**; up to **59.41 tok/s tg** single user and **157.22 aggregated tok/s** on 8 concurrent requests with MTP; **336 aggregated tok/s** on 64 concurrent requests (AR, tq4 KV) · [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-flash-next/QUALITY.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark · [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) | [Quality](docs/models/deepseek-v4-flash/QUALITY.md) |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | **15.27× realtime** · [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) | [Quality](docs/models/qwen3-asr/QUALITY.md) |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis and voice cloning | BF16 [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Up to **2.54× realtime**; **201 ms** to first audio (CustomVoice) · [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) | [Quality](docs/models/qwen3-tts/QUALITY.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | In progress · [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) | [Quality](docs/models/qwen-image-2.1/QUALITY.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA) | In progress · [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) | [Quality](docs/models/minimax-h3/QUALITY.md) |

Peak measured workloads; text pp is autoregressive (AR), while tg uses the
named speculative mode. Peaks include repetitive output; aggregate tg sums
individual request decode rates. Qwen27B's single-user peak uses the short-prompt
C1 workload. Audio excludes loading.
Each model guide lists the required files and complete benchmark settings.

## Philosophy

- Contributions are welcome! We need the help of Strix Halo community to keep improving gufo!
- We would like this to be the one-stop shop for Strix Halo Local AI enthusiasts: batteries included for text, audio, image, and video models.
- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.

## Quickstart

Build this fork from source and serve Flash-Next. The upstream container
image (`ghcr.io/gufo-org/toolboxes/gufo-runtime`) runs upstream Gufo, without
any of this fork's features; this fork does not publish an image yet.

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" \
  --local-dir models/qwen3.8-flash-next
nix build    # or the CMake build in "Build from source" below
./result/bin/gufo serve --sessions 8 llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name workhorse --context 131072 --kv-cache tq4
```

The loader finds the other shards. Build the abliterated adapter with
`tools/qwen-flash/adapters/orca_adapter.py` and add
`--adapter workhorse-abliterated=adapters/orca-abliteration.json` to serve it
beside the base model, and see the
[Flash-Next guide](docs/models/qwen3.8-flash-next/README.md) for the
production configuration, MTP and the speed options.

### Running in a container

If you build your own image, rootless Podman needs `crun` for `--group-add keep-groups`. Your host user must
have read/write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the
`render` and `video` groups; log out and back in after changing membership.
Container groups named `video`/`render` do not preserve host supplementary
groups. Check `id`, `ls -l /dev/kfd /dev/dri/renderD*`, and
`podman info --format '{{.Host.OCIRuntime.Name}}'` if ROCm reports no device.
See [Podman's rootless group-access guidance](https://github.com/containers/podman/blob/main/troubleshooting.md#20-passed-in-devices-or-files-cant-be-accessed-in-rootless-container).

On Fedora or another SELinux-enforcing host, GPU enumeration can succeed while
SELinux blocks mapping `/dev/kfd`, causing ROCr to report a misleading
“Memory critical” error. Check the **host** audit log:

```sh
sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
```

If it shows a denied `map` for the container, Podman documents this fix:

```sh
sudo setsebool -P container_use_devices true
```

This persistently allows containers to access device labels for devices passed
into them; it affects all containers on that host. Review that policy scope
before enabling it. See [Podman's device documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device-container-device-permissions)
and the [SELinux container policy](https://github.com/containers/container-selinux/blob/main/container.te).

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code and OpenAI SDK clients, set the API base URL to
`http://localhost:8080/v1`. Chat Completions supports text, images, tools,
streaming and, on Flash-Next and MiMo models, structured outputs
(`response_format` `json_object` and `json_schema`, `tool_choice` `required`
or a named function, enforced by grammar-constrained sampling); Responses
supports text and streaming. See the
[API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target. CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The currently qualified toolchain is GCC
15.3 and ROCm 7.2.3. Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg | Tokenization, HTTPS, hashing and images |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## Upgrading from earlier fork builds

- **`--prefill-chunk`** now always sets the scheduler's fused-step prompt rows.
  Flash-Next's macro prefill phase is `--prefill-phase-rows`; a configuration
  that passed `--prefill-chunk 2048` for it must switch.
- **CORS** is off by default. Browser clients calling the server directly need
  `--allow-origin https://their.origin`; server-side clients are unaffected.
- **Drain** from another host (or through a same-host reverse proxy, where every
  client looks like loopback) needs `--admin-key-file`; the API key alone no
  longer drains.
- **The metrics listener** answers 404 outside `/metrics` and `/health`.
- **`-flex`** now exists for every served id, so a model id or adapter name
  ending in `-flex` is rejected at startup.
- **`-fast`** ids exist only with `--fast-experts`; with it, adapter and
  served names ending in `-fast` are rejected at startup.
- **`-abliterated`** is the Orca adapter; the huihui adapter is no longer
  documented or recommended.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
