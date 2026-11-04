#!/usr/bin/env bash
# Upload the fork's Flash-Next companion files to a Hugging Face model repo.
#
#   tools/qwen-flash/hf_release.sh REPO [--public] \
#     --adapter orca-abliteration.json \
#     [--decode-twins q4k-gptq2-h5.gguf] [--s4g128 DIR] [--draft-vocab FILE]
#
# The repo is created private unless --public is given; check every source
# file's license before making it public. Needs `hf` from huggingface_hub 2.4
# or later, logged in (`hf auth login`), and runs where the files are. Writes
# SHA256SUMS and a README model card beside the uploads, so a download can be
# verified with `sha256sum -c`.
set -euo pipefail

usage() { sed -n '2,12p' "$0" >&2; exit 2; }
[[ $# -ge 1 ]] || usage
repo=$1
shift
private=--private
adapter="" twins="" s4="" vocab=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --public) private=--no-private ;;
    --adapter) adapter=$2; shift ;;
    --decode-twins) twins=$2; shift ;;
    --s4g128) s4=$2; shift ;;
    --draft-vocab) vocab=$2; shift ;;
    *) usage ;;
  esac
  shift
done
[[ -n $adapter ]] || { echo "--adapter is required" >&2; exit 2; }
command -v hf >/dev/null || { echo "hf (huggingface_hub CLI) not found" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

# The adapter JSON names its direction file relative to itself; ship both.
direction=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["direction_file"])' "$adapter")
mkdir -p "$stage/adapters"
cp "$adapter" "$stage/adapters/orca-abliteration.json"
cp "$(dirname "$adapter")/$direction" "$stage/adapters/$(basename "$direction")"
python3 -I - "$stage/adapters/orca-abliteration.json" "$(basename "$direction")" <<'EOF'
import json, sys
path, name = sys.argv[1], sys.argv[2]
meta = json.load(open(path))
meta["direction_file"] = name
json.dump(meta, open(path, "w"), indent=2)
open(path, "a").write("\n")
EOF
[[ -z $twins ]] || { mkdir -p "$stage/decode-twins"; cp "$twins" "$stage/decode-twins/"; }
[[ -z $s4 ]] || { mkdir -p "$stage/s4g128"; cp -r "$s4"/. "$stage/s4g128/"; }
[[ -z $vocab ]] || { mkdir -p "$stage/mtp"; cp "$vocab" "$stage/mtp/draft-vocab-ranked.txt"; }

(cd "$stage" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort |
  xargs -d '\n' sha256sum > SHA256SUMS)

rows='| `adapters/orca-abliteration.json` + direction | `--adapter NAME=PATH` | rank-1 steering adapter (v2) for the abliterated variant |'
[[ -z $twins ]] || rows+=$'\n'"| \`decode-twins/$(basename "$twins")\` | \`--gguf-decode-override PATH\` | GPTQ Q4_K dense projections and Q5_K head read by decode batches (lossy) |"
[[ -z $s4 ]] || rows+=$'\n''| `s4g128/` | `--dense-prefill w4a4 --dense-prefill-weights DIR` | offline GPTQ S4G128 int4 dense weights (lossy) |'
[[ -z $vocab ]] || rows+=$'\n''| `mtp/draft-vocab-ranked.txt` | `--mtp-draft-vocab FILE` | 32768 ranked token ids for the compact MTP draft head |'

cat > "$stage/README.md" <<EOF
---
base_model: unsloth/Qwen3.8-Flash-Next-GGUF
tags: [gguf, steering-adapter, strix-halo, gufo]
---
# Qwen3.8 Flash-Next companion files for the Gufo Flash-Next fork

Files the Gufo Flash-Next fork reads beside the unmodified
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
UD-Q4_K_XL weights. None of them replaces the weights.

| Path | Gufo option | What it is |
| --- | --- | --- |
$rows

Quality and speed for each option are in the fork's Flash-Next guide.
Verify a download with \`sha256sum -c SHA256SUMS\`.
EOF

hf repos create "$repo" --repo-type model "$private" --exist-ok
hf upload "$repo" "$stage" . --repo-type model \
  --commit-message "Flash-Next companion files"
echo "uploaded to https://huggingface.co/$repo ($private)"
