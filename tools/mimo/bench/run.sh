#!/bin/bash
# run.sh TAG MODEL UPSTREAM EXTRA_JSON WORKERS [FILTER]
# Agent run of the fixed SWE-bench Verified subset through the metrics proxy.
# KEY_FILE (default .gufo-key) authenticates upstream; HEADERS holds extra
# "Name: value" lines for the proxy (OpenRouter app attribution).
set -eu
cd /srv/mimo-bench
TAG=$1; MODEL=$2; UPSTREAM=$3; EXTRA=$4; WORKERS=$5
FILTER=${6:-"^($(./venv/bin/python -c 'import json;print("|".join(json.load(open("instances.json"))))'))$"}
PORT=$((9100 + $(echo -n "$TAG" | cksum | cut -c1-2) % 50))
HEADER_ARGS=()
while IFS= read -r h; do [ -n "$h" ] && HEADER_ARGS+=(--header "$h"); done <<< "${HEADERS:-}"
mkdir -p runs/$TAG
pkill -f "proxy.py --port $PORT" || true
nohup ./venv/bin/python proxy.py --port $PORT --upstream "$UPSTREAM" \
  --key-file "${KEY_FILE:-.gufo-key}" --log runs/$TAG/requests.jsonl --tag "$TAG" \
  --extra "$EXTRA" "${HEADER_ARGS[@]}" > runs/$TAG/proxy.log 2>&1 &
PROXY=$!
trap 'kill $PROXY' EXIT
sleep 2
date -Is > runs/$TAG/started
MSWEA_COST_TRACKING=ignore_errors ./venv/bin/mini-extra swebench \
  --subset verified --split test --filter "$FILTER" -o runs/$TAG -w "$WORKERS" \
  -m "openai/$MODEL" -c swebench.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:$PORT/v1 \
  -c model.model_kwargs.api_key=local \
  -c model.model_kwargs.max_tokens=16384 \
  -c model.model_kwargs.timeout=3600 \
  -c agent.step_limit=100 -c agent.cost_limit=0 > runs/$TAG/agent.log 2>&1
date -Is > runs/$TAG/finished
