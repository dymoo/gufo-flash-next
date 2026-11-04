#!/bin/bash
# phase.sh TAG MODEL UPSTREAM EXTRA_JSON — agent run, official evaluation,
# then the speed microbenchmark for one model.
set -u
cd /srv/mimo-bench
TAG=$1; MODEL=$2; UPSTREAM=$3; EXTRA=$4
./run.sh "$TAG" "$MODEL" "$UPSTREAM" "$EXTRA" 4 || echo "agent rc=$?"
( cd runs/$TAG && ../../venv/bin/python -m swebench.harness.run_evaluation \
    -d SWE-bench/SWE-bench_Verified -p preds.json --max_workers 3 -id "$TAG" > eval.log 2>&1 ) || echo "eval rc=$?"
./venv/bin/python bench_speed.py --url "$UPSTREAM" --key-file .gufo-key --model "$MODEL" \
  --corpus corpus/moby.txt corpus/pride.txt --out runs/$TAG/speed.jsonl > runs/$TAG/speed.log 2>&1 || echo "speed rc=$?"
date -Is > runs/$TAG/phase.done
