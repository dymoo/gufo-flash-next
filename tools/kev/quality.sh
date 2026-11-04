#!/bin/bash
# Kev frozen-suite quality through Gufo's /v1/systemone (kev.benchmark --remote).
set -u
# KEV_GGUF_DIR holds kev-4b-<quant>.gguf, GUFO_SRC a gufo checkout with
# build/gpu-test built, KEV_SRC a checkout of the Kev repository with ../venv.
Q=${1:-q8_0}
G=${KEV_GGUF_DIR:?set KEV_GGUF_DIR}/kev-4b-$Q.gguf
PORT=18093
cd "${GUFO_SRC:?set GUFO_SRC}"
build/gpu-test/gufo serve systemone -m "$G" --port $PORT --max-connections 64 --context 24576 > /tmp/kev-quality-serve.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/ready')" 2>/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 /tmp/kev-quality-serve.log; exit 1; }
  sleep 1
done
cd "${KEV_SRC:?set KEV_SRC}"
for suite in v7/decision-v7 v4/transfer-v4; do
  name=$(basename $suite)
  ../venv/bin/python -m kev.benchmark --remote http://127.0.0.1:$PORT --remote-concurrency 16 --suite evals/$suite --out ../runs/gufo-$Q-$name > ../runs-gufo-$Q-$name.log 2>&1
  echo "== $name rc=$?"; tail -25 ../runs-gufo-$Q-$name.log | grep -v "^\s*$" | tail -18
done
