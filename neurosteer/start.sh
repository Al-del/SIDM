#!/usr/bin/env bash
# usage: ./start.sh [--mock]   (--mock: fake decoder and LLM, runs without model weights)
set -e
cd "$(dirname "$0")"
for arg in "$@"; do
  case "$arg" in
    --mock) export NEUROSTEER_MOCK=1 ;;
    *) echo "unknown option: $arg (usage: ./start.sh [--mock])" >&2; exit 1 ;;
  esac
done
if [ -z "$PY" ]; then
  ENV_PY="$(conda info --base 2>/dev/null)/envs/neurosteer/bin/python"
  if [ -x "$ENV_PY" ]; then PY="$ENV_PY"; else PY="../../../.venv/bin/python"; fi
fi
(cd backend && exec "$PY" app.py) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd frontend
[ -d node_modules ] || npm install
npx next dev -p 3000
