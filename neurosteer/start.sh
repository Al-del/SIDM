#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -z "$PY" ]; then
  ENV_PY="$(conda info --base 2>/dev/null)/envs/neurosteer/bin/python"
  if [ -x "$ENV_PY" ]; then PY="$ENV_PY"; else PY="../../../.venv/bin/python"; fi
fi
(cd backend && "$PY" app.py) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd frontend
[ -d node_modules ] || npm install
npx next dev -p 3000
