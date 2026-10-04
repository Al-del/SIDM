#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="${PY:-../../../.venv/bin/python}"
(cd backend && "$PY" app.py) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd frontend
[ -d node_modules ] || npm install
npx next dev -p 3000
