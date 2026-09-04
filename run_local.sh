#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESOURCE_DIR="${SCENT_STUDIO_RESOURCE_DIR:-$HOME/.scent-molecule-studio/resources}"

if [ ! -f "$RESOURCE_DIR/resource_manifest.json" ]; then
  echo "Private model resources are not configured." >&2
  echo "Prepare the bundle with: $PYTHON_BIN scripts/prepare_resource_bundle.py --source <resource-source> --target $RESOURCE_DIR" >&2
  echo "Or set SCENT_STUDIO_RESOURCE_DIR to an existing bundle." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        "Scent Molecule Studio requires Python 3.10–3.12. "
        f"Current interpreter: {sys.version.split()[0]}"
    )
PY

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 20 or newer is required to build the web interface." >&2
  exit 1
fi

NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo "Node.js 20 or newer is required. Current version: $(node --version)" >&2
  exit 1
fi

cd "$APP_ROOT/frontend"
if [ ! -f node_modules/.package-lock.json ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  npm ci
fi

NEEDS_BUILD=0
if [ ! -f dist/index.html ]; then
  NEEDS_BUILD=1
elif find src -type f -newer dist/index.html -print -quit | grep -q .; then
  NEEDS_BUILD=1
elif [ index.html -nt dist/index.html ] || [ package-lock.json -nt dist/index.html ] || [ vite.config.ts -nt dist/index.html ]; then
  NEEDS_BUILD=1
fi

if [ "$NEEDS_BUILD" -eq 1 ]; then
  npm run build
fi

cd "$APP_ROOT"
exec "$PYTHON_BIN" -m uvicorn olfactory.api:app \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --workers 1
