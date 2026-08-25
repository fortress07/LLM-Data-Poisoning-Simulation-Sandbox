#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python -m pip install -e . >/dev/null
echo "installed poisonlab in editable mode"

python -m poisonlab doctor

if command -v node >/dev/null 2>&1; then
  echo "node found, the html viewer is available"
else
  echo "node not found, reports stay in json and markdown"
fi

echo
echo "next: poisonlab run configs/backdoor.toml --html"
