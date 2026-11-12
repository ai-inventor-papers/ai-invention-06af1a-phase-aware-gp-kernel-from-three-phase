#!/usr/bin/env bash
# Fetch provenance pages for the candidate datasets in parallel.
set -u
export SKILL_DIR="/ai-inventor/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
export F="$SKILL_DIR/scripts/aii_fast_web_fetch.py"
OUT="$(cd "$(dirname "$0")" && pwd)/logs/searches"
mkdir -p "$OUT"

fetch () {
  local tag="$1"; local url="$2"; local chars="${3:-8000}"
  "$PY" "$F" fetch --url "$url" --max-chars "$chars" > "$OUT/fetch_${tag}.txt" 2>&1
  echo "fetched $tag"
}

fetch ieee_pes "https://ieee-pes-data-sharing.org/datasets/detail/4e38bf3e-d4b3-480c-83e4-167b7bf5dece" 12000 &
fetch pandapower_3ph "https://pandapower.readthedocs.io/en/latest/networks/3phase_grids.html" 12000 &
fetch pandapower_runpp "https://pandapower.readthedocs.io/en/latest/powerflow.html" 10000 &
fetch pandapower_kerber "https://pandapower.readthedocs.io/en/latest/networks/kerber.html" 8000 &
fetch pandapower_cigre "https://pandapower.readthedocs.io/en/latest/networks/cigre.html" 8000 &
wait
echo "=== ALL FETCHES DONE ==="
wc -l "$OUT"/fetch_*.txt
