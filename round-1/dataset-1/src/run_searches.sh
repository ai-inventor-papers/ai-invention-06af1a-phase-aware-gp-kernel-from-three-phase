#!/usr/bin/env bash
# Run 8 diverse web searches in parallel and save results.
set -u
export SKILL_DIR="/ai-inventor/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
export S="$SKILL_DIR/scripts/aii_fast_web_search.py"
OUT="$(cd "$(dirname "$0")" && pwd)/logs/searches"
mkdir -p "$OUT"

run () {
  local tag="$1"; shift
  local query="$1"; shift
  local mode="$1"; shift
  "$PY" "$S" --query "$query" --mode "$mode" --max-results 8 > "$OUT/${tag}.txt" 2>&1
  echo "finished $tag"
}

run s1 "IEEE European LV test feeder asymmetric unbalanced distribution network benchmark" general &
run s2 "pandapower runpp_3ph three phase power flow documentation" general &
run s3 "Gaussian process emulator power flow unbalanced low voltage distribution network" general &
run s4 "LinDistFlow linearized power flow distribution grid voltage estimation" general &
run s5 "graph Laplacian kernel Gaussian process network topology power system" general &
run s6 "Kerber Netz low voltage network benchmark pandapower" general &
run s7 "CIGRE low voltage test network benchmark unbalanced three phase feeder" general &
run s8 "synthetic distribution feeder load scenario voltage dataset benchmark" general &
wait
echo "=== ALL DONE ==="
wc -l "$OUT"/*.txt
