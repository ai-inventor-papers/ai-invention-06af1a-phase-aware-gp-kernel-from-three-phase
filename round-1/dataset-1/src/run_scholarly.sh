#!/usr/bin/env bash
# Scholarly-mode searches in parallel: confirm the candidate feeders are established
# benchmarks with citing papers and clear original provenance.
set -u
export SKILL_DIR="/ai-inventor/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
export S="$SKILL_DIR/scripts/aii_fast_web_search.py"
OUT="$(cd "$(dirname "$0")" && pwd)/logs/searches"
mkdir -p "$OUT"

sch () {
  local tag="$1"; local query="$2"
  "$PY" "$S" --query "$query" --mode scholarly --max-results 10 > "$OUT/${tag}.txt" 2>&1
  echo "scholarly done $tag"
}

sch q1 "IEEE European LV test feeder unbalanced three phase power flow" &
sch q2 "pandapower IEEE European low voltage asymmetric feeder voltage profile" &
sch q3 "machine learning surrogate power flow unbalanced distribution grid voltage" &
sch q4 "Gaussian process regression power system state estimation distribution feeder" &
wait
echo "=== ALL SCHOLARLY DONE ==="
wc -l "$OUT"/q*.txt
