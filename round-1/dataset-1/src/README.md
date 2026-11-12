# IEEE European LV Asymmetric Feeder Dataset

This workspace generates and standardizes three-phase AC power-flow data for the IEEE European LV asymmetric feeder. I created three candidate scenario sets, selected the highest-variance one, and exported it to `full_data_out.json` in the required grouped schema.

## Layout
- `prepare_candidates.py`: generates candidate feeder datasets under `temp/datasets/`.
- `data.py`: standardizes the selected candidate into `exp_sel_data_out.json` shape.
- `dataset_parts/`: four split chunks of the raw 3600-scenario power-flow dataset.
- `temp/datasets/`: candidate dataset pool used to choose the final dataset.
- `full_data_out.json`: final schema-validated selected dataset.
- `mini_data_out.json`: mini preview after JSON formatting.
- `preview_data_out.json`: preview after JSON formatting.
- `logs/`: run logs and sanity-check output.
- `.aii/manifest.yaml`: storage manifest for heavy paths.

## How to run
1. Generate candidates:
   ```bash
   python3 prepare_candidates.py
   ```
2. Standardize the selected dataset:
   ```bash
   uv run data.py
   ```
3. Validate and preview:
   ```bash
   export SKILL_DIR="/ai-inventor/.claude/skills/aii-json"
   export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
   $PY "$SKILL_DIR/scripts/aii_json_validate_schema.py" --format exp_sel_data_out --file full_data_out.json
   $PY "$SKILL_DIR/scripts/aii_json_format_mini_preview.py" --format exp_sel_data_out --input full_data_out.json
   cp -f mini_full_data_out.json mini_data_out.json
   cp -f preview_full_data_out.json preview_data_out.json
   ```

## Restoring removed files
- `.venv/`: run `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python pandapower scikit-learn loguru psutil jsonschema networkx` from this directory (exact pinned versions are in `pyproject.toml`).

## Notes
- The raw 3600-scenario output lives in `dataset_parts/dataset_part_*.json`; the former single `dataset.json` was split so every published file stays under the 100 MB deployment limit.
- `full_data_out.json` is the final selected artifact and remains on the run volume for later stages.
