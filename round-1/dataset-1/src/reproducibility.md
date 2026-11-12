# Reproducibility notes

This artifact was created in a workspace folder that is published as part of a public GitHub repository. To reproduce it:

1. Clone the repository and `cd` into this artifact directory. The repository contains this folder as a sibling workspace, so all relative paths below assume you are already inside it.
2. Create a Python 3.12 environment. On Ubuntu, install the system packages needed for pandapower and scientific Python (for example: `python3.12`, `python3.12-venv`, build tools, and the BLAS/LAPACK stack provided by your distribution).
3. Create and activate the venv:
   ```bash
   uv venv .venv --python=3.12
   ```
4. Install the pinned dependencies from `pyproject.toml` (the file was generated from `uv pip freeze --python .venv/bin/python`). The exact pinned versions are recorded there.
5. Run the candidate generation script:
   ```bash
   uv run prepare_candidates.py
   ```
   This created three candidate scenario sets in `temp/datasets/` using pandapower's IEEE European LV asymmetric feeder and `runpp_3ph`. I used the 10-seed, 360-scenario configuration described in `prepare_candidates.py` and sampled household load scaling factors in `[0.5, 1.5]`.
6. Standardize the selected dataset (the final selection is `ieee_european_lv_asymmetric__on_peak_566_correlated`):
   ```bash
   uv run data.py
   ```
7. Validate the output schema and generate preview files:
   ```bash
   export SKILL_DIR="/ai-inventor/.claude/skills/aii-json"
   export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
   $PY "$SKILL_DIR/scripts/aii_json_validate_schema.py" --format exp_sel_data_out --file full_data_out.json
   $PY "$SKILL_DIR/scripts/aii_json_format_mini_preview.py" --format exp_sel_data_out --input full_data_out.json
   cp -f mini_full_data_out.json mini_data_out.json
   cp -f preview_full_data_out.json preview_data_out.json
   ```

Hardware and runtime:
- Hardware: CPU-only container, 2 CPUs, 14 GB RAM, no GPU.
- Candidate generation took about 4 minutes total.
- Standardization and validation took well under a minute.

Outputs you should see:
- `dataset_parts/dataset_part_*.json`: split raw 3600-scenario feeder simulation output.
- `temp/datasets/*.json`: candidate datasets.
- `full_data_out.json`: schema-valid selected dataset used downstream.
- `mini_data_out.json` and `preview_data_out.json`: truncated inspection files.
- The selected dataset is the one used for GP power-flow experiments and should appear in the paper's dataset description and appendix tables.

Notes:
- No private user-uploaded data was used; if a reader wants to reproduce from a different uploaded source, they must supply their own copy.
- All code in this folder uses relative paths anchored on the workspace directory.
