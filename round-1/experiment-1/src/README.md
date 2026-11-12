# Phase-Aware GP Kernel Experiment

This artifact implements the experiment to evaluate a Gaussian Process emulator with a physics-informed, phase-aware kernel derived from the three-phase admittance matrix (Y_abc) of the IEEE European LV distribution network.

## What This Artifact Does

This experiment compares five methods for predicting per-bus phase voltages in an unbalanced low-voltage distribution feeder:
1. **Phase-Aware GP** - Green's function kernel from Y_abc (novel method)
2. **RBF GP** - Generic radial basis function kernel on load injections
3. **Graph-Laplacian GP** - Topological Laplacian pseudoinverse kernel
4. **LinDistFlow + Quadratic** - Linearized power flow baseline
5. **Shuffled-Phase GP** - Control with randomized phase labels

Evaluation metrics: RMSE (accuracy) and NLPD (uncertainty calibration) on far-end bus voltages over multiple random seeds.

## Repository Layout

```
gen_art_experiment_1/
├── method.py                              # Main experiment script
├── full_method_out.json                   # Complete experiment results (exp_gen_sol_out.json format)
├── mini_method_out.json                   # Mini version (same format)
├── preview_method_out.json                # Preview version (same format)
├── full_method_out_detailed.json          # Full metrics and aggregated results
├── mini_method_out_detailed.json          # Mini version of detailed results
├── preview_method_out_detailed.json       # Preview version of detailed results
├── results_rmse_nlpd.png                  # RMSE and NLPD vs training size
├── results_margin.png                     # Margin over baselines
├── reproducibility.md                     # Step-by-step reproduction instructions
├── .aii/manifest.yaml                     # File retention policy
├── pyproject.toml                         # Project config with pinned dependencies
├── requirements.txt                       # Pinned dependencies list
└── .venv/                                 # Virtual environment (not published)
```

## How to Run

### Quick Test (Synthetic Data, ~30 seconds)
```bash
cd gen_art_experiment_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install -e .
python3 method.py
```

### Full Experiment (Real Power Flow, ~15-30 minutes)
See `reproducibility.md` for detailed instructions.

## Key Results

The synthetic test demonstrates the experimental framework works. The full experiment with real pandapower data evaluates:

- Training set sizes: 10, 20, 40, 80, 160 load flow solutions
- Test set: 50+ load scenarios
- 10 random seeds for statistical robustness
- Far-end buses: top 20% by electrical distance from substation

## Restoring Removed Files

The `__pycache__/` directory can be regenerated:
```bash
python3 -m compileall .
```

The virtual environment can be recreated:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Output Files

All results are saved in the JSON output files with the following structure:
- `config`: Experiment configuration
- `network_info`: Network parameters
- `seeds`: Per-seed results
- `aggregated`: Mean/std RMSE and NLPD across seeds

Figures are saved as PNG files in the same directory.