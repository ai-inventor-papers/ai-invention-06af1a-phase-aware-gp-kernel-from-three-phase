#!/usr/bin/env python3
"""Sanity-check the candidate datasets: verify the voltage target carries signal.

Fits a cheap GP/RBF-style surrogate on a small training set and compares its
far-end RMSE against a mean-predictor baseline on the same test rows. A dataset
whose surrogate cannot beat the mean predictor, or whose far-end voltages have
near-zero spread, has no usable signal for the planned experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge

WORKSPACE = Path(__file__).resolve().parent
TEMP_DATASETS_DIR = WORKSPACE / "temp" / "datasets"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "sanity_check.log", rotation="30 MB", level="DEBUG")


def _evaluate(dataset: dict) -> dict[str, float]:
    examples = dataset["examples"]
    X = np.asarray([ex["injection_vector_pq"] for ex in examples], dtype=float)
    Y = np.asarray([ex["far_end_voltage_vector_pu"] for ex in examples], dtype=float)

    split = dataset["split_definitions"][0]
    train_idx = np.asarray(split["train_indices_by_size"]["40"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    mean_pred = np.repeat(Y_train.mean(axis=0, keepdims=True), X_test.shape[0], axis=0)
    rmse_mean = float(np.sqrt(np.mean((Y_test - mean_pred) ** 2)))

    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-4)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gpr.fit(X_train, Y_train)
    pred = gpr.predict(X_test)
    rmse_gp = float(np.sqrt(np.mean((Y_test - pred) ** 2)))

    ridge = Ridge(alpha=1e-3).fit(X_train, Y_train)
    rmse_ridge = float(np.sqrt(np.mean((Y_test - ridge.predict(X_test)) ** 2)))

    return {
        "n_train": float(train_idx.size),
        "n_test": float(test_idx.size),
        "target_spread_std": float(Y_train.std(axis=0).mean()),
        "rmse_mean_predictor": rmse_mean,
        "rmse_rbf_gp": rmse_gp,
        "rmse_ridge": rmse_ridge,
        "gp_skill_over_mean": rmse_mean / rmse_gp if rmse_gp > 0 else float("nan"),
    }


@logger.catch(reraise=True)
def main() -> None:
    results: dict[str, dict[str, float]] = {}
    for path in sorted(TEMP_DATASETS_DIR.glob("*.json")):
        dataset = json.loads(path.read_text(encoding="utf-8"))
        stats = _evaluate(dataset)
        results[dataset["dataset_name"]] = stats
        logger.info(
            "{name}: target spread={spread:.6f} | RMSE mean={m:.6f} | RMSE RBF-GP={g:.6f} | RMSE ridge={r:.6f} | GP skill={s:.2f}x",
            name=dataset["dataset_name"],
            spread=stats["target_spread_std"],
            m=stats["rmse_mean_predictor"],
            g=stats["rmse_rbf_gp"],
            r=stats["rmse_ridge"],
            s=stats["gp_skill_over_mean"],
        )

    out_path = WORKSPACE / "logs" / "sanity_check.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote {}", out_path)


if __name__ == "__main__":
    main()
