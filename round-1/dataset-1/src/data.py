#!/usr/bin/env python3
"""Standardize candidate LV feeder datasets to exp_sel_data_out.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

WORKSPACE = Path(__file__).resolve().parent
SOURCE_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"
LOG_DIR = WORKSPACE / "logs"
DATASET_PARTS_DIR = WORKSPACE / "dataset_parts"
KEEP_TOP_N = 1
FINAL_DATASET_NAME = "ieee_european_lv_asymmetric__on_peak_566_correlated"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "data.log", rotation="30 MB", level="DEBUG")


def _score_candidate(payload: dict[str, Any]) -> float:
    stats = payload.get("summary_stats", {})
    return float(stats.get("mean_far_end_voltage_std", 0.0))


def _load_candidates() -> list[dict[str, Any]]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source directory: {SOURCE_DIR}")
    candidates: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source_path"] = str(path)
        candidates.append(payload)
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {SOURCE_DIR}")
    candidates.sort(key=_score_candidate, reverse=True)
    selected = [c for c in candidates if c["dataset_name"] == FINAL_DATASET_NAME]
    if not selected:
        raise RuntimeError(f"Could not find required dataset {FINAL_DATASET_NAME} in {SOURCE_DIR}")
    return selected[:KEEP_TOP_N]


def _example_to_record(example: dict[str, Any], dataset_payload: dict[str, Any]) -> dict[str, Any]:
    input_obj = {
        "injection_vector_pq": example["injection_vector_pq"],
        "load_scaling_factors": example["load_scaling_factors"],
        "seed": example["seed"],
        "scenario_id": example["scenario_id"],
        "row_index": example["row_index"],
    }
    output_obj = example["voltages"]
    return {
        "input": json.dumps(input_obj, separators=(",", ":")),
        "output": json.dumps(output_obj, separators=(",", ":")),
        "metadata_fold": int(example["seed"]),
        "metadata_seed": int(example["seed"]),
        "metadata_scenario_id": int(example["scenario_id"]),
        "metadata_row_index": int(example["row_index"]),
        "metadata_task_type": "regression",
        "metadata_n_targets": int(len(output_obj["vm_a_pu"]) * 3),
        "metadata_feature_names": dataset_payload["feature_names"],
    }


def _convert_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    examples = [_example_to_record(example, payload) for example in payload["examples"]]
    return {
        "dataset": payload["dataset_name"],
        "examples": examples,
    }


@logger.catch(reraise=True)
def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading candidate datasets from {}", SOURCE_DIR)
    candidates = _load_candidates()
    logger.info("Selected top {} datasets by far-end voltage spread", len(candidates))

    datasets = [_convert_dataset(payload) for payload in candidates]
    output = {
        "metadata": {
            "source_dir": str(SOURCE_DIR),
            "selection_rule": "sorted by summary_stats.mean_far_end_voltage_std descending",
            "keep_top_n": KEEP_TOP_N,
            "n_candidate_files": len(list(SOURCE_DIR.glob("*.json"))),
            "dataset_names": [payload["dataset_name"] for payload in candidates],
            "source_files": [payload["_source_path"] for payload in candidates],
        },
        "datasets": datasets,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote {} with {} datasets", OUTPUT_PATH, len(datasets))


if __name__ == "__main__":
    main()
