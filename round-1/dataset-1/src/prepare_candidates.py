#!/usr/bin/env python3
"""Generate two candidate LV feeder datasets under temp/datasets/."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import resource
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import psutil
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent
TEMP_DATASETS_DIR = WORKSPACE / "temp" / "datasets"
LOG_DIR = WORKSPACE / "logs"
N_EXAMPLES = 360
N_SEEDS = 10
EXAMPLES_PER_SEED = N_EXAMPLES // N_SEEDS
SIGMA = 0.25
SCALE_MIN = 0.5
SCALE_MAX = 1.5
TRAIN_SIZES = [10, 20, 40, 80, 160]
TEST_SIZE = 200
CANDIDATE_SPECS = [
    # (dataset_suffix, ieee_scenario, sampling_mode)
    ("on_peak_566", "on_peak_566", "independent"),
    ("off_peak_1", "off_peak_1", "independent"),
    ("on_peak_566_correlated", "on_peak_566", "correlated"),
]
# Correlated-demand sampling: one shared household activity factor per scenario
# times a per-household idiosyncratic factor, clipped to the same [0.5, 1.5] band.
SIGMA_GLOBAL = 0.12
SIGMA_IDIOSYNCRATIC = 0.20

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "prepare_candidates.log", rotation="30 MB", level="DEBUG")


def _container_ram_bytes() -> int | None:
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            value = Path(path).read_text().strip()
            if value != "max":
                limit = int(value)
                if limit < 1_000_000_000_000:
                    return limit
        except (FileNotFoundError, ValueError):
            continue
    return None


def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return max(1, math.ceil(int(parts[0]) / int(parts[1])))
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _set_memory_limit() -> None:
    total_ram = _container_ram_bytes() or int(psutil.virtual_memory().total)
    ram_budget = int(total_ram * 0.8)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ram_budget * 2, ram_budget * 2))
    except (ValueError, OSError):
        logger.warning("Could not set RLIMIT_AS; continuing without virtual memory cap")



def _load_phase_labels(net: pp.pandapowerNet) -> list[str]:
    phase_labels: list[str] = []
    for _, row in net.asymmetric_load.iterrows():
        if row["p_a_mw"] > 0 or row["q_a_mvar"] > 0:
            phase_labels.append("a")
        elif row["p_b_mw"] > 0 or row["q_b_mvar"] > 0:
            phase_labels.append("b")
        elif row["p_c_mw"] > 0 or row["q_c_mvar"] > 0:
            phase_labels.append("c")
        else:
            raise ValueError("Found asymmetric load with no active phase")
    return phase_labels



def _build_graph_and_far_end_buses(net: pp.pandapowerNet) -> tuple[list[int], list[int], dict[str, float]]:
    graph = nx.Graph()
    for bus_idx in net.bus.index:
        graph.add_node(int(bus_idx))

    for _, row in net.line.iterrows():
        weight = float(np.hypot(row["r_ohm_per_km"], row["x_ohm_per_km"]) * row["length_km"])
        graph.add_edge(int(row["from_bus"]), int(row["to_bus"]), weight=max(weight, 1e-9))

    for _, row in net.trafo.iterrows():
        z_base_ohm = (float(row["vn_lv_kv"]) ** 2) / float(row["sn_mva"])
        r_ohm = float(row["vkr_percent"]) / 100.0 * z_base_ohm
        x_pct = max(float(row["vk_percent"]) ** 2 - float(row["vkr_percent"]) ** 2, 0.0) ** 0.5
        x_ohm = x_pct / 100.0 * z_base_ohm
        weight = float(np.hypot(r_ohm, x_ohm))
        graph.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), weight=max(weight, 1e-9))

    slack_bus = int(net.ext_grid.iloc[0]["bus"])
    distances = nx.single_source_dijkstra_path_length(graph, slack_bus, weight="weight")
    n_far_end = max(1, math.ceil(len(net.bus) * 0.2))
    far_end_bus_indices = [
        int(bus_idx)
        for bus_idx, _ in sorted(distances.items(), key=lambda item: item[1], reverse=True)[:n_far_end]
    ]
    return far_end_bus_indices, [int(b) for b in net.bus.index.tolist()], {str(k): float(v) for k, v in distances.items()}



def _feature_names(n_loads: int) -> list[str]:
    names: list[str] = []
    for load_idx in range(n_loads):
        names.extend([f"load_{load_idx}_p_mw", f"load_{load_idx}_q_mvar"])
    return names



def _build_split_definitions(n_examples: int) -> list[dict[str, Any]]:
    split_definitions: list[dict[str, Any]] = []
    for split_id in range(5):
        rng = np.random.default_rng(10_000 + split_id)
        perm = rng.permutation(n_examples)
        test_indices = perm[:TEST_SIZE].tolist()
        train_pool = perm[TEST_SIZE : TEST_SIZE + max(TRAIN_SIZES)].tolist()
        train_indices_by_size = {
            str(train_size): train_pool[:train_size] for train_size in TRAIN_SIZES
        }
        split_definitions.append(
            {
                "split_id": split_id,
                "random_seed": 10_000 + split_id,
                "test_indices": test_indices,
                "train_indices_by_size": train_indices_by_size,
            }
        )
    return split_definitions



def _sample_scaling_factors(
    rng: np.random.Generator, n_loads: int, sampling_mode: str
) -> np.ndarray:
    """Draw the 55 household load scaling factors for one scenario.

    ``independent``: each household draws an independent log-normal factor.
    ``correlated``: one shared household activity factor per scenario multiplies
    a per-household idiosyncratic factor (realistic co-variation of household
    demand across an area).
    """
    if sampling_mode == "independent":
        raw = rng.lognormal(mean=-0.5 * SIGMA * SIGMA, sigma=SIGMA, size=n_loads)
    elif sampling_mode == "correlated":
        global_factor = rng.lognormal(mean=-0.5 * SIGMA_GLOBAL**2, sigma=SIGMA_GLOBAL)
        idio = rng.lognormal(
            mean=-0.5 * SIGMA_IDIOSYNCRATIC**2, sigma=SIGMA_IDIOSYNCRATIC, size=n_loads
        )
        raw = global_factor * idio
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    return np.clip(raw, SCALE_MIN, SCALE_MAX)


def _generate_candidate_dataset(
    dataset_suffix: str, scenario_name: str, sampling_mode: str
) -> dict[str, Any]:
    net = pn.ieee_european_lv_asymmetric(scenario=scenario_name)
    phase_labels = _load_phase_labels(net)
    far_end_bus_indices, bus_ids, electrical_distances = _build_graph_and_far_end_buses(net)

    load_bus_ids = [int(v) for v in net.asymmetric_load["bus"].tolist()]
    base_p = []
    base_q = []
    for _, row in net.asymmetric_load.iterrows():
        if row["p_a_mw"] > 0 or row["q_a_mvar"] > 0:
            base_p.append(float(row["p_a_mw"]))
            base_q.append(float(row["q_a_mvar"]))
        elif row["p_b_mw"] > 0 or row["q_b_mvar"] > 0:
            base_p.append(float(row["p_b_mw"]))
            base_q.append(float(row["q_b_mvar"]))
        else:
            base_p.append(float(row["p_c_mw"]))
            base_q.append(float(row["q_c_mvar"]))
    base_p_arr = np.asarray(base_p, dtype=float)
    base_q_arr = np.asarray(base_q, dtype=float)

    scenario_examples: list[dict[str, Any]] = []
    mu = -0.5 * SIGMA * SIGMA
    global_row_index = 0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        generated_this_seed = 0
        attempts = 0
        while generated_this_seed < EXAMPLES_PER_SEED:
            attempts += 1
            if attempts > EXAMPLES_PER_SEED * 20:
                raise RuntimeError(
                    f"Too many failed attempts for scenario={scenario_name}, seed={seed}"
                )
            scaling = _sample_scaling_factors(rng, len(net.asymmetric_load), sampling_mode)
            net.asymmetric_load["scaling"] = scaling
            pp.runpp_3ph(net, verbose=False)
            if not net.converged:
                continue

            p_vals = base_p_arr * scaling
            q_vals = base_q_arr * scaling
            injection_vector: list[float] = []
            for p_val, q_val in zip(p_vals.tolist(), q_vals.tolist(), strict=True):
                injection_vector.extend([float(p_val), float(q_val)])

            vm_a = net.res_bus_3ph["vm_a_pu"].astype(float).tolist()
            vm_b = net.res_bus_3ph["vm_b_pu"].astype(float).tolist()
            vm_c = net.res_bus_3ph["vm_c_pu"].astype(float).tolist()
            far_end_vm = []
            for bus_idx in far_end_bus_indices:
                far_end_vm.extend([vm_a[bus_idx], vm_b[bus_idx], vm_c[bus_idx]])

            scenario_examples.append(
                {
                    "row_index": global_row_index,
                    "seed": seed,
                    "scenario_id": generated_this_seed,
                    "load_scaling_factors": scaling.astype(float).tolist(),
                    "injection_vector_pq": injection_vector,
                    "voltages": {
                        "vm_a_pu": vm_a,
                        "vm_b_pu": vm_b,
                        "vm_c_pu": vm_c,
                    },
                    "far_end_voltage_vector_pu": far_end_vm,
                }
            )
            generated_this_seed += 1
            global_row_index += 1

    far_end_matrix = np.asarray([ex["far_end_voltage_vector_pu"] for ex in scenario_examples], dtype=float)
    all_scalings = np.asarray([ex["load_scaling_factors"] for ex in scenario_examples], dtype=float)

    return {
        "dataset_name": f"ieee_european_lv_asymmetric__{dataset_suffix}",
        "source": "pandapower.networks.ieee_european_lv_asymmetric",
        "ieee_test_feeder_source": "https://ieee-pes-data-sharing.org/datasets/detail/4e38bf3e-d4b3-480c-83e4-167b7bf5dece",
        "pandapower_docs": "https://pandapower.readthedocs.io/en/latest/networks/3phase_grids.html",
        "scenario": scenario_name,
        "n_examples": len(scenario_examples),
        "n_buses": len(bus_ids),
        "n_loads": len(load_bus_ids),
        "bus_ids": bus_ids,
        "load_bus_ids": load_bus_ids,
        "load_phase_labels": phase_labels,
        "feature_names": _feature_names(len(load_bus_ids)),
        "far_end_bus_indices": far_end_bus_indices,
        "electrical_distances_ohm_mag": electrical_distances,
        "phase_presence": "all buses represented with three phase voltage magnitudes in res_bus_3ph",
        "split_definitions": _build_split_definitions(len(scenario_examples)),
        "summary_stats": {
            "mean_scale": float(all_scalings.mean()),
            "std_scale": float(all_scalings.std()),
            "min_voltage_pu": float(far_end_matrix.min()),
            "max_voltage_pu": float(far_end_matrix.max()),
            "mean_far_end_voltage_std": float(far_end_matrix.std(axis=0).mean()),
            "mean_total_p_mw": float(np.mean([sum(ex["injection_vector_pq"][0::2]) for ex in scenario_examples])),
            "mean_total_q_mvar": float(np.mean([sum(ex["injection_vector_pq"][1::2]) for ex in scenario_examples])),
        },
        "examples": scenario_examples,
    }


@logger.catch(reraise=True)
def main() -> None:
    _set_memory_limit()
    TEMP_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating candidate datasets into {}", TEMP_DATASETS_DIR)
    num_workers = min(2, _detect_cpus(), len(CANDIDATE_SPECS))
    logger.info("Using {} worker(s) for {} candidate scenarios", num_workers, len(CANDIDATE_SPECS))

    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as executor:
        future_map = {
            executor.submit(_generate_candidate_dataset, *spec): spec
            for spec in CANDIDATE_SPECS
        }
        for future in as_completed(future_map):
            dataset_suffix, scenario_name, sampling_mode = future_map[future]
            result = future.result()
            results[dataset_suffix] = result
            out_path = TEMP_DATASETS_DIR / f"{result['dataset_name']}.json"
            out_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
            logger.info(
                "Saved {} examples for {} ({}) to {}",
                result["n_examples"],
                scenario_name,
                sampling_mode,
                out_path.name,
            )

    logger.info("Generated {} candidate datasets", len(results))
    for dataset_suffix, scenario_name, sampling_mode in CANDIDATE_SPECS:
        stats = results[dataset_suffix]["summary_stats"]
        logger.info(
            "{} ({}) | far-end voltage range [{:.6f}, {:.6f}] | mean far-end std {:.6f}",
            scenario_name,
            sampling_mode,
            stats["min_voltage_pu"],
            stats["max_voltage_pu"],
            stats["mean_far_end_voltage_std"],
        )


if __name__ == "__main__":
    main()
