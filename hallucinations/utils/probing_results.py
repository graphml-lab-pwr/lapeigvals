import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm.auto import tqdm

from hallucinations.dirs import ResultsDir


def load_probing_results(
    root_dir: Path | str,
    res_file_name: str,
    date_threshold: datetime | None = None,
    num_proc: int = 1,
    return_preds: bool = False,
    ignore_errors: bool = False,
) -> list[dict[str, Any]]:
    results = []

    loader_func = partial(
        load_results_for_single_probe,
        date_threshold=date_threshold,
        return_preds=return_preds,
        ignore_errors=ignore_errors,
    )
    results_files = list(Path(root_dir).rglob(res_file_name))
    with Pool(processes=num_proc) as pool:
        for exp_res in tqdm(
            pool.imap(loader_func, results_files),
            total=len(results_files),
            desc="Loading probing results",
        ):
            if exp_res is not None:
                results.extend(exp_res)
    return results


def load_results_for_single_probe(
    res_file: Path | str,
    date_threshold: datetime | None = None,
    return_preds: bool = False,
    ignore_errors: bool = False,
) -> list[dict[str, Any]] | None:
    """
    Structure of the results file:
    {
        "probe_name": [
            {
                "metadata": {"trainable_params": ..., "layer_idx": ..., ...},
                "metrics": {"train_auc": ..., "test_auc": ..., ...},
                "train_proba": list[float],
                "test_proba": list[float],
                "train_preds": list[int],
                "test_preds": list[int],
            },
            ...
        ]
    }
    """
    res_file = Path(res_file)

    file_mtime = datetime.fromtimestamp(res_file.stat().st_mtime)
    if date_threshold is not None and file_mtime < date_threshold:
        print(f"Results for {res_file} are older than {date_threshold.isoformat()}")
        return None

    rr = torch.load(res_file, weights_only=True, mmap=True)

    res_dir = ResultsDir(res_file.parent)

    try:
        ds_metadata = res_dir.dataset_dir.get_dataset_metadata()
        probe_metadata = get_probe_metadata(res_file)
    except FileNotFoundError as e:
        if ignore_errors:
            logger.error(f"Error loading configuration for {res_file}: {e}")
            return None
        else:
            raise e

    def _to_record(probe_name: str, probe_results: dict[str, Any]) -> dict[str, Any]:
        res = {
            "res_dir": res_dir,
            "probe": probe_name,
            **ds_metadata,
            **probe_metadata,
            **probe_results["metadata"],
            **probe_results["metrics"],
        }
        if return_preds:
            res["train_preds"] = probe_results["train_preds"]
            res["test_preds"] = probe_results["test_preds"]
            res["train_proba"] = probe_results["train_proba"]
            res["test_proba"] = probe_results["test_proba"]
        return res

    exp_results = []

    for probe_name, probe_results in rr.items():
        if isinstance(probe_results, list):
            for per_layer_res in probe_results:
                exp_results.append(_to_record(probe_name, per_layer_res))
        else:
            exp_results.append(_to_record(probe_name, probe_results))

    return exp_results


def get_probe_metadata(res_file: Path | str) -> dict[str, Any]:
    if isinstance(res_file, Path):
        fname = res_file.name
    else:
        fname = res_file

    match = re.search(r"attn_vs_laplacian_pca_(\d+)_results\.pkl", fname)
    if match:
        pca_dim = int(match.group(1))
    else:
        pca_dim = None

    return {
        "configured_pca_dim": pca_dim,
    }
