# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear
Security Administration. All rights in the program are reserved by Triad
National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others
acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute copies
to the public, perform publicly and display publicly, and to permit others
to do so.
==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Lightweight, dependency-free (stdlib-only) benchmark-metric logging helpers.

Each pipeline stage calls `log_stage_metrics(...)` at the points where it
already prints its stats/metrics to the console. The metrics are merged into a
per-stage JSON file (``benchmark_<stage>.json``) inside that stage's ``output``
folder. At the end of stage 4 (Deploy), ``generate_benchmark_report.py``
globs every stage's ``output`` folder for these JSON files and assembles a
single, colour-coded Markdown benchmark report.

Design goals:
    * Standard library only (json, os, datetime) - no new dependencies.
    * Fail-safe: a logging error must NEVER break a pipeline run.
    * Additive: stages keep printing exactly as before; we only persist.
"""

import os
import json
import datetime
import numpy as np

# Canonical stage keys used in the benchmark JSON filenames / report sections.
STAGE_DATA_PROCESSING = "2_data_processing"
STAGE_TRAINING = "3_1_training"
STAGE_TESTING = "3_2_testing"
STAGE_PORT_MOOSE = "4_1_port_moose"

# Human-friendly stage titles for the report.
STAGE_TITLES = {
    STAGE_DATA_PROCESSING: "Stage 2 - Data Processing",
    STAGE_TRAINING: "Stage 3.1 - Surrogate Training",
    STAGE_TESTING: "Stage 3.2 - Surrogate Testing",
    STAGE_PORT_MOOSE: "Stage 4.1 - Export to MOOSE",
}

# Quality thresholds (mirror utils.sm_build._print_r2_report conventions).
R2_GOOD = 0.99      # green  at/above this
R2_OK = 0.95        # yellow at/above this, red below
# Relative-RMSE thresholds (fraction, e.g. 0.05 == 5 %).
RELRMSE_GOOD = 0.05
RELRMSE_OK = 0.10

# Bootstrap CI width thresholds (on the same scale as relative RMSE).
# Narrow CI = low uncertainty in the error estimate → good.
BOOTSTRAP_WIDTH_GOOD = 0.01
BOOTSTRAP_WIDTH_OK = 0.03


def _json_safe(obj):
    """Recursively convert numpy / masked / non-serialisable values to plain
    Python so ``json.dump`` never raises."""
    # Lazy import so this module stays import-light if numpy is absent.
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is always present in this repo
        np = None

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ma.MaskedArray):
            return _json_safe(obj.filled(np.nan).tolist())
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        if isinstance(obj, np.generic):
            val = obj.item()
            if isinstance(val, float) and (val != val):  # NaN
                return None
            return val
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    # Last resort: stringify unknown objects (e.g. scaler instances).
    try:
        return str(obj)
    except Exception:
        return None



def benchmark_path(output_dir, stage_key):
    """Return the absolute path of the benchmark JSON for a stage/output dir."""
    return os.path.join(output_dir, "benchmark_{}.json".format(stage_key))


def log_stage_metrics(stage_key, metrics, output_dir="./output"):
    """Merge ``metrics`` into ``benchmark_<stage_key>.json`` inside ``output_dir``.

    Re-runs UPDATE the existing file (shallow-merge on top-level keys) rather
    than clobber it, and always refresh a ``timestamp`` field. Any failure is
    swallowed with a printed warning so the pipeline never crashes on logging.

    Parameters
    ----------
    stage_key : str
        One of the ``STAGE_*`` constants in this module.
    metrics : dict
        JSON-serialisable (best-effort) mapping of metric name -> value.
    output_dir : str
        Directory the calling stage already writes its artefacts to.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = benchmark_path(output_dir, stage_key)

        existing = {}
        if os.path.isfile(path):
            try:
                with open(path, "r") as fh:
                    existing = json.load(fh) or {}
            except Exception:
                existing = {}

        existing.update(_json_safe(metrics))
        existing["stage_key"] = stage_key
        existing["stage_title"] = STAGE_TITLES.get(stage_key, stage_key)
        existing["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(existing, fh, indent=2, sort_keys=False)
        os.replace(tmp, path)
        print("[benchmark] Logged metrics for '{}' -> {}".format(stage_key, path))
    except Exception as exc:  # never break the pipeline because of logging
        print("[benchmark] WARNING: failed to log metrics for '{}': {}".format(stage_key, exc))


# ----------------------------------------------------------------------
# Quality-rating helpers (shared by the console + the Markdown generator)
# ----------------------------------------------------------------------
def r2_rating(value):
    """Return ('good'|'ok'|'bad') for an R-squared value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v >= R2_GOOD:
        return "good"
    if v >= R2_OK:
        return "ok"
    return "bad"


def relrmse_rating(value):
    """Return ('good'|'ok'|'bad') for a relative-RMSE fraction (lower=better)."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if v <= RELRMSE_GOOD:
        return "good"
    if v <= RELRMSE_OK:
        return "ok"
    return "bad"


def bootstrap_width_rating(ci_low, ci_high):
    """Return ('good'|'ok'|'bad') for bootstrap CI width (narrower = better)."""
    try:
        width = abs(float(ci_high) - float(ci_low))
    except (TypeError, ValueError):
        return "unknown"
    if width <= BOOTSTRAP_WIDTH_GOOD:
        return "good"
    if width <= BOOTSTRAP_WIDTH_OK:
        return "ok"
    return "bad"

# NEW for bootstrapping for test result benchmarking
def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=1000, confidence=0.95, seed=42):
    """
    Bootstrap confidence interval for a metric that depends on y_true and y_pred.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (N,)
        Ground truth and predictions on the test set.
    metric_fn : callable
        Function with signature metric_fn(y_true, y_pred) -> float.
        Examples: r2_score, relative_rmse.
    n_boot : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g. 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        point : float          # metric on the original data
        ci_low : float         # lower percentile
        ci_high : float        # upper percentile
        mean : float           # mean of bootstrap distribution
        std : float            # std of bootstrap distribution
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
    boot_stats[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_stats, 100 * alpha))
    ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean": float(np.mean(boot_stats)),
        "std": float(np.std(boot_stats)),
        "n_boot": n_boot,
        "confidence": confidence,
    }


def relative_rmse(y_true, y_pred):
    """Relative RMSE (fraction)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.mean(np.abs(y_true)) + 1e-12
    return rmse / denom


def r2_score(y_true, y_pred):
    """Coefficient of determination."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def bootstrap_relrmse_from_errorinc(errorinc_dict, z_key, n_boot=1000, confidence=0.95, seed=42):
    """Run bootstrap on the per-increment relative RMSE values already computed
    by sma.errorinc for a given output z_key.

    Returns the same dict shape as bootstrap_metric.
    """
    try:
        ei = errorinc_dict
        rel_vals = np.ma.compressed(ei['REL_RMSE'])
        # Use the per-increment relative errors as the population to resample
        # (this approximates the distribution of the mean rel-RMSE).
        def _mean_abs(x):
            return float(np.mean(np.abs(x)))

        # We treat the already-computed per-increment rel-rmse values as observations
        point = float(np.mean(rel_vals))
        rng = np.random.default_rng(seed)
        n = len(rel_vals)
        boot_stats = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_stats[i] = _mean_abs(rel_vals[idx])

        alpha = (1.0 - confidence) / 2.0
        ci_low = float(np.percentile(boot_stats, 100 * alpha))
        ci_high = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
        return {
            "point": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean": float(np.mean(boot_stats)),
            "std": float(np.std(boot_stats)),
            "n_boot": n_boot,
            "confidence": confidence,
        }
    except Exception:
        return None
