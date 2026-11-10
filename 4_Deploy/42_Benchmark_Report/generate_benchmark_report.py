#!/usr/bin/env python3
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

Generate a single, color-coded Markdown BENCHMARK REPORT that summarises the
whole LAROMance pipeline. It globs every stage's ``output`` folder for the
``benchmark_*.json`` files written throughout stages 2 -> 4 (via
``utils.sm_report.log_stage_metrics``) and assembles:

    * Data statistics (sims, total data points, avg increments/sim)
    * Metadata: inputs/outputs + scalers + ranges + ROI
    * Model architecture (grid / mlp / pointnet / moe) and its settings
    * Training metrics (R2, condition number, wall-times)
    * Testing metrics (relative RMSE per output, total + per-increment wall-time)
    * Code-to-code comparison (MOOSE surrogate vs. Python reference), read from
      the last row of a .csv in the ``references`` folder (if present)
    * Ported MOOSE file names

Metrics are color-rated green/yellow/red via colored text with the same
thresholds used elsewhere in the codebase (R2 >=.99 green, >=.95 yellow, else
red). Code-to-code max relative differences use: < 1e-12 green,
1e-12 .. 1e-10 yellow, > 1e-10 red.

Usage (from this directory):
    python generate_benchmark_report.py
    python generate_benchmark_report.py --repo-root /path/to/LAROMance_process \
                                        --output benchmark_report.md \
                                        --references-dir ./references
"""

import os
import sys
import csv
import glob
import json
import math
import argparse
import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from utils import sm_report as smr

# ANSI colors for the short console summary (match sm_build._print_r2_report).
_GREEN, _YEL, _RED, _BOLD, _END = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
_ANSI = {"good": _GREEN, "ok": _YEL, "bad": _RED, "unknown": ""}
# Hex colors used for inline-HTML colored text in the Markdown report.
_HEX = {"good": "#2ca02c", "ok": "#e6a700", "bad": "#d62728", "unknown": "#808080"}

# Code-to-code (MOOSE vs Python) max-relative-difference thresholds.
C2C_GOOD = 1e-12   # below this -> green
C2C_OK = 1e-10     # between C2C_GOOD and C2C_OK -> yellow; above -> red


def c2c_rating(value):
    """Rate a code-to-code max relative difference (smaller is better)."""
    if value is None:
        return "unknown"
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(v):
        return "unknown"
    if v < C2C_GOOD:
        return "good"
    if v <= C2C_OK:
        return "ok"
    return "bad"


def find_benchmark_jsons(repo_root):
    """Return {stage_key: parsed_json} by globbing every stage output folder.

    If more than one file maps to the same stage key (e.g. a stale copy left in
    an old directory), the one with the newest file-modification time wins and a
    warning is printed so the mismatch is visible."""
    found = {}
    found_mtime = {}
    pattern = os.path.join(repo_root, "**", "output", "benchmark_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            mtime = os.path.getmtime(path)
        except Exception as exc:
            print(f"[report] WARNING: could not read {path}: {exc}")
            continue
        key = data.get("stage_key") or os.path.basename(path)[len("benchmark_"):-len(".json")]
        # Keep the file that was most recently WRITTEN (mtime), not the one with
        # the lexicographically-largest timestamp string.
        if key in found:
            older = path if mtime <= found_mtime[key] else found[key]["_source_path_abs"]
            newer = found[key]["_source_path_abs"] if mtime <= found_mtime[key] else path
            print(f"[report] WARNING: multiple '{key}' benchmark files found; "
                  f"using newest ({os.path.relpath(newer, repo_root)}), "
                  f"ignoring {os.path.relpath(older, repo_root)}.")
        if key not in found or mtime > found_mtime[key]:
            data["_source_path"] = os.path.relpath(path, repo_root)
            data["_source_path_abs"] = os.path.abspath(path)
            data["_stage_dir"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            found[key] = data
            found_mtime[key] = mtime
    return found


# ----------------------------------------------------------------------
# Code-to-code comparison (MOOSE vs Python) from references/*.csv
# ----------------------------------------------------------------------
def _to_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _locate_references_dir(user_dir=None):
    """Find the ``references`` folder: explicit arg, else ./references (cwd),
    else <script dir>/references. Returns None if it does not exist."""
    candidates = []
    if user_dir:
        candidates.append(user_dir)
    candidates.append(os.path.join(os.getcwd(), "references"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "references"))
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return None


def _expected_num_outputs(stages):
    """Number of output variables, from the data stage (fallback: testing stage)."""
    data = stages.get(smr.STAGE_DATA_PROCESSING) or {}
    outs = data.get("output_variables")
    if outs:
        return len(outs)
    test = stages.get(smr.STAGE_TESTING) or {}
    rr = test.get("testing_metrics", {}).get("relative_rmse_per_output", {})
    return len(rr) if rr else None


def read_reference_comparison(stages, references_dir=None):
    """Read the last row of the .csv in ``references`` and return the values of
    the last N columns (N = number of outputs).

    Returns ``None`` if the folder / csv is missing or unusable, otherwise
    ``{"file": abs_path, "values": [(column_name, float_or_None), ...]}``."""
    ref_dir = _locate_references_dir(references_dir)
    if ref_dir is None:
        print("[report] No 'references' folder found; skipping code-to-code comparison.")
        return None

    csvs = sorted(glob.glob(os.path.join(ref_dir, "verification_out.csv")))
    if not csvs:
        print(f"[report] No .csv file in {ref_dir}; skipping code-to-code comparison.")
        return None
    if len(csvs) > 1:
        csvs.sort(key=os.path.getmtime)
        print(f"[report] WARNING: multiple .csv files in {ref_dir}; "
              f"using newest ({os.path.basename(csvs[-1])}).")
    path = csvs[-1]

    n_out = _expected_num_outputs(stages)
    if not n_out:
        print("[report] WARNING: could not determine number of outputs; "
              "skipping code-to-code comparison.")
        return None

    try:
        with open(path, "r", newline="") as fh:
            rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    except Exception as exc:
        print(f"[report] WARNING: could not read {path}: {exc}")
        return None
    if not rows:
        print(f"[report] WARNING: {path} is empty; skipping code-to-code comparison.")
        return None

    # First row is a header if any of its cells is non-numeric.
    has_header = any(_to_float(c) is None for c in rows[0] if c.strip())
    header = [c.strip() for c in rows[0]] if has_header else None
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        print(f"[report] WARNING: {path} has a header but no data rows; skipping.")
        return None

    last = [c.strip() for c in data_rows[-1]]
    if n_out > len(last):
        print(f"[report] WARNING: csv has only {len(last)} columns but {n_out} outputs "
              f"expected; using {len(last)}.")
        n_out = len(last)

    tail = last[-n_out:]
    if header and len(header) == len(last):
        names = header[-n_out:]
    else:
        names = [f"output_{i + 1}" for i in range(n_out)]

    return {"file": path,
            "values": [(n, _to_float(v)) for n, v in zip(names, tail)]}


def _abspath(path):
    """Normalise a path to an absolute path for consistent reporting."""
    if not path:
        return None
    return os.path.abspath(path)


def _stage_file(stage, subdir, filename):
    """Resolve ``filename`` to a full path inside a stage's ``subdir`` folder.

    ``stage`` is a parsed benchmark dict (carrying ``_stage_dir``); if the
    filename is already absolute it is returned as-is (normalised)."""
    if not filename:
        return None
    if os.path.isabs(filename):
        return os.path.abspath(filename)
    stage_dir = stage.get("_stage_dir") if isinstance(stage, dict) else None
    if stage_dir:
        return os.path.abspath(os.path.join(stage_dir, subdir, filename))
    return filename


def _fmt(val, fmt="{}"):
    if val is None:
        return "_n/a_"
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


def _rated(value, rating, fmt="{:.4f}"):
    """Return a color-styled (inline-HTML) markdown fragment for a rated metric."""
    color = _HEX.get(rating, _HEX["unknown"])
    return f'<span style="color:{color}">{_fmt(value, fmt)}</span>'


# ----------------------------------------------------------------------
# Markdown section builders
# ----------------------------------------------------------------------
def _group(md, title):
    """Emit a bold full-width group-divider row inside the single metrics table."""
    md.append(f"| **{title}** | |")


def _row(md, label, value):
    md.append(f"| {label} | {value} |")


def build_markdown(stages, c2c=None):
    data = stages.get(smr.STAGE_DATA_PROCESSING) or {}
    train = stages.get(smr.STAGE_TRAINING) or {}
    test = stages.get(smr.STAGE_TESTING) or {}
    port = stages.get(smr.STAGE_PORT_MOOSE) or {}

    md = ["# LAROMance Benchmark Report"]
    md.append(f"_{datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    md.append("| Metric | Value |")
    md.append("|:--|:--|")

    # ---- Data -------------------------------------------------------------
    _group(md, "DATA")
    if data:
        for part in ("train", "test"):
            p = data.get(part)
            if not p:
                continue
            _row(md, f"{part} · simulations", _fmt(p.get("num_simulations")))
            _row(md, f"{part} · data points", _fmt(p.get("total_data_points"), "{:,}"))
            _row(md, f"{part} · increments/sim (avg · max)",
                 f"{_fmt(p.get('avg_increments_per_simulation'), '{:.0f}')} · "
                 f"{_fmt(p.get('max_increments_per_simulation'))}")
        aug = data.get("augmentation")
        if aug:
            _row(md, "augmentation · +pts (total · avg/sim)",
                 "{} · {}".format(
                     _fmt(aug.get("total_points_added"), "{:,}"),
                     _fmt(aug.get("avg_points_added_per_sim"), "{:.0f}")))
        _row(md, "inputs", ", ".join(data.get("input_variables", [])) or "—")
        _row(md, "outputs", ", ".join(data.get("output_variables", [])) or "—")
    else:
        _row(md, "_missing_", "")

    # ---- Training ---------------------------------------------------------
    _group(md, "TRAINING")
    if train:
        tm = train.get("training_metrics", {})
        r2 = tm.get("r2") or tm.get("r2_scaled") or {}
        for k, v in r2.items():
            _row(md, f"R² · `{k}`", _rated(v, smr.r2_rating(v), "{:.4f}"))
        if "M_condition" in tm:
            _row(md, "condition number (M)", _fmt(tm.get("M_condition"), "{:.1f}"))
    else:
        _row(md, "_missing_", "")

    # ---- Testing ----------------------------------------------------------
    _group(md, "TESTING")
    if test:
        tm = test.get("testing_metrics", {})
        rr = tm.get("relative_rmse_per_output", {})
        for k, v in rr.items():
            _row(md, f"rel-RMSE · `{k}` (mean · std)",
                 "{} · {}".format(
                     _rated(v.get("mean"), smr.relrmse_rating(v.get("mean")), "{:.4f}"),
                     _fmt(v.get("std"), "{:.4f}")))
        total = tm.get("total_wall_time_s")
        wpi = tm.get("wall_time_per_increment_s")
        if total is not None or wpi is not None:
            _row(md, "wall-time · (total · per increment)",
                 "{} · {}".format(
                     f"{total:.2f} s" if total is not None else "_n/a_",
                     f"{wpi*1e3:.3f} ms" if wpi is not None else "_n/a_"))
    else:
        _row(md, "_missing_", "")

    # ---- Code-to-code (MOOSE vs Python) -----------------------------------
    if c2c:
        _group(md, "CODE-TO-CODE (MOOSE vs Python)")
        for name, val in c2c["values"]:
            _row(md, f"max rel. diff · `{name}`",
                 _rated(val, c2c_rating(val), "{:.3e}"))
        _row(md, "reference file", f"`{c2c['file']}`")

    # ---- Model & Deployment ----------------------------------------------
    _group(md, "MODEL & DEPLOYMENT")
    if train or port:
        if train:
            _row(md, "architecture", f"`{train.get('model_architecture', '?')}`")
            for k, v in train.get("model_settings", {}).items():
                _row(md, k, _fmt(v))
        if port:
            _row(md, "model name", f"`{_fmt(port.get('model_name'))}`")
            for f in port.get("ported_files", []):
                _row(md, "ported file", f"`{_stage_file(port, 'output', f)}`")
            vt = _stage_file(port, "output", port.get("verification_table"))
            if vt:
                _row(md, "verification table", f"`{vt}`")
    else:
        _row(md, "_missing_", "")

    md.append("")
    key = (f'_color key — R²: <span style="color:{_HEX["good"]}">≥{smr.R2_GOOD}</span> '
           f'<span style="color:{_HEX["ok"]}">≥{smr.R2_OK}</span> '
           f'<span style="color:{_HEX["bad"]}">&lt;{smr.R2_OK}</span> · '
           f'rel-RMSE: <span style="color:{_HEX["good"]}">≤{smr.RELRMSE_GOOD}</span> '
           f'<span style="color:{_HEX["ok"]}">≤{smr.RELRMSE_OK}</span> '
           f'<span style="color:{_HEX["bad"]}">&gt;{smr.RELRMSE_OK}</span>')
    if c2c:
        key += (f' · code-to-code: <span style="color:{_HEX["good"]}">&lt;{C2C_GOOD:.0e}</span> '
                f'<span style="color:{_HEX["ok"]}">{C2C_GOOD:.0e}–{C2C_OK:.0e}</span> '
                f'<span style="color:{_HEX["bad"]}">&gt;{C2C_OK:.0e}</span>')
    key += "_"
    md.append(key)
    return "\n".join(md)


def print_console_summary(stages, c2c=None):
    print(f"\n{_BOLD}===== Benchmark summary ====={_END}")
    train = stages.get(smr.STAGE_TRAINING)
    if train:
        r2 = (train.get("training_metrics", {}).get("r2")
              or train.get("training_metrics", {}).get("r2_scaled") or {})
        for k, v in r2.items():
            rating = smr.r2_rating(v)
            print(f"\tR² {k}: {_ANSI.get(rating,'')}{v:.4f}{_END}")
    test = stages.get(smr.STAGE_TESTING)
    if test:
        rr = test.get("testing_metrics", {}).get("relative_rmse_per_output", {})
        for k, v in rr.items():
            rating = smr.relrmse_rating(v.get("mean"))
            print(f"\trel-RMSE {k}: {_ANSI.get(rating,'')}{v.get('mean'):.4f}{_END}")
    if c2c:
        for name, val in c2c["values"]:
            rating = c2c_rating(val)
            txt = f"{val:.3e}" if val is not None else "n/a"
            print(f"\tcode-to-code max rel. diff {name}: {_ANSI.get(rating,'')}{txt}{_END}")


def main():
    ap = argparse.ArgumentParser(description="Generate the LAROMance benchmark report.")
    default_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    ap.add_argument("--repo-root", default=default_root,
                    help="Repository root to glob for benchmark_*.json (default: auto).")
    ap.add_argument("--output", default="./output/benchmark_report.md",
                    help="Markdown output path (default: ./output/benchmark_report.md).")
    ap.add_argument("--references-dir", default=None,
                    help="Folder holding the MOOSE-vs-Python comparison .csv "
                         "(default: ./references, then <script dir>/references).")
    args = ap.parse_args()

    stages = find_benchmark_jsons(args.repo_root)
    if not stages:
        print(f"[report] No benchmark_*.json files found under {args.repo_root}.")
        print("[report] Run the pipeline stages first (they log metrics automatically).")
        return 1

    c2c = read_reference_comparison(stages, args.references_dir)

    markdown = build_markdown(stages, c2c)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(markdown)
    print(f"[report] Benchmark report written to: {out_path}")
    print_console_summary(stages, c2c)
    return 0


if __name__ == "__main__":
    sys.exit(main())