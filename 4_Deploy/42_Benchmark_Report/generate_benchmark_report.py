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

Generate a single, colour-coded Markdown BENCHMARK REPORT that summarises the
whole LAROMance pipeline. It globs every stage's ``output`` folder for the
``benchmark_*.json`` files written throughout stages 2 -> 4 (via
``utils.sm_report.log_stage_metrics``) and assembles:

    * Data statistics (sims, total data points, avg increments/sim)
    * Metadata: inputs/outputs + scalers + ranges + ROI
    * Model architecture (grid / mlp / pointnet / moe) and its settings
    * Training metrics (R2, condition number, wall-times)
    * Testing metrics (relative RMSE per output, total + per-increment wall-time)
    * Ported MOOSE file names

Metrics are colour-rated green/yellow/red via emoji badges (GitHub-safe) with
the same thresholds used elsewhere in the codebase (R2 >=.99 green, >=.95
yellow, else red).

Usage (from this directory):
    python generate_benchmark_report.py
    python generate_benchmark_report.py --repo-root /path/to/LAROMance_process \
                                        --output benchmark_report.md
"""

import os
import sys
import glob
import json
import argparse
import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from utils import sm_report as smr

# ANSI colours for the short console summary (match sm_build._print_r2_report).
_GREEN, _YEL, _RED, _BOLD, _END = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
_ANSI = {"good": _GREEN, "ok": _YEL, "bad": _RED, "unknown": ""}
# Hex colours used for inline-HTML coloured text in the Markdown report.
_HEX = {"good": "#2ca02c", "ok": "#e6a700", "bad": "#d62728", "unknown": "#808080"}


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
    """Return a colour-styled (inline-HTML) markdown fragment for a rated metric."""
    colour = _HEX.get(rating, _HEX["unknown"])
    return f'<span style="color:{colour}">{_fmt(value, fmt)}</span>'


# ----------------------------------------------------------------------
# Markdown section builders
# ----------------------------------------------------------------------
def section_data_stats(md, stages):
    data = stages.get(smr.STAGE_DATA_PROCESSING)
    md.append("## 1. Data Statistics\n")
    if not data:
        md.append("_No data-processing benchmark found._\n")
        return
    src_files = data.get('source_files', [])
    src_full = [_stage_file(data, "references", f) for f in src_files]
    md.append(f"- **Source file(s):** {', '.join('`%s`' % p for p in src_full) if src_full else '_n/a_'}")
    md.append(f"- **Simulations loaded:** {_fmt(data.get('num_simulations_loaded'))}")
    md.append(f"- **Input variables:** {', '.join(data.get('input_variables', [])) or '_n/a_'}")
    md.append(f"- **Output variables:** {', '.join(data.get('output_variables', [])) or '_n/a_'}")
    if "split_pct_train" in data:
        md.append(f"- **Train/Test split:** {_fmt(data.get('split_pct_train'))}% train")
    md.append("")
    md.append("| Partition | # Simulations | Total data points | Avg increments/sim | Max increments/sim |")
    md.append("|-----------|--------------:|------------------:|-------------------:|-------------------:|")
    for part in ("train", "test"):
        p = data.get(part)
        if not p:
            continue
        md.append("| {} | {} | {} | {} | {} |".format(
            part.capitalize(),
            _fmt(p.get("num_simulations")),
            _fmt(p.get("total_data_points"), "{:,}"),
            _fmt(p.get("avg_increments_per_simulation"), "{:.1f}"),
            _fmt(p.get("max_increments_per_simulation")),
        ))
    md.append("")

    # Optional data-augmentation sub-section
    aug = data.get("augmentation")
    if aug:
        md.append("### Data Augmentation (anchor-point interpolation)\n")
        md.append(f"- **Anchor nodes:** {', '.join(_fmt(n, '{:g}') for n in aug.get('anchor_nodes', []))}")
        md.append(f"- **Simulations augmented:** {_fmt(aug.get('num_simulations'))}")
        md.append("- **Total points:** {} → {} (**+{}** added)".format(
            _fmt(aug.get("total_points_original"), "{:,}"),
            _fmt(aug.get("total_points_augmented"), "{:,}"),
            _fmt(aug.get("total_points_added"), "{:,}"),
        ))
        md.append("- **Points added per simulation:** {} avg (max {})".format(
            _fmt(aug.get("avg_points_added_per_sim"), "{:.1f}"),
            _fmt(aug.get("max_points_added_per_sim")),
        ))
        per_iv = aug.get("per_interval", [])
        if per_iv:
            md.append("")
            md.append("| Anchor interval | Points (orig) | Points (augmented) | Points added |")
            md.append("|-----------------|--------------:|-------------------:|-------------:|")
            for iv in per_iv:
                lo, hi = iv.get("interval", [None, None])
                md.append("| [{:g}, {:g}] | {} | {} | +{} |".format(
                    lo, hi,
                    _fmt(iv.get("points_original"), "{:,}"),
                    _fmt(iv.get("points_augmented"), "{:,}"),
                    _fmt(iv.get("points_added"), "{:,}"),
                ))
        md.append("")


def section_metadata(md, stages):
    train = stages.get(smr.STAGE_TRAINING)
    md.append("## 2. Metadata: Inputs/Outputs, Scalers & Ranges\n")
    if not train:
        md.append("_No training benchmark found._\n")
        return
    roi = train.get("roi", {})
    scalers = train.get("scalers_and_ranges", {})
    md.append("| Variable | Role | Scaler/Map | Range / Settings | ROI |")
    md.append("|----------|------|-----------|------------------|-----|")
    for var, info in scalers.items():
        role = info.get("role", "")
        mp = info.get("map", {})
        if isinstance(mp, dict) and mp:
            scaler_name = ", ".join(mp.keys())
            settings = "; ".join(f"{k}={v}" for k, v in mp.items())
        else:
            scaler_name = str(mp)
            settings = ""
        roi_str = _fmt(roi.get(var))
        md.append(f"| `{var}` | {role} | {scaler_name} | {settings} | {roi_str} |")
    md.append("")


def section_architecture(md, stages):
    train = stages.get(smr.STAGE_TRAINING)
    md.append("## 3. Model Architecture & Settings\n")
    if not train:
        md.append("_No training benchmark found._\n")
        return
    arch = train.get("model_architecture", "unknown")
    md.append(f"- **Architecture:** `{arch}`")
    md.append(f"- **Training data file:** `{_fmt(_abspath(train.get('data_file')))}`")
    settings = train.get("model_settings", {})
    if settings:
        md.append("")
        md.append("| Setting | Value |")
        md.append("|---------|-------|")
        for k, v in settings.items():
            md.append(f"| {k} | {_fmt(v)} |")
    md.append("")


def section_training_metrics(md, stages):
    train = stages.get(smr.STAGE_TRAINING)
    md.append("## 4. Training Metrics\n")
    if not train:
        md.append("_No training benchmark found._\n")
        return
    tm = train.get("training_metrics", {})
    r2 = tm.get("r2") or tm.get("r2_scaled") or {}
    if r2:
        md.append("| Output | R² |")
        md.append("|--------|---:|")
        for k, v in r2.items():
            rating = smr.r2_rating(v)
            md.append(f"| `{k}` | {_rated(v, rating, '{:.4f}')} |")
        md.append("")
        md.append(f'_Colour key: <span style="color:{_HEX["good"]}">R²≥{smr.R2_GOOD}</span> · '
                  f'<span style="color:{_HEX["ok"]}">≥{smr.R2_OK}</span> · '
                  f'<span style="color:{_HEX["bad"]}">&lt;{smr.R2_OK}</span>_\n')
    if "M_condition" in tm:
        md.append(f"- **Condition number (M):** {_fmt(tm.get('M_condition'), '{:.2f}')}")
    wt = train.get("walltimes_s", {})
    for label, key in (("Interpolation matrix build", "interp_matrix"),
                       ("Linear regression", "regression")):
        if key in wt:
            md.append(f"- **{label} wall-time:** {_fmt(wt.get(key), '{:.2f}')} s")
    md.append("")



def section_testing_metrics(md, stages):
    test = stages.get(smr.STAGE_TESTING)
    md.append("## 5. Testing Metrics\n")
    if not test:
        md.append("_No testing benchmark found._\n")
        return
    md.append(f"- **Test data file:** `{_fmt(_stage_file(test, 'references', test.get('test_data_file')))}`")
    md.append(f"- **Test simulations:** {_fmt(test.get('num_test_simulations'))}")
    tm = test.get("testing_metrics", {})
    rr = tm.get("relative_rmse_per_output", {})
    if rr:
        md.append("")
        md.append("| Output | Mean rel-RMSE | Median | Max |")
        md.append("|--------|--------------:|-------:|----:|")
        for k, v in rr.items():
            rating = smr.relrmse_rating(v.get("mean"))
            md.append("| `{}` | {} | {} | {} |".format(
                k,
                _rated(v.get("mean"), rating, "{:.4f}"),
                _fmt(v.get("median"), "{:.4f}"),
                _fmt(v.get("max"), "{:.4f}"),
            ))
        md.append("")
        md.append(f'_Colour key (on mean): <span style="color:{_HEX["good"]}">rel-RMSE≤{smr.RELRMSE_GOOD}</span> · '
                  f'<span style="color:{_HEX["ok"]}">≤{smr.RELRMSE_OK}</span> · '
                  f'<span style="color:{_HEX["bad"]}">&gt;{smr.RELRMSE_OK}</span>_\n')
    md.append(f"- **Total testing wall-time:** {_fmt(tm.get('total_wall_time_s'), '{:.2f}')} s")
    md.append(f"- **Total increments evaluated:** {_fmt(tm.get('total_increments'), '{:,}')}")
    wpi = tm.get("wall_time_per_increment_s")
    if wpi is not None:
        md.append(f"- **Wall-time per increment (single eval):** {wpi*1e3:.4f} ms")
    md.append("")


def section_moose(md, stages):
    port = stages.get(smr.STAGE_PORT_MOOSE)
    md.append("## 6. Ported MOOSE Files\n")
    if not port:
        md.append("_No MOOSE-port benchmark found._\n")
        return
    md.append(f"- **Model name:** `{_fmt(port.get('model_name'))}`")
    md.append(f"- **Architecture:** `{_fmt(port.get('model_architecture'))}`")
    files = port.get("ported_files", [])
    if files:
        md.append("- **Ported files:**")
        for f in files:
            md.append(f"  - `{_stage_file(port, 'output', f)}`")
    md.append(f"- **Verification table:** `{_fmt(_stage_file(port, 'output', port.get('verification_table')))}`")
    md.append("")


def build_markdown(stages):
    md = ["# LAROMance Pipeline - Benchmark Report\n"]
    md.append(f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    found = [smr.STAGE_TITLES.get(k, k) for k in stages]
    md.append(f"_Stages found: {', '.join(found) if found else 'none'}_\n")
    md.append("---\n")
    section_data_stats(md, stages)
    section_metadata(md, stages)
    section_architecture(md, stages)
    section_training_metrics(md, stages)
    section_testing_metrics(md, stages)
    section_moose(md, stages)
    md.append("---\n")
    md.append("_Report assembled by `generate_benchmark_report.py` from per-stage "
              "`benchmark_*.json` artefacts._")
    return "\n".join(md)


def print_console_summary(stages):
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


def main():
    ap = argparse.ArgumentParser(description="Generate the LAROMance benchmark report.")
    default_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    ap.add_argument("--repo-root", default=default_root,
                    help="Repository root to glob for benchmark_*.json (default: auto).")
    ap.add_argument("--output", default="./output/benchmark_report.md",
                    help="Markdown output path (default: ./output/benchmark_report.md).")
    args = ap.parse_args()

    stages = find_benchmark_jsons(args.repo_root)
    if not stages:
        print(f"[report] No benchmark_*.json files found under {args.repo_root}.")
        print("[report] Run the pipeline stages first (they log metrics automatically).")
        return 1

    markdown = build_markdown(stages)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(markdown)
    print(f"[report] Benchmark report written to: {out_path}")
    print_console_summary(stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())

