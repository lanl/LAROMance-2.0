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

LAROMance 2.0 – Stage 0: Demo / Test
------------------------------------

A single, self-contained harness that exercises the entire LAROMance pipeline
(stages 1 -> 5) end-to-end against the shipped synthetic example datasets.

Two mutually-exclusive modes:

    --demo    Interactive guided walkthrough: prints the equivalent terminal
              command for each stage, runs it against the example data,
              narrates the WHY behind each step, summarises the artifacts
              produced, and PAUSES between stages for you to inspect them.
              The demo ALWAYS keeps the generated artifacts so you can explore
              the fully-populated pipeline afterwards.

    --test    Non-interactive smoke test: runs the same sequence, asserts the
              expected artifacts exist, prints a concise PASS summary and sets
              the process exit code (0 = all passed, non-zero otherwise). The
              verbose per-dataset narration is suppressed and the generated
              artifacts are cleaned up afterwards. Add --keep for a deeper test
              that retains the artifacts and prints the full per-stage table so
              you can inspect the files manually.

The training step (stage 3.1) is reproduced *headlessly* via
``utils.sm_build.sm_train`` so that NO GUI toolkit (PySide6) is required. The
harness sets ``QT_QPA_PLATFORM=offscreen`` and ``MPLBACKEND=Agg`` so it runs on
headless servers / CI.

See ``0_Demo_Test/CONTEXT.md`` for the full description.

@author: LAROMance pipeline
"""

import os
import sys

# ----------------------------------------------------------------------
# Headless safety: must be set *before* matplotlib / Qt are imported.
# ----------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import ast
import copy
import json
import pickle
import shutil
import argparse
import subprocess
import numpy as np

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(STAGE_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples", "Viscoplasicity_Creep")
EXAMPLE_PICKLE = os.path.join(
    EXAMPLE_DIR, "Creep_runs_with_dislocation_density_tracking.pickle"
)
EXAMPLE_CONFIG = os.path.join(EXAMPLE_DIR, "Example_Training_Module_Config.json")

STAGE2_DIR = os.path.join(REPO_ROOT, "2_Data_Processing")
STAGE31_DIR = os.path.join(REPO_ROOT, "3_Surrogate_Modeling", "31_Surrogate_Training")
STAGE32_DIR = os.path.join(REPO_ROOT, "3_Surrogate_Modeling", "32_Surrogate_Testing")
STAGE4_DIR = os.path.join(REPO_ROOT, "4_Port_MOOSE", "41_Export_Model")
STAGE5_DIR = os.path.join(REPO_ROOT, "5_Clean_Pipeline")

WORK_DIR = os.path.join(STAGE_DIR, "_work")

# Processing / training parameters.
DOWNSAMPLE_MAX_LENGTH = 150   # max timesteps per sim after down-sampling
SPLIT_PCT = 80
# Anchor points for data augmentation (match the example creep workflow). These
# seed otherwise-empty mesh elements so the regression matrix is non-singular.
ANCHOR_POINTS = "0.0,1e-14,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-2,0.2"
OUTPUT_KEYS = "rhoc,U[evm],U[rhoc]"

SUBSET_NAME = "demo_dataset.pickle"
TRAIN_NAME = "Curated_{}pct_train_demo_dataset.pickle".format(SPLIT_PCT)
TEST_NAME = "Curated_{}pct_test_demo_dataset.pickle".format(100 - SPLIT_PCT)
MOOSE_MODEL_NAME = "DemoCreepModel"


# ----------------------------------------------------------------------
# Small presentation helpers
# ----------------------------------------------------------------------
class C:
    """ANSI colour codes (no-op if output is not a TTY)."""

    _tty = sys.stdout.isatty()
    HEAD = "\033[95m" if _tty else ""
    BLUE = "\033[94m" if _tty else ""
    CYAN = "\033[96m" if _tty else ""
    GREEN = "\033[92m" if _tty else ""
    RED = "\033[91m" if _tty else ""
    YELLOW = "\033[93m" if _tty else ""
    GREY = "\033[90m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""
    DIM = "\033[2m" if _tty else ""
    END = "\033[0m" if _tty else ""


def banner(text):
    """A bold, boxed stage header to clearly separate pipeline stages."""
    line = "═" * 72
    print("\n" + C.HEAD + C.BOLD + "╔" + line + "╗" + C.END)
    print(C.HEAD + C.BOLD + "║  " + text.ljust(70) + "║" + C.END)
    print(C.HEAD + C.BOLD + "╚" + line + "╝" + C.END)


def say_cmd(cmd):
    """Show a shell command the way a user would type it in a terminal."""
    # print(C.DIM + "  (equivalent terminal command)" + C.END)
    print("\n" + C.BLUE + C.BOLD + "  $ " + C.END + C.BLUE + cmd + C.END)


# Set to True in main() when running under --demo. Used by explain() so that the
# verbose pedagogical narration appears only in the interactive demo, never in
# the terse --test smoke run.
DEMO_MODE = False

# Set in main(): controls whether print_summary() shows the full per-stage
# PASS/FAIL table (demo, or --test --keep) or just the concise pass count.
VERBOSE_SUMMARY = True

# Set in main(): True for --demo or --test --keep (artifacts retained); False
# for a plain --test run, where the harness itself cleans up afterwards. Used
# by stage5() so its clean-up message reflects what actually happens.
KEEP_ARTIFACTS = True


def explain(text):
    """Print an explanatory block, but only in --demo mode.

    The FIRST line of the text is treated as a short TITLE and rendered as a
    highlighted header; subsequent ALL-CAPS lines become sub-headers. This
    makes the long pedagogical narration much easier to skim.
    """
    if not DEMO_MODE:
        return
    raw = text.strip("\n").splitlines()
    rule = C.CYAN + "  " + ("─" * 68) + C.END
    print("\n" + rule)
    for i, line in enumerate(raw):
        stripped = line.strip()
        is_header = (
            stripped
            and stripped == stripped.upper()
            and any(ch.isalpha() for ch in stripped)
        )
        if i == 0:
            # Title line.
            print(C.CYAN + C.BOLD + "  " + stripped + C.END)
            print(rule)
        elif is_header:
            print(C.YELLOW + C.BOLD + "  " + line + C.END)
        else:
            print("  " + line)
    print(rule + "\n")


def section(title):
    """A lightweight in-stage sub-section header (shown in both modes)."""
    print("\n" + C.CYAN + C.BOLD + "  ▶ " + title + C.END)


def ok(msg):
    print(C.GREEN + C.BOLD + "  ✓ " + C.END + msg)


def warn(msg):
    print(C.YELLOW + C.BOLD + "  ! " + C.END + msg)


def fail(msg):
    print(C.RED + C.BOLD + "  ✗ " + C.END + msg)


def info(msg):
    """A plain informational line within a stage."""
    print("  " + msg)


def show_output(out, label="console output", n=40):
    """Echo a captured subprocess's console output, nicely framed and indented.

    Long output is tail-trimmed to the last ``n`` lines so the demo stays
    readable while still surfacing what the underlying stage script printed
    (e.g. progress messages, 'building benchmarks.csv', 'Would remove: ...').
    """
    lines = [ln for ln in out.strip().splitlines()]
    if not lines:
        return
    # Drop repetitive, non-informative matplotlib/Qt noise so the meaningful
    # progress messages are not pushed out of the tail window.
    _noise = (
        "FigureCanvasAgg is non-interactive",
        "plt.show(block=False)",
        "Matplotlib is currently using",
        "cannot be shown",
        "coord2index --> Coordinates",
        "vmJ2[MPa] =",
    )
    filtered = [
        ln for ln in lines
        if ln.strip() and not any(tok in ln for tok in _noise)
    ]
    lines = filtered or lines
    trimmed = len(lines) > n
    shown = lines[-n:] if trimmed else lines
    print(C.GREY + "  ┌─ " + label
          + (" (last {} lines)".format(n) if trimmed else "")
          + " " + C.END)
    for ln in shown:
        print(C.GREY + "  │ " + C.END + ln)
    print(C.GREY + "  └" + ("─" * 60) + C.END)


def pause(interactive):
    if interactive:
        try:
            input(C.YELLOW + C.BOLD
                  + "\n  ⏎  Press Enter to continue to the next step… " + C.END)
        except EOFError:
            pass


def assert_exists(path, results):
    """Record a PASS/FAIL for an expected artifact path."""
    exists = os.path.exists(path) and (
        not os.path.isfile(path) or os.path.getsize(path) > 0
    )
    rel = os.path.relpath(path, REPO_ROOT)
    if exists:
        ok("produced: " + rel)
    else:
        fail("MISSING : " + rel)
    results.append((rel, exists))
    return exists


def stage_move(src, dst_dir, dst_name=None):
    """MOVE a produced artifact from one stage's output into the next stage's
    references/ folder, like a factory line: the artifact is consumed and
    transformed, never duplicated. Datasets and SM.pickle files can be large,
    so moving (rather than copying) avoids wasteful duplication on disk.

    Returns the destination path. The source no longer exists afterwards.
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name or os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)
    print("  moved -> {}".format(os.path.relpath(dst, REPO_ROOT)))
    return dst



# ----------------------------------------------------------------------
# Subprocess runner
# ----------------------------------------------------------------------
def run_subprocess(cmd_list, cwd, stdin_text=""):
    """Run a stage script as a subprocess with headless env; return (code, output)."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd_list,
        cwd=cwd,
        input=stdin_text,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def _tail(text, n=25):
    lines = text.strip().splitlines()
    return "\n".join("    | " + ln for ln in lines[-n:])


# ----------------------------------------------------------------------
# Stage 1 - Data Generation (dataset presence check)
# ----------------------------------------------------------------------
def stage1(results, interactive):
    banner("Stage 1 - Data Generation")
    print(
        "  LAROMance ships NO high-fidelity generator; a user integrates their\n"
        "  own model here (see 1_Data_Generation/CONTEXT.md). For the demo we use\n"
        "  the pre-generated synthetic creep dataset shipped under examples/."
    )
    say_cmd("ls examples/Viscoplasicity_Creep/*.pickle")
    assert_exists(EXAMPLE_PICKLE, results)
    assert_exists(EXAMPLE_CONFIG, results)
    pause(interactive)


# ----------------------------------------------------------------------
# Stage 2 - Data Processing
# ----------------------------------------------------------------------
def stage2(results, interactive):
    banner("Stage 2 - Data Processing")

    refs = os.path.join(STAGE2_DIR, "references")
    os.makedirs(refs, exist_ok=True)
    subset_path = os.path.join(refs, SUBSET_NAME)

    explain(
        "PIPELINE STEP - FILES FLOW THROUGH THE STAGES\n"
        "Each LAROMance stage reads its inputs from its own 'references/' folder\n"
        "and writes results to its own 'output/' folder. You advance through the\n"
        "pipeline by passing a stage's output file(s) into the NEXT stage's\n"
        "references/ folder. Think of it like a factory line: from here on the\n"
        "demo MOVES each produced artifact forward (datasets and SM.pickle files\n"
        "can be large, so they are consumed and transformed, never duplicated).\n"
        "The one exception is right here at the start: the SHIPPED example\n"
        "dataset is COPIED into\n"
        "    2_Data_Processing/references/\n"
        "so the original example file stays intact in examples/ for re-runs."
    )

    explain(
        "ABOUT THE DATASET - WHAT THE CREEP SIMULATIONS CONTAIN\n"
        "File: examples/Viscoplasicity_Creep/" + os.path.basename(EXAMPLE_PICKLE) + "\n"
        "\n"
        "Each entry is ONE creep simulation: the time-history of a single\n"
        "material point in a tube wall of a fictitious polycrystalline metal,\n"
        "held under a CONSTANT internal pressure at a CONSTANT temperature.\n"
        "'Creep' = slow, time-dependent plastic flow under a steady load. The\n"
        "pressure is ramped up at the very start, so those initial ramp-up\n"
        "increments are not steady creep; the data-processing step automatically\n"
        "strips them off (you'll see 'Dropping first N increments' in the log).\n"
        "\n"
        "DISLOCATION DENSITY, (briefly)\n"
        "Real metals are crystals full of line defects called DISLOCATIONS;\n"
        "their motion IS plastic deformation. 'Dislocation density' (rhoc, the\n"
        "count of these defects per unit area) is a scalar proxy that tracks how\n"
        "the microstructure evolves as the material creeps - more dislocations\n"
        "generally means a harder, more deformation-resistant lattice.\n"
        "\n"
        "THE VARIABLES (~5000 simulations spanning these)\n"
        "  INPUTS\n"
        "    - vmJ2        von Mises equivalent stress    [MPa]  (from pressure)\n"
        "    - temperature constant temperature of run     [K]\n"
        "    - evm         accumulated equiv. plastic strain [-]  (COUPLED input)\n"
        "    - rhoc        dislocation density            [1/m^2] (COUPLED input)\n"
        "  OUTPUTS (both are RATES, not accumulated values)\n"
        "    - U[evm]      strain rate (plastic)          [1/s]\n"
        "    - U[rhoc]     dislocation-density rate       [1/(m^2 s)]\n"
        "\n"
        "WHY 'COUPLED': the surrogate predicts the RATES, but two of its own\n"
        "inputs (evm and rhoc) are the time-integrals of those rates. So the\n"
        "outputs feed back as inputs over time - this is a COUPLED dataset, and\n"
        "it's exactly why Stage 3.2 later tests forward time-integration.\n"
        "This input/output layout is the format your own Stage 1 generator must\n"
        "emit.\n"
        "\n"
        "WHY WE PLOT AGAINST vmJ2 AND temperature\n"
        "Physics tells us creep strain rate depends most strongly on STRESS and\n"
        "TEMPERATURE. That domain knowledge is why, in the plotting options\n"
        "below, we pick vmJ2 and temperature as the two axes to map the outputs\n"
        "over (--input_keys vmJ2,temperature) - we expect clear, learnable\n"
        "trends there."
    )

    print("  Using the full example creep dataset...")
    shutil.copy(EXAMPLE_PICKLE, subset_path)
    ok("dataset staged: " + os.path.relpath(subset_path, REPO_ROOT))

    cmd = (
        "python load_and_plot_data.py --pickle_files {} "
        "--max_length {} --split_data {} --concat --anchor_points {} "
        "--input_keys vmJ2,temperature --output_keys '{}'".format(
            SUBSET_NAME, DOWNSAMPLE_MAX_LENGTH, SPLIT_PCT, ANCHOR_POINTS, OUTPUT_KEYS
        )
    )
    say_cmd("cd 2_Data_Processing && " + cmd)
    cmd_list = [
        sys.executable,
        "load_and_plot_data.py",
        "--pickle_files", SUBSET_NAME,
        "--max_length", str(DOWNSAMPLE_MAX_LENGTH),
        "--split_data", str(SPLIT_PCT),
        "--concat",
        "--anchor_points", ANCHOR_POINTS,
        "--input_keys", "vmJ2,temperature",
        "--output_keys", OUTPUT_KEYS,
    ]
    # The script prompts for concatenation + augmentation anchors; feed blanks.
    code, out = run_subprocess(cmd_list, STAGE2_DIR, stdin_text="\n\n\n\n")
    show_output(out, label="load_and_plot_data.py console output")
    if code != 0:
        fail("stage 2 script exited with code {}".format(code))
        results.append(("stage2 script exit 0", False))
    else:
        ok("stage 2 script completed")
        results.append(("stage2 script exit 0", True))

    train_pkl = os.path.join(STAGE2_DIR, "output", TRAIN_NAME)
    test_pkl = os.path.join(STAGE2_DIR, "output", TEST_NAME)
    assert_exists(train_pkl, results)
    assert_exists(test_pkl, results)

    explain(
        "WHAT STAGE 2 JUST PRODUCED (look in 2_Data_Processing/output/)\n"
        "--split_data 80 split the data 80% TRAINING / 20% TESTING, each saved\n"
        "as its own curated pickle, plus diagnostic plots worth reading closely:\n"
        "\n"
        "SCATTER PLOTS - each point is the FINAL time increment of one\n"
        "simulation, mapped over vmJ2 (stress) and temperature.\n"
        "  * Why the final increment? In secondary ('steady-state') creep the\n"
        "    strain rate is roughly constant, so the last point is often (not\n"
        "    always, even here) a good stand-in for the steady-state response.\n"
        "  * U[evm] (strain rate) rises DIAGONALLY with both stress and\n"
        "    temperature - exactly the dependence physics predicts, which is why\n"
        "    we chose those two axes.\n"
        "  * U[rhoc] (dislocation-density rate) likewise varies diagonally\n"
        "    across stress and temperature. Reading the data this way is how you\n"
        "    CONFIRM which inputs actually drive each output.\n"
        "\n"
        "TRACE PLOTS - the per-simulation time histories (top row vs. time,\n"
        "bottom row vs. accumulated strain evm). Note the shape: strain rate and\n"
        "dislocation density fall off in a LOG-like, asymptotic way, changing\n"
        "fastest at SMALL evm and flattening later.\n"
        "  => Takeaway for Stage 3: we'll need FINER mesh resolution at small\n"
        "     evm (a log-like node distribution) to capture that early curvature,\n"
        "     and log10 scaling of the outputs to tame their huge dynamic range.\n"
        "\n"
        "WHY WE AUGMENT (--anchor_points / --concat)\n"
        "LAROMance is a GRID-BASED (finite-element mesh) surrogate. It behaves\n"
        "best when data fills the input domain, because empty mesh elements make\n"
        "the regression ill-conditioned (singular matrix -> NaN nodal values).\n"
        "The anchor points inject interpolated samples so otherwise-empty\n"
        "elements are populated, and --concat merges the curated traces into one\n"
        "array ready for training.\n"
        "Compare the `Augmented` vs the original trace plots: notice that in the\n"
        "augmented ones all trace-lines reach the x-axis value of 0.2."
    )
    pause(interactive)
    return train_pkl, test_pkl


# ----------------------------------------------------------------------
# Stage 3.1 - Surrogate Training (headless; GUI bypassed)
# ----------------------------------------------------------------------
def _build_data_for_train(curated, ordered, output_vars):
    """Assemble the DATA dict expected by sm_build.sm_train."""
    data = {
        "x": np.vstack([curated["data"][k] for k in ordered]).T,
        "U": {o["key"]: curated["data"]["U"][o["key"]] for o in output_vars},
    }
    for k in ordered:
        data[k] = curated["data"][k]
    return {"data": data}


def _axis_label(key, as_output=False):
    """Human-readable axis label (with units) for a variable key.

    In the Training Module GUI you set these interactively by DOUBLE-CLICKING a
    plot axis label and typing the desired text. Headlessly we apply the same
    sensible defaults here so the exported PNGs are properly labelled.

    ``as_output=True`` labels the surrogate OUTPUT for that key, which in this
    coupled creep example is the RATE form (e.g. output key 'rhoc' is actually
    the dislocation-density rate, output key 'evm' is the strain rate).
    """
    if as_output:
        return {
            "evm": "U[evm]  (strain rate) [1/s]",
            "rhoc": "U[rhoc]  (dislocation-density rate) [1/(m^2 s)]",
            "U[evm]": "U[evm]  (strain rate) [1/s]",
            "U[rhoc]": "U[rhoc]  (dislocation-density rate) [1/(m^2 s)]",
        }.get(key, key)
    return {
        "vmJ2": "vmJ2  (von Mises stress) [MPa]",
        "temperature": "temperature [K]",
        "evm": "evm  (equiv. plastic strain) [-]",
        "rhoc": "rhoc  (dislocation density) [1/m^2]",
        "U[evm]": "U[evm]  (strain rate) [1/s]",
        "U[rhoc]": "U[rhoc]  (dislocation-density rate) [1/(m^2 s)]",
    }.get(key, key)


def _export_multi_input_plots(SM, DATA2, ordered, outvar, u_phys, uh_phys,
                              out_dir, written, mcolors, tri, Figure, smd, smp):
    """Write the relative-error map (page 1) and 3D results plot (page 2)."""
    var1, var2 = ordered[:2]
    x1 = np.asarray(DATA2["data"][var1]).ravel()
    x2 = np.asarray(DATA2["data"][var2]).ravel()

    # Multi-input: RELATIVE-ERROR MAP (GUI page 1).
    try:
        rel_err = 100 * np.abs((u_phys - uh_phys)
            / np.where(u_phys == 0, np.finfo(float).eps, u_phys))
        fig1 = Figure(figsize=(6, 5))
        ax1 = fig1.add_subplot(111)
        try:
            triang = tri.Triangulation(x1, x2)
            ax1.tricontourf(triang, rel_err, levels=np.logspace(-1, 2, 50),
                cmap="hot", norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2),
                extend="both")
        except (RuntimeError, ValueError) as exc:
            warn("triangulation failed ({}, {}): {}".format(var1, var2, exc))
        sc1 = ax1.scatter(x1, x2, c=rel_err, s=10, edgecolor="k", lw=0.4,
            cmap="hot", norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), zorder=10)
        cbar = fig1.colorbar(sc1, ax=ax1)
        cbar.set_label("Relative Error [%]", rotation=270, labelpad=15)
        try:
            smp.visualize_mesh(
                smd.backtransform_nodes(
                    copy.deepcopy(SM["mesh"]["nodes"]), SM["input_maps"]),
                SM["mesh"]["conn"], dim=2, ax=ax1,
                show_node_numbers=False, show_elem_numbers=False)
        except Exception as exc:  # noqa: BLE001
            warn("mesh overlay failed for '{}': {}".format(outvar, exc))
        # NOTE: set the axis labels/title AFTER the mesh overlay - visualize_mesh
        # resets them to generic 'x'/'y'/'2D Mesh', so relabel here (just as a
        # GUI user double-clicks each label to rename it).
        ax1.set_xlabel(_axis_label(var1)); ax1.set_ylabel(_axis_label(var2))
        ax1.set_title("Relative Error [%]: {}".format(
            _axis_label(outvar, as_output=True)))
        path1 = os.path.join(out_dir, "Training_relerr_{}.png".format(outvar))
        fig1.savefig(path1, dpi=150, bbox_inches="tight", facecolor="#2b2b2b")
        written.append(("relative-error map ({})".format(outvar), path1))
    except Exception as exc:  # noqa: BLE001
        warn("relative-error map failed for '{}': {}".format(outvar, exc))

    # Multi-input: 3D RESULTS PLOT (GUI page 2).
    try:
        n = len(x1)
        idx = np.random.choice(n, min(2000, n), replace=False)
        fig2 = Figure(figsize=(7, 5))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(x1[idx], x2[idx], u_phys[idx], marker="o", color="r",
                    s=20, alpha=0.5, edgecolors="k", label="data")
        ax2.scatter(x1[idx], x2[idx], uh_phys[idx], marker="s", color="k",
                    s=25, alpha=0.4, label="surrogate")
        # NOTE: in the GUI you would DESELECT "nodal values" for this example -
        # the yellow node markers otherwise dominate the z-range and distort the
        # view. We mirror that here by NOT plotting the nodal values, so only the
        # red data circles, the black surrogate squares and the surface remain.
        ax2.legend(loc="best")
        ax2.set_xlabel(_axis_label(var1)); ax2.set_ylabel(_axis_label(var2))
        ax2.set_zlabel(_axis_label(outvar, as_output=True))
        ax2.set_title("3D Results (rate-form output): {}".format(outvar))
        path2 = os.path.join(out_dir, "Training_3d_{}.png".format(outvar))
        fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="#2b2b2b")
        written.append(("3D results plot ({})".format(outvar), path2))
    except Exception as exc:  # noqa: BLE001
        warn("3D results plot failed for '{}': {}".format(outvar, exc))



def _export_training_plots(SM, train_output, DATA2, ordered, output_vars, out_dir):
    """Replicate the Training Module GUI's result plots headlessly and save PNGs.

    The GUI (training_module.py :: update_result_plot) builds, per output:
      * a RELATIVE-ERROR MAP  - a tricontourf of the % error between the training
        data and the surrogate over the two base (plot=True) input dimensions,
        with the data points and the finite-element mesh overlaid;
      * a 3D RESULTS PLOT     - a 3D scatter of training data vs. surrogate
        prediction, plus the calibrated nodal values;
      * for a single-input model, an output-vs-input LINE TRACE instead.
    The GUI exposes an "Export" button (save_current_figure / save_all_figures)
    that writes exactly these figures to disk. We reproduce that export here so
    the demo user can open and inspect the same plots the GUI would show.

    Returns a list of (title, path) for the figures written. Never raises - any
    plotting failure is reported but must not affect the pipeline PASS/FAIL.
    """
    import matplotlib
    matplotlib.use("Agg")
    # Dark background, matching the GUI's dark theme selection. Applied globally
    # so every Figure created below (incl. in _export_multi_input_plots) inherits
    # the dark facecolor on screen AND in the saved PNGs.
    matplotlib.style.use("dark_background")
    # The GUI's dark theme uses a dark GREY panel (not pure black) so that the
    # black surrogate-prediction squares stay visible. Match that here.
    matplotlib.rcParams.update({
        "figure.facecolor": "#2b2b2b",
        "axes.facecolor": "#2b2b2b",
        "savefig.facecolor": "#2b2b2b",
    })
    import matplotlib.colors as mcolors
    import matplotlib.tri as tri
    from matplotlib.figure import Figure
    from utils import sm_data as smd
    from utils import sm_plot as smp

    written = []
    os.makedirs(out_dir, exist_ok=True)
    plot_outputs = [o["key"] for o in output_vars]

    for outvar in plot_outputs:
        if outvar not in SM.get("nodal_values", {}):
            continue
        try:
            u_mapped = np.asarray(train_output["U"][outvar]).ravel()
            uh_mapped = np.asarray(train_output["Usm"][outvar]).ravel()
            u_phys = np.asarray(smd.backtransform_output(
                copy.deepcopy(u_mapped), SM["output_maps"][outvar])).ravel()
            uh_phys = np.asarray(smd.backtransform_output(
                copy.deepcopy(uh_mapped), SM["output_maps"][outvar])).ravel()
        except Exception as exc:  # noqa: BLE001
            warn("could not assemble plot data for '{}': {}".format(outvar, repr(exc)))
            continue

        if len(ordered) == 1:
            # Single-input: output-vs-input LINE TRACE (GUI single-var branch).
            try:
                var = ordered[0]
                x = np.asarray(DATA2["data"][var]).ravel()
                nodes = smd.backtransform_nodes(
                    copy.deepcopy(SM["mesh"]["nodes"]), SM["input_maps"]).flatten()
                nodal_values = smd.backtransform_output(
                    copy.deepcopy(SM["nodal_values"][outvar]),
                    SM["output_maps"][outvar]).flatten()
                fig = Figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                smp.plotxy(x, u_phys, color="g", ax=ax, show=False)
                smp.plotxy(x, uh_phys, color="r", marker="-", ax=ax, show=False)
                smp.plotxy(nodes, nodal_values, color="cyan", marker="o", ax=ax,
                           show=False)
                ax.set_xlabel(var); ax.set_ylabel(outvar)
                ax.set_title("{} vs {}".format(outvar, var))
                path = os.path.join(out_dir, "Training_trace_{}.png".format(outvar))
                fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#2b2b2b")
                written.append(("line trace ({})".format(outvar), path))
            except Exception as exc:  # noqa: BLE001
                warn("line-trace plot failed for '{}': {}".format(outvar, exc))
            continue

        _export_multi_input_plots(
            SM, DATA2, ordered, outvar, u_phys, uh_phys, out_dir, written,
            mcolors, tri, Figure, smd, smp)

    return written



def _export_training_timeseries(SM, train_output, curated, meta_data, output_vars,
                                out_dir, xaxis="evm", max_traces=50):
    """Save the surrogate's TRAINING TIME-SERIES results, unravelled per simulation.

    The Training Module GUI lets the user pick an "x-axis" variable to unravel
    the concatenated training arrays (one long array spanning all simulations)
    back into individual per-simulation time-series. Here that x-axis is the
    deformation-history variable ``evm`` (von Mises strain): unlike wall-clock
    ``time`` it actually tracks deformation, which is the physically meaningful
    "clock" for this mechanics use-case.

    For each output we overlay, per simulation:
        * the GROUND-TRUTH physics data  -> DOTTED line
        * the SURROGATE prediction       -> SOLID line
    plotted against ``evm`` on the x-axis.

    A companion text file explains the x-axis choice and how to read the plots.
    Never raises - any failure is reported but must not break the pipeline.
    """
    import matplotlib
    matplotlib.use("Agg")
    # Match the GUI's dark theme (dark grey panel so black markers stay visible).
    matplotlib.style.use("dark_background")
    matplotlib.rcParams.update({
        "figure.facecolor": "#2b2b2b",
        "axes.facecolor": "#2b2b2b",
        "savefig.facecolor": "#2b2b2b",
    })
    from matplotlib.figure import Figure
    from utils import sm_data as smd

    written = []
    os.makedirs(out_dir, exist_ok=True)

    sim_lengths = list(meta_data.get("sim_lengths", []))
    if not sim_lengths:
        warn("no meta_data['sim_lengths']; cannot unravel training time-series")
        return written

    # Physical (unmapped) x-axis straight from the curated data (index-aligned
    # with the concatenated training arrays).
    try:
        x_full = np.asarray(curated["data"][xaxis]).ravel()
    except Exception as exc:  # noqa: BLE001
        warn("x-axis variable '{}' not available: {}".format(xaxis, exc))
        return written

    # Precompute per-simulation slice boundaries from the concatenation order.
    bounds = np.concatenate([[0], np.cumsum(sim_lengths)]).astype(int)
    n_sims = len(sim_lengths)
    n_show = min(max_traces, n_sims)

    for v in output_vars:
        outvar = v["key"]
        if outvar not in SM.get("nodal_values", {}):
            continue
        try:
            u_phys = np.asarray(smd.backtransform_output(
                copy.deepcopy(np.asarray(train_output["U"][outvar]).ravel()),
                SM["output_maps"][outvar])).ravel()
            uh_phys = np.asarray(smd.backtransform_output(
                copy.deepcopy(np.asarray(train_output["Usm"][outvar]).ravel()),
                SM["output_maps"][outvar])).ravel()
        except Exception as exc:  # noqa: BLE001
            warn("could not assemble time-series for '{}': {}".format(outvar, exc))
            continue

        n = min(len(x_full), len(u_phys), len(uh_phys))
        try:
            fig = Figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            for s in range(n_show):
                lo, hi = bounds[s], min(bounds[s + 1], n)
                if hi <= lo:
                    continue
                xs = x_full[lo:hi]
                # Ground truth: dotted; surrogate: solid. Share a colour per sim.
                line_gt = ax.plot(xs, u_phys[lo:hi], linestyle=":", linewidth=1.5)[0]
                ax.plot(xs, uh_phys[lo:hi], linestyle="-", linewidth=1.2,
                        color=line_gt.get_color(), alpha=0.9)
            # Pick a y-scale that matches the output's own map. U[rhoc] uses a
            # SYMLOG map and genuinely goes negative (its config range is
            # [-1, 0]); a plain log scale would silently DROP every negative
            # sample and make the plot look wrong/empty. symlog keeps them.
            omap = SM.get("output_maps", {}).get(outvar, {})
            maptype = next(iter(omap), None) if isinstance(omap, dict) else None
            finite = np.concatenate([
                u_phys[np.isfinite(u_phys)], uh_phys[np.isfinite(uh_phys)]])
            has_neg = finite.size and np.nanmin(finite) < 0
            if maptype == "symlog" or has_neg:
                ax.set_yscale("symlog")
            else:
                ax.set_yscale("log")
            ax.set_xlabel(_axis_label(xaxis) + "  (deformation-history clock)")
            ax.set_ylabel(_axis_label(outvar, as_output=True))
            ax.set_title("Training time-series: {} vs {}\n"
                         "dotted = ground truth, solid = surrogate"
                         .format(_axis_label(outvar, as_output=True), xaxis))
            path = os.path.join(
                out_dir, "Training_timeseries_{}_vs_{}.png".format(outvar, xaxis))
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#2b2b2b")
            written.append(("time-series ({} vs {})".format(outvar, xaxis), path))
        except Exception as exc:  # noqa: BLE001
            warn("time-series plot failed for '{}': {}".format(outvar, exc))

    # Write the explainer alongside the figures.
    try:
        explainer = os.path.join(out_dir, "Training_timeseries_README.txt")
        with open(explainer, "w") as fh:
            fh.write(
                "TRAINING TIME-SERIES RESULTS\n"
                "============================\n\n"
                "X-AXIS CHOICE: '{xaxis}'\n"
                "---------------------------\n"
                "The surrogate is trained on a CONCATENATED array that merges every\n"
                "simulation into one long sequence. The Training Module GUI lets you\n"
                "select an 'x-axis' variable; that choice unravels the concatenated\n"
                "results back into the individual per-simulation time-series.\n\n"
                "Here the x-axis is the deformation-history variable '{xaxis}'\n"
                "(von Mises strain). We use strain rather than wall-clock 'time'\n"
                "because strain actually tracks the deformation history, which is\n"
                "the physically meaningful 'clock' for this solid-mechanics use-case.\n"
                "(In other use-cases 'time' may instead be the relevant x-axis.)\n\n"
                "HOW TO READ THE PLOTS\n"
                "---------------------\n"
                "  * DOTTED line  = ground truth (physics-based training data)\n"
                "  * SOLID line   = surrogate model prediction\n"
                "Each simulation is drawn in its own colour; a good fit has the\n"
                "solid curve lying on top of its dotted counterpart.\n\n"
                "Showing up to {n_show} of {n_sims} simulations per output.\n"
                .format(xaxis=xaxis, n_show=n_show, n_sims=n_sims)
            )
        written.append(("time-series explainer", explainer))
    except Exception as exc:  # noqa: BLE001
        warn("could not write time-series explainer: {}".format(exc))

    return written


def stage3_train(train_pkl, results, interactive):
    banner("Stage 3.1 - Surrogate Training (headless)")
    from utils import sm_build as smb
    from utils import sm_data as smd
    from utils import fe_shapes as fes

    explain(
        "PIPELINE STEP - MOVE FILES TO THE NEXT STAGE\n"
        "Training (Stage 3.1) consumes the CURATED TRAINING pickle produced by\n"
        "Stage 2. In the manual workflow you MOVE it from\n"
        "    2_Data_Processing/output/" + TRAIN_NAME + "\n"
        "into the training stage's references folder\n"
        "    3_Surrogate_Modeling/31_Surrogate_Training/references/\n"
        "and then load it from the Training Module GUI. Like a factory line, the\n"
        "dataset is MOVED (not copied) - these pickles can be large, so each\n"
        "artifact is consumed and transformed into the next stage's output\n"
        "rather than duplicated. The demo wires this up for you and trains\n"
        "headlessly."
    )

    explain(
        "IMAGINE THE TRAINING GUI IS NOW OPEN IN FRONT OF YOU\n"
        "Stage 3.1 is normally driven through the interactive Training Module\n"
        "GUI (training_module.py). Picture its window as you read along - this is\n"
        "exactly what you would click, in order:\n"
        "\n"
        "1) LOAD THE DATA FILE (top-left of the interface)\n"
        "   The curated training pickle you just produced in Stage 2 is selected\n"
        "   in the file selector at the TOP-LEFT of the window. That loads the\n"
        "   dataset into the module.\n"
        "\n"
        "2) SET THE INPUT & OUTPUT VARIABLES (left side)\n"
        "   On the LEFT you add/remove INPUT variables from the inputs dropdown\n"
        "   menu, and a little BELOW it you add/remove the OUTPUT variables from\n"
        "   the outputs dropdown. For this example the inputs are vmJ2 and\n"
        "   temperature; the outputs are rhoc, U[evm] and U[rhoc].\n"
        "\n"
        "3) CHOOSE SCALERS & THEIR RANGES (left side)\n"
        "   Still on the left, each variable gets a SCALER (the per-axis map,\n"
        "   e.g. minmax/log10) and its RANGE. The exact choices are listed in the\n"
        "   next panel below.\n"
        "\n"
        "4) INSPECT THE DATA (right-hand graphics panel)\n"
        "   Selecting TWO inputs and ONE output in the GUI draws the data as\n"
        "   SCATTER PLOTS in the big panel on the RIGHT. The buttons BELOW that\n"
        "   panel let you manipulate the view (e.g. toggle log scaling, change\n"
        "   axes) so you can sanity-check the data before training.\n"
        "\n"
        "5) APPLY THE ROI CORRECTION (button below the graphics panel)\n"
        "   The region-of-interest (ROI) correction is also done in the GUI, via\n"
        "   the ROI-correction button that appears BELOW the data graphics panel.\n"
        "\n"
        "   TIP - REUSE THIS DEMO'S CONFIG IN THE GUI: the Training Module can\n"
        "   LOAD a config JSON to auto-populate every setting above (inputs,\n"
        "   outputs, scalers/ranges, element counts AND the ROI correction). The\n"
        "   very config this headless run uses -\n"
        "       " + os.path.relpath(EXAMPLE_CONFIG, REPO_ROOT) + "\n"
        "   can be loaded straight into the GUI to reproduce these exact settings\n"
        "   with no manual clicking - a great starting point you can then tweak.\n"
        "\n"
        "The demo performs all of the above for you headlessly, because the GUI\n"
        "needs PySide6; the result is identical to clicking through these steps."
    )

    print(
        "  The training GUI (training_module.py) needs PySide6. We reproduce the\n"
        "  same training headlessly via utils.sm_build.sm_train, driven by the\n"
        "  example config JSON (equivalent GUI action shown below)."
    )
    say_cmd("cd 3_Surrogate_Modeling/31_Surrogate_Training && python training_module.py  # (GUI)")

    # MOVE the curated training pickle from Stage 2's output into this stage's
    # references/ folder (factory-line semantics - the dataset is consumed, not
    # copied), then train from that reference file exactly as a GUI user would.
    refs = os.path.join(STAGE31_DIR, "references")
    train_ref = stage_move(train_pkl, refs, TRAIN_NAME)

    with open(EXAMPLE_CONFIG) as fh:
        cfg = json.load(fh)
    with open(train_ref, "rb") as fh:
        curated = pickle.load(fh)

    # curated train pickle is already concatenated by stage 2 (has 'U').
    if "U" not in curated["data"]:
        curated = smd.concatenate_data(curated, remove_initial_incs=0)

    input_vars = cfg["input_vars"]
    output_vars = cfg["output_vars"]
    base = [v["key"] for v in input_vars if v.get("plot")]
    stack = [v["key"] for v in input_vars if not v.get("plot")]
    ordered = base + stack

    # Element counts taken verbatim from the example GUI config (these are the
    # settings known to produce good results).
    elem = {}
    for v in input_vars:
        elem[v["key"]] = ast.literal_eval(str(v["elem"]))
    elem = {k: elem[k] for k in ordered}

    md = {k: {"type": "power", "param": 1.0} for k in ordered}

    # ------------------------------------------------------------------
    # DEMO WALKTHROUGH - the training choices this headless run takes from
    # the config JSON. In the GUI the user sets each of these by hand in the
    # Training Module; here they are read from
    # Example_Training_Module_Config.json so the result is reproducible.
    # ------------------------------------------------------------------
    if DEMO_MODE:
        lines = [
            "TRAINING CHOICES (from Example_Training_Module_Config.json)",
            "In the GUI you pick these interactively; the demo reads them from",
            "the config so the surrogate is built identically every time.",
            "",
            "INPUT dimensions (the mesh axes):",
        ]
        for v in input_vars:
            role = "base/plot" if v.get("plot") else "stacked"
            lines.append(
                "  - {:<12} map={:<7} elements={:<5} [{}]".format(
                    v["key"], str(v.get("map")), str(v.get("elem")), role))
        lines += [
            "",
            "  * BASE dims (plot=true) form the 2-D mesh that is drawn/plotted:",
            "        " + ", ".join(base),
            "  * STACKED dims (plot=false) are extruded on top to make the",
            "    full N-D mesh: " + (", ".join(stack) if stack else "(none)"),
            "",
            "OUTPUT responses the surrogate predicts:",
        ]
        for v in output_vars:
            role = "plotted" if v.get("plot") else "trained"
            lines.append(
                "  - {:<12} map={:<7} [{}]".format(
                    v["key"], str(v.get("map")), role))
        lines += [
            "",
            "REGRESSION options (args block):",
            "  - sparse / remove_sparse_elements = drop empty mesh elements so",
            "    the least-squares fit of the nodal values stays well-posed.",
            "  - deconcat = {} (keep the concatenated training array as-is).".format(
                cfg.get("args", {}).get("deconcat")),
        ]
        explain("\n".join(lines))

        explain(
            "WHY THESE TRAINING CHOICES (reading back from the Stage 2 data)\n"
            "The table above is not arbitrary - every 'map' and 'elements' value\n"
            "follows from what the Stage 2 plots and the creep physics told us:\n"
            "\n"
            "LOG-TYPE SCALING of the strongly-varying quantities\n"
            "The strain rate and dislocation-density rate span many ORDERS of\n"
            "magnitude and decay log-like over a trajectory (the asymptotic creep\n"
            "behaviour you saw in the trace plots). On a linear scale the tiny\n"
            "late-time values would be swamped by the large early ones, so these\n"
            "get log-type maps: evm uses log10, and rhoc uses symlog (a log that\n"
            "also handles zero/negative values gracefully). The well-behaved\n"
            "drivers vmJ2 and temperature just use plain minmax normalisation.\n"
            "\n"
            "10 x 10 ELEMENTS (minmax) across vmJ2 and temperature\n"
            "The scatter plots showed smooth, diagonal variation of the outputs\n"
            "with stress and temperature - no sharp features. A modest, uniform\n"
            "10x10 grid on these two base axes is enough to capture that gentle\n"
            "2-D variation without over-fitting or wasting nodes.\n"
            "\n"
            "LOG-LIKE NODE DISTRIBUTION on the evm (strain) axis\n"
            "The traces changed fastest at SMALL strain and flattened later, so a\n"
            "uniform grid would under-resolve the early curvature. Instead the\n"
            "evm axis uses explicit, clustered node positions (the anchor-point\n"
            "list) packed densely near small evm and sparser at large evm -\n"
            "spending resolution where the response actually bends.\n"
            "\n"
            "Remember this is a COUPLED dataset: evm and rhoc are both inputs AND\n"
            "integrals of the predicted rates, so getting their discretisation\n"
            "right here is what keeps the Stage 3.2 time-integration accurate."
        )

    # ROI (region of interest) sets the lower/upper bound of each input
    # dimension's mesh. By default we derive it from the data min/max, exactly
    # like the Training Module's get_roi_from_inputs(). However, any per-key ROI
    # declared in the config JSON's top-level "roi" block OVERRIDES the data
    # bounds for that key -- mirroring how a user-set ROI in the GUI is honored.
    #
    # This matters here because of a subtle train/test mismatch: the testing
    # data starts at zero stress, while the training data has its initial time
    # increments removed and therefore may never reach vmJ2 = 0. If we let the
    # training ROI start at the (non-zero) training-data minimum, the surrogate
    # would extrapolate below its mesh for the low-stress test points. Pinning
    # vmJ2 to [0, 512] in the config keeps the mesh anchored at zero stress so
    # train and test share the same input domain.
    roi = {
        k: [float(np.min(curated["data"][k])), float(np.max(curated["data"][k]))]
        for k in ordered
    }
    config_roi = cfg.get("roi") or {}
    roi_overrides = {}
    for k in ordered:
        if k in config_roi:
            lo, hi = config_roi[k]
            roi[k] = [float(lo), float(hi)]
            roi_overrides[k] = roi[k]
    if roi_overrides:
        explain(
            "PIPELINE NOTE - ROI CORRECTION FROM THE CONFIG\n"
            "The config JSON declares an explicit ROI (region of interest) for\n"
            "some input dimensions under its top-level \"roi\" key. These values\n"
            "OVERRIDE the data-derived min/max for the mesh, exactly like setting\n"
            "the ROI by hand in the Training Module GUI.\n"
            "Here vmJ2 is pinned to " + str(roi_overrides.get("vmJ2", "")) + ".\n"
            "Why: the test data starts at zero stress, but the training data has\n"
            "its initial time increments removed and may never reach vmJ2 = 0.\n"
            "Anchoring the lower bound at 0 keeps the surrogate's input domain\n"
            "aligned between training and testing so low-stress points are not\n"
            "extrapolated below the mesh."
        )
        for k, b in roi_overrides.items():
            if DEMO_MODE:
                print("  ROI override from config: {} -> {}".format(k, b))


    mesh_specs = {
        "shapefunc_degree": "linear",
        "element_numbers": elem,
        "mesh_distribution": md,
        "tri_elements": [],
        "refinement_region": None,
        "refinement_configs": None,
        "mesh_filtering": {"density_thresh": 3, "remove_limit": None},
    }
    # Build the 2-D base mesh (plot=True inputs); sm_train extrudes to N-D.
    mesh_specs_2d = copy.deepcopy(mesh_specs)
    mesh_specs_2d["element_numbers"] = {k: elem[k] for k in base}
    mesh_specs_2d["mesh_distribution"] = {k: md[k] for k in base}
    roi_2d = {k: roi[k] for k in base}
    bnodes, bconn = fes.build_hypercube_mesh(mesh_specs_2d, roi_2d, dim_names=base)
    mesh_specs["premade_mesh"] = {
        "nodes": bnodes.tolist(),
        "conn": bconn.tolist(),
        "base_dims": base,
        "stack_dims": stack,
        "ordered_key": ordered,
    }

    args = copy.deepcopy(cfg["args"])
    # Run the training stage *in its own folder*, exactly as a GUI user would:
    # stage scripts hard-code their output directories relative to the stage
    # folder (./output), so we write there instead of a throwaway _work dir.
    train_out_dir = os.path.join(STAGE31_DIR, "output")
    os.makedirs(train_out_dir, exist_ok=True)
    args["output_directory"] = train_out_dir
    args["plot_histograms"] = False
    args["data_file"] = train_ref

    DATA = _build_data_for_train(curated, ordered, output_vars)

    section("Training the surrogate")
    print("  Training surrogate (config mesh, augmented full dataset)...")
    train_output, SM = smb.sm_train(roi, mesh_specs, args, DATA=copy.deepcopy(DATA))

    sm_path = os.path.join(train_out_dir, "SM.pickle")
    assert_exists(sm_path, results)
    has_outputs = bool(SM.get("nodal_values"))
    if has_outputs:
        ok("surrogate outputs: " + ", ".join(SM["nodal_values"].keys()))
    else:
        fail("surrogate has no nodal_values outputs")
    results.append(("stage3 nodal_values present", has_outputs))

    # With proper data augmentation (anchor points + concat in stage 2) the mesh
    # has no empty elements, so the regression is well-conditioned and the nodal
    # values are finite. Verify that explicitly as a correctness check.
    all_finite = True
    for k in SM.get("nodal_values", {}):
        arr = np.asarray(SM["nodal_values"][k], dtype=float)
        if np.isnan(arr).any() or np.isinf(arr).any():
            all_finite = False
            fail("non-finite nodal values for output '{}' "
                 "(data likely under-augmented -> empty mesh elements)".format(k))
    if all_finite:
        ok("all surrogate nodal values are finite (well-conditioned regression)")
    results.append(("stage3 nodal_values finite", all_finite))

    # ------------------------------------------------------------------
    # Save the TRAINING TIME-SERIES results into the training stage's own
    # output/ folder. The concatenated training arrays are unravelled back
    # into per-simulation traces using 'evm' (strain) as the x-axis - the
    # deformation-history variable - with an explainer file describing the
    # choice and how to read them (dotted = ground truth, solid = surrogate).
    # ------------------------------------------------------------------
    ts_meta = curated.get("meta_data", {}) if isinstance(curated, dict) else {}
    ts_figs = _export_training_timeseries(
        SM, train_output, curated, ts_meta, output_vars, train_out_dir,
        xaxis="evm")
    if ts_figs:
        print("  Training time-series saved (x-axis = evm / strain):")
        for title, path in ts_figs:
            print("    - {:<28} {}".format(
                title, os.path.relpath(path, REPO_ROOT)))

    # ------------------------------------------------------------------
    # Export the Training Module's result plots (relative-error maps + 3D
    # results scatter) to PNG, in BOTH --demo and --test. The GUI shows these
    # on screen and offers an "Export" button; headlessly we reproduce the same
    # figures and write them to disk. The verbose explainer is demo-only.
    # ------------------------------------------------------------------
    if DEMO_MODE:
        explain(
            "READING THE TRAINING RESULT PLOTS (now in this stage's output/)\n"
            "First thing to glance at are the R^2 values printed in the console\n" 
            "(GREEN when it's >.99, YELLOW when it's >=.95, RED when it's <.95)\n" 
            "\n"
            "The Training Module GUI then shows the fit quality three ways; the demo\n"
            "exports each to a PNG so you can open and inspect them:\n"
            "\n"
            "TIME-SERIES PLOTS (Training_timeseries_<out>_vs_evm.png)\n"
            "The surrogate is trained on a CONCATENATED array merging every\n"
            "simulation into one long sequence; the GUI's 'x-axis' choice\n"
            "unravels it back into per-simulation traces. Here the x-axis is evm\n"
            "(strain) - the deformation-history 'clock' - not wall-clock time,\n"
            "because strain is the physically meaningful progression here.\n"
            "  * DOTTED line = ground truth (physics data)\n"
            "  * SOLID line  = surrogate prediction (one colour per simulation)\n"
            "Notice the solid curves lie almost PERFECTLY on top of their dotted\n"
            "counterparts across the ENTIRE data domain - and that domain spans\n"
            "7+ ORDERS OF MAGNITUDE in strain rate. Capturing that enormous\n"
            "dynamic range this well is exactly why we chose log-type scaling.\n"
            "\n"
            "RELATIVE-ERROR MAPS (Training_relerr_<out>.png)\n"
            "A 2-D map of the surrogate error (in PERCENT, %) in the RATE-FORM\n"
            "output - i.e. the surrogate's native prediction - at randomly\n"
            "sampled points drawn from across all simulations, laid out over\n"
            "vmJ2 (x) and temperature (y). The errors are largest in the\n"
            "UPPER-LEFT region (low stress, high temperature); everywhere else\n"
            "the fit is tight. Reading the map this way tells you where you might\n"
            "refine the mesh if you needed even better accuracy there.\n"
            "\n"
            "A NOTE ON UNITS (physical vs. scaled space)\n"
            "This map shows the error in PHYSICAL units - i.e. in the real\n"
            "output quantity - not in the transformer/scaled space the\n"
            "surrogate is actually trained in. In the Training Module GUI the\n"
            "user can TOGGLE between the two; here the demo simply shows the\n"
            "physical-units view.\n"
            "\n"
            "3D RESULTS PLOTS (Training_3d_<out>.png)\n"
            "The vertical axis is the RATE-FORM output itself; red circles are\n"
            "the data, black squares the surrogate. Remember this dataset has TWO\n"
            "rate-form outputs - U[evm] (strain rate) and U[rhoc] (dislocation-\n"
            "density rate) - COUPLED to each other through the coupled inputs\n"
            "(evm and rhoc), so you get one 3D plot per output.\n"
            "\n"
            "(Axis labels here are set headlessly; in the GUI you set them by\n"
            "double-clicking a label and typing the text you want.)\n"
            "\n"
            "A NOTE ON THE LIVE GUI: once the Training Module has populated these\n"
            "result plots on screen, they are fully INTERACTIVE - you can pan,\n"
            "ZOOM, and ROTATE the 3D views with the mouse, toggle series on/off,\n"
            "switch axes/scaling, and RENAME any axis or plot title (double-click\n"
            "it). The headless PNGs are static snapshots of those same figures."
        )
    figs = _export_training_plots(
        SM, train_output, DATA, ordered, output_vars, train_out_dir)
    if figs:
        print("  Training plots exported (open these to inspect the fit):")
        for title, path in figs:
            print("    - {:<28} {}".format(
                title, os.path.relpath(path, REPO_ROOT)))
    else:
        warn("no training plots were exported")

    pause(interactive)
    return sm_path


# ----------------------------------------------------------------------
# Stage 3.2 - Surrogate Testing
# ----------------------------------------------------------------------
def stage3_test(sm_path, test_pkl, results, interactive):
    banner("Stage 3.2 - Surrogate Testing")
    refs = os.path.join(STAGE32_DIR, "references")
    os.makedirs(refs, exist_ok=True)
    explain(
        "PIPELINE STEP - MOVE FILES TO THE NEXT STAGE\n"
        "Testing (Stage 3.2) needs TWO inputs MOVED into\n"
        "    3_Surrogate_Modeling/32_Surrogate_Testing/references/ :\n"
        "  1) the trained surrogate SM.pickle from Stage 3.1's output, and\n"
        "  2) the CURATED TESTING pickle (" + TEST_NAME + ") from Stage 2 -\n"
        "     i.e. the 20% held-out, UNSEEN data, so we measure generalisation.\n"
        "\n"
        "The demo MOVES both to the references-directories,\n" 
        "then evaluates the surrogate against the test traces\n"
        "and writes error metrics/plots to this stage's output/.\n"
        "(The SM.pickle is forwarded on again to Stage 4 afterwards.)"
    )
    sm_ref = stage_move(sm_path, refs, "SM.pickle")
    stage_move(test_pkl, refs, TEST_NAME)

    section("Running the coupled surrogate test")
    cmd = (
        "python test_surrogate.py --oracle-mode coupled "
        "--pickle_files {}".format(TEST_NAME)
    )
    say_cmd("cd 3_Surrogate_Modeling/32_Surrogate_Testing && " + cmd)
    cmd_list = [
        sys.executable,
        "test_surrogate.py",
        "--oracle-mode", "coupled",
        "--pickle_files", TEST_NAME,
    ]
    code, out = run_subprocess(cmd_list, STAGE32_DIR, stdin_text="\n\n")
    # Surface what the testing script actually printed (which outputs it is
    # plotting, coupled-validation progress, etc.) so the user can follow along.
    show_output(out, label="test_surrogate.py console output")
    if code != 0:
        fail("stage 3.2 script exited with code {}".format(code))
        results.append(("stage3.2 script exit 0", False))
    else:
        ok("stage 3.2 script completed")
        results.append(("stage3.2 script exit 0", True))

    out_dir = os.path.join(STAGE32_DIR, "output")
    assert_exists(os.path.join(out_dir, "Testing_result_coupled.pickle"), results)

    explain(
        "HOW TO READ THE TESTING RESULTS (and why they differ from training)\n"
        "Stage 3.1 showed how well the surrogate reproduces the data POINT-BY-\n"
        "POINT (per time-step error maps, 3D fits, strain-indexed traces).\n"
        "Stage 3.2 asks a harder, more realistic question: once the surrogate's\n"
        "predictions are fed back into a time-marching simulation, do the\n"
        "accumulated quantities still track the ground truth over the whole\n"
        "deformation history?\n"
        "\n"
        "WHAT THE TEST PLOTS SHOW\n"
        "You will again see MAPS (scatter + surface) and TRACES, but note the\n"
        "crucial difference: they are NOT drawn on the surrogate's raw predicted\n"
        "output. The surrogate predicts RATES (e.g. strain rate evm-dot and\n"
        "dislocation-density rate rhoc-dot). The test plots show the EFFECT of\n"
        "those predicted rates on the COUPLED INPUT PROGRESSION - i.e. the\n"
        "quantities obtained by forward time-integration of the rates.\n"
        "\n"
        "THE COUPLING, STEP BY STEP\n"
        "  1) At each increment the surrogate predicts the strain rate.\n"
        "  2) Multiplying by the time step dt (taken from the reference data so\n"
        "     the increments match exactly) and summing gives ACCUMULATED STRAIN.\n"
        "  3) The same forward integration turns the predicted dislocation-\n"
        "     density rate into ACCUMULATED dislocation density (rhoc).\n"
        "  4) These accumulated quantities feed back as inputs for the next step\n"
        "     (hence 'coupled'), and are compared against the ground-truth\n"
        "     accumulated values - now plotted against TIME on the x-axis of the\n"
        "     6x6 trace figure (one accumulated response per output).\n"
        "\n"
        "WHY THIS TEST IS MORE STRINGENT\n"
        "  * It uses the 20% HELD-OUT split, unseen during training, so it guards\n"
        "    against OVERFITTING.\n"
        "  * Because errors are integrated forward in time, it also exposes\n"
        "    ERROR ACCUMULATION / drift in the strain and dislocation-density\n"
        "    histories - a small per-step bias can compound over a trajectory.\n"
        "  * It therefore measures robustness and MAINTAINED ACCURACY across the\n"
        "    whole time-history, not just isolated per-time-step accuracy.\n"
        "A surrogate that looks great in Stage 3.1 but drifts here is not yet\n"
        "trustworthy for coupled, time-marching simulations.\n"
        "\n"
        "THE REL-RMSE SCATTER PLOTS & HISTOGRAMS\n"
        "Besides the 6x6 traces, this stage writes Testing_REL_RMSE_<out>.png.\n"
        "For each simulation the relative ROOT-MEAN-SQUARE error is computed\n"
        "across ALL its time increments and collapsed into a SINGLE number -\n"
        "one accuracy score for the whole trajectory. In the SCATTER plot that\n"
        "value is the colour of each point, mapped over vmJ2 and temperature,\n"
        "so you can see at a glance where (in stress/temperature space) the\n"
        "surrogate is most/least accurate over entire simulations. That is a\n"
        "direct guide for refining the 2-D base mesh in Stage 3.1 if needed.\n"
        "The accompanying HISTOGRAMS show the distribution of this per-\n"
        "simulation relative-RMSE value across the whole test set - a quick\n"
        "sanity check that the bulk of simulations sit at low error.\n"
        "\n"
        "WHERE THE HOT SPOTS SHOW UP (and that they were already foreseen)\n"
        "In the TOP-LEFT corner of the vmJ2-temperature space (low stress,\n"
        "high temperature) the accumulated error metric here is AGGRAVATED -\n"
        "the per-simulation rel-RMSE is noticeably higher there than elsewhere.\n"
        "That same corner maps directly onto the training-results 2-D plots\n"
        "produced in Stage 3.1: the relative-error map already flagged the\n"
        "UPPER-LEFT region (especially in the STRAIN-RATE output, U[evm]) as\n"
        "the least-accurate zone. So the hot spot is not a surprise - it was\n"
        "already SUSPECTED from the Stage 3.1 training results, and here the\n"
        "forward-integrated (accumulated) error metric simply amplifies it,\n"
        "because small per-step biases in that corner compound over the\n"
        "trajectory. The two stages corroborate each other: Stage 3.1 points\n"
        "at the region; Stage 3.2 confirms and magnifies it in the accumulated\n"
        "metric - and both point to the same place to refine the mesh if even\n"
        "better accuracy is needed there."
    )
    pause(interactive)
    return sm_ref


# ----------------------------------------------------------------------
# Stage 4 - Port to MOOSE
# ----------------------------------------------------------------------
def stage4(sm_path, results, interactive):
    banner("Stage 4 - Port to MOOSE")
    refs = os.path.join(STAGE4_DIR, "references")
    out_dir = os.path.join(STAGE4_DIR, "output")
    os.makedirs(refs, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    explain(
        "PIPELINE STEP - MOVE FILES TO THE NEXT STAGE\n"
        "Porting (Stage 4.1) consumes the trained SM.pickle. In the real\n"
        "pipeline it is MOVED forward from the previous stage's references into\n"
        "    4_Port_MOOSE/41_Export_Model/references/\n"
        "(factory-line semantics - no wasteful copies of the large pickle), then\n"
        "you run the exporter. It fills a MOOSE material-model template with the\n"
        "surrogate's parameters, producing C++ source (.C/.h) plus a\n"
        "benchmarks.csv of input/output pairs you later use (Stage 4.2) to verify\n"
        "the compiled MOOSE model reproduces the surrogate to machine precision."
    )
    stage_move(sm_path, refs, "SM.pickle")

    section("Exporting the surrogate to a MOOSE material model")
    cmd = "python port_surrogate_to_moose.py {}".format(MOOSE_MODEL_NAME)
    say_cmd("cd 4_Port_MOOSE/41_Export_Model && " + cmd)
    cmd_list = [sys.executable, "port_surrogate_to_moose.py", MOOSE_MODEL_NAME]
    code, out = run_subprocess(cmd_list, STAGE4_DIR, stdin_text="\n")
    show_output(out, label="port_surrogate_to_moose.py console output")
    if code != 0:
        fail("stage 4 script exited with code {}".format(code))
        results.append(("stage4 script exit 0", False))
    else:
        ok("stage 4 script completed")
        results.append(("stage4 script exit 0", True))

    assert_exists(os.path.join(out_dir, MOOSE_MODEL_NAME + ".C"), results)
    assert_exists(os.path.join(out_dir, MOOSE_MODEL_NAME + ".h"), results)
    assert_exists(os.path.join(out_dir, "benchmarks.csv"), results)
    pause(interactive)


# ----------------------------------------------------------------------
# Stage 5 - Clean Pipeline (dry run only)
# ----------------------------------------------------------------------
def stage5(results, interactive):
    banner("Stage 5 - Clean Pipeline (dry-run)")
    explain(
        "WHAT STAGE 5 DOES\n"
        "The clean-up script removes generated artefacts from stages 1-4 (the\n"
        "contents of their output/ and references/ folders) while preserving your\n"
        "own source data, documentation (CONTEXT.md/README.md), and the MOOSE\n"
        "base/ templates. Run it to reset the repo to a clean state between runs.\n"
        "\n"
        "DRY-RUN ONLY\n"
        "The demo runs it with --dry-run, which DELETES NOTHING. It only PREVIEWS\n"
        "the files that a real clean-up would remove - listed below as\n"
        "'Would remove: ...'. Drop the --dry-run flag to actually delete them."
    )
    script = os.path.join(STAGE5_DIR, "clean_pipeline.sh")
    assert_exists(script, results)
    section("Previewing files that a clean-up would remove")
    say_cmd("cd 5_Clean_Pipeline && ./clean_pipeline.sh --dry-run")
    code, out = run_subprocess(["bash", "clean_pipeline.sh", "--dry-run"], STAGE5_DIR)
    # Surface the preview so the user actually sees which files WOULD be deleted.
    show_output(out, label="clean_pipeline.sh --dry-run preview", n=80)
    if code != 0:
        fail("clean_pipeline.sh --dry-run exited with code {}".format(code))
        results.append(("stage5 dry-run exit 0", False))
    else:
        if KEEP_ARTIFACTS:
            ok("clean_pipeline.sh --dry-run completed "
               "(preview only, no files deleted)")
        else:
            ok("clean_pipeline.sh completed: artifacts removed")
        results.append(("stage5 dry-run exit 0", True))
    if DEMO_MODE:
        explain(
            "ALL DONE - YOUR PIPELINE IS NOW FULLY POPULATED\n"
            "Every stage above kept its generated artifacts, so you can now open\n"
            "and explore the curated data, plots, trained surrogate, test\n"
            "results and the exported MOOSE model at your own pace.\n"
            "\n"
            "WHEN YOU ARE DONE EXPLORING - HOW TO CLEAN UP\n"
            "Re-run the Stage 5 command WITHOUT the --dry-run flag to actually\n"
            "delete everything the preview listed above:\n"
            "    cd 5_Clean_Pipeline && ./clean_pipeline.sh\n"
            "WARNING: this WIPES ALL generated results across stages 1-4. There\n"
            "is no undo.\n"
            "\n"
            "TIP: work on a FRESH COPY of the pipeline for each new surrogate.\n"
            "That way you can safely clean a working copy, while keeping the\n"
            "fully-populated copy as an ARCHIVE - it captures the entire build\n"
            "flow (data -> surrogate -> MOOSE model), which is often useful later."
        )
    pause(interactive)


# ----------------------------------------------------------------------
# Harness cleanup
# ----------------------------------------------------------------------
def cleanup():
    """Remove all demo-generated artifacts (best-effort), restoring stages 1-5."""
    paths = [
        WORK_DIR,
        os.path.join(STAGE2_DIR, "references", SUBSET_NAME),
        os.path.join(STAGE2_DIR, "output", TRAIN_NAME),
        os.path.join(STAGE2_DIR, "output", TEST_NAME),
        os.path.join(STAGE31_DIR, "references", TRAIN_NAME),
        os.path.join(STAGE32_DIR, "references", "SM.pickle"),
        os.path.join(STAGE32_DIR, "references", TEST_NAME),
        os.path.join(STAGE4_DIR, "references", "SM.pickle"),
    ]
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
    # Remove any files the demo newly created in the stage output dirs.
    for out_dir, before in _OUTPUT_SNAPSHOTS.items():
        try:
            for name in os.listdir(out_dir):
                if name not in before:
                    fp = os.path.join(out_dir, name)
                    if os.path.isfile(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp)
        except OSError:
            pass


# Snapshot of stage output dirs taken before the run so cleanup only removes
# files the demo itself created.
_OUTPUT_SNAPSHOTS = {}


def snapshot_outputs():
    for out_dir in (
        os.path.join(STAGE2_DIR, "output"),
        os.path.join(STAGE31_DIR, "output"),
        os.path.join(STAGE32_DIR, "output"),
        os.path.join(STAGE4_DIR, "output"),
    ):
        if os.path.isdir(out_dir):
            _OUTPUT_SNAPSHOTS[out_dir] = set(os.listdir(out_dir))
        else:
            _OUTPUT_SNAPSHOTS[out_dir] = set()



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_pipeline(interactive):
    results = []
    stage1(results, interactive)
    train_pkl, test_pkl = stage2(results, interactive)
    sm_path = stage3_train(train_pkl, results, interactive)
    sm_path = stage3_test(sm_path, test_pkl, results, interactive)
    stage4(sm_path, results, interactive)
    stage5(results, interactive)
    return results


def print_summary(results):
    banner("Summary - pipeline check results")
    passed = sum(1 for _, ok_ in results if ok_)
    total = len(results)
    if VERBOSE_SUMMARY:
        for name, ok_ in results:
            if ok_:
                tag = C.GREEN + C.BOLD + " ✓ PASS " + C.END
            else:
                tag = C.RED + C.BOLD + " ✗ FAIL " + C.END
            print("  [{}] {}".format(tag, name))
    else:
        # Concise: only call out any failures, otherwise just the count.
        for name, ok_ in results:
            if not ok_:
                print("  [{}] {}".format(
                    C.RED + C.BOLD + " ✗ FAIL " + C.END, name))
    bar_colour = C.GREEN if passed == total else C.RED
    print("\n" + bar_colour + C.BOLD
          + "  {}/{} checks passed.".format(passed, total) + C.END)
    return passed == total


def main():
    parser = argparse.ArgumentParser(
        description="LAROMance Stage 6: run the full pipeline (stages 1-5) "
        "end-to-end against the example data, as an interactive demo or a "
        "non-interactive smoke test."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo", action="store_true",
        help="Interactive guided walkthrough of the full pipeline.",
    )
    mode.add_argument(
        "--test", action="store_true",
        help="Non-interactive smoke test; prints PASS/FAIL and sets exit code.",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="(--test only) Keep generated artifacts and print the full "
             "per-stage PASS/FAIL table for manual inspection.",
    )
    args = parser.parse_args()

    # --demo is always interactive (pauses between stages) and always keeps its
    # artifacts; --test is always non-interactive. The guided-vs-silent
    # distinction IS the demo/test distinction, so no extra flag is needed.
    interactive = args.demo
    keep_artifacts = args.demo or args.keep
    global DEMO_MODE
    DEMO_MODE = bool(args.demo)
    global KEEP_ARTIFACTS
    KEEP_ARTIFACTS = bool(keep_artifacts)
    # In --test, show the full per-stage table only for a deeper (--keep) run.
    global VERBOSE_SUMMARY
    VERBOSE_SUMMARY = bool(args.demo or args.keep)

    if args.demo:
        banner("LAROMance 2.0  ·  Guided Pipeline Demo")
        print(
            C.BOLD + "  Welcome!" + C.END + " This walks you through the complete LAROMance\n"
            "  workflow end-to-end, using the shipped Viscoplasticity/Creep example.\n"
            "\n"
            "  You will go through these stages, one at a time:\n"
            + C.CYAN + "    1  Data Generation   " + C.END + "– locate the raw simulation dataset\n"
            + C.CYAN + "    2  Data Processing   " + C.END + "– curate, split (train/test), augment\n"
            + C.CYAN + "    3.1 Surrogate Training" + C.END + "– build & calibrate the grid surrogate\n"
            + C.CYAN + "    3.2 Surrogate Testing " + C.END + "– coupled, time-marching validation\n"
            + C.CYAN + "    4  Port to MOOSE     " + C.END + "– export a C++ material model\n"
            + C.CYAN + "    5  Clean Pipeline    " + C.END + "– preview a repo clean-up (dry-run)\n"
            "\n"
            "  Tips:\n"
            "    • Blue " + C.BLUE + "$ lines" + C.END + " are the equivalent terminal commands.\n"
            "    • Cyan " + C.CYAN + "─── blocks" + C.END + " explain the WHY behind each step.\n"
            "    • Grey framed text is the real console output of each stage script.\n"
            "    • See README.md (big picture) and 0_Demo_Test/CONTEXT.md (this stage).\n"
            "    • No GUI is required — training runs headlessly."
        )
        pause(interactive)

    all_ok = False
    snapshot_outputs()
    try:
        results = run_pipeline(interactive)
        all_ok = print_summary(results)
    finally:
        if not keep_artifacts:
            cleanup()
            print("\n  (generated artifacts cleaned up; use --keep to retain them)")

    if args.test:
        print(("\n" + C.GREEN + "  RESULT: PASS" + C.END) if all_ok
              else ("\n" + C.RED + "  RESULT: FAIL" + C.END))
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

