"""
© 2026. Triad National Security, LLC. All rights reserved.

LAROMance 2.0 – Stage 0: Demo / Test
------------------------------------

A single, self-contained harness that exercises the entire LAROMance pipeline
(stages 1 -> 5) end-to-end against the shipped synthetic example datasets.

Two mutually-exclusive modes:

    --demo    Interactive guided walkthrough: prints the equivalent terminal
              command for each stage, runs it against the example data,
              summarises the artifacts produced, and pauses between stages.

    --test    Non-interactive smoke test: runs the same sequence, asserts the
              expected artifacts exist, prints a per-stage PASS/FAIL table and
              sets the process exit code (0 = all passed, non-zero otherwise).

Add --yes to suppress all interactive pauses/prompts in either mode.

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
    GREEN = "\033[92m" if _tty else ""
    RED = "\033[91m" if _tty else ""
    YELLOW = "\033[93m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""
    END = "\033[0m" if _tty else ""


def banner(text):
    line = "=" * 72
    print("\n" + C.HEAD + line + C.END)
    print(C.HEAD + C.BOLD + "  " + text + C.END)
    print(C.HEAD + line + C.END)


def say_cmd(cmd):
    print(C.BLUE + "  $ " + cmd + C.END)


def ok(msg):
    print(C.GREEN + "  [OK] " + C.END + msg)


def warn(msg):
    print(C.YELLOW + "  [!]  " + C.END + msg)


def fail(msg):
    print(C.RED + "  [X]  " + C.END + msg)


def pause(interactive):
    if interactive:
        try:
            input(C.YELLOW + "\n  <press Enter to continue> " + C.END)
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
    if code != 0:
        fail("stage 2 script exited with code {}".format(code))
        print(_tail(out))
        results.append(("stage2 script exit 0", False))
    else:
        ok("stage 2 script completed")
        results.append(("stage2 script exit 0", True))

    train_pkl = os.path.join(STAGE2_DIR, "output", TRAIN_NAME)
    test_pkl = os.path.join(STAGE2_DIR, "output", TEST_NAME)
    assert_exists(train_pkl, results)
    assert_exists(test_pkl, results)
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


def stage3_train(train_pkl, results, interactive):
    banner("Stage 3.1 - Surrogate Training (headless)")
    from utils import sm_build as smb
    from utils import sm_data as smd
    from utils import fe_shapes as fes

    print(
        "  The training GUI (training_module.py) needs PySide6. We reproduce the\n"
        "  same training headlessly via utils.sm_build.sm_train, driven by the\n"
        "  example config JSON (equivalent GUI action shown below)."
    )
    say_cmd("cd 3_Surrogate_Modeling/31_Surrogate_Training && python training_module.py  # (GUI)")

    with open(EXAMPLE_CONFIG) as fh:
        cfg = json.load(fh)
    with open(train_pkl, "rb") as fh:
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
    roi = {
        k: [float(np.min(curated["data"][k])), float(np.max(curated["data"][k]))]
        for k in ordered
    }

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
    train_out_dir = os.path.join(WORK_DIR, "stage3_train")
    os.makedirs(train_out_dir, exist_ok=True)
    args["output_directory"] = train_out_dir
    args["plot_histograms"] = False
    args["data_file"] = train_pkl

    DATA = _build_data_for_train(curated, ordered, output_vars)

    print("  Training surrogate (config mesh, augmented full dataset)...")
    _, SM = smb.sm_train(roi, mesh_specs, args, DATA=copy.deepcopy(DATA))

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

    pause(interactive)
    return sm_path


# ----------------------------------------------------------------------
# Stage 3.2 - Surrogate Testing
# ----------------------------------------------------------------------
def stage3_test(sm_path, test_pkl, results, interactive):
    banner("Stage 3.2 - Surrogate Testing")
    refs = os.path.join(STAGE32_DIR, "references")
    os.makedirs(refs, exist_ok=True)
    shutil.copy(sm_path, os.path.join(refs, "SM.pickle"))
    shutil.copy(test_pkl, os.path.join(refs, TEST_NAME))

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
    if code != 0:
        fail("stage 3.2 script exited with code {}".format(code))
        print(_tail(out))
        results.append(("stage3.2 script exit 0", False))
    else:
        ok("stage 3.2 script completed")
        results.append(("stage3.2 script exit 0", True))

    out_dir = os.path.join(STAGE32_DIR, "output")
    assert_exists(os.path.join(out_dir, "Testing_result_coupled.pickle"), results)
    pause(interactive)


# ----------------------------------------------------------------------
# Stage 4 - Port to MOOSE
# ----------------------------------------------------------------------
def stage4(sm_path, results, interactive):
    banner("Stage 4 - Port to MOOSE")
    refs = os.path.join(STAGE4_DIR, "references")
    out_dir = os.path.join(STAGE4_DIR, "output")
    os.makedirs(refs, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(sm_path, os.path.join(refs, "SM.pickle"))

    cmd = "python port_surrogate_to_moose.py {}".format(MOOSE_MODEL_NAME)
    say_cmd("cd 4_Port_MOOSE/41_Export_Model && " + cmd)
    cmd_list = [sys.executable, "port_surrogate_to_moose.py", MOOSE_MODEL_NAME]
    code, out = run_subprocess(cmd_list, STAGE4_DIR, stdin_text="\n")
    if code != 0:
        fail("stage 4 script exited with code {}".format(code))
        print(_tail(out))
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
    script = os.path.join(STAGE5_DIR, "clean_pipeline.sh")
    assert_exists(script, results)
    say_cmd("cd 5_Clean_Pipeline && ./clean_pipeline.sh --dry-run")
    code, out = run_subprocess(["bash", "clean_pipeline.sh", "--dry-run"], STAGE5_DIR)
    if code != 0:
        fail("clean_pipeline.sh --dry-run exited with code {}".format(code))
        print(_tail(out))
        results.append(("stage5 dry-run exit 0", False))
    else:
        ok("clean_pipeline.sh --dry-run completed (no files deleted)")
        results.append(("stage5 dry-run exit 0", True))
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
    stage3_test(sm_path, test_pkl, results, interactive)
    stage4(sm_path, results, interactive)
    stage5(results, interactive)
    return results


def print_summary(results):
    banner("Summary")
    passed = sum(1 for _, ok_ in results if ok_)
    total = len(results)
    for name, ok_ in results:
        tag = (C.GREEN + "PASS" + C.END) if ok_ else (C.RED + "FAIL" + C.END)
        print("  [{}] {}".format(tag, name))
    print("\n  {}/{} checks passed.".format(passed, total))
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
        "--yes", action="store_true",
        help="Suppress all interactive pauses/prompts (useful for CI).",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep generated artifacts instead of cleaning up afterwards.",
    )
    args = parser.parse_args()

    # --test is always non-interactive; --demo is interactive unless --yes.
    interactive = args.demo and not args.yes

    if args.demo:
        banner("LAROMance 2.0 - Pipeline Demo")
        print(
            "  Welcome! This runs the whole LAROMance workflow (stages 1-5) against\n"
            "  the shipped Viscoplasticity/Creep example. See the root README.md for\n"
            "  the big picture and 0_Demo_Test/CONTEXT.md for details on this stage.\n"
            "  No GUI is required - training runs headlessly."
        )
        pause(interactive)

    all_ok = False
    snapshot_outputs()
    try:
        results = run_pipeline(interactive)
        all_ok = print_summary(results)
    finally:
        if not args.keep:
            cleanup()
            print("\n  (generated artifacts cleaned up; pass --keep to retain them)")

    if args.test:
        print(("\n" + C.GREEN + "  RESULT: PASS" + C.END) if all_ok
              else ("\n" + C.RED + "  RESULT: FAIL" + C.END))
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

