#!/usr/bin/env python3
"""
Simple plotting script.

Reads the CSV file ``model_csv.csv`` located in the same directory as this script,
selects two columns (provided via command‑line arguments) and produces a line plot.
Optional ``--logx`` and ``--logy`` switches enable logarithmic scaling of the
corresponding axes.

Usage:
    python plot_model.py X_COLUMN Y_COLUMN [--logx] [--logy]

Example:
    python plot_model.py time temperature --logy
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two columns from one or more CSV files as line plots."
    )
    parser.add_argument(
        "x_column",
        help="Name of the column to use for the x‑axis (common to all CSVs).",
    )
    parser.add_argument(
        "y_column",
        help="Name of the column to use for the y‑axis (common to all CSVs).",
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="One or more CSV files to plot.",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Use logarithmic scale for the x‑axis.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale for the y‑axis.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Minimum value for the x‑axis.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Maximum value for the x‑axis.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Minimum value for the y‑axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Maximum value for the y‑axis.",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Enable dark background theme for the plot.",
    )
    parser.add_argument(
            "--xlabel",
            type=str,
            default=r"$\bar{\varepsilon}_\mathrm{\theta}$ [-]",
            help="String used as x-axis label.",
        )
    parser.add_argument(
            "--ylabel",
            type=str,
            default=r"$\dot{\varepsilon}_\mathrm{pl}$ [s$^{-1}$]",
            help="String used as y-axix label.",
        )
    parser.add_argument(
            "--fontsize",
            type=float,
            default=14,
            help="Fontsize for axis and tick labels.",
        )

    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        sys.stderr.write(f"Error: CSV file not found at {csv_path}\\n")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.stderr.write(f"Error reading CSV: {e}\\n")
        sys.exit(1)
    return df


def main() -> None:
    args = parse_args()

    data_frames = []
    for csv_file in args.csv_files:
        df = load_data(csv_file)
        if args.x_column not in df.columns:
            sys.stderr.write(f"Error: column '{args.x_column}' not found in {csv_file}.\\n")
            sys.exit(1)
        if args.y_column not in df.columns:
            sys.stderr.write(f"Error: column '{args.y_column}' not found in {csv_file}.\\n")
            sys.exit(1)
        data_frames.append(df)

    # Concatenate all series for bounds calculation
    all_x = pd.concat([df[args.x_column] for df in data_frames])
    all_y = pd.concat([df[args.y_column] for df in data_frames])

    if args.dark:
        plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))

    # Plot each CSV with a distinct color and label
    for df, csv_file in zip(data_frames, args.csv_files):
        x = df[args.x_column]
        y = df[args.y_column]
        plt.plot(x, y, marker="o", linestyle="-", label=csv_file.name)

    plt.xlabel(args.x_column)
    plt.ylabel(args.y_column)
    plt.title(f"{args.y_column} vs {args.x_column}")
    plt.legend()

    # Apply logarithmic scaling first so limits are set appropriately
    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")

    # Helper to compute bounds, respecting log scale (positive only)
    def _bounds(series, log_flag, user_min, user_max):
        if user_min is not None:
            lower = user_min
        else:
            lower = series[series > 0].min() if log_flag else series.min()
        if user_max is not None:
            upper = user_max
        else:
            upper = series.max()
        return lower, upper

    x_low, x_high = _bounds(all_x, args.logx, args.xmin, args.xmax)
    y_low, y_high = _bounds(all_y, args.logy, args.ymin, args.ymax)

    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xticks(fontsize=args.fontsize)
    plt.yticks(fontsize=args.fontsize)
    plt.xlabel(args.xlabel, fontsize=args.fontsize)
    plt.ylabel(args.ylabel, fontsize=args.fontsize)

    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()