# Cantera Kinetics Example

This folder demonstrates how to generate training data for chemical kinetics using **Cantera**.

## Files

| File                              | Description |
|-----------------------------------|-------------|
| `generate_cantera_data.py`        | Main script to generate surrogate training data from Cantera simulations |
| `src/gri_autoignition.py`         | Core simulation module that runs constant-volume autoignition cases using the GRI-Mech mechanism |

## Overview

The workflow consists of two main steps:

1. **Design of Experiments (DoE)**: Generate input conditions (`samples.csv`).
2. **Data Generation**: Run Cantera simulations using the generated samples.

## Step 1: Generate Input Samples (`samples.csv`)

Before running the data generator, you need to create a `samples.csv` file containing the input conditions (typically temperature, pressure, and equivalence ratio).

### Recommended ranges

| Variable | Min   | Max    | Unit | Sampling |
|----------|-------|--------|------|----------|
| `T`      | 800   | 2200   | K    | Linear   |
| `P`      | 1e5   | 4e6    | Pa   | Log      |
| `phi`    | 0.4   | 2.2    | —    | Linear   |

### How to generate `samples.csv`

You can use the DoE script located at:

```bash
cd 1_Data_Generation/12_Design_of_Experiment
python doe.py TEMP 800 2200 PRESSURE 1e5 4e6 PHI 0.4 2.2 --number 10000 --log10 PRESSURE
```

> **Tip**: Use logarithmic sampling for pressure (`P`) since it spans multiple orders of magnitude.

## Step 2: Generate Cantera Data

Once `samples.csv` exists, run the data generation script:

```bash
cd 1_Data_Generation/13_Batch_Template_Setup
python generate_cantera_data.py
```

This will execute the simulations defined in `src/gri_autoignition.py` and save the results in a format compatible with LAROMance.

## Using This Example in the Pipeline

To use this Cantera example to generate data locally (instead of using the standard template run folder structure), copy the following files into the relevant stage folders:

### Required Copies

| Source File                              | Destination Folder                                      | Purpose |
|------------------------------------------|----------------------------------------------------------|---------|
| `src/gri_autoignition.py`                | `1_Data_Generation/11_Design_of_Experiments/src/`        | Simulation module used during batch execution |
| `generate_cantera_data.py`               | `1_Data_Generation/13_Batch_Template_Setup/`       | Main data generation script |

After copying these files, you can generate Cantera data directly from the example without relying on the full template-based run folder setup in Stage 1.3.

## Output

The script produces a pickle file (e.g. `data.pickle`) containing the simulation results, which can then be used in:

- **Stage 2** (`2_Data_Processing`) for curation and splitting
- **Stage 3** (`3_Surrogate_Modeling`) for surrogate training

## Notes

- The current implementation uses the **GRI-Mech 3.0** mechanism.
- Simulations are performed in constant-volume homogeneous reactors.
- You can modify `src/gri_autoignition.py` if you want to change the simulation type or mechanism.