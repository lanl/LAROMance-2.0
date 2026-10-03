# Viscoplastic Creep Example Data

This example package contains a single **pickle** file with the results of
**5 000** individual creep simulations of an arbitrary (fictional) metal (polycrystal). The file is intended to be used as the
input for the **02_Data_Processing** stage of the LAROMance pipeline, and can from there be processed through the entire workflow.

## File layout

```
Viscoplasicity_Creep/
├─ README.md                          ← You are reading this file
└─ Creep_runs_with_dislocation_density_tracking.pickle  ← Pickle containing the simulation data
```

The pickle stores a **Python dictionary** where each key is a *simulation ID*
(`int` ranging from `1` to `5000`).  The value for each ID is another dictionary
with the following entries (see `dict_keys([...])` in the original data):

| Key | Description | Units / Type |
|-----|-------------|---------------|
| `vmJ2` | Scalar measure of the von Mises equivalent stress for this simulation. | **MPa** |
| `temperature` | Temperature at which the simulation was run. | **K** |
| `evm` | Effective strain (ε).  This is one of the primary output rates. | **-** |
| `rhoc` | Dislocation density (ρ).  Another primary output rate. | **1/(m^2)** |
| `t` | Physical time vector – the list/array of time points at which the state was recorded. | **s** |
| `dt` | Time‑step size used during the simulation (constant for a given run). | **s** |
| `U` | Nested dictionary storing the *rates* for each time step.  It has two keys:
`evm` | Effective strain‑rate (ε̇).  This is one of the primary output rates. | **1/s** |
`rhoc` | Dislocation density rate (ρ̇).  Another primary output rate. | **1/(m^2·s)** |

All output quantities are **rates**; there are no accumulated strain or density
values in this file.  The separation into `evm` and `rhoc` makes it straightforward
to train a surrogate that predicts either or both rates given the input stress
and temperature.

## Using the data in the LAROMance pipeline

The **02_Data_Processing** stage expects a pickle (or other supported format)
containing the raw simulation results.  Place this data pickle file in the
`references` sub‑folder of `2_Data_Processing` and run the provided utilities:

```bash
cd 2_Data_Processing
python load_and_plot_data.py --pickle_files ./references/Creep_runs_with_dislocation_density_tracking.pickle
```

The processing scripts will:
1. Curate the dataset (remove NaNs, filter outliers, etc.).
2. Optionally augment the data (e.g., interpolate between time steps).
3. Produce visualisations that are saved under `2_Data_Processing/output`.

After this stage the curated data can be fed straight into the **03_Surrogate_Modeling**
pipeline to train a surrogate model for the creep strain‑rate and dislocation‑
density‑rate as functions of `vmJ2` and `temperature`, `evm` (strain), and `rhoc` (dislocation density).

---
