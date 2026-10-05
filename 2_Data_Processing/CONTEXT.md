# 2 – Data Processing
This step loads the raw physics‑based dataset (generated in step 1) of individual time history traces coming from a high-fidelity model. This stage then curates the data by downsampling, cleans, and optionally, augments data with extra points (e.g., where data is sparse). 

> The example data `./pickle`-file in the `../examples/ViscoPlasticity_Creep` folder is already in the right format. Copy it to the `2_Data_Processing/references` folder and then run the `load_and_plot_data.py` script.

## Accepted input format
The pipeline reads a single **`.pickle`** file.  

Inside the pickle there is a top‑level dictionary.

* The top‑level dictionary now contains two primary keys:
  * **"meta_data"** – auxiliary information for each simulation. Currently it stores the number of increments (`"num_increments"`) but can be extended with any other per‑simulation metadata.
  * **"data"** – the actual simulation data. This is itself a dictionary containing the following:
      * **Simulation ID (int)** – uniquely identifies a single simulation instance.  
      * For each ID, the dictionary contains:
        * **Input variables** – stored directly under descriptive keys (e.g., `"temperature"`, `"stress"`).  
        * **Outputs** – grouped under a nested dictionary **`"U"`**. The keys inside **`U`** are the output variable names (e.g., `"strain_rate"`, `"dislocation_density_rate"`).

The surrogate has a grid-based architecture, with its parameter space aligning with the input space. Such a grid-based architecture requires structured data, i.e, all input and output entries hold NumPy arrays or Python lists of the **same length**, representing the time‑series or sample points for that variable.

> **Example (simplified)**  
> ```python
> {
>     "data": {
>         42: {                           # simulation #42
>             "temperature": np.array([...]),
>             "stress": np.array([...]),
>             "U": {
>                 "dislocation_density_rate": np.array([...]),
>                 "strain_rate": np.array([...])
>             }
>         },
>         43: { … }                       # next simulation
>     }
> }
> ```

In short, the pickle must contain a dict with a `"data"` entry, where each integer key maps to a dictionary of **input‑variable arrays** plus a sub‑dictionary **`"U"`** that holds the **output‑variable arrays**. This uniform layout allows the subsequent stages of the workflow to reliably extract inputs, generate designs of experiments, and train surrogate models. Furthermore, a `"meta_data"` entry is optional, but can be convenient for tracking information about simulation runs, e.g., the number of time-steps recorded, or the number of simulations ID's that contained no data.

The best and most realistic example of the expected data format can be found in [examples/Viscoplasicity_Creep](../examples/Viscoplasicity_Creep/).

## Training Format
For Training, the data arrays should be concatenated: all individual simualations assembled into long arrays. This will be handled by executing this stage's python script.

This stage then stores the curated version ready for surrogate‑model training or for testing. This stage can be re-used for both Training and Testing. Training requires concatenation, and testing does not.

Plots are generated to visualize the data with scatter plots and line plots.

## What the Python script does
1. Loads a pickle file containing the original dataset. Make sure the data file is located in `./references` of this Stage.
2. Curates the dataset by down‑sampling to a manageable size.  
3. Optionally, augmentation techniques can be applied to enrich the curated dataset, such as:
   - Noise injection
   - Synthetic data generation
   
   This may produce an augmented dataset that can improve surrogate‑model robustness. An important consideration is data density across the discretized input domain (by the surrogate model's mesh of nodes). Empty elements will result in ill-conditioned regions in the system matrix solved during a linear regression step Refer to the LAROMANCE_USER_MANUAL.html in the 3_Surrogate_Modeling (training) stage. In some cases it may be valid to augment data by interpolating in between existing data points and where gaps exist in a specific mesh (interpolate in between intended nodal positions to prevent emptiness). Care should be taken with this approach, and is best applied in regions where the model is not likely to be queried.
4. Optionally, the data can be concatenated into one large array for Training. This is not needed for Testing. 
5. Stores the curated dataset back to disk in the `./output` folder.
6. Plots the data-set in the form of scatter plots and time-history line traces. Plots are stored in the `./output` folder

## How to run
Make sure the required data file is located in `./references` of this Stage.

Make sure to execture all bash commands in sequence. Do not skip any as this will break the procedure.

First change your workdir to the stage's directory, containing the `load_and_plot_data.py` script:

```bash
cd ./2_Data_Processing
```

And make sure the right conda env is activated (if applicable):
```bash
conda activate laromance
```

Then simply run:

```bash
python load_and_plot_data.py
```

**The script will prompt the intelligent user for a maximum length or a down‑sampling factor if no arguments are supplied.**

Or the user can immediately add flags to the call if the user knows.

```bash
python load_and_plot_data.py --pickle_files Creep_runs_with_dislocation_density_tracking.pickle  --max_length 150 --split_data 80 --concat --anchor_points 0.0,1e-14,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-2,0.2 --input_keys vmJ2,temperature --output_keys rhoc,U[evm],U[rhoc]
```

- `--pickle_files` Pickle files to include in the data processor, can be more than 1
- `--max_length N` Trim each time‑series to __N__ points (e.g., 150).
- `--split_data P` Allocate __P %__ of the data to the training set (remaining to testing).
- `--anchor_points a,b,c,…` Interpolate all fields at the specified anchor values (e.g., `0.0,1e-12,1e-10,…`).
- `--input_keys key1,key2,…` Select the listed columns as __inputs__ for the surrogate model scatter plots (e.g., `vmJ2,temperature`).
- `--output_keys key1,key2,…` Select the listed columns as __outputs__ for the surrogate model (e.g., `rhoc,U[evm],U[rhoc]`).
- `--concat` Concatenate the training data.
- `--correct_rhoc` Only needed for data for incorrectly saved data where the dislocation density should be transformed to dislocation density rates.

---

**Detailed Guidance for intelligent user (human or artificial)**

The script uses `smd.curate_dataset()` which accepts two mutually exclusive parameters that control the down‑sampling:

- `downsample_max_length` – maximum number of timesteps to keep after down‑sampling.  
  - The maximum length will downsample the data accordingly by taking away data points across a log-scaled domain. If not provided, the user can optionally provide the factor.
   - More advanced parameters related to the maximum length setting, can be changed deeper in the code, specifically inside the function `sm_data.curate_dataset()`, where the following can be adjusted:
      ```python
      DownSampler(
                  max_length=downsample_max_length,
                  downsampling_threshold=-20,
                  scale_factor=None,
                  scale_data=True,
                  remove_initial_incs=0
               )
      ```
      1. In particular the ```scale_data``` argument can be set to **False**, to downsample across a linear (rather than logarithmic) domain. 
      2. Also the ```downsampling_threshold``` can be adjusted (this should be considered in the unit of the data domain, e.g., log10-space when ```scale_factor=True```). This parameter will set a balance point at the provided value, and will provide a data set with equal max data points on either side of the ```downsample_threshold``` value provided as argument. Default is set to a very low value of $10^{-20}$ (```downsampling_threshold=-20``` with ```scale_factor=True```), such that the threshold is effectively outside of (most) data and as such is effectively disabled (unused).
- `downsample_factor` – factor by which to reduce the number of points (e.g., 1.3 means ~30 % fewer points).  

The script prints progress information and creates a new pickle file in the same `data_path` directory.
### Interactive workflow

1. **Inspect the data** – after loading, the script prints the average length of simulations:

   **Use this information to decide whether to limit the length or to apply a down‑sampling factor.**

2. **Inline Python-generated prompt** – During this step, the console prompts the user to provide:
   - *Option A*: Provide a **maximum length** (e.g., 500).  If skipped (enter), then go to next step:
   - *Option B*: Provide a **down‑sampling factor** (e.g., 1.5).  

3. **Console prints** information about the dataset: length of simulations and input/output variable labels.

   **Use the information for the next step to know what to plot**

## Plotting (step 5)
Visualizing the curated dataset relies on the helper functions from `sm_plot.py` in the `../utils` folder. 

All plots are saved to `./outputs`.

### Core plotting call for creating scatter plots (called by load_and_plot_data.py):

```python
smp.plotdata(
    data,
    input_keys=["vmJ2", "temperature"],          # variables on the x‑axis
    output_keys=["rhoc", "U[evm]"],              # variables on the y‑axis
    simulation_recording_interval=1,
    inc=-1,                                      # incidence to plot; -1 = last incidence
    cmap='magma',
    storepath="./output",                             # where the figure file will be saved
    dark=True
)
```

- **`input_keys`** – list of dataset keys that define the horizontal axes (e.g., temperature, time, strain).  
- **`output_keys`** – list of keys whose values are shown on the vertical axes.  
- **`inc`** – which simulation incidence to plot. `-1` selects the *last* incidence (useful for a final‑state scatter plot). Change this value (e.g., `0`, `5`, `10`) to visualise earlier incidences.  
- **`cmap`** – colour map for the scatter plot (`'magma'`, `'viridis'`, …).  
- **`storepath`** – directory (relative to the current working directory) where the generated figure will be written.  
- **`dark`** – set `True` for a dark‑theme figure, `False` for light.

### Optional: Plot individual traces

If you want to inspect time‑series traces of specific variables, you can use:

```python
smp.plottraces(
    data,
    input_keys=["t", "evm"],                     # usually time or strain
    output_keys=["rhoc", "U[evm]"],
    ylog=True,                                   # log‑scale on y‑axis
    storepath="./output",
    color='green',
    dark=True
)
```

This produces line‑plots for each simulation in the dataset. It is **not required** for the paper‑ready figures but can be helpful during exploratory analysis.

---

## Modifying the plotting script

- Open `plot_data.py` in an editor.
- Locate the call to `smp.plotdata(...)` and adjust `input_keys`, `output_keys`, and `inc` as needed.
- Save the file and re‑run `python plot_data.py` to generate the updated figure.

---

**Tip:** After editing the keys, verify that they exist in the dataset by printing the first entry:

```python
first_key = list(data.keys())[0]
print("Available keys:", data[first_key].keys())
```

This helps avoid `KeyError` if a typo is introduced.

---

*Make sure the required library `sm_data.py` and `sm_plot.py` is present in this folder (imported locally).*

## Augmentation (optional)
After plotting, the user is asked to augment the data.

The augmentation step enriches the curated dataset by inserting additional data points between a set of user‑defined anchor points. To use it, supply the `--anchor_points` argument when running the script, providing a comma‑separated list of numbers, for example:

```bash
python load_and_plot_data.py --anchor_points 0,1e-5,0.2
```

If `--anchor_points` is omitted, the script will prompt you to enter the list interactively (press __Enter__ to skip). Supplying no anchor points simply bypasses augmentation, allowing the rest of the workflow (curation, plotting, and downstream analysis) to proceed as before.

After the augmentation, the same plotting scripts are used to once again plot the augmented data and stores the .png graphs in the `output` folder.

## Prepare Next Stage
After this step is complete, only a Human user may proceed. Prepare the next step by moving the training pickle to the `../3_Surrogate_Modeling/31_Surrogate_Training/references` and the testing pickle `../3_Surrogate_Modeling/31_Surrogate_Testing/references`.


