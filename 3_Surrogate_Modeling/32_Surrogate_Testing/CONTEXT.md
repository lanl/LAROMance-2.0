# 3.2 - Surrogate Testing
This stage tests the LAROMAnce surrogate built by the training step, using a __TEST__ data set (unseen during training) and a graphical user interface Training Module (instruction manual is here [LAROMANCE_USER_MANUAL.pdf](3_Surrogate_Modeling/31_Surrogate_Training/doc/LAROMANCE_USER_MANUAL.pdf)).

Proper testing overfitting requires a different dataset than used for training. This unseen dataset should not be concatenated, but should per-simulation time-histories. When the data processor of step `2_Data_Processing` is used for splitting a dataset into a training and testing partition, the this data format should correctly be stored in the respective *testing* pickle (see `.pickle` filename). The location of this test-data `.pickle` file should be `./references`.

For the polycrystal creep modeling for which LAROMance is originally designed, this step runs creep simulations and then compares the surrogate prediction to the ground truth in the data. This is not done on the rate-based outputs, but on the accumulated strains. The error metrics are calculated based on the accumulated strain. Since the surrogate predicts, rates, considering the accumulated strain effectively considers the compounded error across time increments, which is a more stringent method than merely comparing rates directly.

The code in this sstage is specifically written for comparing creep simulations. For other types of simulations or evaluations, the user must code those custom in `../../utils/sm_build.py`.


## What the script does

1. Loads the testing data `.pickle`-file and the trained surrogate `SM.pickle` in `./references`. Before running anything, first make sure those files are present.
2. Runs the simulations with the inputs from the data.
3. Calculates error field metrics (e.g., mean squared relative error, i.e., MSRE).
4. Stores the test results in the `output` folder of this stage, including error plots and time-history traces.
   
## How to Run
First move the trained surrogate model `SM.pickle` from the previous steps `output` folder, to the current stage's `references` folder.

```bash
cd ./3_Surrogate_Modeling/32_Surrogate_Testing
```

And make sure the right conda env is activated (if applicable):
```bash
conda activate laromance
```

Then simply run:

```bash
python test_surrogate.py --pickle_files Curated_20pct_test_Creep_runs_with_dislocation_density_tracking.pickle
```


## Oracle Mode
During testing the script can run __oracle (truth‑injection) modes__. An oracle run feeds the surrogate with the *exact* high‑fidelity values for a selected subset of output variables, while the remaining outputs are still predicted by the surrogate. This lets the user probe how tightly the model’s outputs are coupled to one another – for example, the strain‑rate (`evm`) is an output, but the accumulated strain (which depends on the integrated strain‑rate) may also be used as an input to compute the next strain‑rate. By forcing the true strain‑rate and letting the surrogate predict the accumulated strain, or vice‑versa, you can see whether error in one variable propagates to the other.

The script presents an interactive prompt to select one of several __oracle modes__ (`coupled`, `single`, `pairwise`, `all`). Each mode defines a different pattern of forced‑output versus predicted‑output runs. Detailed descriptions of these options follow in the next section.

The `test_surrogate.py` script provides a flexible **`--oracle-mode`** argument that controls how truth‑injection (oracle) runs are executed. Below is a summary of each option and its usage.

### Modes

| Mode | Description | Runs Executed |
|------|-------------|--------------|
| **`coupled`** | Fully coupled run with **no forced outputs**. This is the default when no `--oracle-mode` is supplied. | 1 run (Coupled) |
| **`single`** | Coupled **plus** one run for each individual output that is forced. | Coupled + one run per output |
| **`pairwise`** | Coupled **plus** runs where **exactly one output is *not* forced** (i.e., truth‑injection is applied to all other outputs). | Coupled + one run per “omit‑one” combination |
| **`all`** | Executes **both** the `single` and `pairwise` configurations, covering all individual‑output and all‑but‑one runs, while keeping the Coupled run only once. | Coupled + all single‑output runs + all pairwise runs |

### Example Usage

```bash
# Only Coupled (no truth‑injection)
python test_surrogate.py --oracle-mode coupled

# Coupled + each output forced individually
python test_surrogate.py --oracle-mode single

# Coupled + each configuration where one output is omitted
python test_surrogate.py --oracle-mode pairwise

# Full suite: Coupled, all single‑output, and all‑but‑one runs
python test_surrogate.py --oracle-mode all
```

If **`--oracle-mode`** is omitted, the script will prompt you interactively:

```
Select oracle mode (coupled/single/pairwise/all) [coupled]:
```

You may also override any mode by using the **`--force`** option to explicitly list the outputs to force, e.g.:

```bash
python test_surrogate.py --force rhoc rhow
```

## Output Files

- **Plots**: PNG files are saved in the directory given by `--output-directory` (default: `./output`). Filenames follow the pattern `<output_key>_…‑oracle.png`.
- **Pickles**: Validation results are saved with the new naming convention:
  - Coupled: `test_result_coupled.pickle`

## Prepare Next Stage
A Human user reviews the testing results in the `./output`. Then copy the surrogate model `SM.pickle` to the next stage's `references` folder.