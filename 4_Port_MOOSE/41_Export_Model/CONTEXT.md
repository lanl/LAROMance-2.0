# 4.1 - Export Model

## Purpose
This step generates the MOOSE C++ source (`.C` and `.h`) files for a surrogate
model that has been trained and stored as ``SM.pickle`` in the ``references``
directory. The script also creates an extrapolated mesh and a verification CSV
table.

## How to run the script
The script **port_surrogate_to_moose.py** can be executed from the command line:

```bash
cd ../4_Port_MOOSE/41_Export_Model/
python port_surrogate_to_moose.py [model_name]
```

**Note:** The Python script must be adapted to call the correct `write_MOOSE_*` function located in `../../utils/sm_publish.py`. The appropriate function to use is based on the number of inputs. There are versions for 4D, 5D, and 6D+ (tested up to 8D), which is clear from the function name.

* If *model_name* is supplied, it will be used as the base name for the generated
  MOOSE files (e.g., `MyModel.C` and `MyModel.h`).
* If the argument is omitted, the script will prompt you to enter a name:

```text
Enter a name for the exported MOOSE model:
```

The provided name is also used for the output directory inside `./output`.

## Expected inputs
* ``references/SM.pickle`` – the surrogate model dictionary.
* The script automatically converts any ROI lists to plain Python lists for
  compatibility.

## Generated outputs (placed in `./output`)
* ``SM.pickle`` – the extrapolated surrogate model.
* ``<model_name>.C`` – C++ source code for the MOOSE material.
* ``<model_name>.h`` – Header file for the MOOSE material.
* ``mesh_visualization.png`` – a 2‑D plot of the extrapolated mesh.
* ``benchmarks.csv`` – verification table generated with a generic number of
  input dimensions.

## Dependencies
The script relies on the utilities in ``utils/sm_publish.py`` and
``utils/sm_plot.py`` as well as standard libraries (`pickle`, `copy`, `os`,
`matplotlib`). Ensure the Python environment has these packages installed before
running the script.

## Verification Input Deck Creation
- Adapt the `./references/verification.i` file to correctly pull in the right columns from the `./output/benchmark.csv` file. 
- Compile in MOOSE
- Run the verification.i input deck with the `benchmarks.csv` file in the same folder:

```bash
~/project/bison/bison-opt -i verification.i
```