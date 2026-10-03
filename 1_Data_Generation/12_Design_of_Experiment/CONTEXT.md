# 1.2 - Design of Experiment

This step sets up the design of experiment. The input is provided/configured by the user, and the output is a `samples.csv` file that will contain the sampled values for all input variations to the model generator.

## What the script does
1. Run the script `doe.py` that calls the script `utils/generate_LHS_set_modular.py` to set up the LHS based sampling
2. Use the user's input, either directly through parsed arguments to the `doe.py` call, or if not provided, by prompting the user for the values
3. Generate the `output/samples.csv` file using the user input labels in the header.

## How to run

The script can be used **both** in a fully‑automated mode (providing all
parameters on the command line) **or** interactively (letting the script prompt
for the required values).

### 1️⃣ Automated call – all arguments on the command line

```bash
python doe.py TEMP 600 1023 VMJ2 0 250 RHOC 1e11 1e13 RHOW 1e11 1e13 \
    --number 3000 \
    --log10 RHOC RHOW \
    --visualize \
    --axes TEMP RHOC RHOW
```

* **Triples** ``<label> <min> <max>`` are supplied in the order they should appear in the CSV header.
* ``--number`` (optional) controls the number of Latin‑Hypercube samples – the default is ``1000``.
* ``--log10`` (optional) is a space‑separated list of labels whose bounds should be sampled in logarithmic (base‑10) space. The script converts the sampled values back to linear space before writing ``samples.csv``.
* ``--visualize`` (optional) will open a 3‑D scatter plot of the generated sample set.
* ``--axes`` (optional, requires ``--visualize``) lets you choose which three parameters are shown in the plot, e.g. ``TEMP RHOC RHOW``. If omitted, the first three parameters are used.

### 2️⃣ Interactive mode – no command‑line arguments

```bash
python doe.py
```

Running the script without positional triples will cause it to prompt you for each ``label min max`` set. After entering all parameters, you will be asked to provide (or skip) a space‑separated list of labels to sample in log10 space. You will also be asked whether to visualise the result and, if so, which three axes to display.

Both approaches generate the file ``output/samples.csv`` inside this step’s directory, with a header containing the provided labels.