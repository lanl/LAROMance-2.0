# Training Module Example Datasets

This folder provides a handful of small, synthetic datasets that are **only**
intended for **hands‑on practice** with the **31_Surrogate_Training** GUI (step
3.1 of the LAROMance pipeline).  The data are deliberately lightweight and do
not correspond to any real‑world material model, so they cannot be used for
scientific testing or for exporting to MOOSE/BISON.

## What you will find

* A few pickled dictionaries (`*.pickle`) each containing data of different dimensions, with various inputs and outputs. 
* Most datasets have no physical meaning. They enable a user to explore different data dimensionalities and how to train and visualize the results.
* The 8D dataset are dummy creep runs already pre-processed for training (concatenated, not individual runs). 

## How to use them

1. Launch the training module:

   ```bash
   cd ./3_Surrogate_Modeling/31_Surrogate_Training
   python training_module.py
   ```

2. Click **Load Data** and point to any of the `*.pickle` files in this
   directory.
3. Play with the GUI: adjust hyper‑parameters, explore the visual diagnostics,
   and get a feel for the workflow.

These datasets are **not** meant for downstream steps such as
`02_Data_Processing`, `03_Surrogate_Modeling` testing, or `04_Port_Moose`.  They
serve only as a sandbox for learning the training interface.
