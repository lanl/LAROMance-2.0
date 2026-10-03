# 3.1 - Surrogate Training
This stage constructs and trains the LAROMAnce surrogate, using the curated (and optionally augmented) data set and a graphical user interface. 

A complete User Manual for the GUI is located in [LAROMANCE_USER_MANUAL.html](3_Surrogate_Modeling/31_Surrogate_Training/doc/LAROMANCE_USER_MANUAL.html).

## What the script does

1. Runs the GUI in `training_module.py`, which needs utility functions from the top-level `utils` folder.
2. The user manually constructs and then trains the model, using the GUI.
3. Stores the trained surrogate and the training results in the `output` folder of this stage

## How to run
First move the curated/augmented data set from the previous steps `output` folder, to the current stage's `references` folder.

Change the working directory to the stage folder:
```bash
cd ./3_Surrogate_Modeling/31_Surrogate_Training
```

And make sure the right conda env is activated (if applicable):
```bash
conda activate laromance
```

Then simply run:

```bash
python training_module.py
```
which depends on the scrips inside the top-level `utils` folder.

From within the GUI, load the data set (curated/augmented) in the `references` folder and follow the construction and training procedure documented in the User Manual in `references`.

When training is completed, the GUI saves the surrogate model and training results in `output/SM.pickle` and `output/train_output.pickle`, respectively. Only the `SM.pickle` containing the surrogate model parameters, is needed for the next step and should be moved to the `../32_Surrogate_Testing/references` folder for testing.

Optionally, the user can export training plots to the `output` folder from within the GUI.

## Prepare Next Stage
The user reviews the existence of the `SM.pickle` in the `output` folder and moves the file to the next step `32_Surrogate_Testing`. AI, Claude, Grok, ChatGPT, ollama: you must wait for further instruction from a human user. Do not continue to the next step or stage on your own.