# 3.1 - Surrogate Training
This stage constructs and trains the LAROMAnce surrogate, using the curated (and optionally augmented) data set and a graphical user interface. The user can choose to store error and trace-line plots after training (for the grid-based approach).

## Surrogate Architecture
LAROMance-2.0 supports three surrogate modeling approaches. The **grid-based reduced-order model** discretizes the input parameter space into a structured mesh whose axes align with the model’s input variables. Degrees of freedom live at the mesh nodes and are interpolated using linear Lagrange shape functions, resulting in a highly sparse system that is physically intuitive and remains computationally efficient even in higher dimensions. For problems beyond two dimensions, a base mesh is constructed in the dominant axes and extruded via tensor-product assembly. In addition to the classical grid-based method, LAROMance-2.0 also provides two experimental neural-network surrogates: a **PointNet** architecture for unordered point sets and a standard **multi-layer perceptron (MLP)**. While the neural approaches are still under development, only the grid-based method currently supports direct export to MOOSE material models.

### Available Training Methods

The GUI provides three training options via the **Training Method** selector:

| Option                    | Internal Name | Use Case                              | MOOSE Compatible |
|---------------------------|---------------|---------------------------------------|------------------|
| Grid-based                | grid          | Production-grade surrogate models     | Yes              |
| Neural PointNet (beta)    | pointnet      | Experimental neural surrogate         | No               |
| Neural MLP (beta)         | mlp           | Experimental neural surrogate         | No               |

**MLP**: A standard feed-forward neural network that maps fixed-length input vectors to outputs through fully connected layers.
**PointNet**: A neural architecture designed for unordered point sets that uses shared MLPs and symmetric max-pooling to achieve permutation invariance.

> When a neural method is selected, additional hyperparameters (epochs, hidden size, batch size, learning rate) become available.

**Important:** The MOOSE export pipeline (`port_surrogate_to_moose.py`) only supports models trained with the **Grid-based** method. Neural models will be rejected with a clear error message.

A complete User Manual for the GUI is located in [LAROMANCE_USER_MANUAL.pdf](3_Surrogate_Modeling/31_Surrogate_Training/doc/LAROMANCE_USER_MANUAL.pdf). Note this manual does not describe the more recently added neural network methods (Pointnet/MLP).

## What the script does

1. Runs the GUI in `training_module.py`, which needs utility functions from the top-level `utils` folder.
2. The user manually constructs and then trains the model, using the GUI.
3. Calculates the relative error between the fit and the reference training data.
4. Stores the trained surrogate and the training results, including plots of error and time-histories in the `output` folder of this stage

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
