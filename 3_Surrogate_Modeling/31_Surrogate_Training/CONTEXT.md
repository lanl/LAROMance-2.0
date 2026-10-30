# 3.1 - Surrogate Training
This stage constructs and trains the LAROMAnce surrogate, using the curated (and optionally augmented) data set and a graphical user interface. The user can choose to store error and trace-line plots after training (for the grid-based approach).

## Surrogate Architecture
LAROMance-2.0 supports three surrogate modeling approaches. The **grid-based reduced-order model** discretizes the input parameter space into a structured mesh whose axes align with the model’s input variables. Degrees of freedom live at the mesh nodes and are interpolated using linear Lagrange shape functions, resulting in a highly sparse system that is physically intuitive and remains computationally efficient even in higher dimensions. For problems beyond two dimensions, a base mesh is constructed in the dominant axes and extruded via tensor-product assembly. In addition to the classical grid-based method, LAROMance-2.0 also provides two experimental neural-network surrogates: a **PointNet** architecture for unordered point sets and a standard **multi-layer perceptron (MLP)**. While the neural approaches are still under development, only the grid-based method currently supports direct export to MOOSE material models.

### Available Training Methods

The GUI provides three training options via the **Training Method** selector:

**Grid**: Finite-element mesh interpolation-based method that calibrates the output value of preconfigured nodal coordinates across inputs space, then interpolates between nodes. 
**MLP**: A standard feed-forward neural network that maps fixed-length input vectors to outputs through fully connected layers.
**PointNet**: A neural architecture designed for unordered point sets that uses shared MLPs and symmetric max-pooling to achieve permutation invariance.
**MoE** (Mixture of Experts): A model that splits work among several specialist MLP "experts", while a learned gating network decides, for each input, which experts to use and how to weight their outputs.  

> When a neural method is selected, additional hyperparameters (epochs, hidden size, batch size, learning rate) become available.

**Neural Network Hyperparameters:**

| Hyperparameter   | What it controls                          | Too Low                                      | Too High                                      | Rule of Thumb |
|------------------|-------------------------------------------|----------------------------------------------|-----------------------------------------------|---------------|
| **Hidden Size**  | Model capacity (number of neurons)        | Underfitting – model too simple to capture trends | Overfitting + slower training + more parameters | Start with 64–256. Increase if training error stays high. |
| **Batch Size**   | Number of samples per gradient update     | Noisy gradients, slower but sometimes better generalization | Smoother but potentially poorer generalization | 512–2048 is usually a good range for surrogate modeling. |
| **Learning Rate**| Step size of weight updates               | Very slow convergence                        | Divergence or unstable training               | Start at `1e-3`. Reduce to `5e-4` or `1e-4` if training is unstable. |
| **Epochs**       | Number of full passes over the training data | Underfitting (model hasn’t learned enough)   | Overfitting (model memorizes training data)   | 30–100 for most surrogate problems. Use early stopping if possible. |
| **Num Experts**       |Number of parallel expert MLPs the gating network can choose from (num_experts) | Limited specialization – each expert must cover a large part of the input space, so the model behaves like a single MLP   | More parameters and memory, some experts rarely get used (expert collapse / load imbalance), and the gate may overfit on small datasets   | 4–8 for small or medium surrogate problems. Increase only if the data has distinct regimes, and monitor expert utilization. |
| **Top k**       |Number of experts activated per input (sparse routing, top_k); the gate's outputs are renormalized over these k experts | k=1: cheapest and most specialized, but routing is hard and non-smooth, gradients to the gate are poor, and training can be unstable  | k close to num_experts: nearly dense MoE, so there is no compute savings and less specialization   | k=2 is the common default (with num_experts ≥ 4). Must satisfy 1 ≤ k ≤ num_experts. |

**Rules of Thumb**

- **Start simple**: Begin with `Hidden Size = 128`, `Batch Size = 1024`, `Learning Rate = 1e-3`, and `Epochs = 50`.
- **Hidden Size**: Increase only if the training loss plateaus at a high value. Doubling it significantly increases training time.
- **Batch Size**: Larger batches (≥ 1024) tend to work well for surrogate modeling because the loss landscape is relatively smooth. Very small batches (< 256) are rarely beneficial here.
- **Learning Rate**: If the loss oscillates or increases, reduce the learning rate by a factor of 2–5. If the loss decreases very slowly, try increasing it slightly.
- **Epochs**: Monitor the training loss. If it stops improving for many epochs, further training is usually not helpful (and can lead to overfitting).
- **Interaction**: These parameters are coupled. A larger hidden size often benefits from a slightly lower learning rate and more epochs.

A more visual User Manual for the GUI is located in [LAROMANCE_USER_MANUAL.pdf](3_Surrogate_Modeling/31_Surrogate_Training/doc/LAROMANCE_USER_MANUAL.pdf). Note this manual does not describe the more recently added neural network methods (Pointnet/MLP).

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

---

## Developer Notes on MLP Architecture Extension
The **MLP-based architectures** coded in this workflow are exemplary, experimental, and intentionally relatively simple in order to facilitate a range of use cases out-of-the-box. They offer an alternative to the more targeted and custom-coded **grid-based** surrogate architecture. Whereas the latter was originally designed for data with a unidirectional character (e.g., thermo-mechanical creep deformation under constant unidirectional loading), the MLP architectures may represent "spiky" data (many peaks and valleys that change direction quickly) more easily, yet are here only provided in an "exploratory", yet generic format. The MLP-code utilized standard functions from the PyTorch library, which makes extension and customization directly accessible to the user. 

The main code of the MLP models lives in the utility scripts `utils/mlp_utils.py` and `utils/mlp_models.py` and is based on PyTorch.

### Adding new per-point architectures
Any architecture that maps per-point input features `(N, n_inputs)` to outputs `(N, n_outputs)`, or batches of point sets `(B, N, n_inputs)`, can be added with two steps:

1. Implement a new `torch.nn.Module` in `utils/mlp_models.py`.
2. Register it in `MODEL_REGISTRY` so it can be selected through `get_model`.

No changes to `utils/mlp_utils.py` are needed for such models. Examples include residual MLPs, alternative activation functions, dropout variants, and other point-wise or set-based networks.

### Adding sequence-aware or graph-based architectures
The data used in the training step of this workflow is assumed to be a concatenation of multiple time series (preprocessed as such by stage 2_Data_Processing), and the chronological ordering of neighboring points can be reconstructed from the time array. Sequence-aware architectures (e.g., 1D CNNs/TCNs, recurrent networks, or transformers) and graph-based models (e.g., GNNs) can therefore also be explored. The flat, concatenated storage format does not need to change. Instead, the structure must be supplied to the model at batch time:

- **Series identification:** A series identifier (or series boundaries derived from the time array) is needed to group points into ordered series or windows. Points should be sorted by time within each series.
- **Dataset:** A new `Dataset` class (replacing `PointDataset`) should return ordered series or windows, e.g., tensors of shape `(L, n_inputs)` and `(L, n_outputs)`, rather than independent points.
- **Collate function and loss:** Variable-length series require padding, a mask, and a masked loss. Fixed-length windows avoid this. To make data arrays consistent between runs, consider the augmentation tools in stage `2_Data_Processing`.
- **Training loop:** Graph models typically require their own batching (e.g., PyTorch Geometric or DGL), additional inputs such as `edge_index`, and a modified training/evaluation step. They also introduce extra dependencies, such as `torch-geometric`.

Note that the existing `PointNetRegressor` uses global max-pooling, which is permutation-invariant and therefore does not exploit temporal ordering.

### Practical Considerations
- **Train/validation splits:** For sequence or graph models, split by series (not by individual rows) to avoid leakage between neighboring time steps.
- **Non-uniform time steps:** Standard convolutions assume uniform spacing. For adaptive time steps, make sure to provide `t` or `Δt` as an input feature if the time increments are not uniform.
- **Graph structure:** Time ordering alone yields only a chain graph. Richer graphs (e.g., spatial neighbors) require additional information such as data-point connectivity in other dimensions.
