# 3 - Surrogate Modeling

The **3_Surrogate_Modeling** stage consists of two key steps that together build and test a surrogate model for the simulation data.

## 31_Surrogate_Training
Use a graphical user interface to visualize data, construct the grid-based surrogate, and then calibrate the nodal values across the data-set. The GUI tool allows for inspecting training results with error metrics.

### Training Methods

Stage 31 (`training_module.py`) offers three training back-ends:

- **Grid-based (LAROMance)** – a piecewise polynomial approach. Produces a mesh + nodal values that can be exported to MOOSE.
- **Neural PointNet (beta)** – uses a PointNet architecture. Currently for research only. No MOOSE FE-solver models in this format yet.
- **Neural MLP (beta)** – standard multi-layer perceptron. Currently for research/experimentation only. No MOOSE FE-solver models in this format yet.

Only the **Grid-based** method is compatible with the MOOSE export script (`4_Port_MOOSE/41_Export_Model/port_surrogate_to_moose.py`).

## 32_Surrogate_Testing
Test the surrogate model error in an accumulated fashion, and on unseen data during training, to test for overfitting.

**Execute first training, and then testing.**