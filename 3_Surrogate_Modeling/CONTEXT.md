# 3 - Surrogate Modeling

The **3_Surrogate_Modeling** stage consists of two key steps that together build and test a surrogate model for the simulation data.

## 31_Surrogate_Training
Use a graphical user interface to visualize data, construct the grid-based surrogate, and then calibrate the nodal values across the data-set. The GUI tool allows for inspecting training results with error metrics.

## 32_Surrogate_Testing
Test the surrogate model error in an accumulated fashion, and on unseen data during training, to test for overfitting.

**Execute first training, and then testing.**