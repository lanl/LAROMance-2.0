# 1 – Data Generation

In this stage, the user prepares the run folders and SLURM scripts needed to generate data using the particular model loaded in this stage **1 - Data Generation**. 

This stage consists of three steps:

- 11_Generative_Model_Summary : summarize the model and its applicability range and dependent/independent/coupled variables.
- 12_Design_of_Experiment: design of experiment that uses sampling methods such as Latin Hypercube Sampling to generate a .csv file with parametric variations
- 13_Batch_Template_Setup: This is a manual process the user should take care in setting up. A template run folder is created with the model files and placeholder value, together with the SLURM script to create a batch of simulation run folders on the HPC, using the template folder here created.

The simulation outputs should be collected in the .pickle file needed by step **2 - Data Processing**.

>*LAROMance does not ship with a high-fidelity data-generation model. The user must integrate their own model into this step, ensuring that the data is in a specific format, explained in `../2_Data_Processing/CONTEXT.md`. Published LAROMance models have used VPSC and EVP-FFT crystal plasticity codes to generate data. One example is the open-sourced Los Alamos code "LApx".*