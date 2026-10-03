## LAROMance 2.0 – Fast, Flexible, Surrogate Modeling Workflow for Nonlinear Constitutive Equations
*Release O# O5190  
Date: October 3rd 2026  
Developers: Andre Ruybalid, Laurent Capolungo  (Los Alamos National Laboratory | New Mexico, USA)  
Author: Andre Ruybalid | andreruybalid@gmail.com*

---

Welcome to the **LAROMance 2.0** codebase: an end‑to‑end, modular pipeline that **generates**, **curates**, and **cleans** high‑dimensional, nonlinear time‑series data, then builds **ultra‑fast surrogate models** to capture complex multi‑variable relationships. Although this workflow, which was developed at Los Alamos National Laboratory, is not restricted to any particular simulation framework, it was originally developed for creep modeling in **MOOSE** and **BISON**, supporting nuclear‑fuel‑system performance analysis [1]. By compressing heavyweight high‑fidelity simulations into quick surrogates, LAROMance delivers turnkey material‑behavior predictions for safety‑critical scenarios (nuclear accidents, aerospace loading, etc.) with **~1000× speed‑ups**. The repo ships with exemplary synthetic datasets, located in `./examples/`, and a ready‑to‑run training GUI.

[1] A.P. Ruybalid *et al.*, *Ann. Nucl. Energy* **2026**, [DOI:10.1016/j.anucene.2026.112681](https://doi.org/10.1016/j.anucene.2026.112681)  

> **Want the math & training details?** Check out `3_Surrogate_Modeling/31_Surrogate_Training/references/LAROMANCE_USER_MANUAL.html` for the full technical training module manual and its info sections.

*LAROMance: Los Alamos Reduced Order Model for Advanced Nonlinear Constitutive Equations.*

## Installation

The project is written in Python (≥ 3.10).  The easiest way to get a clean,
reproducible environment is to use a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/lanl/LAROMance-2.0.git
cd LAROMance-2.0

# 2. Create and activate a virtual environment (recommended)
conda create -n laromance python=3.10 -y   # you can choose any 3.10+ version
conda activate laromance                 # ← switch into the env

#  Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# 3. Install the core dependencies
pip install --upgrade pip
pip install \
    PyQt5 \
    numpy \
    matplotlib \
    scipy \
    pandas \
    pyDOE2 \
    tqdm \
    rtree \
    scikit-learn
```

## Stages and Folder Structure
The framework is split into four self‑contained stages—each in its own folder (`1_Data_Generation`, `2_Data_Processing`, …). 

Follow the numbered folder structure to build a surrogate from scratch. You will need a data generation model first in order to build your database (no generation model is here provided). Example data is located in the `./examples` folder.

Every stage ships a `CONTEXT.md` that details the stage's workflow, scripts, and usage, so you can jump in at any point and run a complete sub‑pipeline. The workflow follows a nexted, numbered folder structure, which form consecutive stages to complete the surrogate building process. The folder structure, with self-explanatory naming, and what each stage and sub-step does, is overviewed as follows:

```text
LAROMance-2.0
|
├─ 1_Data_Generation            – Generate data (e.g., High‑Fidelity Polycrystal Modeling)
│   ├─ 11_Generation_Model_Summary   – Read model scripts and input decks
│   ├─ 12_Design_of_Experiments       – Latin Hypercube Sampling (or your own) and configure the parameter space
│   └─ 13_Batch_Template_Setup       – Create the template folder, HPC‑bash injection script, and SLURM launch scripts ready for copy‑‑paste to the cluster
|
├─ 2_Data_Processing              – Curate, filter, augment, and visualise the data from step 1
|
├─ 3_Surrogate_Modeling            – Build, train, and test a fast surrogate model
│   ├─ 31_Surrogate_Training          – Use the GUI to train the model; produce training diagnostics/metrics
│   └─ 32_Surrogate_Testing           – Evaluate on (preferably unseen) test data; plot testing metrics
|
└─ 4_Port_Moose                    – Export the trained surrogate as a MOOSE material model

   *The utilities are modular; to retarget another FEM/physics solver, replace the template files and add a small wrapper (e.g., new helper functions or classes in `utils/sm_publish.py`) that writes the appropriate input files for the target code.
```

## How to Use
The process is modular, so the user can step into any stage, provided data from previous stage is already available.
1. Start with any stage and follow the `CONTEXT.md` file within each step's folder.
2. Dependencies (python scripts) are located in the `utils` folder.
3. The workflow is designed such that the input for the current stage is the output of the previous stage. 
   1. To stay structured, after every step, make sure to inspect output file saved in the current stage's `output` 
   2. and then **move** the output file from the current step's `output` file to the `references` folder of the next step. Like a product in an assembly line, the objects created during each stage move and transform along the pipeline, each stage's `output` becoming the input (`references`) for the next.
4. After review and moving of any output, proceed to the next stage or step.

## Examples

Below are brief pointers to the example folders that illustrate how to use each
stage of the pipeline.  Detailed usage instructions are provided in the
`README.md` files inside those folders.

* **Viscoplastic‑Creep data** – Located in [examples/Viscoplasicity_Creep](examples/Viscoplasicity_Creep).  The README explains how to load the 5,000‑simulation pickle into the 02_Data_Processing step and then run the Training GUI (31_Surrogate_Training) followed by testing (32_Surrogate_Testing) and porting the MOOSE (FEM framework). 
  > Tip: run the full pipeline on the creep example (5 k sims) to see everything in action
* **Training‑Module practice data** – Found in [examples/Training_Module](examples/Training_Module).  This folder contains small synthetic datasets meant solely for exploring the GUI of the training module (step 3.1).  See its README for more info.
* There is currently no dummy data generator model for stage **1-Data Generation** for testing or as example. 


## Limitations
This workflow is documented for the intended four-step path. Scripts and utilies are thin wrappers and can break if intputs, file layouts, or call order differ from what the CONTEXT.md files describe. It is not a hardened library: edge cases, unusual model design, and non-standard data shapes are not fully guarded. Treat this README.md and CONTEXT.md files within each stage as the supported interface, and adjust the utilities if your case falls outside that path. 

## BSD-3 License
This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.