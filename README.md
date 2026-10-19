## LAROMance 2.0 – Fast, Flexible, Surrogate Modeling Workflow for Nonlinear Time-Series Data
*"LAROMance": Los Alamos Reduced Order Model for Advanced Nonlinear Constitutive Equations.*

*Release O# O5190  
Date: October 3rd 2026  
Developers: Andre Ruybalid, Laurent Capolungo  (Los Alamos National Laboratory | New Mexico, USA)  
Author: Andre Ruybalid | andreruybalid@gmail.com*

---

Welcome to the **LAROMance 2.0** codebase: an end‑to‑end, physics-AI pipeline that **generates**, **cleans** and **curates** high‑dimensional, nonlinear time‑series data, e.g., thousands of tensile‑simulation stress‑strain curves (time-histories) generated from varying input/model parameters, then builds **ultra‑fast surrogate models** to capture complex multi‑variable relationships. Although this workflow (developed at Los Alamos National Laboratory) is not restricted to any particular simulation framework, it was originally designed for creep modeling in **MOOSE** and **BISON**, supporting nuclear‑fuel‑system performance analysis [1]. By compressing heavyweight high‑fidelity simulations into quick surrogates, LAROMance delivers turnkey material‑behavior predictions for safety‑critical scenarios (nuclear accidents, aerospace loading, etc.) with **~1000× speed‑ups**. The repo ships with exemplary synthetic datasets, located in `./examples/`, and a ready‑to‑run training GUI.

[1] A.P. Ruybalid *et al.*, *Ann. Nucl. Energy* **2026**, [DOI:10.1016/j.anucene.2026.112681](https://doi.org/10.1016/j.anucene.2026.112681)  

 > **Want the math & training details?** Check out [LAROMANCE_USER_MANUAL.pdf](3_Surrogate_Modeling/31_Surrogate_Training/doc/LAROMANCE_USER_MANUAL.pdf) for the full technical training module manual and its info sections.

*This repo's "main" branch is stable. The "devel" branch is for developers and may break in-between commits. Users are advised to work with "main". Fearure branches are for new experimental features and are not expected to behave well.* 

## Forking the Repository
If you plan to **develop, extend, or adapt** this pipeline, the recommended way to work is from your **own fork** rather than cloning this repository directly. A fork is your personal copy of the project on GitHub that you can freely push to, while still staying linked to this original repository (the *upstream*) so you can pull in future updates.

1. **Create your fork.** Go to the repository page at [https://github.com/lanl/LAROMance-2.0](https://github.com/lanl/LAROMance-2.0) and click the **Fork** button (top‑right). GitHub creates a copy under your own account, e.g. `https://github.com/<your-username>/LAROMance-2.0`.

2. **Clone *your* fork** to your computer (note it uses *your* username, not `lanl`):
   ```bash
   git clone https://github.com/<your-username>/LAROMance-2.0.git
   cd LAROMance-2.0
   ```
   Cloning automatically sets up a remote named `origin` that points to **your fork** — this is where you will push your own changes.

3. **Link the original repository as `upstream`** so you can fetch new updates from it:
   ```bash
   git remote add upstream https://github.com/lanl/LAROMance-2.0.git
   # Verify both remotes are set correctly:
   git remote -v
   #   origin    https://github.com/<your-username>/LAROMance-2.0.git  (your fork)
   #   upstream  https://github.com/lanl/LAROMance-2.0.git             (original)
   ```

4. **Day-to-day use.** Push your work to your fork, and pull in upstream updates when you want the latest changes:
   ```bash
   # Push your commits to your own fork:
   git push origin main

   # Fetch and merge the latest updates from the original repository:
   git fetch upstream
   git merge upstream/main
   ```

> With this setup, `origin` is *yours* (push freely) and `upstream` is the original project (pull updates only). You can also open a Pull Request from your fork on GitHub if you'd like to contribute changes back.

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
    PySide6 \
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

Every stage ships a `CONTEXT.md` that details the stage's workflow, scripts, and usage, so you can jump in at any point and run a complete sub‑pipeline. The workflow follows a nested, numbered folder structure that forms the consecutive stages to complete the surrogate building process. The folder structure, with self-explanatory naming, and what each stage and sub-step does, is overviewed as follows:

```text
LAROMance-2.0
|
├─ 0_Demo_Test                     – Self-contained demo/smoke test that runs the whole pipeline (stages 1→5) headlessly against the shipped Viscoplasticity/Creep example data. Run `0_Demo_Test/run_test_demo.py --demo` for a guided tour or `--test` for a pass/fail check.
|
├─ 1_Data_Generation               – Generate data (e.g., High‑Fidelity Polycrystal Modeling)
│   ├─ 11_Generation_Model_Summary – Read model scripts and input decks
│   ├─ 12_Design_of_Experiments    – Latin Hypercube Sampling (or your own) and configure the parameter space
│   └─ 13_Batch_Template_Setup     – Create the template folder, HPC‑bash injection script, and SLURM launch scripts ready for copy‑‑paste to the cluster
|
├─ 2_Data_Processing               – Curate, filter, augment, and visualise the data from step 1
|
├─ 3_Surrogate_Modeling            – Build, train, and test a fast surrogate model
│   ├─ 31_Surrogate_Training          – Use the GUI to train the model; produce training diagnostics/metrics
│   └─ 32_Surrogate_Testing           – Evaluate on (preferably unseen) test data; plot testing metrics
|
└─ 4_Port_Moose                    – Export the trained surrogate as a MOOSE material model
│   ├─ 41_Export_Model             – Uses a MOOSE material file template to fill with the surrogate parameters, and generates a benchmark file with inputs/outputs to test the eventual MOOSE model outputs against.
│   └─ 42_Analyze_Benchmark_Test   – Manual step: run the MOOSE verification test with the benchmark file (outside of this workflow) and then analyze/plot some results. The difference between benchmark and MOOSE output should approach machine-precision.
|
└─5_Clean_Pipeline                 – Clean generated artefacts from previous stages (remove `output` and `references` directories, keep documentation, utils and example data). Run `5_Clean_Pipeline/clean_pipeline.sh` after completing stage 4.2 or when restarting the workflow from an archived copy.
 
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
5. Step 5_Clean_Pipeline is optional, but convenient for cleaning all generated files in the pipeline (in the `output` and `references` folders), including any data `.pickle`.
   
>Pro-Tip: Keep one copy of the entire tree per material model, started empty and archived once filled, and do not mix files from different models in one pipeline. Use a fresh (empty) copy of the working tree for each new surrogate model. That way, the folder chain becomes the record of the model build. Use step 5_Clean_Pipeline to clean your pipeline for a fresh restart.

## Quick Start
New to LAROMance? Run the whole pipeline end-to-end against the shipped
Viscoplasticity/Creep example with a single command:

```bash
cd 0_Demo_Test
python run_test_demo.py --demo
```

This guided demo walks you through every stage (1 → 5), narrating the *why*
behind each step and keeping all generated artifacts so you can explore them
afterwards. For a quick, non-interactive health check use
`python run_test_demo.py --test`. See [0_Demo_Test/CONTEXT.md](0_Demo_Test/CONTEXT.md)
for details.

## Training Methods

The workflow and the training module (`training_module.py`) currently supports three different surrogate training approaches:

| Method                    | Description                                      | MOOSE Export |
|---------------------------|--------------------------------------------------|--------------|
| **Grid-based**            | Classical piecewise-polynomial LAROMance method  | ✅ Supported |
| **Neural PointNet (beta)**| PointNet architecture for point-cloud style data | ❌ Not supported |
| **Neural MLP (beta)**     | Standard multi-layer perceptron                  | ❌ Not supported |

**MLP**: A standard feed-forward neural network that maps fixed-length input vectors to outputs through fully connected layers.  
**PointNet**: A neural architecture designed for unordered point sets that uses shared MLPs and symmetric max-pooling to achieve permutation invariance.

> **Note:** Only the **Grid-based** method currently supports export to MOOSE C++ code via `port_surrogate_to_moose.py`. The two neural-network options are experimental.


## Examples
Below are brief pointers to the example folders that illustrate how to use each
stage of the pipeline.  Detailed usage instructions are provided in the
`README.md` files inside those folders.

* **Viscoplastic‑Creep data** – Located in [examples/Viscoplasicity_Creep](examples/Viscoplasicity_Creep). The README explains how to load the 5,000‑strong creep simulation data `.pickle` into the `2_Data_Processing` step and then open the Training Module in `31_Surrogate_Training`, test its performance in `32_Surrogate_Testing`, and finally, create the MOOSE material model files (__MOOSE__ is a multi-physics FEM-framework widely used by the nuclear engineerin community). 
  > Tip: run the entire pipeline in 0_Demo_Test. Follow the CONTEXT.md to run the python script through the entire pipeline, using this example dataset.
* **Training‑Module practice data** – Found in [examples/Training_Module](examples/Training_Module).  This folder contains small synthetic datasets meant solely for exploring the GUI of the Training Modulein `31_Surrogate_Training`.  See its README for more info.
* There is currently no dummy data generator model for stage **1-Data Generation** for testing or as example. 

## Limitations
This workflow is documented for the intended four-step path, and originally for thermomechanical creep data. Scripts and utilies are thin wrappers and can break if intputs, file layouts, or call order differ from what the CONTEXT.md files describe. It is not a hardened library: edge cases, unusual model design, and non-standard data shapes are not fully guarded. Treat this README.md and CONTEXT.md files within each stage as the supported interface, and adjust the utilities if your case falls outside that path. 

## Contributing & Support
If this workflow is useful to you, a **GitHub star** on the repository is genuinely appreciated and helps others discover the project. Please also **reference this repository** in any work that builds on it.

**Forks are welcome** for those who wish to develop, extend, or adapt the pipeline to their own material models. If you have **ideas, suggestions, or questions**, they are welcome via the email address provided at the top of this README (Andre Ruybalid | andreruybalid@gmail.com).

##  Software Notice
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

## BSD-3 License
This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.