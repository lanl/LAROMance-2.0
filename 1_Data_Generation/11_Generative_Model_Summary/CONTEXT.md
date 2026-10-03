# 1.1 – Generative Model Summary
To an intelligent user (artificial/human):

Load the `referencs` folder with the a model description e.g., a .pdf-file, and/or model files of the data-generating model the surrogate will be calibrated on. This may include source files (e.g. in a `src` sub-folder) as well as input files (`input` subfolder) or documentation about the 

Use the model materials inside the `reference` folder to make a summary of the model, as follows:
1. first find a .pdf or other textual document that describes the model as well as model srouce code or model input files (in folders named `src` and/or `input`).
2. summarize the outputs the model is intended to predict as output, as well as the inputs on which the output depends. 
3. summarize the applicable ranges for the input variables if this is found in the documentation or the associated input/model files
4. summarize the data used to calibrate the model
5. create a file called `output/model_summary.md` to list the above points

Do not proceed to the next step without human instruction.
