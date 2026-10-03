# 4.2 - Analyze Benchmark Test (Port MOOSE)

This step is currently **manual**.  

After the surrogate has been exported to a MOOSE material (step 41) the user runs the verification simulations and then examines the results.

The utility functions in `utils/sm_analyze.py` can be used to visualise the
benchmark output.  A typical call looks like:

```python
import os
from utils import sm_analyze as sma

output_directory = "./output"
filepath = os.path.join(output_directory, "verification_cube_out.csv")

# Produce a 3‑D plot of the rate difference (example variable name)
sma.moose_verification(
    filepath=filepath,
    x='vmJ2',
    y='temperature',
    z='rhom_rate_diff'
)
```

Replace the column names (`vmJ2`, `temperature`, `rhom_rate_diff`) with the
appropriate fields from your CSV.  The function will generate the requested
visualisation in the `output` folder.

Further automation can be added later, but for now the analysis is performed
manually by the user using the `sm_analyze` helpers.

## Congratulations!
You have completed the workflow and should now have a verified surrogate. 

There is no constitutive model. Bend the code. -AR