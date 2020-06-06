# jPCA for Neural Data Analysis in Python
This is an implementation of the "jPCA" technique from "Neural Population Dynamics During Reaching" by Churchland, Cunningham et al (Nature 2012). It closely mirrors the original MATLAB codepack published by Prof. Churchland. More information about the JPCA technique, and the MATLAB implementation, can be found at: https://churchland.zuckermaninstitute.columbia.edu/content/code

### **Note: We can't make guarantees about the accuracy of this package, though our results match the MATLAB implementation for the datasets we have tested. For serious analyses, we recommend also using the original MATLAB implementation to verify results.**

# Installation
Clone this repo, and navigate to the directory, then try `pip install .` Feel free to reach out with installation issues.

# Useage
We've put a simple example notebook in the "example" folder. You will need to format your data as a list, where each entry of the list is a T x N array (T is the number of time-steps, N is the number of neurons or units). The `fit` function takes care of preprocessing like subtracting the cross-condition mean and performing regular PCA as a preprocessing step.

```python

import numpy as np
import jPCA
from jPCA.util import load_churchland_data, plot_projections

# Load publicly available data from Mark Churchland's group
path = "/Users/Bantin/Documents/Stanford/Linderman-Shenoy/jPCA_ForDistribution/exampleData.mat"
datas, times = load_churchland_data(path)

# Create a jPCA object
jpca = jPCA.JPCA(num_jpcs=2)

# Fit the jPCA object to data
(projected, 
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(datas, times=times, tstart=-50, tend=150)

# Plot the projected data
plot_projections(projected)
```
