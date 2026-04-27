import numpy as np
import pandas as pd
import pathlib as pl
import pickle

import candas as can
import gumbi as gmb
from candas.test import FluorescenceData, QuantStudio

# Setup paths
code_pth = pl.Path(__file__).parent
fig_pth = code_pth.parent
data_pth = fig_pth / 'data'
gen_pth = code_pth / 'generated'
gen_pth.mkdir(exist_ok=True)

# Load endpoints
endpoints = pd.read_pickle(gen_pth / 'endpoints.pkl')

# Filter and fit GP for zero copies
zero_endpoints = endpoints.query("SNV_lg10_Copies == -2")
ds_zero = gmb.DataSet(zero_endpoints, outputs=["SignalDifference"])
gp_zero = gmb.GP(ds_zero).fit(
    continuous_dims=["WT_lg10_Copies", "Blocker μM"],
    linear_dims=["WT_lg10_Copies", "Blocker μM"],
)

# Save zero-copy GP model
with open(gen_pth / 'gp_zero_model.pkl', 'wb') as f:
    pickle.dump(gp_zero, f)

# Save zero endpoints
zero_endpoints.to_pickle(gen_pth / 'zero_endpoints.pkl') 