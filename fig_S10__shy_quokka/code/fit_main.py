import numpy as np
import pandas as pd
import pathlib as pl
import pickle
import scipy as sp
import logging

import candas as can
import gumbi as gmb
from candas.test import FluorescenceData, QuantStudio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
code_pth = pl.Path(__file__).parent
fig_pth = code_pth.parent
data_pth = fig_pth / 'data'
gen_pth = fig_pth / 'generated'
gen_pth.mkdir(exist_ok=True)

logger.info("Loading and processing data...")
# Load and process data
cmax = 50
JG075F = (
    QuantStudio(
        data_pth / "JG075F L-MMMMx blocker tripartite competition matrix.xlsx", "JG075F"
    )
    .import_data()
    .format_reactions()
    .index_reactions()
    .subtract_background()
    .normalize_reactions(cmax=cmax)
    .invert_fluorophore("FAM")
)

logger.info("Denoting reaction conditions...")
# Denote reaction conditions
JG075F.reactions.wide = (
    JG075F.reactions.wide.drop(columns=["Sample"]).merge(
        pd.read_csv(data_pth / "JG075F Plate Map.csv"), on="Well"
    )
)

JG075F.reactions.neaten()
JG075F.extract_endpoints(cmax=cmax)
endpoints = JG075F.endpoints

logger.info("Filtering and fitting GP model...")
# Filter and fit GP
nonblank_endpoints = endpoints.query("SNV_lg10_Copies > 0")
ds = gmb.DataSet(nonblank_endpoints, outputs=["SignalDifference"])
gp = gmb.GP(ds).fit(
    continuous_dims=["WT_lg10_Copies", "SNV_lg10_Copies", "Blocker μM"],
    linear_dims=["WT_lg10_Copies", "SNV_lg10_Copies", "Blocker μM"],
)

logger.info("Saving GP model and endpoints...")
# Save GP model
with open(gen_pth / 'gp_model.pkl', 'wb') as f:
    pickle.dump(gp, f)

# Save endpoints
endpoints.to_pickle(gen_pth / 'endpoints.pkl')
nonblank_endpoints.to_pickle(gen_pth / 'nonblank_endpoints.pkl')

logger.info("Calculating uncertainty parameters...")
# Calculate aleatoric uncertainty
σ_aleatoric = np.sqrt(gp.stdzr["SignalDifference"]["σ2"]) * gp.MAP["σ"]
min_detectable_diff = sp.stats.norm(scale=σ_aleatoric * np.sqrt(2), loc=0).isf(0.05)

# Save uncertainty parameters
uncertainty_params = {
    'σ_aleatoric': σ_aleatoric,
    'min_detectable_diff': min_detectable_diff
}
with open(gen_pth / 'uncertainty_params.pkl', 'wb') as f:
    pickle.dump(uncertainty_params, f)
    
# Save stdzr
with open(gen_pth / 'stdzr.pkl', 'wb') as f:
    pickle.dump(gp.stdzr, f)

logger.info("Fit complete!") 