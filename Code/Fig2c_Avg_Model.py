# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.6 ('candas')
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Cairo')  # for saving SVGs that Affinity Designer can parse
import matplotlib.pyplot as plt
import seaborn as sns
import dill

import candas as can
import gumbi as gmb
from candas.lims import Librarian, library
from candas.learn import ParameterSet

base_pth, code_pth, data_pth, rslt_pth, fig_pth = can.utils.setup_paths(make_missing=False)
plt.style.use(str(can.style.breve))

# %config InlineBackend.figure_format = 'retina'

# +
ps = ParameterSet.load(data_pth / 'ADVI_ParameterSets_220528.pkl')
data = (ps.wide
        .query('Metric == "mean"')
        .astype({'BP': float})
        .groupby(['Target'])
        .mean()
        .reset_index()
        )

ds = gmb.DataSet(
    data = data,
    outputs = ['F0_lg', 'r', 'K', 'm'],
    log_vars = ['BP', 'K', 'm', 'r'],
    logit_vars = ['GC'],
    )
# -

# # Average

# +
recalculate = True

if recalculate:
    gp = gmb.GP(ds).fit(
        outputs='r',
        continuous_dims=['BP', 'GC'],
        # categorical_dims=['Reporter'],
        )

    with open(rslt_pth / 'Model_Avg.pkl', 'wb') as f:
        dill.dump(gp, f)
else:
    with open(rslt_pth / 'Model_Avg.pkl', 'rb') as f:
        gp = dill.load(f)
# -

predictions_dict = {}

# +
output = 'r'
reporter = 'HEX'

limits = gp.parray(GC=[0.2, 0.8], BP=[10, 600])
XY = gp.prepare_grid(limits=limits)
r = gp.predict_grid(
    output=output,
    with_noise=False,
    # categorical_levels={'Reporter': reporter}
)
BP, GC = XY['BP'], XY['GC']

predictions_dict['r'] = r
predictions_dict['BP'] = BP
predictions_dict['GC'] = GC

pp = gmb.ParrayPlotter(x=GC, y=BP, z=r, 
                    #    x_scale='standardized',
                       y_scale='standardized'
                       )

cmap=sns.color_palette('flare_r', as_cmap=True)
# rnorm = mpl.colors.Normalize()
# rnorm(r.μ)
rnorm = mpl.colors.Normalize(vmin=0.23, vmax=1.03)

cs = pp(plt.contourf, levels=np.arange(0.20, 1.05, 0.05), cmap='flare_r', norm=rnorm)
pp.colorbar(cs)

ax=plt.gca()

yticks = gp.parray(BP=[20, 30, 50, 100, 200, 300, 500])
ax.set_yticks(yticks['BP'].z.values())
ax.set_yticklabels(yticks.values())

gc = (ds.wide
      # .query('Reporter == @reporter')
      .GC
     )
bp = (ds.wide.z
      # .query('Reporter == @reporter')
      .BP
     )

ax.scatter(gc, bp, c='0.5', cmap='flare_r', norm=rnorm, s=1)

ax.set_xlim(limits['GC'].values())

cs = ax.contour(GC.values(), BP.z.values(), r.σ, levels=[0.05, 0.10, 0.15, 0.20, 0.25], colors='0.2', linestyles='--')
ax.clabel(cs);
# -

for gc in [0.25, 0.5, 0.75]:
    x_pa, y_upa = gp.get_conditional_prediction(GC=gc)
    gmb.ParrayPlotter(x_pa.z, y_upa).plot()
    predictions_dict[f'GC {gc:.2f}'] = [x_pa, y_upa]

for bp in [30, 100, 300]:
    x_pa, y_upa = gp.get_conditional_prediction(BP=bp)
    gmb.ParrayPlotter(x_pa.z, y_upa).plot()
    predictions_dict[f'BP {bp}'] = [x_pa, y_upa]

# +
## Dill needs to "warm up" for some reason??

for value in predictions_dict.values():
    dump = dill.dumps(value)
    loaded = dill.loads(dump)

# +
with open(rslt_pth / 'Model_Avg_predictions.pkl', 'wb') as f:
    dill.dump(predictions_dict, f)
    
with open(rslt_pth / 'Model_Avg_predictions.pkl', 'rb') as f:
    loaded = dill.load(f)
    
loaded
# -


