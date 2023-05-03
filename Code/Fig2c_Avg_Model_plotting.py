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
mpl.use('Cairo')  # for saving SVGs that Affinity Designer can parse
import matplotlib.pyplot as plt
import seaborn as sns
import dill

from candas.utils import setup_paths
from candas.style import breve
from candas.learn import ParameterSet
import gumbi as gmb

base_pth, code_pth, data_pth, rslt_pth, fig_pth = setup_paths(make_missing=False)
plt.style.use(str(breve))

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

with open(rslt_pth / 'Model_Avg_predictions.pkl', 'rb') as f:
    predictions_dict = dill.load(f)

# +

width = 3.45
height = 3.31
figsize=(width, height)
spotsize=20
linewidth=2
ticklabelsize=8
labelsize=10
titlesize=labelsize+2

# Set rcParams for plotting
mpl.rc('xtick', labelsize=ticklabelsize)
mpl.rc('ytick', labelsize=ticklabelsize)
mpl.rc('axes', labelsize=labelsize, titlesize=titlesize, linewidth=1)

fig, axs = plt.subplots(2, 2, figsize=figsize, sharey='row', sharex='col')
r_ax = axs[1,0]
gc_ax = axs[0,0]
bp_ax = axs[1,1]
cax = axs[0,1]

r = predictions_dict['r']
BP = predictions_dict['BP']
GC = predictions_dict['GC']
reporter = 'HEX'

plt.sca(r_ax)
pp = gmb.ParrayPlotter(x=GC, y=BP, z=r, 
                    #    x_scale='standardized',
                       y_scale='standardized'
                       )

cmap=sns.color_palette('flare_r', as_cmap=True)
# rnorm = mpl.colors.Normalize()
# rnorm(r.μ)
rnorm = mpl.colors.Normalize(vmin=0.23, vmax=1.03)

cs = pp(plt.contourf, levels=np.arange(0.20, 1.05, 0.05), cmap='flare_r', norm=rnorm)
pp.colorbar(cs, ax=cax, fraction=0.55)

gc = (ds.wide
    #   .query('Reporter == @reporter')
      .GC
     )
bp = (ds.wide.z
    #   .query('Reporter == @reporter')
      .BP
     )
rs = (ds.wide
    #   .query('Reporter == @reporter')
      .r
     )

r_ax.scatter(gc, bp, 
            #  c='0.5', s=1, alpha=0.5,
             c=rs, edgecolor='0.8', linewidths=0.5, s=spotsize,
             cmap='flare_r', norm=rnorm)

r_ax.set_xlim([GC.values().min(), GC.values().max()])

cs = r_ax.contour(GC.values(), BP.z.values(), r.σ, 
                  levels=[0.05, 0.10, 0.15, 0.20, 0.25],
                  colors='0.2', linestyles='--', linewidths=1)
r_ax.clabel(cs, fontsize=ticklabelsize);

plt.sca(bp_ax)
for gc in [0.25, 0.5, 0.75]:
    x_pa, y_upa = predictions_dict[f'GC {gc:.2f}']
    gmb.ParrayPlotter(y_upa, x_pa.z).plot(ci=None)
    
    ci = 0.682

    palette = sns.cubehelix_palette()
    kwargs = dict(lw=2, facecolor=palette[1], zorder=-1, alpha=0.5)

    b = y_upa.dist.ppf((1 - ci) / 2)
    m = y_upa.dist.ppf(0.5)
    u = y_upa.dist.ppf((1 + ci) / 2)

    fill_between_styles = ["fill", "band"]
    errorbar_styles = ["errorbar", "bar"]
    plt.fill_betweenx(x_pa.z.values(), b, u, **kwargs)
    
    
plt.sca(gc_ax)
for bp in [30, 100, 300]:
    x_pa, y_upa = predictions_dict[f'BP {bp}']
    gmb.ParrayPlotter(x_pa, y_upa, ).plot(ci=0.682)
    
gc_ax.set_xlabel('')
bp_ax.set_ylabel('')

rlim = [0.4, 1.1]
gc_ax.set_ylim(rlim)
gc_ax.set_yticks(np.arange(*rlim, 0.2))
bp_ax.set_xlim(rlim)
bp_ax.set_xticks(np.arange(*rlim, 0.2))
axs[0,1].axis('off')

yticks = BP.parray(BP=[10, 20, 30, 50, 100, 200, 300, 500])
r_ax.set_yticks(yticks['BP'].z.values())
r_ax.set_yticklabels(map(int, yticks.values()))
r_ax.set_ylabel('Length (bp)')

xticks = GC.parray(GC=[0.25, 0.5, 0.75])
r_ax.set_xticks(xticks['GC'].values())
r_ax.set_xticklabels(map(int, 100*xticks.values()))
r_ax.set_xlabel('GC content (%)')

mar_l=0.6
mar_r=0.1
mar_t=0.22
mar_b=0.48

plt.subplots_adjust(
    left=mar_l / width,
    right=1 - mar_r / width,
    top=1 - mar_t / height,
    bottom=mar_b / height,
)

plt.savefig(fig_pth / 'Fig2c Avg r model.png', dpi=300, transparent=True);
plt.savefig(fig_pth / 'Fig2c Avg r model.svg', dpi=300, transparent=True);
