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
#     display_name: manuscript_venv
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
# -

with open(rslt_pth / 'Model_Avg_predictions.pkl', 'rb') as f:
    avg_model_r = dill.load(f)['r']

# +
ps = ParameterSet.load(data_pth / 'ADVI_ParameterSets_220528.pkl')
def make_pair(row):
    return '-'.join(sorted([row.FPrimer, row.RPrimer]))

data = (ps.wide
        .query('Metric == "mean"')
        .astype({'BP': float})
        .assign(PrimerPair = lambda df: df.apply(make_pair, axis=1))
        .groupby(['Target', 'PrimerPair','Reporter'])
        .mean()
        .reset_index()
        )

ds = gmb.DataSet(
    data = data,
    outputs = ['F0_lg', 'r', 'K', 'm'],
    log_vars = ['BP', 'K', 'm', 'r'],
    logit_vars = ['GC'],
    )

selected = (data
 .groupby(['PrimerPair', 'Reporter'])
 .size()
 .reset_index()
 .rename(columns={0:'Observations'})
 .sort_values('Observations', ascending=False)
 .reset_index(drop=True)
).iloc[[0, 1, 4, 5, 6, 8, 38, 39, 42]]

# +
with open(rslt_pth / 'Model_LMC_predictions.pkl', 'rb') as f:
    predictions_dict = dill.load(f)
    
all_r = [predictions_dict[f'r{i}'] for i in range(9)]
BP = predictions_dict['BP']
GC = predictions_dict['GC']

limits = BP.parray(GC=[0.2, 0.8], BP=[10, 600])

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

fig, axs = plt.subplots(3, 3, figsize=figsize, sharey=True, sharex=True)

rnorm = mpl.colors.Normalize(vmin=0.23, vmax=1.03)

for i, (r, ax, row) in enumerate(zip(all_r, axs.flat, selected.itertuples())):
    
    plt.sca(ax)
    pp = gmb.ParrayPlotter(x=GC, y=BP, z=r, 
                        #    x_scale='standardized',
                           y_scale='standardized'
                           )

    step = 0.05
    levels=np.arange(np.floor(rnorm.vmin/step), np.ceil(rnorm.vmax/step)+1)*step
    cs = pp(plt.contourf, levels=levels, cmap='flare_r', norm=rnorm)
    # pp.colorbar(cs)

    yticks = BP.parray(BP=[10, 30, 100, 300])
    ax.set_yticks(yticks['BP'].z.values())
    ax.set_yticklabels(map(int, yticks.values()))

    xticks = GC.parray(GC=[0.25, 0.5, 0.75])
    ax.set_xticks(xticks['GC'].values())
    ax.set_xticklabels(map(int, 100*xticks.values()))
    
    # if i%3 != 0:
    #     ax.set_ylabel('')
    # if i//3 != 2:
    #     ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlabel('')

    gc = (ds.wide
          .query('Reporter == @row.Reporter and PrimerPair == @row.PrimerPair')
          .GC
         )
    bp = (ds.wide.z
          .query('Reporter == @row.Reporter and PrimerPair == @row.PrimerPair')
          .BP
         )
    rs = (ds.wide
          .query('Reporter == @row.Reporter and PrimerPair == @row.PrimerPair')
          .r
         )

    ax.scatter(gc, bp, 
            #    c='0.5', s=1,
               c=rs, edgecolor='0.8', linewidths=0.5, s=spotsize,
               cmap='flare_r', norm=rnorm)

    ax.set_xlim([0.2, 0.8])

    cs = ax.contour(GC.values(), BP.z.values(), r.σ, levels=[0.05, 0.10, 0.15, 0.20, 0.25],
                    colors='0.2', linestyles='--', linewidths=1)
    ax.clabel(cs, fontsize=ticklabelsize);
    
    
axs[1,0].set_ylabel('Length (bp)')
axs[-1,1].set_xlabel('GC content (%)')
    

mar_l=0.55
mar_r=0.1
mar_t=0.22
mar_b=0.48

plt.subplots_adjust(
    left=mar_l / width,
    right=1 - mar_r / width,
    top=1 - mar_t / height,
    bottom=mar_b / height,
)

plt.savefig(fig_pth / 'Fig2d LMC r model.png', dpi=300, transparent=True);
plt.savefig(fig_pth / 'Fig2d LMC r model.svg', dpi=300, transparent=True);


# +
stdzr = ds.stdzr

BP_vec = BP[0,:]
GC_vec = GC[:,0]
BP_idx = 1
GC_idx = 0
BP_ticks = gmb.parray(BP=[10, 30, 100, 300], stdzr=stdzr)
GC_ticks = gmb.parray(GC=[0.25, 0.5, 0.75], stdzr=stdzr)

rlim = rmin, rmax= 0.23, 1.03
r_ticks = gmb.parray(r=[0.25, 0.5, 0.75, 1.0], stdzr=stdzr)

# +
figsize = width, height = 7.083, 3
linewidth=0.5
ticklabelsize=6
labelsize=8
titlesize=labelsize+2

# Set rcParams for plotting
mpl.rc('xtick', labelsize=ticklabelsize)
mpl.rc('ytick', labelsize=ticklabelsize)
mpl.rc('axes', labelsize=labelsize, titlesize=titlesize, linewidth=linewidth)

# rnorm = mpl.colors.Normalize()
# rnorm(np.stack([r.μ for r in all_r]));
rnorm = mpl.colors.Normalize(vmin=0.23, vmax=1.03)

fig, axs = plt.subplots(3, 10, figsize=figsize)

for i, (r, row, selection) in enumerate(zip(all_r, axs.T, selected.itertuples())):
    ax = row[0]
    plt.sca(ax)
    pp = gmb.ParrayPlotter(x=GC, y=BP, z=r, 
                        #    x_scale='standardized',
                           y_scale='standardized'
                           )

    step = 0.05
    levels=np.arange(np.floor(rnorm.vmin/step), np.ceil(rnorm.vmax/step)+1)*step
    cs = pp(plt.contourf, levels=levels, cmap='flare_r', norm=rnorm)

    ax.set_yticks(BP_ticks.z.values())
    ax.set_xticks(GC_ticks.values())
    ax.set_xticklabels(map(int, 100*GC_ticks.values()))

    gc = (ds.wide
          .query('Reporter == @selection.Reporter and PrimerPair == @selection.PrimerPair')
          .GC
         )
    bp = (ds.wide.z
          .query('Reporter == @selection.Reporter and PrimerPair == @selection.PrimerPair')
          .BP
         )
    rs = (ds.wide
          .query('Reporter == @selection.Reporter and PrimerPair == @selection.PrimerPair')
          .r
         )

    # ax.plot(gc, bp, ls='none', marker='.', mfc='0.5', mec='none', ms=5) #, cmap='flare_r', norm=rnorm)
    ax.scatter(gc, bp, c=rs, edgecolor='0.8', linewidths=0.5, s=10, cmap='flare_r', norm=rnorm)

    ax.set_xlim(limits['GC'].values())

    cs = ax.contour(GC.values(), BP.z.values(), r.σ, levels=[0.05, 0.10, 0.15, 0.20, 0.25], colors='0.2', linestyles='--', linewidths=0.5)
    ax.clabel(cs, fontsize=ticklabelsize-2);
    
    ax = row[1]
    plt.sca(ax)
    marginal = r.mean(axis=BP_idx)
    ax = gmb.ParrayPlotter(GC_vec, marginal).plot(line_kws={'lw': 0.5})
    ax.scatter(gc, rs, c=rs, edgecolor='0.8', linewidths=0.5, s=10, cmap='flare_r', norm=rnorm, zorder=-10)
    gmb.ParrayPlotter(GC_vec, avg_model_r.mean(axis=1).μ).plot(line_kws={'lw': 0.5, 'ls':'--', 'color':'0.4', 'zorder':-20})
    ax.set_xlim(limits['GC'].values())
    ax.set_ylim(rlim)
    ax.set_yticks(r_ticks.values())
    ax.set_xticks(GC_ticks.values())
    ax.set_xticklabels(map(int, 100*GC_ticks.values()))
    
    ax = row[2]
    plt.sca(ax)
    marginal = r.mean(axis=GC_idx)
    ax = gmb.ParrayPlotter(BP_vec.z, marginal).plot(line_kws={'lw': 0.5})
    ax.scatter(bp, rs, c=rs, edgecolor='0.8', linewidths=0.5, s=10, cmap='flare_r', norm=rnorm, zorder=-10)
    gmb.ParrayPlotter(BP_vec.z, avg_model_r.mean(axis=0).μ).plot(line_kws={'lw': 0.5, 'ls':'--', 'color':'0.4', 'zorder':-20})
    ax.set_xlim(limits['BP'].z.values())
    ax.set_ylim(rlim)
    ax.set_yticks(r_ticks.values())
    ax.set_xticks(BP_ticks.z.values())
    ax.set_xticklabels(map(int, BP_ticks.values()))

for ax in axs.flat:
    ax.tick_params(axis='both', which='both', length=2, width=0.5)
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

axs[0,0].set_ylabel('Length (bp)', labelpad=1)
axs[0,0].set_xlabel('GC content (%)', labelpad=1)
axs[0,0].set_yticklabels(map(int, BP_ticks.values()))

axs[1,0].set_ylabel('Rate', labelpad=1)
axs[1,0].set_yticklabels(map('{:.2f}'.format, r_ticks.values()))
axs[1,0].set_xlabel('GC content (%)', labelpad=1)

axs[2,0].set_ylabel('Rate', labelpad=1)
axs[2,0].set_yticklabels(map('{:.2f}'.format, r_ticks.values()))
axs[2,0].set_xlabel('Length (bp)', labelpad=1);

"""Add fake colorbar"""

y = np.linspace(*rlim, 100)
x = np.zeros_like(y)
cax = axs[0,-1]
cax.imshow(y[::-1,None], cmap='flare_r', norm=rnorm, aspect=15, extent=[0,1,rlim[0],rlim[1]])
cax.yaxis.tick_right()
cax.yaxis.set_label_position("right")
cax.set_yticks(r_ticks.values())
cax.set_yticklabels(map('{:.2f}'.format, r_ticks.values()))
cax.set_ylabel('Rate')
cax.set_xticks([]);
        
mar_l=0.4
mar_r=-0.15
mar_t=0.05
mar_b=0.35

plt.subplots_adjust(
    hspace=1,
    left=mar_l / width,
    right=1 - mar_r / width,
    top=1 - mar_t / height,
    bottom=mar_b / height,
)

pos = cax.get_position()
x0 = axs[1,-1].get_position().x0
cax.set_position((x0, pos.y0, pos.width, pos.height))
for ax in axs[1:,-1]:
    ax.remove()
    
plt.savefig(fig_pth / 'SFigX lmc model surfaces and marginals.svg', dpi=300, transparent=True);
