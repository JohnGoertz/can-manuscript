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
import candas as can
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Cairo')  # for saving SVGs that Affinity Designer can parse
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

base_pth, code_pth, data_pth, rslt_pth, fig_pth = can.utils.setup_paths(make_missing=False)
plt.style.use(str(can.style.breve))

# %config InlineBackend.figure_format = 'retina'
# -

JG045A = (can.test.QuantStudio(data_pth / 'JG045' / 'JG045A TMCC1 Gen3 EvaGreen.xlsx', 'JG045A')
        .import_data()
        .format_reactions()
        .index_reactions()
        .subtract_background(cycle_end=7)
        .get_derivatives()
        .annotate_reactions()
        .trim_reactions()
        )

# +
targets = [ 'BP50_GC25', 'BP50_GC75', 'BP280_GC30', 'BP280_GC70']
rates = [0.98, 0.67, 0.70, 0.49]
outliers = ['N8','P13','O14', 'G13', 'G14', 'A17', 'B17', 'C17', 'D17', 'E17', 'F17', 'G17']

reactions = JG045A.reactions.data.query('Target in @targets and WellPosition not in @outliers and Cycle<60')

width = 3.45
height = 1.65
figsize=(width, height)
spotsize=8**2
linewidth=2
ticklabelsize=8
labelsize=10
titlesize=labelsize+2

fig, axs = plt.subplots(2, 2, figsize = (width, height), sharex=True, sharey='row')

for target, rate, ax in zip(targets, rates, axs.flat):
    sns.lineplot(
        data=reactions.query('Target==@target'),
        x="Cycle",
        y="Fluorescence",
        units="Reaction",
        estimator=None,
        hue="lg10_Copies",
        palette="ch:0",
        legend=False,
        ax=ax,
        linewidth=1,
    )
    
    parts = target.split('_')
    bp = parts[0][2:]+' bp'
    gc = parts[1][2:]+'% GC'
    name = parts[2] if len(parts)>2 else ''
    title = f'{name}\n{bp}\n{gc}'
    ax.text(.975, .025, title, fontsize=ticklabelsize,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(.025, .95, f'r={rate}', fontsize=ticklabelsize,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    
for ax in axs.flat:
    ax.set_xlabel('Cycle', fontsize=labelsize)
    ax.set_xticks([0,20,40, 60])
    # ax.set_xlim([0, 41])
    # ax.set_ylim([-1.1,1.1])
    ax.set_ylabel('', fontsize=labelsize)
    ax.set_yticks(np.arange(5))
    # ax.set_yticklabels([1.0, 0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=ticklabelsize)
    plt.setp(ax.spines.values(), linewidth=1)

plt.setp(axs[0,:], ylim=[-0.1, 1.5])
plt.setp(axs[1,:], ylim=[-0.1, 4.1])

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

fig.text(0.07, 0.55, 'Fluorescence', fontsize=labelsize, ha='center', va='center', rotation='vertical')


plt.savefig(fig_pth / 'JG045A reaction rates.png', dpi=300, transparent=True);
plt.savefig(fig_pth / 'JG045A reaction rates.svg', dpi=300, transparent=True);


# -


