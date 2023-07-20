import pathlib as pl
import numpy as np
import pymc3 as pm
import arviz as az
import pickle
import sys

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

code_pth = pl.Path(__file__).parent.absolute()
base_pth = code_pth.parent
data_pth = base_pth / 'Data'
rslt_pth = base_pth / 'Results'

SEED=2020

tar_idx = int(sys.argv[1])-1

TMCC1_targets = [
    'BP88_GC43_ISO', 'BP88_GC43_WT', 'BP30_GC43', 'BP40_GC43',
    'BP55_GC43', 'BP160_GC43', 'BP200_GC43', 'BP240_GC43', 'BP88_GC15',
    'BP88_GC25', 'BP88_GC35', 'BP88_GC55', 'BP88_GC65', 'BP88_GC75',
    'BP88_GC80', 'BP88_GC85', 'BP160_GC10', 'BP160_GC20', 'BP160_GC60',
    'BP160_GC80', 'BP280_GC70', 'BP280_GC30', 'BP500_GC40',
    'BP50_GC25', 'BP50_GC60', 'BP50_GC75', 'BP500_GC60'
]

ARG1_targets = [
    'BP108_GC48_ISO', 'BP88_GC43', 'BP30_GC43', 'BP108_GC48_WT', 'BP108_GC25'
]

GBP6_targets = [
    'BP74_GC53_WT', 'BP74_GC53_ISO', 'BP88_GC43', 'BP30_GC53'
]

lens = np.cumsum(list(map(len,[TMCC1_targets,ARG1_targets,GBP6_targets])))-1

if tar_idx in range(lens[0]):
    gene = 'TMCC1'
    tar = TMCC1_targets[tar_idx]
elif tar_idx in range(lens[0],lens[1]):
    gene = 'ARG1'
    tar = ARG1_targets[tar_idx-lens[0]]
elif tar_idx in range(lens[1],lens[2]):
    gene = 'GBP6'
    tar = GBP6_targets[tar_idx-lens[1]]

with open(rslt_pth / gene / f'{tar}_model.pkl','rb') as buff:
    model = pickle.load(buff)
    
with model:
    try:
        trace = pm.sample(
            4000,
            target_accept=0.95,
            max_treedepth=14,
            random_seed=SEED,
        )
    except:
        trace = pm.sample(
            4000,
            init = 'adapt_diag',
            target_accept=0.95,
            max_treedepth=14,
            random_seed=SEED,
        )
        
    
    pm.save_trace(trace, directory=rslt_pth / gene / tar, overwrite=True)

    summary = az.summary(trace).sort_values('ess_mean')

summary.to_pickle(rslt_pth / gene / f'{tar}_summary.pkl', protocol=4)
    
