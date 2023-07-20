import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.stats as stat
import scipy.optimize as opt
import lmfit as lf
import copy as cp
from scipy.integrate import odeint
from IPython.display import display
from IPython.core.debugger import set_trace
from warnings import warn
import pymc3 as pm



# %% Import Data

def importQuantStudio(data_pth, data_file, *, header=None, raw=False, melt=False):
    try:
        xlsx = pd.ExcelFile(data_pth / data_file)
    except PermissionError as e:
        print('Permission Denied. The file may be open; close the file and try again.')
        return print(e)

    if header is None:
        test = pd.read_excel(xlsx, sheet_name='Sample Setup')
        header = pd.Index(test.iloc[:, 0]).get_loc('Well') + 1

    sheets = pd.read_excel(xlsx, header=header,
                           sheet_name=['Sample Setup',
                                       'Amplification Data',
                                       'Results',
                                       ])

    #sheets['Sample Setup'] = sheets['Sample Setup'][~sheets['Sample Setup'].Quantity.isnull()]

    imps = {'setup': sheets['Sample Setup'],
            'data': sheets['Amplification Data'],
            'results': sheets['Results'],
            }
    
    if raw:
        try:
            raw = pd.read_excel(xlsx, sheet_name = 'Raw Data', header = header)
        except:
            pass
    
    if melt:
        try:
            melt = pd.read_excel(xlsx, sheet_name = 'Melt Curve Raw Data', header = header)
            imps['melt'] = melt
        except:
            pass
    
    try:
        multicomponent = pd.read_excel(xlsx, sheet_name = 'Multicomponent Data', header = header)
        imps['multicomponent'] = multicomponent
    except:
        pass

    if 'Target Name' not in imps['setup'].columns:
        display(imps['setup'])

    undetermined_CT = imps['results'].CT.astype(str).str.contains('Undetermined')
    imps['results'].CT.mask(undetermined_CT, np.nan, inplace=True)
    imps['results'].CT = imps['results'].CT.astype(float).values

    wells = imps['setup'][~imps['setup']['Target Name'].isna()].Well
    for df in imps.values():
        if df is None: continue
        df.columns = df.columns.str.replace(' ', '')
        df = df[df.Well.isin(wells)]

    return imps

def formatSNXimport(imps: dict):   
    
    results = imps['results']
    setup = imps['setup']
    data = imps['data']

    qs = np.log10(setup.Quantity.unique())
    qs = qs[~np.isnan(qs)]
    cmap = {q: sns.cubehelix_palette(as_cmap=True)((q - qs.min()) / len(qs)) for q in qs}
    cmap[np.nan] = (0.5,0.5,0.5)

    agg_data = data.groupby(['Well', 'TargetName']).agg(list)
    for column in agg_data.columns:
        agg_data[column] = agg_data[column].apply(np.array)
    
    rxns = (setup[['Well', 'WellPosition', 'TargetName', 'Reporter', 'Quantity']]
            .merge(
                agg_data,
                on=['Well', 'TargetName']
            )
            .merge(
                results[['Well', 'Reporter','CT','Omit']],
                on=['Well', 'Reporter']
            )
            .assign(
                Tar_Q=lambda df: df.Quantity.map(np.log10),
                color=lambda df: df.Tar_Q.map(cmap)
            )
            .rename(columns={'TargetName': 'Tar','Omit':'outlier'})
            .reset_index()
            .drop(columns=['Quantity', 'index'])
           )

    return rxns

def get_inflection_pts(sig,threshold=0.01):
    c = np.arange(len(sig))+1
    c_ = np.linspace(1,len(sig),1000)
    splder = sp.interpolate.UnivariateSpline(c,sig,s=0.01).derivative()(c_)
    pks,_ = sp.signal.find_peaks(splder,height=threshold)
    return c_[pks]

def plot_inflection_pts(tar=None,threshold=0.01):
    for row in rxns.itertuples():
        if tar in [None,row.Tar]:
            sig = row.Rn
            c = np.arange(len(sig))+1
            c_ = np.linspace(1,len(sig),1000)
            splder = sp.interpolate.UnivariateSpline(c,sig,s=0.01).derivative()(c_)
            pks,_ = sp.signal.find_peaks(splder,height=threshold)
            pks = get_inflection_pts(sig, threshold)
            plt.plot(c_,splder,color=row.color)
            for i,pk in enumerate(pks):
                plt.plot(c_[pk],splder[pk],'x',color=f'C{i}')

def get_reaction_bounds(rxn,threshold=0.01):
    pks = get_inflection_pts(rxn.Rn, threshold)
    start = 3
    end = min(int(np.ceil(pks[0]))+25,len(rxn.Rn)) if len(pks)>0 else rxn.Cycle[-1]
            
    return [start, end]

def isflat(sig):
    
    c = np.arange(len(sig))+1
    c_ = np.linspace(1,len(sig),1000)
    splder = sp.interpolate.UnivariateSpline(c,sig,s=0.01).derivative()(c_)
    pks,_ = sp.signal.find_peaks(splder,height=0.05)
    
    return len(pks)==0

def trim_rows(row,column):
    start = np.argwhere(np.array(row.Cycle)==row.Bounds[0])[0,0] if row.Bounds[0]>=row.Cycle[0] else [0]
    end = np.argwhere(np.array(row.Cycle)==row.Bounds[1])[0,0] if row.Bounds[1]<=row.Cycle[-1] else [-1]
    return row[column][start:end]

def apply_reaction_bounds(rxns,threshold=0.01):
    rxns = rxns.assign(Bounds=lambda df: df.apply(lambda row: get_reaction_bounds(row,threshold), axis=1))
    return rxns

def trim_reactions(rxns):
    for column in ['Rn','DeltaRn','Cycle']:
        rxns[column] = rxns.apply(lambda row: trim_rows(row,column), axis=1)
    return rxns

def estimate_parameters(rxn,F0_offset=10.5):
    F0_lg = rxn.Tar_Q-F0_offset
    F0 = 10**F0_lg

    c = np.linspace(0,rxn.Cycle[-1],1000)
    spl = sp.interpolate.UnivariateSpline(rxn.Cycle,rxn.Rn,s=0.01)(c)
    τ = c[np.argmin(np.abs((spl-spl[0])-(spl[-1]-spl[0])/2))]
    τ_ = τ+np.log2(10)*(rxn.Tar_Q-5)

    r = -np.log(F0)/τ/np.log(2)
    ρ = -np.log(r*np.log(2))/np.log(τ)

    K = rxn.Rn[-1]

    bkg = rxn.Rn-rxn.DeltaRn
    bkg_β = (bkg[10]-bkg[0])/10
    bkg_α = bkg[0]
    bad_bkg = bkg_β>0.02

    m = sp.stats.linregress(rxn.Cycle[-10:],rxn.Rn[-10:]).slope
    m_ = sp.stats.linregress(rxn.Cycle[-10:],rxn.DeltaRn[-10:]).slope
    
    return {'F0_lg':F0_lg,'F0':F0,'τ':τ,'τ_':τ_,'r':r,'ρ':ρ,'K':K,'m':m,
            'm_':m_,'bkg_α':bkg_α,'bkg_β':bkg_β,'bad_bkg':bad_bkg}

def get_param_estimates(rxns,F0_offset=10.5):
    return rxns.apply(lambda rxn: estimate_parameters(rxn,F0_offset), axis=1, result_type = 'expand')

skip = lambda x: x

transforms = {
    'r':[skip,skip], 
    'ρ':[sp.special.logit,sp.special.expit], 
    'τ':[np.log,np.exp], 
    'τ_':[np.log,np.exp], 
    'K':[np.log,np.exp], 
    'm':[np.log,np.exp], 
    'offset':[skip,skip], 
    'Q':[skip,skip], 
    'BP':[np.log,np.exp], 
    'GC':[sp.special.logit,sp.special.expit]
}

pymc_transforms = {
    'r':[skip,skip], 
    'ρ':[pm.math.logit,pm.math.invlogit], 
    'τ':[pm.math.log,pm.math.exp], 
    'τ_':[pm.math.log,pm.math.exp], 
    'K':[pm.math.log,pm.math.exp], 
    'm':[pm.math.log,pm.math.exp], 
    'offset':[skip,skip], 
    'Q':[skip,skip], 
    'BP':[pm.math.log,pm.math.exp], 
    'GC':[pm.math.logit,pm.math.invlogit]
}

def build_standardizers(stdzr, transforms=transforms, pymc_transforms=pymc_transforms, WT_BP=88, WT_GC=43, prefix=None):
    def stdz(val, param, lg_Q=None, stdzr=stdzr, transforms=transforms):
        transform = transforms[param][0] if param in transforms.keys() else lambda x: x
        val = val if param != 'τ' else val+np.log2(10)*(lg_Q-5)
        val_z = (transform(val)-stdzr[param]['μ'])/stdzr[param]['σ']
        return val_z

    def unstdz(val_z, param, lg_Q=None, stdzr=stdzr, transforms=transforms):
        untransform = transforms[param][1] if param in transforms.keys() else lambda x: x
        val = untransform(val_z*stdzr[param]['σ']+stdzr[param]['μ'])
        val = val if param != 'τ' else val-np.log2(10)*(lg_Q-5)
        return val
    
    def pm_stdz(*args,**kwargs):
        kwargs.setdefault('transforms',pymc_transforms)
        return stdz(*args,**kwargs)
    
    def pm_unstdz(*args,**kwargs):
        kwargs.setdefault('transforms',pymc_transforms)
        return unstdz(*args,**kwargs)
    
    def get_BP(tar, WT_BP=WT_BP):
        if ('WT' in tar) | ('ISO' in tar):
            return float(WT_BP)
        else:
            return float(tar.split(' ')[-1].split('_')[0][2:])

    def get_GC(tar, WT_GC=WT_GC):
        if ('WT' in tar) | ('ISO' in tar):
            return float(WT_GC)/100
        else:
            return float(tar.split(' ')[-1].split('_')[1][2:])/100
        
    return stdz, unstdz, pm_stdz, pm_unstdz, get_BP, get_GC

def formatMeltCurveImport(imps: dict):
    
    results = imps['results']
    setup = imps['setup']
    data = imps['melt'][['Well','Temperature','Fluorescence','Derivative']]

    qs = np.log10(setup.Quantity.unique())
    qs = qs[~np.isnan(qs)]
    cmap = {q: sns.cubehelix_palette(as_cmap=True)((q - qs.min()) / len(qs)) for q in qs}
    cmap[np.nan] = (0.5,0.5,0.5)
    
    def collapse(df):
        return pd.DataFrame({c: [np.array(df[c])] for c in df.columns if c != 'Well'})

    rxns = (setup[['Well', 'WellPosition', 'TargetName', 'Reporter', 'Quantity']]
            .merge(
                (data
                 .groupby('Well')
                 .apply(lambda df: collapse(df))
                 .reset_index()
                 .drop(columns='level_1')
                ),
                on=['Well']
            )
            .merge(
                results[['Well','Tm1','Tm2','Tm3']],
                on=['Well']
            )
            .assign(
                Tar_Q=lambda df: df.Quantity.map(np.log10),
                color=lambda df: df.Tar_Q.map(cmap)
            )
            .rename(columns={'TargetName': 'Tar'})
            .reset_index()
            .drop(columns=['Quantity', 'index'])
           )

    return rxns
    

def formatMUXimport(imps: dict):

    rxns = formatSNXimport(imps)

    rel_q = (rxns.groupby('Well')
                .apply(lambda x: (x[x.Tar=='WT'].Tar_Q.values -
                                  x[x.Tar!='WT'].Tar_Q.values))
                .apply(lambda x: x[0] if len(x)>0 else np.nan))
    rel_qs = rel_q.dropna().unique()
    extent = np.max(np.abs(rel_qs))*2+1

    cmap = {q: mpl.cm.get_cmap('twilight')((q-rel_qs.min())/extent) for q in rel_qs}

    return rxns.assign(Rel_Q=lambda df: df.Well.map(rel_q),
                       color=lambda df: df.Rel_Q.map(cmap))

# %% Calculate CTs from thresholds

def calcCTs(dRn, thresholds, n_t, n_q, n_r, n_c, interp_step=0.01):
    c = np.arange(n_c) + 1
    ct_manual = np.zeros([n_t, n_q, n_r])
    for (i, j, k) in np.ndindex(n_t, n_q, n_r):
        # Interpolate the amplification curves to get fractional CTs
        c_interp = np.arange(0, n_c + interp_step, interp_step) + 1
        dRn_interp = np.interp(c_interp, c, dRn[i, j, k])
        # Find the first fractional cycle over the threshold
        ct = np.argmax(dRn_interp > thresholds[i, j, k]) * interp_step + 1
        # If nothing found, np.argmax will return 0, so enter nan as CT
        ct_manual[i, j, k] = np.where(ct - 1, ct, np.nan)
    return ct_manual


# %% Plot each target in own axis

def setupOverlay(targets, legend_kws=None, t_map=None, show_legend=True, fig=None):
    n_t = len(targets)

    if legend_kws is None: legend_kws = {}

    if fig is None:
        fig = plt.figure(constrained_layout=True,
                         figsize=[20, 8])
    if t_map is not None:
        # Use the coordinates specified in t_map
        nrows = max([v[0] for v in t_map.values()])
        ncols = max([v[1] for v in t_map.values()])
        gs = fig.add_gridspec(nrows + 1, ncols + 1)
        gs_map = {t: gs[coord] for (t, coord) in t_map.items()}
    else:
        # Make the GridSpec layout as square as possible
        n_t_ = n_t + 1
        if n_t_ > 25:
            sq = n_t_ ** 0.5
            nrows = int(sq)
            ncols = n_t_ // nrows + int(n_t_ % nrows != 0)
        elif n_t_ <= 4:
            ncols = n_t_
            nrows = 1
        else:
            ncols = 5
            nrows = n_t_ // ncols + int(n_t_ % ncols != 0)
        gs = fig.add_gridspec(nrows, ncols)
        gs_map = {t: gs[i] for i, t in enumerate(targets)}
        gs_map['legend'] = gs[n_t]

        # Plot with smaller x- and y-tick labels and axis labels
    with plt.rc_context({'axes.labelsize': 'small',
                         'axes.labelweight': 'normal',
                         'xtick.labelsize': 'small',
                         'ytick.labelsize': 'small'}):
        # Create axes for each target
        axs = {t: fig.add_subplot(gs_map[t]) for t in targets}

    with plt.rc_context({'axes.titlesize': 'large'}):
        for t in targets: axs[t].set_title(t)

    if show_legend:
        # Add legend in blank gridspec location
        # Add a subplot in the location
        axs['legend'] = fig.add_subplot(gs_map['legend'])
        # plot dummy data with appropriate labels, add legend, then remove axes

        for k, v in legend_kws.items():
            axs['legend'].plot([], [], **v, label=k)
        axs['legend'].legend(loc='center', fontsize=16)
        axs['legend'].set_frame_on(False)
        axs['legend'].axes.get_xaxis().set_visible(False)
        axs['legend'].axes.get_yaxis().set_visible(False)

    return fig, axs


# %% Plot efficiencies from CT fits
# Function for determining efficiency from CT fits with uncertainty
def calc_E(ct_mat, quant_list, n_reps):
    ct_unroll = np.reshape(ct_mat, len(quant_list) * n_reps)
    q_unroll = np.repeat(quant_list, n_reps)
    mask = ~np.isnan(ct_unroll)
    fit_stats = stat.linregress(q_unroll[mask], ct_unroll[mask])
    slope = fit_stats.slope
    E = 10 ** (-1 / slope) - 1  # calculation of the efficiency based on the slope of the fit
    dE = (fit_stats.stderr / slope ** 2) * E
    return {'E': 2 * E,
            'dE': 2 * dE,
            'slope': fit_stats.slope,
            'intercept': fit_stats.intercept}


# Plot efficiencies with uncertainty
def plotBox(ax, x, E, dE, style=None, label=None):
    if style is None: style = {'color': 'k'}
    le = E - dE
    ue = E + dE
    ax.fill([x - 0.25, x + 0.25, x + 0.25, x - 0.25], [le, le, ue, ue],
            alpha=0.5,
            **style)
    ax.plot([x - 0.25, x + 0.25], [E, E], lw=2, **style)
    ax.plot(x, E, '.', ms=20, **style, label=label)
    return


def plotEfit(targets, lg_q, CTs, t_colors=None, style='absolute'):
    n_t, n_q, n_r = np.shape(CTs)
    target_stats = {t: calc_E(CTs[i, ::], lg_q, n_r) for i, t in enumerate(targets)}

    # Overlay CT efficiency fits
    fig_CT_eff, ax_CT_eff = plt.subplots(1, 2, figsize=[12, 4])

    if t_colors is None:
        t_colors = {t: {'color': 'C{:}'.format(i)} for i, t in enumerate(targets)}

    for i, t in enumerate(targets):
        eff = target_stats[t]
        ax_CT_eff[0].plot(lg_q, CTs[i, ::],
                          linestyle='none',
                          marker='.',
                          **t_colors[t])
        ax_CT_eff[0].plot(lg_q, eff['intercept'] + eff['slope'] * lg_q,
                          **t_colors[t],
                          label=t)
    ax_CT_eff[0].legend()
    plt.setp(ax_CT_eff[0], **{
        'xticks': lg_q,
        'title': 'Efficiency fits to CTs',
        'xlabel': 'log$_{10}$ Copies',
        'ylabel': 'C$_{T}$'
    })

    if style.casefold() in ('%', 'percent'):
        for v in target_stats.values():
            v['E'] /= 0.02
            v['dE'] /= 0.02

    for i, t in enumerate(targets):
        plotBox(ax_CT_eff[1], i + 1, target_stats[t]['E'], target_stats[t]['dE'], style=t_colors[t])

    if style.casefold() == 'absolute':
        ax_CT_eff[1].set_ylim(0.95, 3.05)
        ax_CT_eff[1].set_ylabel('Efficiency (abs)')
    elif style.casefold() in ('%', 'percent'):
        ax_CT_eff[1].set_ylim(0.5, 1.5)
        ax_CT_eff[1].set_ylabel('Efficiency (%)')

    plt.setp(ax_CT_eff[1], **{
        'title': 'Efficiencies from CT Fits',
        'xlabel': 'Target',
        'xlim': [0.5, n_t + 0.5],
        'xticks': np.arange(n_t) + 1,
        'xticklabels': targets
    })
    return {'target_stats': target_stats,
            'fig_CT_eff': fig_CT_eff,
            'ax_CT_eff': ax_CT_eff, }


# %% Equation Definitions

# Define the (differential) equations to be fit to the data

# Convert the growth rate from base-e to base-2
lg2_e = np.log2(np.exp(1))


# Logistic growth equation with growth rate r and carrying capacity K
def growth(t, params):
    pop0 = params['pop0'].value
    r = params['r'].value
    K = params['K'].value
    return K / (1 + (K - pop0) / pop0 * np.exp(-r / lg2_e * t))


# Logistic growth ODE
def growth_deq(pop, t, params):
    r = params['r'].value
    K = params['K'].value
    x = pop
    return r / lg2_e * x * (1 - x / K)


# Solve the logistic growth ODE
def growth_deq_sol(t, pop0, params):
    return odeint(growth_deq, pop0, t, args=(params,))


# Error in the growth model
def growth_residual(params, t, data):
    pop0 = params['pop0'].value
    model = growth_deq_sol(t, pop0, params)
    return (model - data).ravel()  # *np.gradient(data.ravel())


# Linear drift equation
def drift(t, params):
    intercept = params['intercept'].value
    slope = params['slope'].value
    return intercept + slope * t


# A mixture model for the drift and growth equations (solutions)
def mix_model(t, params):
    return drift(t, params) * growth(t, params)


# Error in the growth model
def mix_residual(params, t, data):
    model = np.reshape(mix_model(t, params), [-1, 1])
    return (model - data).ravel()  # *np.gradient(data.ravel())


# Logistic growth ODE where K (carrying capacity) is a linear function of time
def mix_deq(vals, t, params):
    pop, K = vals
    r = params['r'].value
    m = params['slope'].value

    dx = r / lg2_e * pop * (1 - pop / K)
    dK = m

    return [dx, dK]


# Solve the ODE with tight absolute tolerance
def mix_deq_sol(t, vals0, params):
    abserr = 1.0e-12
    relerr = 1.0e-6
    return odeint(mix_deq, vals0, t, args=(params,),
                  atol=abserr, rtol=relerr)


# Error in the growth/drift ODE model.
# Assess only the full model, not the drift component alone (model[:,1])
def mix_deq_res(params, t, data):
    pop0 = params['pop0'].value
    K0 = params['intercept'].value
    model = mix_deq_sol(t, [pop0, K0], params)
    return (model[:, 0] - data[:, 0]).ravel()

def logistic_eqn(x,x0,k,xmax,xmin):
    return (xmax-xmin)/(1+np.exp(-k*(x-x0)))+xmin


def logistic_DR(k,width=0.8):
    return -2/k*np.log(2/(1+width)-1)


# %% Fitting Strategy
# Fit differential equations directly (True) or use the "empirical" mixture model (False).
# The direct diffeq approach doesn't improve the fit greatly, and takes ~twice as long, but is more explainable.

def driftgrowth(rxn, pop0_offset=0, diffeq=True, signal='Sig'):

    c = rxn.cycle
    data = np.reshape(getattr(rxn,signal), [-1, 1])

    # Fit a simple logistic growth ODE model with no drift to provide estimates for the drift model
    init_growth = lf.Parameters()
    init_growth.add('r', value=1, min=0.1, max=2, vary=True)  # Could maybe be a wider range
    # Offset which relates known initial copy number to estimated initial fluorescence intensity
    pop0_guess = rxn.Tar_Q - pop0_offset
    init_growth.add('pop0', value=10 ** (pop0_guess), min=10 ** (pop0_guess - 1), max=10 ** (pop0_guess + 1), vary=True)
    init_growth.add('K', value=float(data[-1]) * 0.8, min=1e-1, max=1.5)

    init_result = lf.minimize(growth_residual, init_growth, args=(c, data), method='leastsq')
    final = data + init_result.residual.reshape(data.shape)

    drift0 = len(c) - 5 if rxn.Tar_Q == 2 else len(c) - 10

    init_drift = np.polyfit(c[drift0:], data[drift0:], 1)[:, 0]

    # Use outputs from the preliminary growth fit as initial guesses for the mixture model
    mixture = lf.Parameters()
    # Should probably limit variation to a narrower relative range for both pop0 and r
    mixture['r'] = cp.copy(init_result.params['r'])
    mixture['pop0'] = cp.copy(init_result.params['pop0'])
    # Allow the drift parameters to vary only slightly (20%) from their values found by the linear fit
    # Otherwise the intercept and r in particular become far too correlated
    slope = float(init_drift[0])
    intercept = float(init_drift[1])
    if np.abs(slope) < 1e-15:
        mixture.add('slope', value=slope, min=-1e-10, max=1e-10)
    else:
        mixture.add('slope', value=slope, min=slope * 0.8, max=slope * 1.2)
    mixture.add('intercept', value=intercept, min=intercept * 0.8, max=intercept * 1.2)

    if diffeq:
        mix_result = lf.minimize(mix_deq_res, mixture, args=(c, data), method='leastsq')
        pop0 = mix_result.params['pop0'].value
        K0 = mix_result.params['intercept'].value
        model = mix_deq_sol(c, [pop0, K0], mix_result.params)
        final = model[:, 0]
    else:
        mixture.add('K', value=1, vary=False)
        mix_result = lf.minimize(mix_residual, mixture, args=(c, data), method='leastsq')
        final = data[:, 0] + mix_result.residual.reshape(data[:, 0].shape)  # /np.gradient(data.ravel())
    return {'params': mix_result,
            'fit': final}

def fit_driftgrowth(rxn, diffeq=False, pop0_offset=0, signal='Sig'):

    # Perform the fits
    mix_result, final_fit = driftgrowth(rxn, pop0_offset=pop0_offset, diffeq=diffeq, signal=signal).values()

    # Find the "true" carrying capacity CC defined by the value of the drift function at the critical time t_c
    dy = np.diff(final_fit)
    c = np.arange(len(getattr(rxn,signal)))
    t_c = c[np.argmax(dy) + 1]
    CC = drift(t_c, mix_result.params)

    # Flag the results to ignore anything for which the software couldn't find a CT,
    # or had very low signal change (CC<0.01), or was just a bad fit (reduced chi-squared > 0.001)
    is_bad = bool(np.isnan(rxn.CT) | (CC < 0.01) | (mix_result.redchi > 1e-3))

    final_params = {
        'Name': rxn.Index,
        'Fit': final_fit,
        'Sig0': mix_result.params['pop0'].value,
        'lg_Sig0': np.log10(mix_result.params['pop0'].value),
        'r': mix_result.params['r'].value,
        'b': mix_result.params['intercept'].value,
        'K': mix_result.params['K'].value,
        'm': mix_result.params['slope'].value,
        't_c': t_c,
        'CC': CC,
        'bad': is_bad,
        'chi2': mix_result.redchi,
        'aic': mix_result.aic,
        'bic': mix_result.bic,
    }

    return final_params





# %% Plot solo and multiplexed data and fits

def getRxnCmap(rxns,col):
    return rxns.apply(lambda x: (x[col], x.color), axis=1).unique()

def dummyLegendKws(rxns,col):
    # Define appearance of labels in the legend
    plt_kws = {int(q): {
        'color': c,
        'ls': '',
        'marker': 'o'
    } for q, c in getRxnCmap(rxns,col)}
    return plt_kws

def plotSNXdata(c, rxns, axs, **plt_kwargs):

    if 'zorder' not in plt_kwargs.keys():
        plt_kwargs['zorder'] = 1
    if 'ls' not in plt_kwargs.keys():
        plt_kwargs['ls'] = 'None'
    if 'marker' not in plt_kwargs.keys():
        plt_kwargs['marker'] = '.'

    plt_kwargs.pop('color',None)

    # Plot the raw data
    for row in rxns.itertuples():
        axs[row.Tar].plot(c, row.Sig, color=row.color, **plt_kwargs)

    plt.setp(axs.values(),
             xlim=[0, c.max()],
             xticks=np.arange(0, c.max() + 1, 10),
             xticklabels='',
             # xlabel = 'Cycle',
             ylim=[0, 1],
             yticks=np.arange(0, 1.01, 0.2),
             yticklabels='',
             # ylabel = 'Norm Signal',
             )

    if 'ISO' in axs.keys():
        plt.setp(axs['ISO'],
                 xticklabels=np.arange(0, c.max() + 1, 10),
                 xlabel='Cycle',
                 yticklabels=np.around(np.arange(0, 1.01, 0.2), 1),
                 ylabel='Norm Signal',
                 )

    return axs

def plotSNXfits(c, rxns, axs, grey = None, **plt_kwargs):
    if grey is None: grey=[0.6, 0.6, 0.6]

    if 'zorder' not in plt_kwargs.keys():
        plt_kwargs['zorder'] = 2
    if 'lw' not in plt_kwargs.keys():
        plt_kwargs['lw'] = 2
    plt_kwargs.pop('color',None)

    # Plot the fits, ignoring "bad" fits
    for row in rxns.query('~bad').itertuples():
        axs[row.Tar].plot(c, row.Fit, color=row.color, **plt_kwargs)

    for t in rxns.Tar.unique():
        for row in rxns.query('Tar=="ISO"').itertuples():
            axs[t].plot(c, row.Fit, color=grey, zorder=-1)

    return


def plotMUXdata(c, rxns, axs, pos_reporter='FAM', neg_reporter='HEX', **plt_kwargs):

    if 'lw' not in plt_kwargs.keys():
        plt_kwargs['lw'] = 2
    plt_kwargs.pop('color',None)

    for well, group in rxns.groupby('Well'):
        tar = group.query('Tar!="WT"').Tar.values[0]
        pos = group.query('Reporter == @pos_reporter')
        neg = group.query('Reporter == @neg_reporter')
        axs[tar].plot(c, pos.Sig.values[0], color=pos.color.values[0], **plt_kwargs)
        axs[tar].plot(c, -neg.Sig.values[0], color=neg.color.values[0], **plt_kwargs)
        axs[tar].axhline(0, color='k')

    plt.setp(axs.values(),
             xlim=[0, c.max()],
             xticks=np.arange(0, c.max() + 1, 10),
             xticklabels='',
             ylim=[-1, 1],
             yticks=np.arange(-1., 1.01, 0.5),
             yticklabels='',
    )

    if 'ISO' in axs.keys():
        plt.setp(axs['ISO'],
                 xticks=np.arange(0, c.max() + 1, 10),
                 xticklabels=np.arange(0, c.max() + 1, 10),
                 xlabel='Cycle',
                 yticklabels=np.array([1, 0.5, 0.0, 0.5, 1.0]),
                 ylabel=f'$\leftarrow {pos_reporter} \quad | \quad {neg_reporter} \\rightarrow$'
        )


    return axs


# %% Plot Fit parameters by target and quantity

def plotParam(FitParams, param, threshold, x='Target', hue='Quantity', order=None, ax=None):

    bad = pd.Series(index=FitParams.index, dtype=bool)
    if threshold is np.ndarray:
        assert len(threshold) == 2, 'Threshold must have at most two bounds'
        out = (FitParams[param] < threshold[0]) | (FitParams[param] > threshold[1])
    else:
        out = (np.abs(FitParams[param]) > np.abs(threshold))
    na = FitParams[param].isin([np.nan, np.inf, -np.inf])
    bad = bad | out | na | FitParams.bad

    good = ~bad

    if ax is None:
        _, ax = plt.subplots(1,1)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    sns.violinplot(x=x, y=param, data=FitParams[good],
                   order=order,
                   color='white', inner=None, cut=0, ax=ax, zorder=0)

    for j, val in enumerate(order):
        this = FitParams[good]
        this = this[this[x] == val]
        dot_zorder = this.sort_values(by=[hue]).index
        ax.scatter(x=[j for _ in range(len(this))], y=this[param][dot_zorder], c=this.color[dot_zorder], s=10 ** 2,
                   zorder=5)
        med = this[param].median()
        ax.plot([j - 0.25, j + 0.25], [med, med], 'k', zorder=1)

    '''
    sns.stripplot(x = x, y = p, data = FitParams[good],
                  order=order, 
                  size = 10, alpha = 1, jitter = False, hue = hue, ax = ax)
    '''
    ax.set_xlabel('')
    ax.set_ylabel('')

    return ax



# %% Compare endpoints
def compareEndpoints(rxns, endpt=-15, k_i = 2, skip_rxns=None):

    if skip_rxns is not None:
        rxns = rxns[~skip_rxns]

    diff = (rxns
            .groupby('Well')
            .apply(lambda x: (x[x.Tar=='WT'].Sig.values[0] -
                              x[x.Tar!='WT'].Sig.values[0])[endpt])
            )

    tar_qs = (rxns
              .query('Tar=="WT"')
              .groupby('Well')
              .Tar_Q
              .first()
              )

    combined = (rxns
                 .assign(Endpoint=rxns.Well.map(diff),
                         Tar_Q=rxns.Well.map(tar_qs))
                 .query('Tar!="WT"')[['Tar', 'Endpoint', 'Tar_Q', 'color']]
                 )

    endpoint_lst = []
    for ref, points in combined.groupby('Tar'):

        Tar_Qs = points.Tar_Q.values
        Diffs = points.Endpoint.values

        x0_i = Tar_Qs[np.argmin(np.abs(Diffs))]
        xmax_i = Diffs.max()
        xmin_i = Diffs.min()

        params = ['x0', 'k', 'x_min', 'x_max']
        p0 = [x0_i, k_i, xmax_i, xmin_i]
        bounds = ([x0_i - 1, k_i - 1, xmax_i / 2, -1],
                  [x0_i + 1, k_i + 1, 1, xmin_i / 2])

        p_opt, _ = opt.curve_fit(logistic_eqn, Tar_Qs, Diffs, p0=p0, bounds=bounds)

        for i,p in enumerate(p_opt):
            p = np.around(p,2)
            if any([p in bounds[0]]):
                warn(f'Final value of {params[i]} at lower bound for {ref}. Fit may be inaccurate.')
            if any([p in bounds[1]]):
                warn(f'Final value of {params[i]} at upper bound for {ref}. Fit may be inaccurate.')

        endpoint_lst.append({
            'Ref': ref,
            'Ref_Q': rxns.query('Tar==@ref').Tar_Q.unique()[0],
            'Tar_Qs': Tar_Qs,
            'Diffs': Diffs,
            'logistic_ps': p_opt,
            'k': p_opt[1],
            'DR': logistic_DR(p_opt[1]),
            'color': points.color.values
        })

    return pd.DataFrame(endpoint_lst)


def plotEndpoints(endpoints, axs, grey = None):
    if grey is None: grey=[0.6, 0.6, 0.6]

    q_min = endpoints.Tar_Qs.map(min).min()
    q_max = endpoints.Tar_Qs.map(max).max()
    x = np.linspace(q_min, q_max, 100)

    # For each competition experiment, plot the difference between endpoint FAM and HEX intensities
    for row in endpoints.itertuples():
        t = row.Ref
        axs[t].scatter(row.Tar_Qs, row.Diffs, color=row.color, lw=2, zorder=3)

        axs[t].axhline(0, color=[0.5, 0.5, 0.5], zorder=1)
        axs[t].axvline(row.Ref_Q, color=[0.5, 0.5, 0.5], zorder=1)

        axs[t].annotate(f'{row.k:.2f}', xy=(0.025, 0.875), xycoords='axes fraction')
        axs[t].plot(x, logistic_eqn(x, *row.logistic_ps), color='k', zorder=2)

        if t == 'ISO':
            for t_ in endpoints.Ref:
                if t_ == 'ISO': continue
                axs[t_].plot(x, logistic_eqn(x, *row.logistic_ps), color=grey, zorder=0)

    plt.setp(axs.values(),
             ylim=[-1., 1.],
             yticks=np.arange(-1., 1.01, 0.5),
             yticklabels='',
             xticks=np.arange(min(x), max(x) + 1, 1),
             xticklabels='',
             )

    if 'ISO' in axs.keys():
        plt.setp(axs['ISO'],
                 xlabel='log10 WT Copies',
                 xticklabels=np.arange(2, 8 + 1),
                 ylim=[-1., 1.],
                 yticks=np.arange(-1., 1.01, 0.5),
                 yticklabels=np.array([-1, -0.5, 0.0, 0.5, 1.0]),
                 ylabel='FAM - HEX'
                 )

    axs['legend'].get_legend().set_title('log10 Ratio')
    axs['legend'].set_ylim([1, 2])
    axs['legend'].set_xlim([1, 2])

    return axs


# %%