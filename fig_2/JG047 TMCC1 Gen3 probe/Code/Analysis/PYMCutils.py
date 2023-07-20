import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.stats as stat
import pickle


import pymc3 as pm
import arviz as az
import theano
import theano.tensor as tt


def nc_Normal(name, mu, sigma, **kws):
    nc = pm.Normal(str(name+'_nc'), mu=0, sigma=1, **kws)
    rv = pm.Deterministic(name, mu + sigma*nc)
    return rv, nc

def sc_Exponential(name, mu, **kws):
    nc = pm.Exponential(str(name+'_nc'), lam=1, **kws)
    rv = pm.Deterministic(name, mu*nc)
    return rv, nc


def myRidgeplot(trace, params, axs, forest=True, yticklabels=None, transform_dict=None, **kwargs):
    
    kwargs.setdefault('ridgeplot_overlap', 15)
        
    kws=[k for k in kwargs.keys()]
    
    if 'colors' in kws:
        colors = kwargs['colors']
    
    for i,var in enumerate(params):
        ax=axs.flat[i]
        
        if transform_dict is not None:
            if var in transform_dict.keys():
                tfrm = transform_dict[var]
            else:
                tfrm = None
        elif 'transform' not in kws:
            tfrm = {True: sp.special.expit,
                    False: None}['lgt' in var]
        else:
            kwargs.setdefault('transform', None)
            tfrm = kwargs['transform']
        
        kwargs.update({'transform': tfrm})
            
        if 'colors' not in kws:
            kwargs.update({'colors':'cycle'})
        else:
            kwargs.update({'colors':colors})
            
        if 'linewidth' not in kws:
            kwargs.update({'linewidth' : 0})

        arr = trace[var]
        if len(arr.shape)>1:
            arr = [arr[:,j] for j in range(arr.shape[1])]
            
        az.plot_forest(arr, 
                       kind='ridgeplot',
                       combined=True,
                       ax=ax,
                       **kwargs
                      );
        
        if forest:
            kwargs.update({'colors':'k'})
            kwargs.update({'linewidth' : None})
            az.plot_forest(arr, 
                           kind='forestplot',
                           combined=True,
                           ax=ax,
                           **kwargs
                          );

        ax.set_title(var, fontsize=20)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    return axs


def compareRidgeplots(var_names, axs, trace_lst: list, df_lst: list, kwarg_lst=None):
    
    n_items = len(trace_lst)
    assert n_items == len(df_lst)
    if kwarg_lst is None:
        kwarg_lst =[{} for _ in range(n_items)]
    assert n_items == len(kwarg_lst)
    
    n_rxns = list(map(len,df_lst))
    longest = np.argmax(n_rxns)
    ys = []
    offset = []
    start=0
    
    for i,(trace,df,kwargs) in enumerate(zip(trace_lst, df_lst, kwarg_lst)):
        df = df.reset_index()
        kwargs.setdefault('ridgeplot_overlap',13)
        
        if i<n_items-1:
            kwargs.setdefault('colors','k')
            kwargs.setdefault('linewidth',1)
            kwargs.setdefault('ridgeplot_alpha',0)
            kwargs['forest'] = False
            
            myRidgeplot(trace, var_names, axs, **kwargs)
            for ax in axs:
                for line in ax.lines:
                    plt.setp(line,'zorder',plt.getp(line,'zorder')-100)
        else:
            kwargs.setdefault('colors',list(df.color.values))
            myRidgeplot(trace, var_names, axs, **kwargs)
        
        stop = start+n_rxns[i]*2
        #ys.append([plt.getp(line,'ydata')[0] for line in ax.lines[start:stop:2]][::-1])
        ys.append(axs[0].get_yticks()[::-1])
        start += n_rxns[i]*2
        
    #print(ys)
    offset = {row.Well: ys[longest][idx] for idx,row in df_lst[longest].reset_index().iterrows()}
    
    for i,df in enumerate(df_lst):
        df = df.reset_index()
        if n_rxns[i] == n_rxns[longest]:
            continue
        line_start = np.sum([(n_rxns[j])*2 for j in range(i+1)])
        kde_start = np.sum([(n_rxns[j]) for j in range(i+1)])-1
        for ax in axs:
            for idx,row in df.iterrows():
                if i!=n_items-1:
                    for inc in range(2):
                        # Move the KDE outlines
                        line = ax.lines[line_start-(idx*2+inc+1)]
                        xydata = plt.getp(line,'xydata')
                        line.set_ydata(xydata[:,1]-ys[i][idx]+offset[row.Well])
                else:
                    # Move the mean dot
                    line = ax.lines[-1-idx]
                    xydata = plt.getp(line,'xydata')
                    line.set_ydata(xydata[:,1]-ys[i][idx]+offset[row.Well])
                    for inc in range(2):
                        # Move the IQR and CI bars
                        line = ax.collections[-(idx*2+inc+1)]
                        verts = line.get_paths()[0].vertices
                        verts[:,1] -= ys[i][idx]
                        verts[:,1] += offset[row.Well]
                    # Move the KDEs
                    col = ax.collections[kde_start-idx]
                    verts = col.get_paths()[0].vertices
                    verts[:,1] -= ys[i][idx]
                    verts[:,1] += offset[row.Well]


    ext = max(ys[longest])+max([kw['ridgeplot_overlap'] for kw in kwarg_lst])
    axs[0].set_ylim([-ext*0.025, ext*1.025])
    axs[0].set_yticks(ys[longest])
    return
    
    
def plot_VSUP(X, Y, Z=None, Z_av=None, Z_err=None, ax=None, 
              cmap='pink', bkg=0., av_kws=None, er_kws=None,
              av_levels=8, er_levels=4, av_cbar=True, er_cbar=True,
              continuous=False
             ):
    
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(10,4))
    if Z is not None:
        Z_av = Z.mean(axis=0)
        Z_err = Z.std(axis=0)
    
    if av_kws is None: av_kws={}
    av_kws.setdefault('zlim',[Z_av.min(),Z_av.max()])
    zlim = av_kws['zlim']
    Z_av = np.clip(Z_av,*zlim)
    av_clim = zlim+np.diff(zlim)*0.25*[-1,1]
    av_kws.setdefault('vmin',av_clim[0])
    av_kws.setdefault('vmax',av_clim[1])
    av_kws.setdefault('norm',mpl.colors.Normalize(*av_clim))
    
    if er_kws is None: er_kws={}
    er_kws.setdefault('vmin',Z_err.min())
    er_kws.setdefault('vmax',Z_err.max())
    er_kws.setdefault('norm',None)
    er_clim = [er_kws['vmin'],er_kws['vmax']]
    Z_err = np.clip(Z_err,*er_clim)
    
    
    av_kws.setdefault('label','Mean')
    er_kws.setdefault('label','Standard Deviation')
    av_label = av_kws['label']
    er_label = er_kws['label']
    del av_kws['label']
    del er_kws['label']
    
    clist = np.vstack([[bkg,bkg,bkg,alpha] for alpha in np.linspace(0,0.6,256)])
    ercmap = mpl.colors.LinearSegmentedColormap.from_list('VSUP', clist, er_levels)
    
    if continuous:
        X = np.interp(np.linspace(0,1,len(X)+1),np.linspace(0,1,len(X)),X)
        Y = np.interp(np.linspace(0,1,len(Y)+1),np.linspace(0,1,len(Y)),Y)
        av_pc = ax.pcolormesh(X, Y , Z_av, shading='flat', cmap=cmap, **av_kws)
        av_cbar = plt.colorbar(av_pc, ax=ax, pad=0.1)
        av_cbar.set_label('Mean')
        plt.gcf().canvas.draw()
        colors = av_pc.get_facecolor()
        alphas = ((Z_err-er_kws['vmin'])/(er_kws['vmax']-er_kws['vmin'])).ravel()
        av_pc.set_facecolor(np.vstack([colors[:,i]*(1-alphas)+bkg*alphas for i in range(3)]).T)
        er_pc = None
        er_cbar = None
    else:
        del av_kws['zlim']
        av_pc = ax.contourf(X, Y , Z_av, cmap=cmap, levels=np.linspace(*zlim,av_levels+1), **av_kws)
        av_kws['zlim'] = zlim
        sm = plt.cm.ScalarMappable(cmap=ercmap, norm=er_kws['norm'])
        sm.set_clim(*er_clim)
        sm.set_array(Z_err)
        er_pc = ax.contourf(X, Y , Z_err, cmap=ercmap, levels=np.linspace(*er_clim,er_levels+1), **er_kws)
    
    
    if er_cbar:
        er_cbar = plt.colorbar(ax=ax, mappable=sm,
                               boundaries=np.linspace(*er_clim,er_levels+1), format='%.2f')
        er_cbar.set_label(er_label)
    if av_cbar:
        pad = 0.1 if er_cbar else 0.05
        av_cbar = plt.colorbar(av_pc, ax=ax, pad=pad, format='%.2f')
        av_cbar.set_label(av_label)
    
    return av_kws, er_kws