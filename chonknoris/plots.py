from matplotlib import pyplot,cm 
import numpy as np 

def plot_metrics(metrics, tags=None, logscale=None, s0=0, linewidth=3, color_train=None, color_val=None, fsx=7, fsy=5, sharex=False, sharey=False):
    from matplotlib import pyplot
    if tags is None: tags = [col[6:] for col in metrics.columns if "train_" in col]
    fig,ax = pyplot.subplots(nrows=1,ncols=len(tags),figsize=(fsy*len(tags),fsx),sharex=sharex,sharey=sharey)
    ax = np.atleast_1d(ax)
    assert ax.ndim==1
    epochs = metrics.index[s0:]
    for i,tag in enumerate(tags):
        tvals = metrics["train_"+tag][s0:]
        vvals = metrics["val_"+tag][s0:]
        ax[i].set_ylabel(tag,fontsize="xx-large")
        ax[i].plot(epochs,tvals,label="train",linewidth=linewidth,color=color_train)
        ax[i].plot(epochs,vvals,label="val",linewidth=linewidth,color=color_val)
        ax[i].set_xlabel("epoch",fontsize="xx-large")
        ax[i].legend(fontsize="xx-large")
        if (logscale is True) or (isinstance(logscale,list) and logscale[i]) or (logscale is None and (tvals>0).all() and (vvals>0).all()):
            ax[i].set_yscale("log",base=10)
    fig.tight_layout()
    return fig,ax

def _grid_plot_parser(v, nrows, ncols):
    if v is None or np.isscalar(v): v = np.tile([v],(nrows,ncols))
    v = np.atleast_2d(np.array(v))
    assert v.ndim==2 
    if v.shape[0]==1:
        assert v.shape[1]==ncols 
        v = np.tile(v,(nrows,1))
    if v.shape[1]==1:
        assert v.shape[0]==nrows 
        v = np.tile(v,(1,ncols))
    return v

def plot_contourfs(data, cmap=cm.plasma, contourf_level=100, titles=None, xlabels=None, ylabels=None, fsx=5, fsy=6, antialiased=False):
    assert isinstance(data,list) and all(isinstance(dat,list) for dat in data) and all(len(data[0])==len(dat) for dat in data)
    nrows,ncols = len(data),len(data[0])
    titles = _grid_plot_parser(titles,nrows,ncols)
    xlabels = _grid_plot_parser(xlabels,nrows,ncols)
    ylabels = _grid_plot_parser(ylabels,nrows,ncols)
    assert titles.shape==(nrows,ncols)
    fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(fsy*ncols,fsx*nrows))
    ax = np.atleast_1d(ax).reshape((nrows,ncols))
    for i in range(nrows):
        for j in range(ncols):
            d = data[i][j]
            assert isinstance(d,tuple) and len(d)==3
            ax[i,j].set_title(titles[i,j],fontsize="xx-large")
            cf = ax[i,j].contourf(d[0],d[1],d[2],cmap=cmap,levels=contourf_level,antialiased=antialiased)
            fig.colorbar(cf)
            ax[i,j].set_xlabel(xlabels[i,j],fontsize="xx-large")
            ax[i,j].set_ylabel(ylabels[i,j],fontsize="xx-large")
    fig.tight_layout()
    return fig,ax

def plot_band_strand(data, 
        plot_strands=False, 
        plot_median=True, 
        plot_mean=False, 
        qlows=[0.01,0.05,0.1,0.25], 
        qhighs=[0.99,0.95,0.9,0.75], 
        plt_band_alphas=[0.1,0.2,0.3,0.4], 
        sharex=None, 
        sharey=None, 
        xlabels=None, 
        ylabels=None, 
        titles=None, 
        colors=None, 
        ylogscale=None, 
        xlogscale=None, 
        marker='o', 
        LWTHIN=0.05, 
        LWTHICK=3, 
        fsx=6, 
        fsy=6, 
        use_nan_funcs=False, 
        legends=True):
    assert isinstance(data,list) and all(isinstance(dat,list) for dat in data) and all(len(data[0])==len(dat) for dat in data)
    nrows,ncols = len(data),len(data[0])
    titles = _grid_plot_parser(titles,nrows,ncols)
    fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(fsy*ncols,fsx*nrows),sharex=sharex,sharey=sharey)
    ax = np.atleast_1d(ax).reshape((nrows,ncols))
    ylogscale = _grid_plot_parser(ylogscale,nrows,ncols)
    xlogscale = _grid_plot_parser(xlogscale,nrows,ncols)
    titles = _grid_plot_parser(titles,nrows,ncols)
    xlabels = _grid_plot_parser(xlabels,nrows,ncols)
    ylabels = _grid_plot_parser(ylabels,nrows,ncols)
    quantile_func = np.nanquantile if use_nan_funcs else np.quantile
    mean_func = np.nanmean if use_nan_funcs else np.mean
    for i in range(nrows):
        for j in range(ncols):
            assert isinstance(data[i][j],list)
            for c in range(len(data[i][j])):
                d = data[i][j][c] 
                assert isinstance(d,tuple)
                if len(d)==3:
                    xtrend,ytrend,label = d
                    kwargs = {
                        "marker":marker,
                        "color":colors[c] if colors is not None else None,
                        "linewidth":LWTHICK,
                    }
                elif len(d)==4:
                    xtrend,ytrend,label,kwargs = d
                else:
                    assert False, "data must be a tuple of length 3 or 4"
                ytrend = np.atleast_2d(ytrend)
                assert ytrend.ndim==2
                if xtrend is None: xtrend = np.arange(ytrend.shape[1])
                assert xtrend.ndim==1 and len(xtrend)==ytrend.shape[1]
                if label is None: label=""
                if plot_strands:
                    ax[i,j].plot(
                        xtrend,
                        ytrend.T,
                        color=kwargs["color"] if "color" in kwargs else None,
                        linewidth = kwargs["linewidth"] if "linewidth" in kwargs else LWTHIN,
                        linestyle = kwargs["linestyle"] if "linestyle" in kwargs else None,
                        label=label,
                    )
                    label = None
                if plot_median: 
                    ax[i,j].plot(
                        xtrend,
                        quantile_func(ytrend,.5,0),
                        linestyle = kwargs["linestyle"] if "linestyle" in kwargs else None,
                        label=label,
                        marker=kwargs["marker"] if "marker" in kwargs else marker,
                    )
                    label = None
                for qlow,qhigh,plt_band_alpha in zip(qlows,qhighs,plt_band_alphas):
                    ax[i,j].fill_between(
                        xtrend,
                        quantile_func(ytrend,qlow,0),
                        quantile_func(ytrend,qhigh,0),
                        color=kwargs["color"] if "color" in kwargs else None,
                        alpha=plt_band_alpha,
                        label=label,
                    )
                    label = None
                if plot_mean:
                    ax[i,j].plot(
                        xtrend,
                        mean_func(ytrend,0),
                        marker=kwargs["marker"] if "marker" in kwargs else marker,
                        color=kwargs["color"] if "color" in kwargs else None,
                        linewidth=kwargs["linewidth"] if "linewidth" in kwargs else LWTHIN,
                        label=label,
                        linestyle = kwargs["linestyle"] if "linestyle" in kwargs else None,
                    )
                    label = None
            if ylogscale[i,j]: ax[i,j].set_yscale("log",base=10)
            if xlogscale[i,j]: ax[i,j].set_xscale("log",base=10)
            ax[i,j].set_xlabel(xlabels[i,j],fontsize="xx-large")
            ax[i,j].set_ylabel(ylabels[i,j],fontsize="xx-large")
            ax[i,j].set_title(titles[i,j],fontsize="xx-large")
            if legends:
                ax[i,j].legend(frameon=False,fontsize="xx-large")
    fig.tight_layout()
    return fig,ax