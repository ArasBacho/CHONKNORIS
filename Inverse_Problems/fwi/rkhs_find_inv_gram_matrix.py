import fastgps 
import qmcpy as qp
import torch 
import numpy as np 
from util import Timer,get_torch_device_backend
from acoustic_forward_solver import acoustic_forward_solver,FWIPARAMETERS, downsample_v

def calc_Thetainv_v(pname, vtype, device, kernel, noise, requires_grad_noise):
    ROOT = os.path.dirname(os.path.realpath(__file__))
    OUTDIR = "%s/data/%s/%s"%(ROOT,pname,vtype)
    p = FWIPARAMETERS[pname]

    data = torch.load("%s/split_train_val.pt"%OUTDIR,weights_only=True)
    logv_t = torch.log(data["velocity_train"][:NGPFITS].to(device))
    print("logv_t.shape = %s"%str(tuple(logv_t.shape)))

    xticks = torch.arange(p["nx"],device=device) 
    zticks = -torch.arange(p["nx"],device=device) 
    zmesh,xmesh = torch.meshgrid(zticks,xticks,indexing="ij")
    cpts = torch.stack([xmesh.flatten(),zmesh.flatten()],dim=1)

    xticks_fine = torch.linspace(xticks.min(),xticks.max(),128,device=device)
    zticks_fine = torch.linspace(zticks.min(),zticks.max(),128,device=device)
    zmesh_fine,xmesh_fine = torch.meshgrid(zticks_fine,xticks_fine,indexing="ij")
    cpts_fine = torch.stack([xmesh_fine.flatten(),zmesh_fine.flatten()],dim=1)

    opt_scales = torch.empty((NGPFITS,1))
    opt_lengthscales = torch.empty((NGPFITS,2))
    opt_noises = torch.empty((NGPFITS,1))
    hparamnames = [
        ("loss",False),
        ("scale",True),
        ("lengthscales",True),
        ("noise",True),
    ]
    ncols = 3+len(hparamnames)
    fig,ax = pyplot.subplots(nrows=NGPFITS,ncols=ncols,figsize=(5*ncols,5*NGPFITS),sharex="col",sharey="col")
    ax = ax.reshape((NGPFITS,ncols))
    for i in range(NGPFITS):
        logvmesh_i = logv_t[i]
        print("i = %d"%i)
        fgp = fastgps.StandardGP(
            kernel(
                d = 2, 
                torchify = True, 
                device = device,
            ),
            {"x": cpts,"y":logvmesh_i.flatten()},
            noise=noise,
            requires_grad_noise=requires_grad_noise,
        )
        fit_data_fgp = fgp.fit(
            loss_metric = "MLL",
            store_hists = True,
            verbose = 25,
        )
        opt_scales[i] = fgp.kernel.base_kernel.scale.data.detach().to("cpu")
        opt_lengthscales[i] = fgp.kernel.base_kernel.lengthscales.detach().data.to("cpu")
        opt_noises[i] = fgp.noise.detach().to("cpu")
        logvhatmesh_i = fgp.post_mean(cpts).reshape(xmesh.shape).to("cpu")
        logvhatmesh_fine_i = fgp.post_mean(cpts_fine).reshape(xmesh_fine.shape).to("cpu")
        ax[i,0].imshow(torch.exp(logvmesh_i).to("cpu"),cmap=CMAP)
        ax[i,1].imshow(torch.exp(logvhatmesh_i),cmap=CMAP)
        ax[i,2].contourf(xmesh_fine.to("cpu"),zmesh_fine.to("cpu"),torch.exp(logvhatmesh_fine_i),levels=CLEVELS,cmap=CMAP)
        for j in range(3):
            ax[i,j].axis('off')
        for j,(hparam,log10ed) in enumerate(hparamnames):
            ax[i,3+j].plot(fit_data_fgp['iteration'],fit_data_fgp[hparam],linewidth=LW)
    chosen_scale = opt_scales.median(dim=0).values
    chosen_lengthscales = opt_lengthscales.median(0).values
    chosen_noise = opt_noises.median(0).values
    print("\nchosen_scale = %s"%str(chosen_scale))
    print("chosen_lengthscales = %s"%str(chosen_lengthscales))
    print("chosen_noise = %s"%str(chosen_noise))
    ax[0,0].set_title(r"$v$",fontsize=FONTSIZE)
    ax[0,1].set_title(r"GP $\hat{v}$",fontsize=FONTSIZE)
    ax[0,2].set_title(r"GP $\hat{v}$",fontsize=FONTSIZE)
    for j,(hparam,log10ed) in enumerate(hparamnames):
        ax[0,3+j].set_title(hparam,fontsize=FONTSIZE)
        if log10ed:
            ax[0,3+j].set_yscale("log",base=10)
    for i in range(NGPFITS):
        ax[i,4].axhline(y=chosen_scale,color=COLORS[0],alpha=ALPHA,linestyle='--')
        for k in range(len(chosen_lengthscales)):
            ax[i,5].axhline(y=chosen_lengthscales[k],color=COLORS[k],alpha=ALPHA,linestyle='--')
        ax[i,6].axhline(y=chosen_noise,color=COLORS[0],alpha=ALPHA,linestyle='--')
        for j in range(ncols):
            xmin,xmax = ax[i,j].get_xlim()
            ymin,ymax = ax[i,j].get_ylim()
            if j>=3 and hparamnames[j-3][1]:
                ax[i,j].set_aspect(abs((xmax-xmin)/(np.log10(ymax)-np.log10(ymin))))
            else:
                ax[i,j].set_aspect(abs((xmax-xmin)/(ymax-ymin)))
    fig.savefig("%s/RKHS_v_custom.png"%OUTDIR,dpi=256,bbox_inches="tight")
    fgp_final = fastgps.StandardGP(
        kernel(
                d = 2, 
                scale = chosen_scale.to(device),
                lengthscales = chosen_lengthscales.to(device),
                torchify = True, 
                device = device,
            ),
            {"x": cpts,"y":logvmesh_i.flatten()},
            noise=chosen_noise,
            requires_grad_noise=False,
        )
    Thetainv = fgp_final.get_inv_log_det_cache()()[0].detach().to("cpu")
    save_data_custom = { 
        "kernel": kernel.__name__,
        "Thetainv": Thetainv,
        "scale": chosen_scale,
        "lengthscales": chosen_lengthscales,
        "noise": chosen_noise,
    }
    torch.save(save_data_custom,'%s/RKHS_v_custom.pt'%OUTDIR)
    save_data_eye = { 
        "Thetainv": torch.eye(p["nx"]*p["nx"]),
    }
    torch.save(save_data_eye,'%s/RKHS_v_eye.pt'%OUTDIR)

def calc_Thetainv_w(pname, vtype, device, kernel, noise, requires_grad_noise):
    ROOT = os.path.dirname(os.path.realpath(__file__))
    OUTDIR = "%s/data/%s/%s"%(ROOT,pname,vtype)
    p = FWIPARAMETERS[pname]

    data = torch.load("%s/split_train_val.pt"%OUTDIR,weights_only=True)
    w_t = data["wavefield_train"][:NGPFITS].to(device)
    print("w_t.shape = %s"%str(tuple(w_t.shape)))

    xticks = torch.arange(p["nx"],device=device) 
    tticks = -torch.arange(p["nt"]-1,device=device) 
    tmesh,xmesh = torch.meshgrid(tticks,xticks,indexing="ij")
    cpts = torch.stack([xmesh.flatten(),tmesh.flatten()],dim=1)

    xticks_fine = torch.linspace(xticks.min(),xticks.max(),128,device=device)
    tticks_fine = torch.linspace(tticks.min(),tticks.max(),128,device=device)
    tmesh_fine,xmesh_fine = torch.meshgrid(tticks_fine,xticks_fine,indexing="ij")
    cpts_fine = torch.stack([xmesh_fine.flatten(),tmesh_fine.flatten()],dim=1)

    opt_scales = torch.empty((NGPFITS,5,1))
    opt_lengthscales = torch.empty((NGPFITS,5,2))
    opt_noises = torch.empty((NGPFITS,5,1))
    hparamnames = [
        ("loss",False),
        ("scale",True),
        ("lengthscales",True),
        ("noise",True),
    ]
    ncols = 3+len(hparamnames)
    figs,axs = [None]*5,[None]*5
    for i in range(5):
        figs[i],axs[i] = pyplot.subplots(nrows=NGPFITS,ncols=ncols,figsize=(5*ncols,5*NGPFITS),sharex="col",sharey="col")
        axs[i] = axs[i].reshape((NGPFITS,ncols))
    for i in range(NGPFITS):
        wmesh_i = w_t[i]
        print("i = %d"%i)
        fgp = fastgps.StandardGP(
            kernel(
                d = 2,
                shape_scale = [5,1],
                shape_lengthscales = [5,2],
                torchify = True, 
                device = device,
            ),
            {"x": cpts,"y":wmesh_i.flatten(start_dim=1)},
            noise = noise,
            shape_noise = [5,1],
            requires_grad_noise = requires_grad_noise,
        )
        fit_data_fgp = fgp.fit(
            loss_metric = "MLL",
            store_hists = True,
            verbose = 25,
        )
        opt_scales[i] = fgp.kernel.base_kernel.scale.data.detach().to("cpu")
        opt_lengthscales[i] = fgp.kernel.base_kernel.lengthscales.data.detach().to("cpu")
        opt_noises[i] = fgp.noise.data.detach().to("cpu")
        whatmesh_i = fgp.post_mean(cpts).reshape([5]+list(xmesh.shape)).to("cpu")
        whatmesh_fine_i = fgp.post_mean(cpts_fine).reshape([5]+list(xmesh_fine.shape)).to("cpu")
        for l in range(5):
            ax = axs[l]
            ax[i,0].imshow(wmesh_i[l].to("cpu"),cmap=CMAP)
            ax[i,1].imshow(whatmesh_i[l],cmap=CMAP)
            ax[i,2].contourf(xmesh_fine.to("cpu"),tmesh_fine.to("cpu"),whatmesh_fine_i[l],levels=CLEVELS,cmap=CMAP)
            for j in range(3):
                ax[i,j].axis('off')
            for j,(hparam,log10ed) in enumerate(hparamnames):
                ax[i,3+j].plot(fit_data_fgp['iteration'],fit_data_fgp[hparam] if hparam=="loss" else fit_data_fgp[hparam][:,l],linewidth=LW)
    chosen_scale = opt_scales.median(dim=0).values
    chosen_lengthscales = opt_lengthscales.median(0).values
    chosen_noise = opt_noises.median(dim=0).values
    print("\nchosen_scale = %s"%str(chosen_scale))
    print("chosen_lengthscales = %s"%str(chosen_lengthscales))
    print("chosen_noise = %s"%str(chosen_noise))
    for l in range(5):
        fig,ax = figs[l],axs[l]
        ax[0,0].set_title(r"$w_%d$"%l,fontsize=FONTSIZE)
        ax[0,1].set_title(r"GP $\hat{w}_%d$"%l,fontsize=FONTSIZE)
        ax[0,2].set_title(r"GP $\hat{w}_%d$"%l,fontsize=FONTSIZE)
        for j,(hparam,log10ed) in enumerate(hparamnames):
            ax[0,3+j].set_title(hparam,fontsize=FONTSIZE)
            if log10ed:
                ax[0,3+j].set_yscale("log",base=10)
        for i in range(NGPFITS):
            ax[i,4].axhline(y=chosen_scale[l],color=COLORS[0],alpha=ALPHA,linestyle='--')
            for k in range(len(chosen_lengthscales[l])):
                ax[i,5].axhline(y=chosen_lengthscales[l,k],color=COLORS[k],alpha=ALPHA,linestyle='--')
            ax[i,6].axhline(y=chosen_noise[l],color=COLORS[0],alpha=ALPHA,linestyle='--')
            for j in range(ncols):
                xmin,xmax = ax[i,j].get_xlim()
                ymin,ymax = ax[i,j].get_ylim()
                if j>=3 and hparamnames[j-3][1]:
                    ax[i,j].set_aspect(abs((xmax-xmin)/(np.log10(ymax)-np.log10(ymin))))
                else:
                    ax[i,j].set_aspect(abs((xmax-xmin)/(ymax-ymin)))
        fig.savefig("%s/RKHS_w_custom.%d.png"%(OUTDIR,l),dpi=256,bbox_inches="tight")
    fgp_final = fgp = fastgps.StandardGP(
            kernel(
                d = 2,
                scale = chosen_scale.to(device),
                lengthscales = chosen_lengthscales.to(device),
                torchify = True, 
                device = device,
            ),
            {"x": cpts,"y":wmesh_i.flatten(start_dim=1)},
            noise = chosen_noise.to(device),
            shape_noise = [5,1],
            requires_grad_noise = False,
        )
    Thetainv = fgp_final.get_inv_log_det_cache()()[0].detach().to("cpu")
    save_data_custom = { 
        "kernel": kernel.__name__,
        "Thetainv": Thetainv,
        "scale": chosen_scale,
        "lengthscales": chosen_lengthscales,
        "noise": chosen_noise,
    }
    torch.save(save_data_custom,'%s/RKHS_w_custom.pt'%OUTDIR)
    save_data_eye = { 
        "Thetainv": torch.ones((5,1,1))*torch.eye((p["nt"]-1)*p["nx"]),
    }
    torch.save(save_data_eye,'%s/RKHS_w_eye.pt'%OUTDIR)

if __name__ == "__main__":
    import time
    import os 
    import matplotlib
    from matplotlib import pyplot
    from cycler import cycler
    pyplot.style.use('seaborn-v0_8-whitegrid')
    import pandas as pd

    NGPFITS = 1
    FONTSIZE = "xx-large"
    CMAP = "gnuplot2"
    CLEVELS = 100
    LW = 3
    ALPHA = 0.5
    ROOT = os.path.dirname(os.path.realpath(__file__))
    COLORS = ["xkcd:"+color[:-1] for color in pd.read_csv(ROOT+"/../xkcd_colors.txt",comment="#",header=None).iloc[:,0].tolist()][::-1]
    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=COLORS)

    DEVICE,TORCH_BACKEND = get_torch_device_backend()
    torch.set_default_dtype(torch.float64)
    print("DEVICE = %s\n"%DEVICE) 

    pname = "RES14"
    # pname = "RES10"
    # pname = "RES7"
    # pname = "RES5"
    vtype = "Style_B"
    
    # calc_Thetainv_v(
    #     pname,
    #     vtype,
    #     DEVICE,
    #     kernel = qp.KernelMatern12,
    #     noise = 1e-8,
    #     requires_grad_noise = False,
    # )
    calc_Thetainv_w(
        pname,
        vtype,
        DEVICE,
        kernel = qp.KernelMatern12,
        noise = 1e-8,
        requires_grad_noise = False,
    )
