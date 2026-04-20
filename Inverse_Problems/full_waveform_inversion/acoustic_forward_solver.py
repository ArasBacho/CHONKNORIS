"""
Translation of FD Acoustic Modeling Lab by Xin Wang from 
    
    https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html

Original author's information copied below 

    Copyright (C) 2010 Center for Subsurface Imaging and Fluid Modeling (CSIM),
    King Abdullah University of Science and Technology, All rights reserved.

    author:   Xin Wang
    email:    xin.wang@kaust.edu.sa
    date:     Sep 26, 2012
    purpose:  2DTDFD solution to acoustic wave equation with accuracy of 2-8 use the absorbing boundary condition

Debugging command from https://docs.pytorch.org/docs/stable/bottleneck.html.
Make sure to set PLOT=False for debugging. 

    python -m torch.utils.bottleneck fwi/acoustic_forward_solver.py
"""

import torch
import numpy as np 
from util import get_torch_device_backend,Timer

def ricker(f, dt, device=None):
    """
    Ricker wavelet of centralfrequency f.
    
    Args:
        f (float): central frequency in Hz with f << 1/(2dt).
        dt (float) sampling interval in sec.
    
    Returns:
        w (torch.Tensor): the Ricker wavelet
    """
    device = torch.device(device) if device is not None else torch.get_default_device()
    nw = 2.2/f/dt
    nw = int(2*np.floor(nw/2)+1)
    nc = int(np.floor(nw/2))
    k = torch.arange(1,nw+1,device=device)
    alpha = (nc-k+1)*f*dt*np.pi
    beta = alpha**2
    w = (1-2*beta)*torch.exp(-beta)
    return w

def _AbcCoef2D(vel, nbc, dx):
    """
    Artifician Boundary Condition (ABC) coefficients in 2 dimensions 
    
    Args:
      (torch.Tensor): velocity map with shape (nzbc, nxbc). 
        nbc (int): number of boundary layers 
        dx (float): mesh width in both x and z
    
    Returns:
        damp (torch.Tensor): dampening tensor with shape (nzbc, nxbc)
    """
    device = vel.device
    nzbc,nxbc = vel.shape[-2:]
    velmin = vel.amin((-2,-1),keepdim=True)
    #nz = nzbc-2*nbc
    #nx = nxbc-2*nbc
    a = (nbc-1)*dx
    kappa = 3*velmin*np.log(10000000)/(2*a)
    # setup 1D BC damping array
    damp1d = kappa[...,0]*(torch.arange(nbc,device=device)*dx/a)**2
    damp1d_flip = torch.flip(damp1d,dims=(-1,))
    # setup 2D BC damping array
    #damp = torch.zeros(vel.shape[:-2]+(nzbc,nxbc),device=device)
    # divide the whole area to 9 zones, and 5th is the target zone
    #  1   |   2   |   3
    #  ------------------
    #  4   |   5   |   6
    #  ------------------
    #  7   |   8   |   9
    b1s = (1,)*(vel.ndim-2)
    # fill zone 1, 4, 7 and 3, 6, 9
    zones_147 = damp1d_flip[...,None,:].repeat(b1s+(nzbc,1))
    zones_369 = damp1d[...,None,:].repeat(b1s+(nzbc,1))
    #damp[...,:,:nbc] = damp1d_flip[...,None,:]
    #damp[...,:,(nx+nbc):(nx+2*nbc)] = damp1d[...,None,:]
    # fill zone 2 and 8
    zone_2 = damp1d_flip[...,:,None].repeat(b1s+(1,nxbc-2*nbc))
    zone_8 = damp1d[...,:,None].repeat(b1s+(1,nxbc-2*nbc))
    #damp[...,:nbc,nbc:(nbc+nx)] = damp1d_flip[...,:,None]
    #damp[...,(nbc+nz):(nz+2*nbc),nbc:(nbc+nx)] = damp1d[...,:,None]
    zone_5 = torch.zeros(vel.shape[:-2]+(nzbc-2*nbc,nxbc-2*nbc),device=device)
    zones_258 = torch.cat([zone_2,zone_5,zone_8],dim=-2)
    damp = torch.cat([zones_147,zones_258,zones_369],dim=-1)
    return damp

def _adjust_sr(coord_sx, coord_sz, coord_gx, coord_gz, dx, nbc):
    """ set and adjust the free surface position """
    isx = torch.round(coord_sx/dx).to(int)+nbc
    isz = torch.round(coord_sz/dx).to(int)+nbc
    igx = torch.round(coord_gx/dx).to(int)+nbc
    igz = torch.round(coord_gz/dx).to(int)+nbc
    isz = isz+(torch.abs(coord_sz)<0.5)*1
    igz = igz+(torch.abs(coord_gz)<0.5)*1
    return isx,isz,igx,igz

def _expand_source(s0, nt):
    nt0 = s0.numel()
    if nt0<nt:
        s = torch.zeros(nt)
        s[:nt0] = s0
    else:
        s = s0
    return s

def _padvel(v0, nbc):
    onespad = (1,)*(v0.ndim-2)
    v = torch.cat([v0[...,:,0,None].repeat(onespad+(1,nbc)),v0,v0[...,:,-1,None].repeat(onespad+(1,nbc))],-1)
    v = torch.cat([v[...,None,0,:].repeat(onespad+(nbc,1)),v,v[...,None,-1,:].repeat(onespad+(nbc,1))],-2)
    return v

def a2d_mod_abc24(v, nbc, dx, nt, dt, s, coord_sx, coord_sz, coord_gx, coord_gz, isFS):
    """
    Acoustic 2 dimensional model with Artificial Boundary Conditions (ABC) 
    with 2nd order finite difference in time and 4th order finite difference in space. 

    Args:
        v (torch.Tensor): velocity map
        nbc (int): number of boundary coefficients 
        dx (float): mesh width for both x and z dimensions 
        nt (int): number of time steps 
        dt (float): mesh width in time 
        s (torch.Tensor): source e.g. a Ricker wavelet 
        coord_sx (torch.Tensor): x coordinates of source locations
        coord_sz (torch.Tensor): z coordinates of source locations
        coord_gx (torch.Tensor): coordinate grid of x values
        coord_gz (torch.Tensor): coordinate grid of z values 
        isFS (bool): if True, use the free surface condition 
    
    Returns: 
        seis (torch.Tensor): seismogram
    """
    assert isinstance(s,torch.Tensor) and s.ndim==1
    assert isinstance(coord_sx,torch.Tensor) and coord_sx.ndim==1 
    assert isinstance(coord_sz,torch.Tensor) and coord_sz.ndim==1
    ns = len(coord_sx)
    assert coord_sx.shape==coord_sz.shape==(ns,)
    assert isinstance(coord_gx,torch.Tensor) and coord_gx.ndim==1 
    assert isinstance(coord_gz,torch.Tensor) and coord_gz.ndim==1 
    ng = coord_gx.numel()
    assert coord_gx.shape==coord_gz.shape==(ng,)
    assert isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[-2:]==(ng,ng)
    device = v.device
    seiss = [None]*nt#torch.zeros(v.shape[:-2]+(ns,)+(nt,ng),device=device)
    c1 = -2.5
    c2 = 4.0/3.0
    c3 = -1.0/12.0
    # setup ABC and temperary variables
    v = _padvel(v,nbc)
    abc = _AbcCoef2D(v,nbc,dx)
    alpha = (v*dt/dx)**2
    kappa = abc*dt
    temp1 = 2+2*c1*alpha-kappa
    temp2 = 1-kappa
    beta_dt = (v*dt)**2
    s = _expand_source(s,nt)
    isx,isz,igx,igz = _adjust_sr(coord_sx,coord_sz,coord_gx,coord_gz,dx,nbc)
    p1 = torch.zeros(v.shape[:-2]+(ns,)+v.shape[-2:],device=device)
    p0 = torch.zeros(v.shape[:-2]+(ns,)+v.shape[-2:],device=device)
    # Time Looping
    ins = torch.arange(ns,device=device)
    #circshift = lambda x,shifts: torch.roll(x,shifts,dims=(-2,-1))
    _idx0 = torch.arange(nbc-1,nbc-3,-1,device=device)
    _idx1 = torch.arange(nbc+1,nbc+3,device=device)
    r0 = torch.arange(p1.size(-2),device=device)
    r0_n2 = torch.roll(r0,-2)
    r0_n1 = torch.roll(r0,-1)
    r0_1 = torch.roll(r0,1) 
    r0_2 = torch.roll(r0,2)
    r1 = torch.arange(p1.size(-1),device=device) 
    r1_n2 = torch.roll(r1,-2)
    r1_n1 = torch.roll(r1,-1)
    r1_1 = torch.roll(r1,1) 
    r1_2 = torch.roll(r1,2)
    for it in range(nt):
        # p = temp1[...,None,:,:]*p1-temp2[...,None,:,:]*p0+alpha[...,None,:,:]*\
        #     (c2*(circshift(p1,[0,1])+circshift(p1,[0,-1])+circshift(p1,[1,0])+circshift(p1,[-1,0]))\
        #     +c3*(circshift(p1,[0,2])+circshift(p1,[0,-2])+circshift(p1,[2,0])+circshift(p1,[-2,0])))
        p = temp1[...,None,:,:]*p1-temp2[...,None,:,:]*p0+alpha[...,None,:,:]*\
            (c2*(p1[...,:,r1_1]+p1[...,:,r1_n1]+p1[...,r0_1,:]+p1[...,r0_n1,:])\
            +c3*(p1[...,:,r1_2]+p1[...,:,r1_n2]+p1[...,r0_2,:]+p1[...,r0_n2,:]))
        p[...,ins,isz,isx] = p[...,ins,isz,isx]+beta_dt[...,isz,isx]*s[it]
        if isFS:
            p[...,:,nbc,:] = 0.0
            p[...,:,_idx0,:] = -p[...,:,_idx1,:]
        #seis[...,:,it,:] = p[...,:,igz,igx]
        seiss[it] = p[...,:,igz,igx]#.clone()
        p0 = p1
        p1 = p
    seis = torch.stack(seiss,dim=-2)
    return seis

FWIPARAMETERS = {
    "RES70": {
        "name": "RES70",
        "nx": 70,
        "dx": 10,
        "nbc": 120,
        "nt": 1001,
        "dt": 0.001,
        "freq": 15,
        "coord_sx": [10,175,350,530,700],
        "coord_sz": [10,10,10,10,10],},
    "RES14": {
        "name": "RES14",
        "nx": 70//5,
        "dx": 10,
        "nbc": 120//5,
        "nt": 1001//5+1,
        "dt": 0.001,
        "freq": 25,
        "coord_sx": [10//5,175//5,350//5,530//5,700//5],
        "coord_sz": [2,2,2,2,2],},
    "RES10": {
        "name": "RES10",
        "nx": 70//7,
        "dx": 10,
        "nbc": 120//7,
        "nt": 1001//7+1,
        "dt": 0.001,
        "freq": 35,
        "coord_sx": [10/7,175/7,350/7,530/7,700/7],
        "coord_sz": [1,1,1,1,1],},
    "RES7": {
        "name": "RES7",
        "nx": 70//10,
        "dx": 10,
        "nbc": 120//10,
        "nt": 1001//10+1,
        "dt": 0.001,
        "freq": 35,
        "coord_sx": [10/10,175/10,350/10,530/10,700/10],
        "coord_sz": [1,1,1,1,1],},
    "RES5": {
        "name": "RES5",
        "nx": 70//14,
        "dx": 10,
        "nbc": 120//14,
        "nt": 1001//14+1,
        "dt": 0.001,
        "freq": 35,
        "coord_sx": [10/14,175/14,350/14,530/14,700/14],
        "coord_sz": [10/14,10/14,10/14,10/14,10/14],},
}

def acoustic_forward_solver(v, p):
    """ 
    See Section 3: Seismic Forward Modeling Details of 
    
        Deng, Chengyuan, et al. 
        "OpenFWI: Large-scale multi-structural benchmark datasets for full waveform inversion." 
        Advances in Neural Information Processing Systems 35 (2022): 6007-6020.
        https://proceedings.neurips.cc/paper_files/paper/2022/file/27d3ef263c7cb8d542c4f9815a49b69b-Supplemental-Datasets_and_Benchmarks.pdf
    """
    device = v.device
    assert isinstance(v,torch.Tensor) and v.ndim>=2
    assert v.shape[-2:]==(p["nx"],p["nx"])
    seis_24 = a2d_mod_abc24(
        v = v.to(device),
        nbc = p["nbc"],
        dx = p["dx"],
        nt = p["nt"],
        dt = p["dt"],
        s = ricker(p["freq"],p["dt"],device=device),
        coord_sx = torch.tensor(p["coord_sx"],device=device),
        coord_sz = torch.tensor(p["coord_sz"],device=device),
        coord_gx = torch.arange(1,p["nx"]+1,device=device)*p["dx"],
        coord_gz = torch.ones(p["nx"],device=device)*p["dx"],
        isFS = False
    )
    assert seis_24.shape==(v.shape[:-2]+(len(p["coord_sx"]),p["nt"],p["nx"]))
    seis_24 = seis_24[...,:-1,:]
    return seis_24

def downsample_v(v, p):
    assert (70/p["nx"])%1==0, "p[nx] = %d does not evenly divide 70"
    by_x = 70//p["nx"]
    v_downsampled = v[...,::by_x,::by_x]
    return v_downsampled

if __name__ == "__main__":
    import time
    import os 

    PLOT = True
    RIDXS = [i for i in range(2)]
    DEVICE,TORCH_BACKEND = get_torch_device_backend()
    torch.set_default_dtype(torch.float64)
    ROOT = os.path.dirname(os.path.realpath(__file__))
    timer = Timer(TORCH_BACKEND)

    # load data
    # vtype = "FlatVel_A"
    # vtype = "CurveVel_A"
    # vtype = "CurveFault_A"`
    # vtype = "Style_A"
    vtype = "Style_B"
    vel_wavefield_paths = ('%s/model/model2.npy'%vtype,'%s/data/data2.npy'%vtype)
    velocity = torch.from_numpy(np.load(ROOT+'/train_samples/'+vel_wavefield_paths[0])).to(torch.get_default_dtype())[:,0,:,:] # [km/s] (r,x,z)
    print("velocity.shape = %s"%str(tuple(velocity.shape)))
    print("velocity.dtype = %s\n"%str(velocity.dtype))
    wavefield = torch.from_numpy(np.load(ROOT+'/train_samples/'+vel_wavefield_paths[1])).to(torch.get_default_dtype()) # (r,t,x)
    print("wavefield.shape = %s"%str(tuple(wavefield.shape)))
    print("wavefield.dtype = %s\n"%str(wavefield.dtype))
    
    pnames = [
        # "RES70",
        # "RES14",
        # "RES10",
        # "RES7",
        "RES5",
    ]
    for pname in pnames:
        p = FWIPARAMETERS[pname]
        print("pname = %s"%p["name"]) 
        v_coarse = downsample_v(velocity[RIDXS],p)
        print("\tv_coarse.shape = %s"%str(tuple(v_coarse.shape)))
        print("\tv_coarse.dtype = %s"%str(v_coarse.dtype))
        timer.tic()
        what_coarse = acoustic_forward_solver(v_coarse.to(DEVICE),p=p).cpu()
        print("\twhat_coarse.shape = %s"%str(tuple(what_coarse.shape)))
        print("\twhat_coarse.dtype = %s"%str(what_coarse.dtype))
        print("\tforward map time: %.1e"%timer.toc())
        if p["name"] in ["RES70"]:
            what = what_coarse
            w = wavefield[RIDXS,:]
            error = torch.abs(what-w)
            print("\tMAE = %.1e"%error.mean())
            print("\tRMSE = %.1e"%torch.sqrt((error**2).mean()))
            if PLOT:
                from matplotlib import pyplot 
                pyplot.style.use('seaborn-v0_8-whitegrid')
                nrows = 3 
                ncols = what_coarse.size(-3)
                fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(5*ncols,5*nrows))
                ax = ax.reshape((nrows,ncols))
                for i,(name,_data) in enumerate(zip(["what","w","error"],[what[0],w[0],error[0]])):
                    for j in range(ncols):
                        cax = ax[i,j].imshow(_data[j],cmap="gnuplot2")
                        ax[i,j].set_aspect(_data[j].size(1)/_data[j].size(0))
                        fig.colorbar(cax)
                        ax[0,j].set_title(r"source index %d"%j,fontsize="xx-large")
                    ax[i,0].set_ylabel(name,fontsize="xx-large")
        else:
            if PLOT:
                from matplotlib import pyplot 
                pyplot.style.use('seaborn-v0_8-whitegrid')
                nrows = 1
                ncols = what_coarse.size(-3)
                fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(5*ncols,5*nrows))
                ax = ax.reshape((nrows,ncols))
                for i,(name,_data) in enumerate(zip(["what"],[what_coarse[0]])):
                    for j in range(ncols):
                        cax = ax[i,j].imshow(_data[j],cmap="gnuplot2")
                        ax[i,j].set_aspect(_data[j].size(1)/_data[j].size(0))
                        fig.colorbar(cax)
                        ax[0,j].set_title(r"source index %d"%j,fontsize="xx-large")
                    ax[i,0].set_ylabel(name,fontsize="xx-large")
        if PLOT:
            fig.savefig(ROOT+"/out/afs.%s.%s.png"%(p["name"],vtype),bbox_inches="tight")
