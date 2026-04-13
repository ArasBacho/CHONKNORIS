from acoustic_forward_solver import acoustic_forward_solver,acoustic_forward_solver_coarse
from util import Timer,get_torch_device_backend
import torch

def rmse_acoustic_forward_solver(v, w):
    R = v.shape[0]
    assert v.shape==(R,70,70)
    assert w.shape==(R,5,1000,70)
    what = acoustic_forward_solver(v)
    rmse = torch.sqrt(((w-what)**2).sum((-3,-2,-1)))
    assert rmse.shape==(R,)
    return rmse 

def _resid2_acoustic_forward_solver(v, w):
    assert v.shape==(70,70)
    assert w.shape==(5,1000,70)
    what = acoustic_forward_solver(v)
    resid2 = (w-what)**2
    assert resid2.shape==(5,1000,70)
    return resid2 

def _rmse_acoustic_forward_solver(v, w):
    resid2 = _resid2_acoustic_forward_solver(v,w)
    rmse = torch.sqrt(resid2.sum((-2,-1)).sum(-1,keepdim=True))
    assert rmse.shape==(1,)
    return rmse,rmse

def _rmse5_acoustic_forward_solver(v,w):
    resid2 = _resid2_acoustic_forward_solver(v,w)
    rmse = torch.sqrt(resid2.sum((-2,-1)))
    assert rmse.shape==(5,)
    return rmse,rmse

def _rmseall_acoustic_forward_solver(v,w):
    resid2 = _resid2_acoustic_forward_solver(v,w)
    rmse = torch.sqrt(resid2)
    assert rmse.shape==(5,1000,70)
    return rmse,rmse

vjac_rmse_acoustic_forward_solver = torch.vmap(torch.func.jacrev(_rmse_acoustic_forward_solver,has_aux=True))

vjac_rmse5_acoustic_forward_solver = torch.vmap(torch.func.jacrev(_rmse5_acoustic_forward_solver,has_aux=True))

vjac_rmseall_acoustic_forward_solver = torch.vmap(torch.func.jacfwd(_rmseall_acoustic_forward_solver,has_aux=True))

def rmse_acoustic_forward_solver_coarse(v, w):
    R = v.shape[0]
    assert v.shape==(R,14,14)
    assert w.shape==(R,5,190,14)
    what = acoustic_forward_solver_coarse(v)
    rmse = torch.sqrt(((w-what)**2).sum((-3,-2,-1)))
    assert rmse.shape==(R,)
    return rmse 

def _resid2_acoustic_forward_solver_coarse(v, w):
    assert v.shape==(14,14)
    assert w.shape==(5,190,14)
    what = acoustic_forward_solver_coarse(v)
    resid2 = (w-what)**2
    assert resid2.shape==(5,190,14)
    return resid2 

def _rmse_acoustic_forward_solver_coarse(v, w):
    resid2 = _resid2_acoustic_forward_solver_coarse(v,w)
    rmse = torch.sqrt(resid2.sum((-2,-1)).sum(-1,keepdim=True))
    assert rmse.shape==(1,)
    return rmse,rmse

def _rmse5_acoustic_forward_solver_coarse(v,w):
    resid2 = _resid2_acoustic_forward_solver_coarse(v,w)
    rmse = torch.sqrt(resid2.sum((-2,-1)))
    assert rmse.shape==(5,)
    return rmse,rmse

def _rmseall_acoustic_forward_solver_coarse(v,w):
    resid2 = _resid2_acoustic_forward_solver_coarse(v,w)
    rmse = torch.sqrt(resid2)
    assert rmse.shape==(5,190,14)
    return rmse,rmse

vjac_rmse_acoustic_forward_solver_coarse = torch.vmap(torch.func.jacrev(_rmse_acoustic_forward_solver_coarse,has_aux=True))

vjac_rmse5_acoustic_forward_solver_coarse = torch.vmap(torch.func.jacrev(_rmse5_acoustic_forward_solver_coarse,has_aux=True))

vjac_rmseall_acoustic_forward_solver_coarse = torch.vmap(torch.func.jacfwd(_rmseall_acoustic_forward_solver_coarse,has_aux=True))

if __name__ == "__main__":
    import time
    import os
    import numpy as np

    ROOT = os.path.dirname(os.path.realpath(__file__))
    R = 2

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(17)
    DEVICE,TORCH_BACKEND = get_torch_device_backend()
    print("DEVICE = %s\n"%str(DEVICE))
    print("TORCH_BACKEND = %s"%str(TORCH_BACKEND.__name__))
    timer = Timer(TORCH_BACKEND)

    # load data
    vel_wavefield_paths = ('FlatVel_A/model/model2.npy','FlatVel_A/data/data2.npy')
    #vel_wavefield_paths = ('CurveVel_A/model/model2.npy','CurveVel_A/data/data2.npy')
    #vel_wavefield_paths = ('CurveFault_A/vel2_1_0.npy','CurveFault_A/seis2_1_0.npy')
    velocity = torch.from_numpy(np.load(ROOT+'/train_samples/'+vel_wavefield_paths[0])).to(torch.get_default_dtype())[:,0,:,:] # [km/s] (r,x,z)
    print("velocity.shape = %s"%str(tuple(velocity.shape)))
    print("velocity.dtype = %s\n"%str(velocity.dtype))
    wavefield = torch.from_numpy(np.load(ROOT+'/train_samples/'+vel_wavefield_paths[1])).to(torch.get_default_dtype()) # (r,t,x)
    print("wavefield.shape = %s"%str(tuple(wavefield.shape)))
    print("wavefield.dtype = %s\n"%str(wavefield.dtype))

    
    # # FINE
    
    # v = velocity[:R].to(DEVICE)
    # v.requires_grad_(True) 
    # w = wavefield[:R].to(DEVICE)
    # w.requires_grad_(False) 
    
    # timer.tic()
    # rmse = rmse_acoustic_forward_solver(v,w) 
    # assert rmse.shape==(R,) 
    # print("time: %.1e"%timer.toc())

    # timer.tic()
    # vjac,rmse = vjac_rmse_acoustic_forward_solver(v,w)
    # assert vjac.shape==(R,1,70,70)
    # assert rmse.shape==(R,1)
    # print("time: %.1e"%timer.toc())

    # timer.tic()
    # vjac,rmse = vjac_rmse5_acoustic_forward_solver(v,w)
    # assert vjac.shape==(R,5,70,70)
    # assert rmse.shape==(R,5)
    # print("time: %.1e"%timer.toc())

    # # timer.tic()
    # # vjac,rmse = vjac_rmseall_acoustic_forward_solver(v,w)
    # # assert vjac.shape==(R,5,1000,70,70,70)
    # # assert rmse.shape==(R,5,1000,70)
    # # print("time: %.1e"%timer.toc())
    

    # COARSE
    
    v_coarse = velocity[:R,::5,::5].to(DEVICE)
    v_coarse.requires_grad_(True) 
    w_coarse = wavefield[:R,:,::5,::5][:,:,10:,:].to(DEVICE)
    w_coarse.requires_grad_(False) 

    # timer.tic()
    # rmse_coarse = rmse_acoustic_forward_solver_coarse(v_coarse,w_coarse) 
    # assert rmse_coarse.shape==(R,) 
    # print("\ntime: %.1e"%timer.toc())

    # timer.tic()
    # vjac_coarse,rmse_coarse = vjac_rmse_acoustic_forward_solver_coarse(v_coarse,w_coarse)
    # assert vjac_coarse.shape==(R,1,14,14)
    # assert rmse_coarse.shape==(R,1)
    # print("time: %.1e"%timer.toc())

    timer.tic()
    vjac_coarse,rmse_coarse = vjac_rmseall_acoustic_forward_solver_coarse(v_coarse,w_coarse)
    assert vjac_coarse.shape==(R,5,190,14,14,14)
    assert rmse_coarse.shape==(R,5,190,14)
    print("time: %.1e"%timer.toc())
    print(vjac_coarse.isnan().sum())
    idxs = torch.stack(torch.where(vjac_coarse.isnan()))
    print(idxs.shape)
    print(idxs[2].unique())

    pass
