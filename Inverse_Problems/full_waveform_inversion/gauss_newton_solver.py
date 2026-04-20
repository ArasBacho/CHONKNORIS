import torch 
import numpy as np
from util import Timer,get_torch_device_backend
from acoustic_forward_solver import acoustic_forward_solver, downsample_v, FWIPARAMETERS

def _F_func(logv, w, p):
    assert logv.ndim>=2 and w.ndim>=3 and w.ndim==(logv.ndim+1)
    what = acoustic_forward_solver(torch.exp(logv),p)
    F = w-what
    return F,F

F_func = lambda logv,w,p: _F_func(logv,w,p)[0]

J_F_func = torch.vmap(torch.func.jacfwd(_F_func,has_aux=True),in_dims=(0,0,None))

def _JtThetainvF_F_func(logv, w, Thetainv, p):
    assert logv.ndim==2 and w.ndim==3 and Thetainv.ndim==3
    y,vjp_fn = torch.func.vjp(lambda logv,w: F_func(logv,w,p),logv,w)
    thetainvy = torch.einsum("rij,rj->ri",Thetainv,y.flatten(start_dim=1)).reshape(y.shape)
    JtThetainvF,_ = vjp_fn(thetainvy)
    return JtThetainvF,y

JtThetainvF_F_func = torch.vmap(_JtThetainvF_F_func,in_dims=(0,0,None,None))

def fwi_gn_solver(p, w, v0, Thetainv_v, Thetainv_w,  
                  vref = None, 
                  num_newton_iter = 3, 
                  relaxation = 1e-5, 
                  lr = 1, 
                  verbose = 1, 
                  verbose_indent = 4, 
                  wiggle_factors_relaxation = (1/2,2), 
                  wiggle_factors_lr = (1/2,2), 
                  range_relaxations = (0,np.inf), 
                  range_lrs = (0,np.inf), 
                  store_Linvs = False, 
                  predict = None, 
                  print_rmse_flow = False,
                  compute_cond_nums = False,
                  relax_step = True,
                  ):
    device = v0.device
    if "cuda" in str(device):
        torch_backend = torch.cuda 
    elif "cpu" in str(device):
        torch_backend = torch.cpu
    else:
        raise Exception("unrecognized device")
    timer = Timer(torch_backend)
    timer.tic()
    assert torch.get_default_dtype()==torch.float64,"gauss newton fwi solver only works with torch.float64 precision, use torch.set_default_dtype(torch.float64)"
    assert predict is None or callable(predict) 
    if callable(predict): assert store_Linvs is False, "cannot store Linvs when predict is callable"
    import gc
    gc.collect()
    assert v0.ndim==3 and w.ndim==4
    _R,Ns,Nt,Nx = w.shape
    assert v0.shape==(_R,Nx,Nx)
    assert w.shape==(_R,Ns,Nt,Nx)
    logv = torch.log(v0.flatten(start_dim=-2).clone())
    if isinstance(vref,torch.Tensor):
        assert vref.shape==(_R,Nx,Nx)
        vref = vref.flatten(start_dim=-2)
        vref_norm = torch.linalg.norm(vref,dim=-1)
        assert vref_norm.shape==(_R,)
    if verbose:
        _strvi = " "*verbose_indent
        _vstr = _strvi+"%-15s| %-10s| %-65s|"%("iter of %-6d"%num_newton_iter,"time","RMSE")
        if isinstance(vref,torch.Tensor):
            _vstr += " %-52s|"%"L2 relative error"
        if compute_cond_nums:
            _vstr += " %-52s|"%"condition numbers"
        _vstr += " %-52s| %-52s|"%("relaxations","lrs")
        print(_vstr,flush=True)
        _vstr = _strvi+"%-15s| %-10s| %-13s%-13s%-13s%-13s%-13s|"%(" "*15," "*10,"5%","median","mean","95%","finite %")
        if isinstance(vref,torch.Tensor):
            _vstr += " %-13s%-13s%-13s%-13s|"%("5%","median","mean","95%")
        if compute_cond_nums:
            _vstr += " %-13s%-13s%-13s%-13s|"%("5%","median","mean","95%")
        _vstr += 2*(" %-13s%-13s%-13s%-13s|"%("5%","median","mean","95%"))
        print(_vstr,flush=True)
        _vstr = _strvi+"-"*(len(_vstr)-verbose_indent)
        print(_vstr,flush=True)
    F_fn = lambda logv,w: F_func(logv,w,p)
    J_F_fn = lambda logv,w: J_F_func(logv,w,p)
    JtThetainvF_F_fn = lambda logv,w,Thetainv: JtThetainvF_F_func(logv,w,Thetainv,p)
    assert Thetainv_v.device==device and Thetainv_v.shape==(Nx*Nx,Nx*Nx)
    assert Thetainv_w.device==device and Thetainv_w.shape==(5,Nt*Nx,Nt*Nx)
    eyeNxNx = torch.eye(Nx**2,device=device)
    _Rrange = torch.arange(_R,device=device)
    relaxation = relaxation*torch.ones(_R,device=device)
    lr = lr*torch.ones(_R,device=device)
    vs = torch.nan*torch.ones((_R,num_newton_iter+1,Nx,Nx))
    relaxations = torch.nan*torch.ones((_R,num_newton_iter+1))
    lrs = torch.nan*torch.ones((_R,num_newton_iter+1))
    rmses = torch.nan*torch.ones((_R,num_newton_iter+1))
    updated = torch.zeros((_R,num_newton_iter),dtype=bool)
    times = torch.nan*torch.ones(num_newton_iter+1)
    if store_Linvs:
        Linvs = torch.nan*torch.ones((_R,num_newton_iter,Nx**2,Nx**2))
    if compute_cond_nums:
        cond_nums = torch.nan*torch.ones((_R,num_newton_iter))
    if isinstance(vref,torch.Tensor):
        l2rerrors = torch.zeros((_R,num_newton_iter+1))
    for i in range(num_newton_iter+1):
        assert logv.shape==(_R,Nx*Nx)
        if i==num_newton_iter:
            F = F_fn(logv.reshape((-1,Nx,Nx)),w)
            assert F.shape==(_R,Ns,Nt,Nx)
            F = F.reshape((_R,Ns*Nt*Nx))
        elif predict is None:
            J,F = J_F_fn(logv.reshape((-1,Nx,Nx)),w)
            assert J.shape==(_R,Ns,Nt,Nx,Nx,Nx)
            J = J.reshape((_R,Ns*Nt*Nx,Nx*Nx))
            assert F.shape==(_R,Ns,Nt,Nx)
            F = F.flatten(start_dim=1)
            ThetainvF = torch.einsum("rij,krj->kri",Thetainv_w,F.reshape((_R,Ns,Nt*Nx))).flatten(start_dim=1)
            JtThetainvF = torch.einsum("rji,rj->ri",J,ThetainvF)
            assert JtThetainvF.shape==(_R,Nx*Nx)
            ThetainvJ = torch.einsum("rij,krjl->kril",Thetainv_w,J.reshape((_R,Ns,Nt*Nx,Nx*Nx))).reshape((_R,Ns*Nt*Nx,Nx*Nx))
            JtThetainvJ = torch.einsum("rkj,rkl->rjl",J,ThetainvJ)
            assert JtThetainvJ.shape==(_R,Nx*Nx,Nx*Nx)
        elif callable(predict):
            JtThetainvF,F = JtThetainvF_F_fn(logv.reshape((-1,Nx,Nx)),w,Thetainv_w)
            assert JtThetainvF.shape==(_R,Nx,Nx)
            JtThetainvF = JtThetainvF.reshape(_R,Nx*Nx)
            assert F.shape==(_R,Ns,Nt,Nx)
            F = F.reshape((_R,Ns*Nt*Nx))
        else:
            raise Exception("invalid parsing, check code")
        rmse = torch.linalg.norm(F,dim=1)
        vs[:,i] = torch.exp(logv).reshape((_R,Nx,Nx)).cpu()
        relaxations[:,i] = relaxation.cpu()
        lrs[:,i] = lr.cpu()
        rmses[:,i] = rmse.cpu()
        times[i] = timer.toc()
        timer.tic()
        if isinstance(vref,torch.Tensor):
            l2rerrors[:,i] = torch.linalg.norm(vref-torch.exp(logv),dim=-1)/vref_norm
        if verbose and (i%verbose==0 or i==num_newton_iter):
            _vstr = _strvi+"%-15d| %-10.2e| %-13.2e%-13.2e%-13.2e%-13.2e%-13.1f|"%(i,times[i]*_R,
                    torch.nanquantile(rmses[:,i],.05),torch.nanquantile(rmses[:,i],.5),torch.nanmean(rmses[:,i]),torch.nanquantile(rmses[:,i],.95),100*torch.mean(torch.isfinite(rmses[:,i]).to(torch.float)))
            if isinstance(vref,torch.Tensor):
                _vstr += " %-13.2e%-13.2e%-13.2e%-13.2e|"%(
                    torch.nanquantile(l2rerrors[:,i],.05),torch.nanquantile(l2rerrors[:,i],.5),torch.nanmean(l2rerrors[:,i]),torch.nanquantile(l2rerrors[:,i],.95))
            if compute_cond_nums:
                if i==0:
                    _vstr += " %-13.2e%-13.2e%-13.2e%-13.2e|"%(torch.nan,torch.nan,torch.nan,torch.nan)
                else:
                    _vstr += " %-13.2e%-13.2e%-13.2e%-13.2e|"%(
                        torch.nanquantile(cond_nums[:,i-1],.05),torch.nanquantile(cond_nums[:,i-1],.5),torch.nanmean(cond_nums[:,i-1]),torch.nanquantile(cond_nums[:,i-1],.95))
            _vstr += " %-13.2e%-13.2e%-13.2e%-13.2e|"%(
                torch.nanquantile(relaxations[:,i],.05),torch.nanquantile(relaxations[:,i],.5),torch.nanmean(relaxations[:,i]),torch.nanquantile(relaxations[:,i],.95))
            _vstr += " %-13.2e%-13.2e%-13.2e%-13.2e|"%(
                torch.nanquantile(lrs[:,i],.05),torch.nanquantile(lrs[:,i],.5),torch.nanmean(lrs[:,i]),torch.nanquantile(lrs[:,i],.95))
            print(_vstr,flush=True)
        if i==num_newton_iter: break
        if not relax_step:
            Thetainv_v_logv = torch.einsum("ij,rj->ri",Thetainv_v,logv)
        def try_relax_lr(relaxation_try, lr_try):
            if relax_step:
                rhs = JtThetainvF
            else:
                rhs = JtThetainvF+relaxation_try[:,None]*Thetainv_v_logv
            logv_try = torch.empty((_R,Nx*Nx),device=device)
            rmse_try = torch.empty(_R,device=device)
            if predict is None: # here L is the Cholesky factor
                L,fails = torch.linalg.cholesky_ex(JtThetainvJ+relaxation_try[:,None,None]*Thetainv_v,upper=False)
                success = fails==0 
                logdelta_try = torch.cholesky_solve(rhs[success,:,None],L[success],upper=False)[...,0]
            else: # here L is the inverse Cholesky factor
                logdelta_try,success = predict(relaxation_try,torch.exp(logv),w,rhs)
                assert logdelta_try.shape==(success.sum(),Nx*Nx)
            logv_try[success] = logv[success]-lr_try[success,None]*logdelta_try
            F_try = F_fn(logv_try[success].reshape((-1,Nx,Nx)),w[success])
            rmse_try[success] = torch.linalg.norm(F_try.flatten(start_dim=1),dim=1)
            if print_rmse_flow:
                with np.printoptions(formatter={"float":lambda x: "%-10.2e"%x}):
                    print(" "*verbose_indent+"\trelax *%-10.2flr *%-10.2frmse_try %s"%(relaxation_try[0]/relaxation[0],lr_try[0]/lr[0],str(rmse_try.cpu().numpy())),flush=True)
            improved_rmse = rmse_try[success]<=rmse[success]
            success[success.clone()] = improved_rmse
            if predict is None:
                L[~success] = torch.nan
            else:
                L = torch.empty(0)
            logv_try[~success] = torch.nan
            rmse_try[~success] = torch.inf
            return L,logv_try,rmse_try
        relaxation_low = torch.maximum(relaxation*wiggle_factors_relaxation[0],torch.tensor(range_relaxations[0]))
        relaxation_high = torch.minimum(relaxation*wiggle_factors_relaxation[1],torch.tensor(range_relaxations[1]))
        relaxation_options = torch.stack([relaxation_low,relaxation,relaxation_high],dim=0)
        lr_low = torch.maximum(lr*wiggle_factors_lr[0],torch.tensor(range_lrs[0]))
        lr_high = torch.minimum(lr*wiggle_factors_lr[1],torch.tensor(range_lrs[1]))
        lr_options = torch.stack([lr_high,lr,lr_low],dim=0)
        _i0,_i1 = torch.meshgrid(torch.arange(len(relaxation_options),device=device),torch.arange(len(lr_options),device=device),indexing="ij")
        relaxation_options = relaxation_options[_i0.flatten()]
        lr_options = lr_options[_i1.flatten()]
        L_try,logv_try,rmse_try = [None]*len(relaxation_options),[None]*len(relaxation_options),[None]*len(relaxation_options)
        for j in range(len(relaxation_options)):
            L_try[j],logv_try[j],rmse_try[j] = try_relax_lr(relaxation_options[j],lr_options[j])
        L_try,logv_try,rmse_try = torch.stack(L_try,dim=0),torch.stack(logv_try,dim=0),torch.stack(rmse_try,dim=0)
        best_idxs = rmse_try.argmin(0)
        best_logv = logv_try[best_idxs,_Rrange]
        best_rmse = rmse_try[best_idxs,_Rrange]
        best_relaxation = relaxation_options[best_idxs,_Rrange]
        best_lr = lr_options[best_idxs,_Rrange]
        improved_rmse = best_rmse<rmse
        if print_rmse_flow:
            with np.printoptions(formatter={"float":lambda x: "%-10.2e"%x,"bool":lambda x: "%-10s"%x}):
                print(" "*verbose_indent+"\t"+" "*27+"current_rmse %s"%(rmse.cpu().numpy()),flush=True)
                print(" "*verbose_indent+"\t"+" "*30+"best_rmse %s"%(best_rmse.cpu().numpy()),flush=True)
                print(" "*verbose_indent+"\t"+" "*26+"improved_rmse %s"%(improved_rmse.cpu().numpy()),flush=True)
            print(flush=True)
        logv[improved_rmse] = best_logv[improved_rmse]
        rmse[improved_rmse] = best_rmse[improved_rmse]
        relaxation[improved_rmse] = best_relaxation[improved_rmse]
        lr[improved_rmse] = best_lr[improved_rmse]
        relaxation[~improved_rmse] = wiggle_factors_relaxation[1]*relaxation[~improved_rmse]
        lr[~improved_rmse] = lr[~improved_rmse]*wiggle_factors_lr[0]
        if store_Linvs:
            best_L = L_try[best_idxs,_Rrange]
            Linvs[improved_rmse,i,:,:] = torch.linalg.solve_triangular(best_L[improved_rmse],eyeNxNx,upper=False).cpu()
        if compute_cond_nums:
            best_L = L_try[best_idxs,_Rrange]
            best_Theta_improved = torch.einsum("rij,rkj->rik",best_L[improved_rmse],best_L[improved_rmse])
            cond_nums[improved_rmse,i] = torch.linalg.cond(best_Theta_improved).cpu()
        updated[:,i] = improved_rmse.cpu()
    logv = logv.detach()
    data = {
        "vs": vs,
        "rmses": rmses,
        "times": times,
        "relaxations": relaxations,
        "lrs": lrs,
        "updated": updated, 
    }
    if store_Linvs:
        data["Linvs"] = Linvs
    if compute_cond_nums:
        data["cond_nums"] = cond_nums
    if isinstance(vref,torch.Tensor):
        data["l2rerrors"] = l2rerrors
    return torch.exp(logv).reshape((_R,Nx,Nx)),data

if __name__ == "__main__":
    import time
    import os
    import numpy as np
    import sys 

    outfile = open("gauss_newton_solver.csv","w") # recommend the rainbow CSV VSCode extension
    sys.stdout = outfile # Uncomment to send to log file which is easier to read

    pname = "RES5"
    vtype = "Style_B"
    custom_rkhs_v = False
    custom_rkhs_w = False
    R = 10
    ROOT = os.path.dirname(os.path.realpath(__file__))
    OUTDIR = "%s/data/%s/%s"%(ROOT,pname,vtype)
    torch.set_default_dtype(torch.float64)
    DEVICE,TORCH_BACKEND = get_torch_device_backend()
    print("DEVICE = %s\n"%str(DEVICE),flush=True)
    timer = Timer(TORCH_BACKEND)

    p = FWIPARAMETERS[pname]
    data = torch.load("%s/split_train_val.pt"%OUTDIR,weights_only=True)
    v_all = data["velocity_train"]
    w_all = data["wavefield_train"]
    v = v_all[:R].to(DEVICE)
    w = w_all[:R].to(DEVICE)
    Thetainv_v = torch.load("%s/RKHS_v_%s.pt"%(OUTDIR,"custom" if custom_rkhs_v else "eye"),weights_only=True)["Thetainv"].to(DEVICE)
    Thetainv_w = torch.load("%s/RKHS_w_%s.pt"%(OUTDIR,"custom" if custom_rkhs_w else "eye"),weights_only=True)["Thetainv"].to(DEVICE)

    # F_func
    timer.tic()
    y = F_func(torch.log(v),w,p) 
    print("y.shape = %s"%str(tuple(y.shape)),flush=True)
    print("time: %.1e\n"%timer.toc(),flush=True)
    # J_F_func
    timer.tic()
    J,_y = J_F_func(torch.log(v),w,p)
    print("J.shape = %s"%str(tuple(J.shape)),flush=True)
    print("time: %.1e\n"%timer.toc(),flush=True)
    assert not J.isnan().any()
    assert torch.allclose(_y,y)
    # JtF_F_func
    timer.tic()
    JtthetainvF,_y = JtThetainvF_F_func(torch.log(v),w,Thetainv_w,p)
    print("JtthetainvF.shape = %s"%str(tuple(JtthetainvF.shape)),flush=True)
    print("time: %.1e\n"%timer.toc(),flush=True)
    thetainvy = torch.einsum("rij,krj->kri",Thetainv_w,y.flatten(start_dim=2)).reshape(y.shape)
    assert torch.allclose(_y,y)
    assert torch.allclose(JtthetainvF,(J*thetainvy[...,None,None]).sum((1,2,3)))
    # fwi_gn_solver
    print("Gauss-Newton Solver",flush=True)
    vref = v.clone()
    v0 = torch.tile(v_all.mean(0,keepdim=True),(R,1,1)).to(DEVICE)
    print("here")
    vhat,data_gn = fwi_gn_solver(p,w,v0,Thetainv_v,Thetainv_w,vref=vref,compute_cond_nums=False,
                              num_newton_iter = 10,
                              verbose = 1,
                               relaxation = 1e1,
                            #  relaxation = 5e-5,
                            #    relaxation = 1e-10,
                            #   relax_step = False,
                            #     # wiggle_factors_relaxation = (1/2,2),
                            #     wiggle_factors_relaxation = (2,1/2),
                            # #   wiggle_factors_relaxation = (1,1/2),
                            # #   wiggle_factors_relaxation = (1,1),
                            # #   wiggle_factors_lr = (1,1),
                            # range_relaxations = (1e-3,np.inf),
                              )
    
    outfile.close()