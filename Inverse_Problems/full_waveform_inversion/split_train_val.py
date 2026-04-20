from gauss_newton_solver import FWIPARAMETERS, downsample_v
from acoustic_forward_solver import acoustic_forward_solver
import torch 
import numpy as np 
import os 

def split_train_val(pname, vtype, device, force=False):
    ROOT = os.path.dirname(os.path.realpath(__file__))
    OUTDIR = "%s/data/%s/%s"%(ROOT,pname,vtype)
    if not os.path.isdir(OUTDIR): os.makedirs(OUTDIR)
    outfile = OUTDIR+"/split_train_val.pt"
    if not os.path.isfile(outfile) or force:
        p = FWIPARAMETERS[pname]
        velocity_full = torch.cat([torch.from_numpy(np.load('train_samples/%s/model/model%d.npy'%(vtype,i))) for i in range(1,3)],axis=0).to(torch.get_default_dtype())[:,0,:,:] # [km/s] (r,x,z)
        velocity = downsample_v(velocity_full,p)
        print("velocity.shape = %s"%str(tuple(velocity.shape)))
        print("velocity.dtype = %s"%str(velocity.dtype))
        wavefield = acoustic_forward_solver(velocity.to(device),p).to("cpu")
        print("wavefield.shape = %s"%str(tuple(wavefield.shape)))
        print("wavefield.dtype = %s"%str(wavefield.dtype))
        R = velocity.size(0)
        R_val = R//5 
        R_train = R-R_val
        _order = torch.randperm(R,generator=torch.Generator().manual_seed(7))
        tidxs,vidxs = _order[:R_train],_order[R_train:]
        velocity_train,velocity_val = velocity[tidxs],velocity[vidxs]
        print("velocity_train.shape = %s"%str(tuple(velocity_train.shape)))
        print("velocity_val.shape = %s"%str(tuple(velocity_val.shape)))
        wavefield_train,wavefield_val = wavefield[tidxs],wavefield[vidxs]
        print("wavefield_train.shape = %s"%str(tuple(wavefield_train.shape)))
        print("wavefield_val.shape = %s"%str(tuple(wavefield_val.shape)))
        data = {
            "velocity": velocity,
            "wavefield": wavefield,
            "velocity_train": velocity_train,
            "velocity_val": velocity_val,
            "wavefield_train": wavefield_train,
            "wavefield_val": wavefield_val,
            "tidxs": tidxs,
            "vidxs": vidxs,
            }
        torch.save(data,outfile)
    else:
        data = torch.load(outfile,weights_only=True)
    for key,val in data.items():
        print("%s: %s"%(key,str(tuple(val.shape))))
    R = data["velocity"].size(0)
    assert R==1000

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64) 

    device = "cuda:3"

    pnames = [
        # "RES70",
        #"RES14",
        # "RES10",
        # "RES7",
        "RES5",
    ]
    vtype = "Style_B"
    # generate reference data
    for pname in pnames:
        print(pname)
        print("~"*50)
        split_train_val(pname,vtype,device,force=False)
        print()