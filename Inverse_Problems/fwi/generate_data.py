""" check memory with du -hd 2"""

from gauss_newton_solver import fwi_gn_solver, FWIPARAMETERS
import torch 
import numpy as np 
import os
import gc

def generate_data(pname, tag_rkhs_v, tag_rkhs_w, batch_size, start_idx, end_idx, cudaidx, num_newton_iter, relaxation, wiggle_factors_relaxation, wiggle_factors_lr):
    device = "cuda:%d"%cudaidx
    data = torch.load(OUTDIR+"/split_train_val.pt",weights_only=True)
    Thetainv_v = torch.load("%s/RKHS_v_%s.pt"%(OUTDIR,"custom" if tag_rkhs_v else "eye"),weights_only=True)["Thetainv"].to(device)
    Thetainv_w = torch.load("%s/RKHS_w_%s.pt"%(OUTDIR,"custom" if tag_rkhs_w else "eye"),weights_only=True)["Thetainv"].to(device)
    vtmean = data["velocity_train"].mean(0)
    velocity = data["velocity"].clone()
    wavefield = data["wavefield"].clone()
    p = FWIPARAMETERS[pname]
    i0 = start_idx
    while i0<end_idx:
        i1 = i0+batch_size
        print("block %4d:%4d, processing realizations %4d:%4d"%(start_idx,end_idx,i0,i1),flush=True)
        v_true = velocity[i0:i1].clone().to(device)
        w = wavefield[i0:i1].to(device) 
        vhat = torch.tile(vtmean,(v_true.size(0),1,1)).clone().to(device)
        v,data = fwi_gn_solver(p,w,vhat,Thetainv_v,Thetainv_w,vref=v_true,store_Linvs=True,compute_cond_nums=True,verbose=25,num_newton_iter=num_newton_iter,relaxation=relaxation,wiggle_factors_relaxation=wiggle_factors_relaxation,wiggle_factors_lr=wiggle_factors_lr)
        data["v_true"] = v_true.cpu()
        data["w_true"] = w.cpu()
        torch.save(data,FDIR+'/nk.%d.%d.pt'%(i0,i1))
        i0 = i1

if __name__=="__main__":
    import shutil
    import sys

    outfile = open("generate_data.csv","w") # recommend the rainbow CSV VSCode extension
    sys.stdout = outfile # Uncomment to send to log file which is easier to read

    torch.set_default_dtype(torch.float64) 

    pname = "RES14"
    vtype = "Style_B"
    custom_rkhs_v = False
    custom_rkhs_w = False
    num_newton_iter = 400
    relaxation = 1e1
    tag = "NEWTRY_RELAX_%.0e"%relaxation
    wiggle_factors_relaxation = (1/2,2) 
    wiggle_factors_lr = (1/2,2)

    ROOT = os.path.dirname(os.path.realpath(__file__))
    OUTDIR = "%s/data/%s/%s/"%(ROOT,pname,vtype)
    FDIR = "%s/NK.%s.RKHS_v_%s.RKHS_w_%s"%(OUTDIR,tag,"custom" if custom_rkhs_v else "eye","custom" if custom_rkhs_w else "eye")
    if not os.path.isdir(FDIR): os.mkdir(FDIR)
    
    # generate reference data 
    R = 1000 
    # multi-GPU Newton-Kantorovich solves
    batch_size = 100
    cudaidxs = [1,2,3,4,5]
    nprocesses = len(cudaidxs)
    idxs = torch.linspace(0,R,nprocesses+1).to(int)
    torch.cuda.empty_cache()
    print("idxs = %s\n"%idxs,flush=True)
    processes = []
    if True: # multi-GPU
        for i in range(nprocesses):
            p = torch.multiprocessing.Process(
                target = generate_data,
                args = (pname,custom_rkhs_v,custom_rkhs_w,batch_size,idxs[i],idxs[i+1],cudaidxs[i],num_newton_iter,relaxation,wiggle_factors_relaxation,wiggle_factors_lr),
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else: # single GPU for debugging 
        for i in range(nprocesses):
            generate_data(pname,custom_rkhs_v,custom_rkhs_w,batch_size,idxs[i],idxs[i+1],cudaidxs[i],num_newton_iter,relaxation,wiggle_factors_relaxation,wiggle_factors_lr)
    print(flush=True)

    # aggregate outputs
    nkfiles = np.array([file for file in os.listdir(FDIR) if file[:2]=="nk" and file[-3:]==".pt"],dtype="object")
    sidxs = np.array([int(file.split(".")[1]) for file in nkfiles],dtype=int)
    nkfiles = nkfiles[sidxs.argsort()]
    data = {key:[None]*len(nkfiles) for key in list(torch.load("%s/%s"%(FDIR,nkfiles[0]),weights_only=True).keys())}
    for i,file in enumerate(nkfiles):
        print(file,flush=True)
        new_data = torch.load("%s/%s"%(FDIR,file),weights_only=True)
        for key,val in new_data.items():
            data[key][i] = val.cpu()
    print(flush=True)
    for key,val in data.items():
        if key=="times": continue 
        data[key] = torch.cat(data[key],dim=0)
    data["times"] = torch.stack(data["times"],dim=0).sum(0)/data["vs"].size(0)
    torch.save(data,FDIR+'.pt')
    shutil.rmtree(FDIR)
    print(flush=True)
    for key,val in data.items():
        print("%s: %s"%(key,str(tuple(val.shape))),flush=True)
    
    outfile.close()
