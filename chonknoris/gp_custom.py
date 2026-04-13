import torch 
import gpytorch 
from .util import _LightingBase,train_val_split
from .datasets import DatasetClassic
import numpy as np 
import warnings
import lightning

class IndepVecGPSharedCustom(torch.nn.Module):
    """
    >>> gp = IndepVecGPSharedCustom(d_in=4,d_out=1)
    >>> gp.fit(x=torch.rand(3,4),y=torch.rand(3))
    >>> gp.forward(torch.rand(10,4)).shape
    torch.Size([10])

    >>> gp = IndepVecGPSharedCustom(d_in=3,d_out=2)
    >>> gp.fit(x=torch.rand(4,3),y=torch.rand(4,2))
    >>> gp.forward(torch.rand(10,3)).shape
    torch.Size([10, 2])

    >>> gp = IndepVecGPSharedCustom(d_in=3,d_out=2,
    ...     mean_module = gpytorch.means.LinearMean(input_size=3,batch_shape=(2,)),
    ...     covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=3,batch_shape=(2,)),batch_shape=(2,)))
    >>> gp.fit(x=torch.rand(4,3),y=torch.rand(4,2))
    >>> gp.forward(torch.rand(10,3)).shape
    torch.Size([10, 2])
    """
    def __init__(self, d_in, d_out, mean_module=None, covar_module=None, fixed_noise=True, noise_lb=1e-6):
        super().__init__()
        assert isinstance(d_in,int) and isinstance(d_out,int) 
        assert fixed_noise, "IndepVecGPSharedCustom currently only supports fixed_noise"
        self.d_in = d_in
        self.d_out = d_out
        self.noise = noise_lb
        # self.mean_module = mean_module if mean_module is not None else gpytorch.means.LinearMean(input_size=self.d_in)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ZeroMean(input_size=self.d_in)
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=3/2,ard_num_dims=self.d_in))
    def mean_func(self, x):
        mu = self.mean_module(x)
        assert mu.ndim==1 or mu.ndim==2
        if mu.ndim==1: mu = mu[None,:]
        assert mu.size(0) in [1,self.d_out]
        return mu
    def covar_func(self, x1, x2):
        kmat = self.covar_module(x1,x2).to_dense() 
        assert kmat.ndim==2 or kmat.ndim==3
        if kmat.ndim==2: kmat = kmat[None,:,:]
        assert kmat.size(0) in [1,self.d_out]
        return kmat
    def fit(self, x, y):
        self.ydim = y.ndim
        assert self.ydim==1 or self.ydim==2
        if y.ndim==1: y = y[:,None]
        y = y.T 
        n = x.size(0)
        assert x.ndim==2 and y.ndim==2 and x.size(1)==self.d_in and y.size(0)==self.d_out and y.size(1)==n
        # x is n x d_in and y is d_out x n
        mu = self.mean_func(x) # d_out x n 
        kmat = self.covar_func(x,x) # d_out x n x n
        nrange = torch.arange(n)
        kmat[:,nrange,nrange] = kmat[:,nrange,nrange]+self.noise
        L = torch.linalg.cholesky(kmat)
        self.coeffs = torch.cholesky_solve((y-mu)[...,None],L)[:,:,0] # d_out x n
        self.x = x
    def forward(self, xnew):
        assert xnew.ndim==2 and xnew.size(1)==self.d_in
        munew = self.mean_func(xnew) # d_out x nnew
        if not hasattr(self,"coeffs"):
            yhat = munew.T
        else:
            knew = self.covar_func(xnew,self.x) # d_out x nnew x n
            delta = torch.einsum("ijk,ik->ij",knew,self.coeffs) # d_out x nnew
            yhat = munew+delta
            yhat = yhat.T # nnew x d_out
            if self.ydim==1: yhat = yhat[:,0]
        return yhat

class LightningGPCustom(_LightingBase):
    """
    >>> x = torch.rand(16,2)
    >>> y = torch.rand(16)
    >>> (xt,xv,yt,yv),vidx = train_val_split(x,y,val_frac=1/4)
    >>> dataset_t = DatasetClassic(x=xt,y=yt)
    >>> dataset_v = DatasetClassic(x=xv,y=yv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> gp = IndepVecGPSharedCustom(d_in=2,d_out=1)
    >>> lgp = LightningGPCustom(gp)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lgp,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)
    
    >>> x = torch.rand(16,2)
    >>> y = torch.rand(16,3)
    >>> (xt,xv,yt,yv),vidx = train_val_split(x,y,val_frac=1/4)
    >>> dataset_t = DatasetClassic(x=xt,y=yt)
    >>> dataset_v = DatasetClassic(x=xv,y=yv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> gp = IndepVecGPSharedCustom(d_in=2,d_out=3)
    >>> lgp = LightningGPCustom(gp)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lgp,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)
    """ 
    DEFAULT_LR = 1.
    def __init__(self, gp, folds=4, rng_seed=None, use_l2rerror_loss=True, **super_kwargs):
        super().__init__(**super_kwargs)
        self.gp = gp
        self.folds = folds
        self.rng_seed = rng_seed
        self.use_l2rerror_loss = use_l2rerror_loss
        assert isinstance(self.gp,IndepVecGPSharedCustom)
    def forward(self, x):
        return self.gp.forward(x)
    def _common_step(self, batch, tag):
        assert len(batch)==2
        x = batch[0]
        y = batch[-1]
        if y.ndim==1: y = y[:,None]
        if "train" in tag:
            device = y.device
            n = y.size(0)
            fsize = max(1,int(np.ceil(n//self.folds)))
            bools = torch.ones(n,dtype=bool,device=device)
            generator = torch.Generator()
            if self.rng_seed is not None:
                generator = generator.manual_seed(self.rng_seed)
            perm = torch.randperm(n,generator=generator).to(device)
            avg_l2error = 0. 
            avg_l2rerror = 0.
            for i in range(self.folds):
                bools[:] = True
                bools[(i*fsize):((i+1)*fsize)] = False
                pidxs_fit = perm[bools]
                self.gp.fit(x[pidxs_fit],y[pidxs_fit])
                pidxs_pred = perm[~bools]
                yhat = self.gp.forward(x[pidxs_pred])
                ytrue = y[pidxs_pred]
                l2error = torch.linalg.norm(yhat-ytrue,dim=1)
                l2rerror = l2error/torch.linalg.norm(ytrue,dim=1)
                self.use_l2rerror_loss
                avg_l2error += (torch.mean(l2error) if l2error.numel()>0 else 0)
                avg_l2rerror += (torch.mean(l2rerror) if l2rerror.numel()>0 else 0)
            self.gp.fit(x,y)
        else:
            assert "val" in tag
            yhat = self.gp.forward(x)
            l2error = torch.linalg.norm(yhat-y,dim=1)
            l2rerror = l2error/torch.linalg.norm(y,dim=1)
            avg_l2error = torch.mean(l2error)
            avg_l2rerror = torch.mean(l2rerror)
        self.log(tag+"avg_l2error",avg_l2error,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"avg_l2rerror",avg_l2rerror,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        return avg_l2rerror if self.use_l2rerror_loss else avg_l2error
    def to(self, device):
        super().to(device)
        self.gp = self.gp.to(device)
        if hasattr(self.gp,"coeffs"): self.gp.coeffs = self.gp.coeffs.to(device)
        if hasattr(self.gp,"x"): self.gp.x = self.gp.x.to(device)
    def training_step(self, batch, batch_idx):
        self.gp.train()
        return super().training_step(batch,batch_idx)
    def validation_step(self, batch, batch_idx):
        self.gp.eval()
        with gpytorch.settings.fast_pred_var():
            return super().validation_step(batch,batch_idx)