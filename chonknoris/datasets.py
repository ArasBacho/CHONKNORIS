import torch 
import pyKoLesky.cholesky

class DatasetClassic(torch.utils.data.Dataset):
    """
    >>> dataset = DatasetClassic(x=torch.rand(6,4),y=torch.rand(6,5))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=3,collate_fn=tuple)
    >>> for x,y in dataloader:
    ...     assert x.shape==(3,4) and y.shape==(3,5)
    """
    def __init__(self, x, y):
        assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor)
        assert x.size(0)==y.size(0)
        self.x = x
        self.y = y
        self.i = torch.arange(self.y.size(0))
    def to(self, whereto):
        self.x = self.x.to(whereto)
        self.y = self.y.to(whereto)
        if not isinstance(whereto,torch.dtype):
            self.i = self.i.to(whereto)
    def __getitems__(self, i):
        i = torch.tensor(i,dtype=torch.int)
        i = self.i[i]
        return self.x[i],self.y[i]
    def __len__(self):
        return len(self.i)

class DatasetOpLearn(torch.utils.data.Dataset):
    """
    >>> dataset = DatasetOpLearn(u=torch.rand(9,4),x=torch.rand(6,2),v=torch.rand(9,6))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=3,collate_fn=tuple)
    >>> for u,x,v in dataloader:
    ...     assert isinstance(u,list) and len(u)==1
    ...     assert isinstance(x,list) and len(x)==1
    ...     assert u[0].shape==(3,4) and x[0].shape==(3,2) and v.shape==(3,)
    """
    def __init__(self, u, x, v):
        assert isinstance(x,torch.Tensor) and isinstance(u,torch.Tensor) and isinstance(v,torch.Tensor)
        assert x.ndim==2 and u.ndim==2 and v.ndim==2
        assert x.size(0)==v.size(1) and u.size(0)==v.size(0)
        self.u = u
        self.x = x 
        self.v = v
        self.i_r,self.i_c = torch.cartesian_prod(torch.arange(self.u.size(0)),torch.arange(x.size(0))).T
    def to(self, whereto):
        self.x = self.x.to(whereto)
        self.u = self.u.to(whereto)
        self.v = self.v.to(whereto)
        if not isinstance(whereto,torch.dtype):
            self.i_r = self.i_r.to(whereto)
            self.i_c = self.i_c.to(whereto)
    def __getitems__(self, i):
        i = torch.tensor(i,dtype=torch.int)
        ir,ic = self.i_r[i],self.i_c[i]
        return [self.u[ir]],[self.x[ic]],self.v[ir,ic]
    def __len__(self):
        return len(self.i_c)

class DatasetLowerTriMatOpLearn(torch.utils.data.Dataset):
    """
    >>> dataset = DatasetLowerTriMatOpLearn(v=torch.rand(3,4,5),Linvs=torch.rand(3,4,5,5),relaxations=torch.rand(1))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,collate_fn=tuple)
    >>> for inputs,Linv in dataloader:
    ...     assert inputs.shape==(6,5) and Linv.shape==(6,5/2*(1+5))

    >>> dataset = DatasetLowerTriMatOpLearn(v=torch.rand(3,4,5),Linvs=torch.rand(3,4,5,5),relaxations=torch.rand(6))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,collate_fn=tuple)
    >>> for inputs,Linv in dataloader:
    ...     assert inputs.shape==(6,6) and Linv.shape==(6,5/2*(1+5))

    >>> dataset = DatasetLowerTriMatOpLearn(v=torch.rand(3,4,5),Linvs=torch.rand(3,4,5,5),relaxations=torch.rand(1),u=torch.rand(3,7))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,collate_fn=tuple)
    >>> for inputs,Linv in dataloader:
    ...     assert inputs.shape==(6,12) and Linv.shape==(6,5/2*(1+5))
    
    >>> dataset = DatasetLowerTriMatOpLearn(v=torch.rand(3,4,5),Linvs=torch.rand(3,4,5,5),relaxations=torch.rand(6),u=torch.rand(3,7))
    >>> dataset.to("cpu")
    >>> dataset.to(torch.float64)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,collate_fn=tuple)
    >>> for inputs,Linv in dataloader:
    ...     assert inputs.shape==(6,13) and Linv.shape==(6,5/2*(1+5))

    >>> n = 7
    >>> grid = torch.linspace(0,1,n+2)[1:-1]
    >>> sparse_perm,sparse_lengths = pyKoLesky.cholesky.maximin(x=grid[:,None],initial=torch.tensor([0.,1.])[:,None])
    >>> sparse_pattern = pyKoLesky.cholesky.sparsity_pattern(grid[:,None],lengths=sparse_lengths,rho=1)
    >>> dataset = DatasetLowerTriMatOpLearn(v=torch.rand(3,4,5),Linvs=torch.tril(torch.rand((3,4,n,n))),relaxations=torch.rand(6),sparse=True,sparse_perm=sparse_perm,sparse_pattern=sparse_pattern)
    >>> dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,collate_fn=tuple)
    >>> for inputs,Linv in dataloader:
    ...     assert inputs.shape==(6,6) and Linv.shape==(6,dataset.nelemLinv)
    """
    def __init__(self, v, Linvs, relaxations, u=None, sparse=False, sparse_perm=None, sparse_pattern=None, log_diag=False):
        assert v.ndim==3 and Linvs.ndim==4 and relaxations.ndim==1 and v.size(0)==Linvs.size(0) and v.size(1)==Linvs.size(1) and Linvs.size(2)==Linvs.size(3)
        if u is not None:
            assert u.ndim==2 and u.size(0)==v.size(0)
        self.u = u 
        self.v = v 
        self.relaxations = relaxations
        common_v0 = (Linvs[0,0,:,:]==Linvs[:,0,:,:]).all()
        eye = torch.eye(Linvs.size(-1),dtype=Linvs.dtype,device=Linvs.device)
        if not sparse:
            if (relaxations==0).all():
                assert len(relaxations)==1
                self.Linvs = Linvs[:,:,None,:,:]
            else:
                Ls = torch.linalg.solve_triangular(Linvs,eye,upper=False)
                Thetas = torch.einsum("rikp,rimp->rikm",Ls,Ls)
                Thetas = Thetas[:,:,None,:,:]+self.relaxations[:,None,None]*eye
                Ls = torch.linalg.cholesky(Thetas)
                self.Linvs = torch.linalg.solve_triangular(Ls,eye,upper=False)
            self._lti0,self._lti1 = torch.tril_indices(self.Linvs.size(-1),self.Linvs.size(-1))
            if log_diag:
                Nrange = torch.arange(self.Linvs.size(-1),device=Linvs.device)
                diag = self.Linvs[...,Nrange,Nrange]
                assert (diag>0).all()
                self.Linvs[...,Nrange,Nrange] = torch.log(diag)
            self.Linvs = self.Linvs[...,self._lti0,self._lti1]
            self.nelemLinv = len(self._lti0)
        else: # sparse Cholesky
            assert not log_diag, "log_diag not yet implemented for sparse=True"
            assert isinstance(sparse_pattern,dict)
            assert isinstance(sparse_perm,torch.Tensor) and sparse_perm.ndim==1 and sparse_perm.size(0)==Linvs.size(-1)
            self.nelemLinv = sum(len(idxs) for idxs in sparse_pattern.values())
            Ls = torch.linalg.solve_triangular(Linvs,eye,upper=False)
            Thetas = torch.einsum("rikp,rimp->rikm",Ls,Ls)
            Thetas = Thetas[:,:,None,:,:]+self.relaxations[:,None,None]*eye
            self.Linvs = torch.zeros((Thetas.size(0),Thetas.size(1),Thetas.size(2),self.nelemLinv))
            # could be made more efficient by enablsing batching for pyKoLesky.cholesky.sparse_cholesky 
            for r in range(Thetas.size(0)):
                for k in range(Thetas.size(1)):
                    for l in range(Thetas.size(2)):
                        Uinv_sparse = pyKoLesky.cholesky.sparse_cholesky(Theta=Thetas[r,k,l],Perm=sparse_perm,sparsity=sparse_pattern)
                        Linv_sparse_coalesce = Uinv_sparse.T.coalesce()
                        self.Linvs[r,k,l,:] = Linv_sparse_coalesce.values()
            self._lti0,self._lti1 = Linv_sparse_coalesce.indices()
        if common_v0:
            i_r_0,i_k_0,i_l_0 = torch.cartesian_prod(torch.arange(1),torch.arange(1),torch.arange(self.relaxations.size(0))).T
            i_r_p,i_k_p,i_l_p = torch.cartesian_prod(torch.arange(self.Linvs.size(0)),torch.arange(1,self.v.size(1)),torch.arange(self.relaxations.size(0))).T
            self.i_r,self.i_k,self.i_l = torch.hstack([i_r_0,i_r_p]),torch.hstack([i_k_0,i_k_p]),torch.hstack([i_l_0,i_l_p])
        else:
            self.i_r,self.i_k,self.i_l = torch.cartesian_prod(torch.arange(self.Linvs.size(0)),torch.arange(self.v.size(1)),torch.arange(self.relaxations.size(0))).T
        self.prepend_relaxations = self.relaxations.size(0)>1
    def to(self, whereto):
        if self.u is not None: 
            self.u = self.u.to(whereto)
        self.v = self.v.to(whereto)
        self.Linvs = self.Linvs.to(whereto)
        self.relaxations = self.relaxations.to(whereto)
        if not isinstance(whereto,torch.dtype):
            self.i_r = self.i_r.to(whereto)
            self.i_k = self.i_k.to(whereto)
            self.i_l = self.i_l.to(whereto)
            self._lti0 = self._lti0.to(whereto)
            self._lti1 = self._lti1.to(whereto)
    def __getitems__(self, i):
        i = torch.tensor(i,dtype=torch.int)
        ir,ik,il = self.i_r[i],self.i_k[i],self.i_l[i]
        inputs,Linvs = self.v[ir,ik,:],self.Linvs[ir,ik,il]
        if self.u is not None: 
            us = self.u[ir]
            inputs = torch.hstack([inputs,us])
        if self.prepend_relaxations:
            relaxations = self.relaxations[il]
            inputs = torch.hstack([relaxations[:,None],inputs])
        return inputs,Linvs
    def __len__(self):
        return len(self.i_r)