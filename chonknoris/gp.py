import gpytorch 
import torch 
import lightning
from .util import _LightingBase,train_val_split,ParallelPartialKernel
from .datasets import DatasetClassic
import warnings

class GP(gpytorch.models.ExactGP):
    """
    https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    
    >>> gp = GP(x=torch.rand(3,4),y=torch.rand(3))
    >>> gp.forward(torch.rand(10,4))
    MultivariateNormal(loc: torch.Size([10]))
    >>> gp = gp.eval()
    >>> gp(torch.rand(10,4))
    MultivariateNormal(loc: torch.Size([10]))
    """
    def __init__(self, x, y, mean_module=None, covar_module=None, likelihood=None, fixed_noise=True, noise_lb=1e-8):
        assert x.ndim==2 and y.ndim==1 and len(x)==len(y) 
        likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(noise_lb))
        if fixed_noise:
            likelihood.noise = noise_lb
            likelihood.noise_covar.raw_noise.requires_grad_(False)
        super().__init__(x,y,likelihood)
        self.d_in = x.size(1)
        self.d_out = 1
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.LinearMean(input_size=self.d_in)
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=self.d_in))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

class MultiTaskVecGP(gpytorch.models.ExactGP):
    """
    https://docs.gpytorch.ai/en/v1.10/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
    
    >>> gp = MultiTaskVecGP(x=torch.rand(3,4),y=torch.rand(3,5))
    >>> gp.forward(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    >>> gp = gp.eval()
    >>> gp(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    """
    def __init__(self, x, y, mean_module=None, covar_module=None, likelihood=None, fixed_noise=True, noise_lb=1e-8):
        assert x.ndim==2 and y.ndim==2 and len(x)==len(y)
        likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.size(1),noise_constraint=gpytorch.constraints.GreaterThan(noise_lb),has_task_noise=False,has_global_noise=True)
        if fixed_noise:
            likelihood.raw_noise = torch.nn.Parameter(-torch.inf*torch.ones_like(likelihood.raw_noise),requires_grad=False)
        super().__init__(x,y,likelihood)
        self.d_in = x.size(1)
        self.d_out = y.size(1)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.MultitaskMean(gpytorch.means.LinearMean(input_size=self.d_in),num_tasks=self.d_out)
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.MultitaskKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(input_size=self.d_in,ard_num_dims=self.d_in)),num_tasks=self.d_out)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,covar_x)

class IndepVecGPShared(gpytorch.models.ExactGP):
    """
    https://github.com/PieterjanRobbe/gpytorch_parallel_partial_emulation_notebooks/blob/main/Parallel_Partial_GP_Regression.ipynb
    
    >>> gp = IndepVecGPShared(x=torch.rand(3,4),y=torch.rand(3,5))
    >>> gp.forward(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    >>> gp = gp.eval()
    >>> gp(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    """
    def __init__(self, x, y, mean_module=None, covar_module=None, likelihood=None, fixed_noise=True, noise_lb=1e-8):
        assert x.ndim==2 and y.ndim==2 and len(x)==len(y)
        likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.size(1),noise_constraint=gpytorch.constraints.GreaterThan(noise_lb),has_task_noise=False,has_global_noise=True)
        if fixed_noise:
            likelihood.raw_noise = torch.nn.Parameter(-torch.inf*torch.ones_like(likelihood.raw_noise),requires_grad=False)
        super().__init__(x,y,likelihood)
        self.d_in = x.size(1)
        self.d_out = y.size(1)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.MultitaskMean(gpytorch.means.LinearMean(input_size=self.d_in),num_tasks=self.d_out)
        self.covar_module = covar_module if covar_module is not None else ParallelPartialKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.d_in),num_tasks=self.d_out)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class IndepVecGP(gpytorch.models.ExactGP):
    """
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Batch_Independent_Multioutput_GP.html
    
    >>> gp = IndepVecGP(x=torch.rand(3,4),y=torch.rand(3,5))
    >>> gp.forward(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    >>> gp = gp.eval()
    >>> gp(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    """
    def __init__(self, x, y, mean_module=None, covar_module=None, likelihood=None, fixed_noise=True, noise_lb=1e-8):
        assert x.ndim==2 and y.ndim==2 and len(x)==len(y)
        likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.size(1),noise_constraint=gpytorch.constraints.GreaterThan(noise_lb),has_task_noise=False,has_global_noise=True)
        if fixed_noise:
            likelihood.raw_noise = torch.nn.Parameter(-torch.inf*torch.ones_like(likelihood.raw_noise),requires_grad=False)
        super().__init__(x,y,likelihood)
        self.d_in = x.size(1)
        self.d_out = y.size(1)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.LinearMean(input_size=self.d_in,batch_shape=torch.Size([self.d_out]))
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=self.d_in,batch_shape=torch.Size([self.d_out])),batch_shape=torch.Size([self.d_out]))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(gpytorch.distributions.MultivariateNormal(mean_x,covar_x))

class IndepVecVGP(gpytorch.models.ApproximateGP):
    """
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
    
    >>> gp = IndepVecVGP(n=10,d_in=4,d_out=5,num_inducing_pts=3)
    >>> gp.forward(torch.rand(10,4))
    MultivariateNormal(loc: torch.Size([5, 10]))
    >>> gp(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    """
    def __init__(self, n, d_in, d_out, num_inducing_pts, mean_module=None, covar_module=None, variational_strategy=None, fixed_noise=True, noise_lb=1e-8):
        if variational_strategy is not None:
            variational_strategy = variational_strategy
        else:
            inducing_points = torch.rand(d_out,num_inducing_pts,d_in)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_pts,batch_shape=torch.Size([d_out]))
            variational_strategy = gpytorch.variational.VariationalStrategy(self,inducing_points,variational_distribution,learn_inducing_locations=True)
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strategy,num_tasks=d_out)
        super().__init__(variational_strategy)
        self.n = n
        self.d_in = d_in
        self.d_out = d_out
        self.fixed_noise = fixed_noise
        self.noise_lb = noise_lb
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.LinearMean(input_size=self.d_in,batch_shape=torch.Size([self.d_out]))
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=self.d_in,batch_shape=torch.Size([self.d_out])),batch_shape=torch.Size([self.d_out]))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

class VecVGPLatents(gpytorch.models.ApproximateGP):
    """
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
    
    >>> gp = VecVGPLatents(n=10,d_in=4,d_out=5,num_inducing_pts=3,num_latents=2)
    >>> gp.forward(torch.rand(10,4))
    MultivariateNormal(loc: torch.Size([2, 10]))
    >>> gp(torch.rand(10,4))
    MultitaskMultivariateNormal(mean shape: torch.Size([10, 5]))
    """
    def __init__(self, n, d_in, d_out, num_inducing_pts, num_latents, mean_module=None, covar_module=None, variational_strategy=None, fixed_noise=True, noise_lb=1e-8):
        if variational_strategy is not None:
            variational_strategy = variational_strategy
        else:
            inducing_points = torch.rand(num_latents,num_inducing_pts,d_in)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_pts,batch_shape=torch.Size([num_latents]))
            variational_strategy = gpytorch.variational.VariationalStrategy(self,inducing_points,variational_distribution,learn_inducing_locations=True)
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(variational_strategy,num_tasks=d_out,num_latents=num_latents,latent_dim=-1)
        super().__init__(variational_strategy)
        self.n = n
        self.d_in = d_in
        self.d_out = d_out
        self.fixed_noise = fixed_noise
        self.noise_lb = noise_lb
        self.num_latents = num_latents
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.LinearMean(input_size=self.d_in,batch_shape=torch.Size([self.num_latents]))
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=self.d_in,batch_shape=torch.Size([self.num_latents])),batch_shape=torch.Size([self.num_latents]))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

class LightningGP(_LightingBase):
    """
    >>> x = torch.rand(16,2)
    >>> y = torch.rand(16)
    >>> (xt,xv,yt,yv),vidx = train_val_split(x,y,val_frac=1/4)
    >>> dataset_t = DatasetClassic(x=xt,y=yt)
    >>> dataset_v = DatasetClassic(x=xv,y=yv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> gp = GP(x=xt,y=yt)
    >>> lgp = LightningGP(gp)
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
    >>> gps = [
    ...     MultiTaskVecGP(x=xt,y=yt),
    ...     IndepVecGP(x=xt,y=yt),
    ...     IndepVecGPShared(x=xt,y=yt),
    ...     IndepVecVGP(n=xt.size(0),d_in=x.size(1),d_out=y.size(1),num_inducing_pts=2),
    ...     VecVGPLatents(n=xt.size(0),d_in=x.size(1),d_out=y.size(1),num_inducing_pts=2,num_latents=4),]
    >>> for gp in gps:
    ...     lgp = LightningGP(gp)
    ...     with warnings.catch_warnings():
    ...         warnings.simplefilter("ignore")
    ...         trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...         trainer.fit(lgp,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)
    """ 
    DEFAULT_LR = 0.1
    def __init__(self, gp, **super_kwargs):
        super().__init__(**super_kwargs)
        self.gp = gp
        if isinstance(self.gp,(GP,IndepVecGP,MultiTaskVecGP,IndepVecGPShared)):
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood,self.gp)
        elif isinstance(self.gp,IndepVecVGP) or isinstance(self.gp,VecVGPLatents):
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.gp.d_out,noise_constraint=gpytorch.constraints.GreaterThan(self.gp.noise_lb),has_task_noise=False,has_global_noise=True)
            self.mll = gpytorch.mlls.VariationalELBO(likelihood,self.gp,num_data=self.gp.n)
            if self.gp.fixed_noise:
                self.mll.likelihood.raw_noise = torch.nn.Parameter(-torch.inf*torch.ones_like(self.mll.likelihood.raw_noise),requires_grad=False)
        else:
            raise Exception("LightningGP mll parser not implemented for %s"%(type(self.gp).__name__))
    # def to(self, device):
    #     super().to(device)
    #     self.gp = self.gp.to(device)
    #     self.mll.likelihood = self.mll.likelihood.to(device)
    def forward_mvn(self, *inputs):
        #assert all(inp.ndim==2 for inp in inputs)
        return self.gp(*inputs)
    def forward(self, *inputs):
        outs = self.forward_mvn(*inputs)
        return outs.mean
    def _common_step(self, batch, tag):
        v = batch[-1]
        vhat_mvn = self.forward_mvn(*batch[:-1])
        loss = -self.mll(vhat_mvn,v)
        self.log(tag+"loss",loss,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        vhat = vhat_mvn.mean
        v = v.reshape((len(v),-1))
        vhat = vhat.reshape((len(v),-1))
        assert v.shape==vhat.shape
        err = vhat-v
        mse = torch.mean(err**2)
        self.log(tag+"rmse",torch.sqrt(mse),logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        if self.compute_l2errors:
            l2error = torch.linalg.norm(err,dim=1)
            self.log(tag+"avg_l2error",torch.mean(l2error),logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
            l2rerror = l2error/torch.linalg.norm(v,dim=1)
            self.log(tag+"avg_l2rerror",torch.mean(l2rerror),logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        return loss
    def training_step(self, batch, batch_idx):
        self.gp.train()
        self.mll.likelihood.train()
        return super().training_step(batch,batch_idx)
    def validation_step(self, batch, batch_idx):
        self.gp.eval()
        self.mll.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            return super().validation_step(batch,batch_idx)
    def eval(self):
        self.gp.eval()
        self.mll.likelihood.eval()
        return super().eval()
