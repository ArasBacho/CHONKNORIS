import torch
import numpy as np
import os 
import lightning
import gpytorch
import linear_operator
from matplotlib import pyplot 

class _LightingBase(lightning.LightningModule):
    def __init__(self, lr=None, get_opt=None, automatic_optimization=True, zero_grad_set_to_none=True, compute_l2errors=True):
        super().__init__()
        self.lr = lr if lr is not None else self.DEFAULT_LR
        self.get_opt = get_opt
        self.automatic_optimization = automatic_optimization
        self.zero_grad_set_to_none = zero_grad_set_to_none
        self.compute_l2errors = compute_l2errors
    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            return self._common_step(batch,tag="train_")
        else:
            opt = self.optimizers()
            def closure():
                loss = self._common_step(batch,tag="train_")
                opt.zero_grad(set_to_none=self.zero_grad_set_to_none)
                self.manual_backward(loss)
                return loss
            opt.step(closure=closure)
    def validation_step(self, batch, batch_idx):
        with torch.inference_mode(True):
          return self._common_step(batch,tag="val_")
    def configure_optimizers(self):
        return self.get_opt(self.parameters()) if self.get_opt is not None else torch.optim.Adam(self.parameters(),amsgrad=True,lr=self.lr)

def train_val_split(*xs, val_frac=1/4, shuffle=True, rng_shuffle_seed=None):
    """
    >>> (xt,xv,yt,yv),vidx = train_val_split(torch.rand(8,4),torch.rand(8,3),val_frac=1/4)
    >>> assert xt.shape==(6,4) and xv.shape==(2,4) and yt.shape==(6,3) and yv.shape==(2,3)
    >>> (xt,xv),vidx = train_val_split(torch.rand(8,4),shuffle=False,val_frac=1/8)
    >>> assert xt.shape==(7,4) and xv.shape==(1,4)
    """
    assert (isinstance(x,torch.Tensor) for x in xs)
    R = xs[0].size(0)
    assert R>=2
    assert all(xs[0].size(0)==R for x in xs)
    if shuffle:
        rng = np.random.Generator(np.random.PCG64(rng_shuffle_seed))
        tv_idx = torch.from_numpy(rng.permutation(R))
    else:
        tv_idx = torch.arange(R)
    n_train = R-max(1,int(val_frac*R))
    tidx = tv_idx[:n_train]
    vidx = tv_idx[n_train:]
    x_split = (xtv for x in xs for xtv in (x[tidx],x[vidx]))
    return x_split,vidx

def parse_metrics(path):
    import pandas as pd
    newpath = path[:-4]+"_parsed.csv"
    if not os.path.isfile(path):
        assert os.path.isfile(newpath)
        return pd.read_csv(newpath).drop('epoch',axis=1)
    metrics = pd.read_csv(path)
    tags = [col[6:] for col in metrics.columns if "train_" in col]
    metrics_train = metrics.iloc[~np.isnan(metrics["train_"+tags[0]].values)]
    metrics_val = metrics.iloc[~np.isnan(metrics["val_"+tags[0]].values)]
    parsed_metrics = {}
    for tag in tags:
        parsed_metrics["train_"+tag] = metrics_train["train_"+tag].values
        parsed_metrics["val_"+tag] = metrics_val["val_"+tag].values
    parsed_metrics = pd.DataFrame(parsed_metrics)
    parsed_metrics["epoch"] = np.arange(metrics["epoch"][0],metrics["epoch"][0]+len(parsed_metrics))
    if os.path.isfile(newpath):
        parsed_metrics_old = pd.read_csv(newpath)
        if parsed_metrics_old["epoch"][len(parsed_metrics_old)-1]==(parsed_metrics["epoch"][0]-1): # append
            parsed_metrics = pd.concat([parsed_metrics_old,parsed_metrics])
        if len(parsed_metrics)==0:
            parsed_metrics = parsed_metrics_old
    parsed_metrics.reset_index(drop=True,inplace=True)
    parsed_metrics.to_csv(newpath,index=False)
    return parsed_metrics.drop('epoch',axis=1)

class ParallelPartialKernel(gpytorch.kernels.Kernel):
    r"""
    A special :class:`gpytorch.kernels.MultitaskKernel` where tasks are assumed
    to be independent, and a single, common kernel is used for all tasks.
    https://github.com/cornellius-gp/gpytorch/pull/2470

    Given a base covariance module to be used for the data, :math:`K_{XX}`,
    this kernel returns :math:`K = I_T \otimes K_{XX}`, where :math:`T` is the
    number of tasks.

    .. note::

        Note that, in this construction, it is crucial that all coordinates (or
        tasks) share the same kernel, with the same kernel parameters. The
        simplification of the inter-task kernel leads to computational
        savings if the number of tasks is large. If this were not the case
        (for example, when using the batch-independent Gaussian Process
        construction), then each task would have a different design correlation
        matrix, requiring the inversion of an `n x n` matrix at each
        coordinate, where `n` is the number of data points. Furthermore, when
        training the Gaussian Process surrogate, there is only one set of
        kernel parameters to be estimated, instead of one for every coordinate.

    :param ~gpytorch.kernels.Kernel covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks.
    :param dict kwargs: Additional arguments to pass to the kernel.

    Example:
    """

    def __init__(
        self,
        covar_module: gpytorch.kernels.Kernel,
        num_tasks: int,
        **kwargs,
    ):
        super(ParallelPartialKernel, self).__init__(**kwargs)
        self.covar_module = covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ParallelPartialKernel does not accept the last_dim_is_batch argument.")
        covar_x = linear_operator.to_linear_operator(self.covar_module.forward(x1, x2, **params))
        res = linear_operator.operators.BlockInterleavedLinearOperator(covar_x.repeat(self.num_tasks, 1, 1))
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this parallel
        partial kernel returns an `(n*num_tasks) x (m*num_tasks)`
        block-diagonal covariance matrix with `num_tasks` blocks of shape
        `n x m` on the diagonal.
        """
        return self.num_tasks

class TorchNumThreadsContext:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.prev_num_threads = None
    def __enter__(self):
        self.prev_num_threads = torch.get_num_threads()
        torch.set_num_threads(self.num_threads)
    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_num_threads(self.prev_num_threads)