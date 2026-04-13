import torch
import lightning
import warnings 
import typing 
from .util import _LightingBase,train_val_split
from .datasets import DatasetClassic,DatasetOpLearn
import neuralop

class MLP(torch.nn.Module):
    """
    >>> mlp = MLP(mlp_layer_nodes=[3,4,5])
    >>> mlp(torch.rand(2,3)).shape
    torch.Size([2, 5])
    """
    def __init__(self, mlp_layer_nodes:list, activation_function:torch.nn.Module=torch.nn.Tanh(), activate_last_layer:bool=False, scale_last_layer:bool=True, bias_last_layer:bool=True, weight_init_scheme:callable=None, batch_norm=None):
        super().__init__()
        num_layers = len(mlp_layer_nodes)-1
        self.nn_layer_nodes = mlp_layer_nodes
        self.output_nodes = self.nn_layer_nodes[-1]
        layers = []
        use_batch_norm = (batch_norm is not None) and (batch_norm is not False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",".*Initializing zero-element tensors is a no-op*") # occurs when setting a layer with 0 inputs
            for i in range(num_layers):
                layer = torch.nn.Linear(self.nn_layer_nodes[i],self.nn_layer_nodes[i+1])
                if callable(weight_init_scheme):
                    weight_init_scheme(layer.weight)
                layer.bias.data.fill_(0. if self.nn_layer_nodes[i]>0 else 1.)
                if use_batch_norm: 
                    if callable(batch_norm):
                        batch_norm_layer = batch_norm(self.nn_layer_nodes[i+1])
                    else:
                        batch_norm_layer = torch.nn.BatchNorm1d(self.nn_layer_nodes[i+1])
                    layers.extend([layer,batch_norm_layer,activation_function])
                else:
                    layers.extend([layer,activation_function])
        if use_batch_norm:
            self.nn_sequential = torch.nn.Sequential(*(layers if activate_last_layer else layers[:-2]))
        else:    
            self.nn_sequential = torch.nn.Sequential(*(layers if activate_last_layer else layers[:-1]))
        self.logscale,self.bias = torch.tensor(0.),torch.tensor(0.)
        if scale_last_layer: self.logscale = torch.nn.parameter.Parameter(self.logscale)
        if bias_last_layer: self.bias = torch.nn.parameter.Parameter(self.bias)
    def output_transform(self, x:torch.Tensor):
        return x
    def forward(self, x):
        x = self.nn_sequential(x).squeeze()
        x = self.output_transform(x)
        return torch.exp(self.logscale)*x+self.bias

class DeepONet(torch.nn.Module):
    """
    >>> don = DeepONet(branches_layers_nodes=[[3,4,10],[4,5,10]],trunks_layers_nodes=[[5,6,10],[6,7,10]])
    >>> don([torch.rand(2,3),torch.rand(2,4)],[torch.rand(2,5),torch.rand(2,6)]).shape
    torch.Size([2])
    """
    def __init__(self, branches_layers_nodes:list, trunks_layers_nodes:list, branch_activation_function:torch.nn.Module=torch.nn.Tanh(), trunk_activation_function:torch.nn.Module=torch.nn.Tanh(), scale_last_layer:bool=True, bias_last_layer:bool=True):
        super().__init__()
        self.branches_layers_nodes,self.trunks_layers_nodes = branches_layers_nodes,trunks_layers_nodes
        self.n_branches,self.n_trunks = len(branches_layers_nodes),len(trunks_layers_nodes)
        self.combining_neurons = trunks_layers_nodes[0][-1]
        self.output_nodes = branches_layers_nodes[0][-1]/self.combining_neurons
        assert self.output_nodes%1==0; self.output_nodes = int(self.output_nodes)
        assert all(branches_layers_nodes[i][-1]==(self.output_nodes*self.combining_neurons) for i in range(self.n_branches)), "Each branch net must have the same number of outputs"
        assert all(trunks_layers_nodes[i][-1]==self.combining_neurons for i in range(self.n_trunks)), "Each trunk net must have the same number of outputs as each branch net"
        self.branch_nets = torch.nn.ModuleList([MLP(branches_layers_nodes[i],activation_function=branch_activation_function,activate_last_layer=False,scale_last_layer=False,bias_last_layer=False) for i in range(self.n_branches)])
        self.trunk_nets = torch.nn.ModuleList([MLP(trunks_layers_nodes[i],activation_function=trunk_activation_function,activate_last_layer=True,scale_last_layer=False,bias_last_layer=False) for i in range(self.n_trunks)])
        self.logscale,self.bias = torch.tensor(0.),torch.tensor(0.)
        if scale_last_layer: self.logscale = torch.nn.parameter.Parameter(self.logscale)
        if bias_last_layer: self.bias = torch.nn.parameter.Parameter(self.bias)
    def output_transform(self, x:torch.Tensor):
        return x
    def forward_branch_nets(self, x_branches:typing.List[torch.Tensor]):
        assert len(x_branches)==self.n_branches
        y_branches_prod = 1.
        for i,branch_net in enumerate(self.branch_nets):
            y_branches_prod *= branch_net(x_branches[i])
        return y_branches_prod
    def forward_trunck_nets(self, x_trunks:typing.List[torch.Tensor]):
        assert len(x_trunks)==self.n_trunks
        y_trunks_prod = 1.
        for i,trunk_net in enumerate(self.trunk_nets):
            y_trunks_prod *= trunk_net(x_trunks[i])
        return y_trunks_prod
    def forward_combine(self, y_branches_prod, y_trunks_prod):
        out = (y_branches_prod.reshape(-1,self.combining_neurons,self.output_nodes)*y_trunks_prod.reshape(-1,self.combining_neurons,1)).sum(1).squeeze()
        out_tf = self.output_transform(out) 
        out = torch.exp(self.logscale)*out_tf+self.bias
        return out
    def forward(self, x_branches:typing.List[torch.Tensor], x_trunks:typing.List[torch.Tensor]):
        y_branches_prod = self.forward_branch_nets(x_branches)
        y_trunks_prod = self.forward_trunck_nets(x_trunks)
        out = self.forward_combine(y_branches_prod,y_trunks_prod)
        return out

class LightningNN(_LightingBase):
    """
    >>> x = torch.rand(16,2)
    >>> y = torch.rand(16)
    >>> (xt,xv,yt,yv),vidx = train_val_split(x,y,val_frac=1/4)
    >>> dataset_t = DatasetClassic(x=xt,y=yt)
    >>> dataset_v = DatasetClassic(x=xv,y=yv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> nn = MLP(mlp_layer_nodes=[2,3,1])
    >>> lnn = LightningNN(nn)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lnn,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)

    >>> x = torch.rand(16,2)
    >>> y = torch.rand(16,4)
    >>> (xt,xv,yt,yv),vidx = train_val_split(x,y,val_frac=1/4)
    >>> dataset_t = DatasetClassic(x=xt,y=yt)
    >>> dataset_v = DatasetClassic(x=xv,y=yv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> nn = MLP(mlp_layer_nodes=[2,3,4])
    >>> lnn = LightningNN(nn)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lnn,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)

    >>> x = torch.rand(4,2)
    >>> u = torch.rand(16,3)
    >>> v = torch.rand(16,4)
    >>> (ut,uv,vt,vv),vidx = train_val_split(u,v,val_frac=1/4)
    >>> dataset_t = DatasetOpLearn(ut,x,vt)
    >>> dataset_v = DatasetOpLearn(uv,x,vv)
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> nn = DeepONet(branches_layers_nodes=[[3,4,5]],trunks_layers_nodes=[[2,3,4,5]])
    >>> lnn = LightningNN(nn,compute_l2errors=False,use_l2rerror_loss=False)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lnn,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)

    >>> in_channels = 3 
    >>> out_channels = 5
    >>> dataset_t = DatasetClassic(x=torch.rand(15,in_channels,16),y=torch.rand(15,out_channels,16))
    >>> dataset_v = DatasetClassic(x=torch.rand(10,in_channels,8),y=torch.rand(10,out_channels,8))
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> nn = neuralop.models.FNO(n_modes=(32,),in_channels=in_channels,out_channels=out_channels,hidden_channels=7,n_layers=9)
    >>> lnn = LightningNN(nn)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lnn,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)
 
    >>> in_channels = 3 
    >>> out_channels = 5
    >>> dataset_t = DatasetClassic(x=torch.rand(15,in_channels,16,8),y=torch.rand(15,out_channels,16,8))
    >>> dataset_v = DatasetClassic(x=torch.rand(10,in_channels,8,16),y=torch.rand(10,out_channels,8,16))
    >>> dataloader_t = torch.utils.data.DataLoader(dataset_t,batch_size=len(dataset_t),collate_fn=tuple,shuffle=False)
    >>> dataloader_v = torch.utils.data.DataLoader(dataset_v,batch_size=len(dataset_v),collate_fn=tuple,shuffle=False)
    >>> nn = neuralop.models.FNO(n_modes=(32,32),in_channels=in_channels,out_channels=out_channels,hidden_channels=7,n_layers=9)
    >>> lnn = LightningNN(nn)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     trainer = lightning.Trainer(max_epochs=2,accelerator="cpu",enable_progress_bar=False)
    ...     trainer.fit(lnn,train_dataloaders=dataloader_t,val_dataloaders=dataloader_v)
    """
    DEFAULT_LR = 1e-3
    def __init__(self, nn, use_l2rerror_loss=True, **super_kwargs):
        super().__init__(**super_kwargs)
        self.nn = nn
        assert use_l2rerror_loss==False or use_l2rerror_loss==True or use_l2rerror_loss=="both"
        self.use_l2rerror_loss = use_l2rerror_loss
        if self.use_l2rerror_loss == True or self.use_l2rerror_loss == "both": assert self.compute_l2errors, "use_l2rerror_loss=True requires compute_l2errors=True"
        assert not (isinstance(self.nn,DeepONet) and self.compute_l2errors), "set compute_l2errors=False when using DeepONet"
    # def to(self, device):
    #     super().to(device)
    #     self.nn = self.nn.to(device)
    def forward(self, *inputs):
        #assert all(inp.ndim==2 for inp in inputs)
        return self.nn(*inputs)
    def _common_step(self, batch, tag):
        v = batch[-1]
        vhat = self.forward(*batch[:-1])
        v = v.reshape((len(v),-1))
        vhat = vhat.reshape((len(v),-1))
        assert v.shape==vhat.shape
        err = vhat-v
        mse = torch.mean(err**2)
        self.log(tag+"loss",mse,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"rmse",torch.sqrt(mse),logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        if self.compute_l2errors:
            l2error = torch.linalg.norm(err,dim=1)
            self.log(tag+"avg_l2error",torch.mean(l2error),logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
            l2rerror = l2error/torch.linalg.norm(v,dim=1)
            mean_l2rerror = torch.mean(l2rerror)
            self.log(tag+"avg_l2rerror",mean_l2rerror,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        if self.use_l2rerror_loss == False:
            loss = mse 
        elif self.use_l2rerror_loss == True: 
            loss = mean_l2rerror
        elif self.use_l2rerror_loss == "both":
            loss = mse+mean_l2rerror
        return loss
    