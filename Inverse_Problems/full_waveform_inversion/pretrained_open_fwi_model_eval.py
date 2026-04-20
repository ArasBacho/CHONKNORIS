import OpenFWI.transforms as T
from OpenFWI import network
from util import get_torch_device_backend

import torch 
from torchvision.transforms import Compose
import json 
import os
import numpy as np

class PretrainedOpenFWIModelEval(object):
    def __init__(self, model_name="fva_l2.pth", dataset="flatvel-a", model="InversionNet", device="cpu"):
        ROOT = os.path.dirname(os.path.realpath(__file__))
        resume = ROOT+"/pth_open_fwi_models/%s"%model_name
        k = 1
        up_mode = None
        sample_spatial = 1. 
        sample_temporal = 1
        with open(ROOT+'/OpenFWI/dataset_config.json') as f:
            self.ctx = json.load(f)[dataset]
        self.transform_data = Compose([
            T.LogTransform(k=k),
            T.MinMaxNormalize(T.log_transform(self.ctx['data_min'],k=k),T.log_transform(self.ctx['data_max'],k=k))
        ])
        self.model = network.model_dict[model](upsample_mode=up_mode, 
                sample_spatial=sample_spatial, sample_temporal=sample_temporal).to(device)
        checkpoint = torch.load(resume, map_location='cpu',weights_only=False)
        self.model.load_state_dict(network.replace_legacy(checkpoint['model']))
        self.model.eval()
    def __call__(self, p0):
        p0_tf = self.transform_data(p0.cpu()).to(p0.device)
        with torch.no_grad():
            chat_tf = self.model(p0_tf)
        chat = T.tonumpy_denormalize(chat_tf.cpu(),self.ctx['label_min'],self.ctx['label_max'],exp=False)
        chat = torch.from_numpy(chat[:,0]).to(torch.get_default_dtype()).to(p0.device)
        return chat

if __name__=="__main__":
    ROOT = os.path.dirname(os.path.realpath(__file__))

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

    DEVICE,_ = get_torch_device_backend()
    print(DEVICE)
    model = PretrainedOpenFWIModelEval()
    model.model.to(DEVICE) 

    velocity_hat = model(wavefield[:2].to(DEVICE))
    print(velocity_hat.shape)
    print(velocity_hat.device) 

    from matplotlib import pyplot 
    fig,ax = pyplot.subplots(nrows=2,ncols=2,figsize=(10,5)) 
    for i in range(2):
        cax = ax[i,0].imshow(velocity[i].cpu())
        fig.colorbar(cax)
        cax = ax[i,1].imshow(velocity_hat[i].cpu())
        fig.colorbar(cax)
    fig.savefig("out/pretrained_open_fwi_model_eval.pdf",bbox_inches="tight")
