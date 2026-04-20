import sys
sys.path.append('../../')
sys.path.append('../')
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

model_type = "FNO"
assert model_type in ["FNO","TNO"]
problem_type = "Darcy"
assert problem_type  in ["elliptic","Burgers","Darcy"]

########################################################################
'''
Uncomment the model you want to test and comment the other two. Use FNO1D and FNO (2D) accordingly
'''
if model_type=="FNO":
    if problem_type=="Darcy":
        from models.FNO.FNOneuralop_lightning import FNOModule
    else:
        from models.FNO1D.FNO1Dneuralop_lightning import FNOModule
elif model_type=="TNO":
    from models.Transformer.Transformer_lightning import SimpleEncoderModule
########################################################################

from datasets import MetaDataModule
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################################################
'''
Change checkpoint_path accordingly
'''
checkpoint_path = '../checkpoints/%s_%s_checkpoint.ckpt'%(model_type,problem_type)
########################################################################
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


# api = wandb.Api()
########################################################################
'''
Change 'config.json' accordingly
'''

import json
with open('../configs/%s_%s_config.json'%(model_type,problem_type),'r',encoding='utf-8') as f:
    config = json.load(f)
    config = {key:val['value'] for key,val in config.items()}


#run = api.run('edoardo-calvello/DeepNewtonOperator_Burgers/04mggjms')
#config = run.config
########################################################################


########################################################################
'''
uncomment based on model choice. FNO module refers to both FNO1D and FNO2D, 
so it can be used for both. SimpleEncoderModule is the transformer model.
'''

if model_type=="FNO":
    valid_model_args = {key: value for key, value in config.items() if key in FNOModule.__init__.__code__.co_varnames}
    model = FNOModule(**valid_model_args)
elif model_type=="TNO":
    valid_model_args = {key: value for key, value in config.items() if key in SimpleEncoderModule.__init__.__code__.co_varnames}
    model = SimpleEncoderModule(**valid_model_args)

########################################################################

model.load_state_dict(checkpoint['state_dict'],strict=False)
model.to(device)

datamodule = MetaDataModule(**config)
datamodule.setup(stage='test') 
test_data_loader = datamodule.test_dataloader() 
test_sample_rate = 1
test_data_loader = test_data_loader[test_sample_rate]


model.eval()


errors = []
predictions = []
forward_times = []

for sample in test_data_loader.dataset:
    x, y, coords_x, coords_y = sample
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    coords_x = torch.from_numpy(coords_x).to(device)
    coords_y = torch.from_numpy(coords_y).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_pred = model(x.unsqueeze(0), coords_x=coords_x.unsqueeze(0))
        if model_type=="FNO":
            y_pred = y_pred.permute(0,2,1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - t0)
        
        error_sample = torch.sqrt(torch.mean(torch.abs(y_pred.to(device) - (y.unsqueeze(0)))**2))/torch.sqrt(
            torch.mean(torch.abs(y.unsqueeze(0))**2))
    
        predictions.append(y_pred.to('cpu').numpy())
        errors.append(error_sample.to('cpu').numpy())


avg_error = np.mean(errors)
avg_forward_time = np.mean(forward_times)
median_forward_time = np.median(forward_times)

print(f"Average relative L2 error: %.1e"%avg_error)
print(f"Average forward pass time: %.1e"%avg_forward_time)
print(f"Median forward pass time: %.1e"%median_forward_time)

predictions = np.array(predictions)
errors = np.array(errors)

# Find the indices of the samples with the median and lowest error
median_idx = np.argsort(errors)[len(errors) // 2]
median_error= errors[median_idx]
min_error_idx = np.argmax(errors)
min_error = errors[min_error_idx]

print(f"Median relative L2 error: {median_error:.6f}")