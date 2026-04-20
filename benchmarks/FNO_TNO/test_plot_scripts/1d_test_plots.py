import sys
sys.path.append('../../')
sys.path.append('../')
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

########################################################################
'''
Uncomment the model you want to test and comment the other two. Use FNO1D and FNO (2D) accordingly
'''
#from models.FNO.FNOneuralop_lightning import FNOModule
from models.FNO1D.FNO1Dneuralop_lightning import FNOModule
#from models.Transformer.Transformer_lightning import SimpleEncoderModule
########################################################################

from datasets import MetaDataModule
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################################################
'''
Change checkpoint_path accordingly
'''
checkpoint_path = '../../paper_run_scripts/1d_experiments_FNO/lightning_logs/04mggjms/checkpoints/epoch=99-step=42500.ckpt'
########################################################################
checkpoint = torch.load(checkpoint_path, map_location=device)


api = wandb.Api()
########################################################################
'''
Change 'config.json' accordingly
'''

import json
with open('config.json') as f:
    config = json.load(f)



#run = api.run('edoardo-calvello/DeepNewtonOperator_Burgers/04mggjms')
#config = run.config
########################################################################


########################################################################
'''
uncomment based on model choice. FNO module refers to both FNO1D and FNO2D, 
so it can be used for both. SimpleEncoderModule is the transformer model.
'''

#valid_model_args = {key: value for key, value in config.items() if key in SimpleEncoderModule.__init__.__code__.co_varnames}
valid_model_args = {key: value for key, value in config.items() if key in FNOModule.__init__.__code__.co_varnames}

#model = SimpleEncoderModule(**valid_model_args)
model = FNOModule(**valid_model_args)
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
        #####################################################################
        '''
        For FNO1D and FNO2D, uncomment permute
        '''
        y_pred = model(x.unsqueeze(0), coords_x=coords_x.unsqueeze(0)) #.permute(0,2,1)
        #####################################################################
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - t0)
        
        error_sample = torch.sqrt(torch.mean(torch.abs(y_pred.to(device) - (y.unsqueeze(0)))**2))/torch.sqrt(
            torch.mean(torch.abs(y.unsqueeze(0))**2))
    
        predictions.append(y_pred.to('cpu').numpy())
        errors.append(error_sample.to('cpu').numpy())


avg_error = np.mean(errors)
avg_forward_time = np.mean(forward_times)

print(f"Average relative L2 error: {avg_error:.6f}")
print(f"Average forward pass time: {avg_forward_time * 1e3:.3f} ms")

predictions = np.array(predictions)
errors = np.array(errors)

# Find the indices of the samples with the median and lowest error
median_idx = np.argsort(errors)[len(errors) // 2]
median_error= errors[median_idx]
min_error_idx = np.argmax(errors)
min_error = errors[min_error_idx]

print(f"Median relative L2 error: {median_error:.6f}")