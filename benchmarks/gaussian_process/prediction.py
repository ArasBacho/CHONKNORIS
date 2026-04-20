# Add argparse at the very top
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Benchmark GPR with Optuna")
parser.add_argument("name", type=str, help="Name of the experiment/problem")
parser.add_argument("--save", action="store_true", help="Whether to save predictions and errors")
parser.add_argument("--time", action="store_true", help="Whether to time the prediction")
args = parser.parse_args()

experiment_name = args.name

valid_names = ['burgers_pde', 'darcy_pde_2d', 'elliptic_pde', 
               'seismic_res5', 'seismic_res7', 'seismic_res10', 'seismic_res14',
               "inverse_scattering", "Calderon"]

if experiment_name not in valid_names:
    raise ValueError(f"Invalid experiment name. Must be one of: {valid_names}")
#%%

import sys
from pathlib import Path

root = Path(__file__).resolve().parent

for folder in ["utils", "data"]:
    path_to_add = root / folder
    if str(path_to_add) not in sys.path:
        sys.path.append(str(path_to_add))

#%%

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils_jax import GlobalStandardScaler, relative_l2_loss, l2_loss
from utils_jax import BenchmarkGPR  
from utils_jax import PCA, matern_kernel_create, dot_kernel

import optuna
from utils_jax import cross_val_score


#%% Load dataset
folder = 'data'
pb_name = experiment_name
data = jnp.load(folder + '/' + pb_name + '.npz')

print("Loading dataset:", pb_name)

u_t = jnp.array(data['u_t'])
v_t = jnp.array(data['v_t'])
u_v = jnp.array(data['u_v'])
v_v = jnp.array(data['v_v'])
#%%
if pb_name == 'darcy_pde_2d':
    # Extract the first channel for 2D data
    u_t =  u_t[:, 0, :]
    u_v =  u_v[:, 0, :]
elif pb_name == 'inverse_scattering':
    n_train = 9750
    n_test = 10999# Use all test data

    u_t = u_t[:n_train]
    v_t = v_t[:n_train]

    u_v = u_v[:n_test]
    v_v = v_v[:n_test]
elif pb_name == 'Calderon':
    n_train = 7500 # Use 7500 training samples
    n_test = 10999# Use all test data
    u_t = u_t[:n_train]
    v_t = v_t[:n_train]
    u_v = u_v[:n_test]
    v_v = v_v[:n_test]


print("Input shape:", u_t.shape)
print("Output shape:", v_t.shape)


assert u_t.shape[1:] == u_v.shape[1:], "Train and test inputs must have the same shape"
assert v_t.shape[1:] == v_v.shape[1:], "Train and test outputs must have the same shape"

#%%

input_dim = u_t.shape[1:] 
output_dim = v_t.shape[1:]  

u_t = u_t.reshape(u_t.shape[0], -1)  # Flatten the input
v_t = v_t.reshape(v_t.shape[0], -1)  # Flatten the output
u_v = u_v.reshape(u_v.shape[0], -1)  # Flatten the input
v_v = v_v.reshape(v_v.shape[0], -1)  # Flatten the output


print("Flattened input shape:", u_t.shape)
print("Flattened output shape:", v_t.shape)

print("Training samples:", u_t.shape[0])
print("Test samples:", u_v.shape[0])
#%%

results_folder ='results'
# Load the model parameters
import json

print(results_folder + '/' + pb_name + '_model_params.json')
with open(results_folder + '/' + pb_name + '_model_params.json', 'r') as f:
    model_params = json.load(f)

print("Loaded model parameters:")
print(model_params)


#%%

matern_kernel = matern_kernel_create(nu=model_params["nu"])  # Default value for nu
def kernel(x,y, param):
    length_scale = param[0]
    weight_matern = param[1]
    weight_dot = 1.0 - weight_matern
    return weight_matern * matern_kernel(x, y, length_scale) + weight_dot * dot_kernel(x, y)

if pb_name == 'inverse_scattering':
    use_relative_l2_loss = True  # Use relative L2 loss for inverse_scattering problem
else:
    use_relative_l2_loss = False

batch_size = None  # Set to None for no batching
model = BenchmarkGPR(
    kernel=kernel,
    parameters=jnp.array([model_params['length_scale'], model_params['weight_matern']]),  # Initial guess for length_scale
    alpha= model_params['alpha'],
    pca_input_components=model_params['pca_input_components'],
    pca_output_components = model_params['pca_output_components'],
    normalize= model_params['normalize'],
    batch_size=batch_size,  # Set to None for no batching
    jit_kernel=True,
    relative_l2=use_relative_l2_loss,  # Use relative L2 loss if specified
)


#%%


print("Initial guess parameters:")
print("PCA input components:", model.pca_input_components)
print("Alpha:", model.alpha)
print("Length scale:", model.parameters[0])
print("Normalize:", model.normalize)
print("PCA output components:", model.pca_output_components)
print("Weight Matern:", model.parameters[1])
print("Weight Dot:", 1.0 - model.parameters[1])

print("Using the relative L2:", use_relative_l2_loss)


#%%
model.fit(u_t, v_t)  # Fit the model with the training data and weights if provided

#%%
print("Cross validation loss ", jnp.mean(cross_val_score(model, u_t, v_t, cv=5, scoring=None)))  # Uses model.score
pred = model.predict(u_v)
print("Test relative L2 loss:", relative_l2_loss(v_v, pred))
print("Test L2 loss:", l2_loss(v_v, pred))

# %% Optionally, save the predictions and errors
if args.save:
    print("Saving predictions and errors to:", results_folder + '/' + pb_name + '_predictions.npz')
    import numpy as np
    # Save the predictions and errors
    np.savez(results_folder + '/' + pb_name + '_predictions.npz', 
            v_v=v_v, 
            pred=pred, 
            rel_l2_loss=relative_l2_loss(v_v, pred), 
            l2_loss=l2_loss(v_v, pred))

# %%

import time
import json
import os
import numpy as np

import time
import json
import os
import numpy as np

if args.time:
    print("Timing prediction")

    n_runs = 100
    times = []

    # # One-time setup / warm-up
    model.fit(u_t, v_t)
    _ = model.predict(u_v[0][None]).block_until_ready()

    for r in range(n_runs):
        # model.fit(u_t, v_t)
        # _ = model.predict(u_v[0][None]).block_until_ready()
        
        start = time.perf_counter()

        u = u_v[r + 1][None]
        _ = model.predict(u).block_until_ready()
        end = time.perf_counter()
        run_time = (end - start)
        times.append(run_time)

        #print(f"Run {r + 1}: {run_time:.6f} s/pred")

    times_np = np.array(times)

    stats = {
        "n_runs": n_runs,
        "n_predictions_per_run": 1,
        "times_per_prediction": [float(t) for t in times],
        "time_per_prediction_median": float(np.median(times_np)),
        "time_per_prediction_mean": float(times_np.mean()),
        "time_per_prediction_std": float(times_np.std()),
        "time_per_prediction_min": float(times_np.min()),
        "time_per_prediction_max": float(times_np.max()),
    }

    print("\nSummary:")
    print(json.dumps(stats, indent=2))

    save_dir = results_folder
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, pb_name + "_timing.json")
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved timing results to: {save_path}")

    



