# Add argparse at the very top
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Benchmark GPR with Optuna")
parser.add_argument("name", type=str, help="Name of the experiment/problem")
parser.add_argument("--n_trials", type=int, default=1000, help="Number of trials for Optuna optimization")
args = parser.parse_args()

experiment_name = args.name
n_trials = args.n_trials

#%%

import sys
from pathlib import Path

root = Path(__file__).resolve().parent

for folder in ["utils", "data"]:
    path_to_add = root / folder
    if str(path_to_add) not in sys.path:
        sys.path.append(str(path_to_add))

#%%
# %load_ext autoreload
# %autoreload 2

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
#%% first look at the pca on the input data
n_input = int(jnp.min(jnp.array(u_t.shape)))
pca = PCA(n_components=n_input)
pca.fit(u_t)
variance_input = pca.explained_variance_ratio_
machine_precision_input = int(jnp.min(jnp.array([jnp.where( 1- jnp.cumsum(pca.explained_variance_ratio_) <= 1e-15)[0][0] + 1, u_t.shape[0], u_t.shape[1]])))
plt.plot(jnp.arange(n_input) + 1, 1- jnp.cumsum(pca.explained_variance_ratio_))
plt.xticks([machine_precision_input])
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Explained Variance (input)')
plt.yscale("log")

#%% Now look at the pca on the output data
n_output = int(jnp.min(jnp.array(v_t.shape)))
pca = PCA(n_components=n_output)
pca.fit(v_t)
variance_output = pca.explained_variance_ratio_
machine_precision_output = int(jnp.min(jnp.array([jnp.where( 1- jnp.cumsum(pca.explained_variance_ratio_) <= 1e-15)[0][0] + 1, v_t.shape[0], v_t.shape[1]])))
plt.plot(jnp.arange(n_output) + 1, 1- jnp.cumsum(pca.explained_variance_ratio_))
plt.xticks([machine_precision_output])
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Explained Variance (output)')
plt.yscale("log")

#%% During the cross validation, we only have 80% of the data for training, so we need to adjust the maximum number of components accordingly
max_pca_input = int(jnp.min(jnp.array([machine_precision_input,u_t.shape[1],int(u_t.shape[0])*4/5])))#int(jnp.min(jnp.array([machine_precision_input, int(u_t.shape[0])*4/5])))
max_pca_output = int(jnp.min(jnp.array([machine_precision_output, v_t.shape[1], int(v_t.shape[0])*4/5])))#int(jnp.min(jnp.array([machine_precision_output, int(v_t.shape[0])*4/5])))

if pb_name == 'Calderon': # Calderon has a lot of input features, so we limit the PCA input components to 100 to fit in memory
    max_pca_input = 100
    print(f"Limiting PCA input components to {max_pca_input} for inverse_scattering problem to fit in memory")
    print("Missing variance is", 1- jnp.sum(variance_input[:max_pca_input]))
elif pb_name == 'inverse_scattering': # inverse_scattering has a lot of input features, so we limit the PCA input components to 100 to fit in memory
    max_pca_input = 100
    print(f"Limiting PCA input components to {max_pca_input} for inverse_scattering problem to fit in memory")
    print("Missing variance is", 1- jnp.sum(variance_input[:max_pca_input]))

#%%
initial_guess = {
    "alpha": 1e-8,
    "length_scale": 1.0,
    "normalize": False,
    "pca_input_components": max_pca_input,
    "pca_output_components": max_pca_output,
    "weight_matern": 0.5,  
    "nu": "1.5"  # Default value for nu
}



matern_kernel = matern_kernel_create(nu=initial_guess["nu"])  # Default value for nu
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
    parameters=jnp.array([initial_guess['length_scale'], initial_guess['weight_matern']]),  # Initial guess for length_scale
    alpha= initial_guess['alpha'],
    pca_input_components=initial_guess['pca_input_components'],
    pca_output_components = initial_guess['pca_output_components'],
    normalize= initial_guess['normalize'],
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
print("Initial cross validation loss ", jnp.mean(cross_val_score(model, u_t, v_t, cv=5, scoring=None)))  # Uses model.score


#%%

#%% Optuna objective function

def objective(trial):
    try:
        pca_input_components = trial.suggest_int('pca_input_components', 0, max_pca_input)
        alpha = trial.suggest_float('alpha', 1e-11, 1e0, log=True)
        length_scale = trial.suggest_float('length_scale', 1e-5, 1e5, log=True)
        normalize = trial.suggest_categorical('normalize', [True, False])
        pca_output_components = trial.suggest_int('pca_output_components', 0, max_pca_output)
        weight_matern = trial.suggest_float('weight_matern', 0.0, 1.0)

        nu = trial.suggest_categorical('nu', ['0.5', '1.5', '2.5', 'inf'])

        if pb_name == 'inverse_scattering':
            use_relative_l2_loss = trial.suggest_categorical('use_relative_l2_loss', [True, False])
        else:
            use_relative_l2_loss = False

        parameters = jnp.array([length_scale, weight_matern])
        matern_kernel = matern_kernel_create(nu=nu)  # Create the Matern kernel with the suggested nu

        def kernel(x,y, param):
            length_scale = param[0]
            weight_matern = param[1]
            weight_dot = 1.0 - weight_matern
            return weight_matern * matern_kernel(x, y, length_scale) + weight_dot * dot_kernel(x, y)


        model = BenchmarkGPR(
            kernel=kernel,
            parameters=parameters,
            alpha=alpha,
            normalize=normalize,
            pca_input_components=pca_input_components,
            pca_output_components=pca_output_components,
            batch_size=batch_size,  # Set to None for no batching
            relative_l2=use_relative_l2_loss,  # Use relative L2 loss if specified
        )

        score = cross_val_score(model, u_t, v_t, cv=5, scoring = None)  # Uses model.score
        score = jnp.mean(score)
        if jnp.isnan(score) or jnp.isinf(score):
            return jnp.inf
        return score

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return jnp.inf

#%%
if pb_name == 'inverse_scattering':
    initial_guess = {
        "alpha": 1e-10,
        "length_scale": 1.0,
        "normalize": False,
        "pca_input_components": max_pca_input,
        "pca_output_components": max_pca_output,
        "weight_matern": 0.5,  
        "nu": "1.5",  # Default value for nu
        "relative_l2_loss": True  # Use relative L2 loss for inverse_scattering problem
    }
else:
    initial_guess = {
        "alpha": 1e-8,
        "length_scale": 1.0,
        "normalize": False,
        "pca_input_components": max_pca_input,
        "pca_output_components": max_pca_output,
        "weight_matern": 0.5,  
        "nu": "1.5",  # Default value for nu
    }


#%%

study = optuna.create_study(
    study_name=experiment_name,
    sampler=optuna.samplers.TPESampler(seed=42),
    direction="minimize",
    load_if_exists=True,   # lets you resume from the same DB
)
# Add initial guess as first trial
study.enqueue_trial(initial_guess)


#study.optimize(objective, n_trials=n_trials, callbacks=[save_current_best_if_improved])
study.optimize(objective, n_trials=n_trials)
print("Best trial:")
print("  Value: {}".format(study.best_trial.value))
print("  Params: ")
for key, value in study.best_trial.params.items():
    print("    {}: {}".format(key, value))

#%% Fit the model with the best parameters
best_params = study.best_trial.params

nu = best_params['nu']
matern_kernel = matern_kernel_create(nu=nu)  # Create the Matern kernel with the best nu
def kernel(x,y, param):
    length_scale = param[0]
    weight_matern = param[1]
    weight_dot = 1.0 - weight_matern
    return weight_matern * matern_kernel(x, y, length_scale) + weight_dot * dot_kernel(x, y)

#%%
parameters = jnp.array([best_params['length_scale'], best_params['weight_matern']])
best_params["parameters"] = list([best_params['length_scale'], best_params['weight_matern']])


#%%
model = BenchmarkGPR(
    kernel = kernel,
    alpha=best_params['alpha'],
    parameters=parameters,
    normalize=best_params['normalize'],
    pca_input_components=best_params['pca_input_components'],
    pca_output_components=best_params['pca_output_components'],
    batch_size=batch_size,  # Set to None for no batching
    relative_l2= best_params.get('use_relative_l2_loss', False),  # Use relative L2 loss if specified
)

#%%
print("Cross-validated relative L2 loss (from the trials): ", study.best_trial.value)
print("Cross-validated relative L2 loss (measured after the fact):", jnp.mean(cross_val_score(model, u_t, v_t, cv=5, scoring=None)))


#%%
model.fit(u_t, v_t)
pred = model.predict(u_v)
print("Test relative L2 loss:", relative_l2_loss(v_v, pred))
print("Test L2 loss:", l2_loss(v_v, pred))
#%%
# Save the model parameters and predictions
import numpy as np
import os
import json


results_folder ='results'
os.makedirs(results_folder, exist_ok=True)  # ensures the folder exists

# Save the model parameters and predictions
with open(results_folder + '/' + pb_name + '_model_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# Save the predictions and errors
np.savez(results_folder + '/' + pb_name + '_predictions.npz', 
         v_v=v_v, 
         pred=pred, 
         rel_l2_loss=relative_l2_loss(v_v, pred), 
         l2_loss=l2_loss(v_v, pred))
