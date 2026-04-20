#%%
import sys
from pathlib import Path

root = Path(__file__).resolve().parent

for folder in ["utils", "data"]:
    path_to_add = root / folder
    if str(path_to_add) not in sys.path:
        sys.path.append(str(path_to_add))

#%%
import numpy as np
from utils import relative_l2_loss, itemize_relative_l2_loss

import pandas as pd

import joblib

from utils import BenchmarkGPR


#%%

problems = ['burgers_pde', 'darcy_pde_2d', 'elliptic_pde',
             'seismic_res5', 'seismic_res7', 'seismic_res10', 'seismic_res14',
               "inverse_scattering", "Calderon"]
res = {}


for pb_name in problems:


    res_optuna = np.load(f'results/' + pb_name + '_predictions.npz')

    print(f"Problem: {pb_name}")
    print("Optuna relative L2 loss:", res_optuna['rel_l2_loss'])

    pred = res_optuna['pred']
    v_v = res_optuna['v_v']

    errors = itemize_relative_l2_loss(v_v, pred)
    print("Errors shape:", errors.shape)

    # Convert to Series
    error_series = pd.Series(errors)

    res[pb_name] = error_series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])


df_summary = pd.DataFrame(res).T
print(df_summary)
# %%

df_summary

# %%
