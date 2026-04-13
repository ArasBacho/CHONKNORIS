This folder implements the Gaussian Process Operator Learning framework described in the paper “Learning Operators with Gaussian Processes” (https://arxiv.org/abs/2304.13202).
It provides scripts to train, evaluate, and benchmark the model on the problems considered in the paper.

Usage:

**0. Data extraction**

Before running any experiment, extract the dataset:
unzip data.zip

**1.Training**


To train the model and perform hyperparameter tuning, run:

python train_pb.py <problem_name> –-n_trials <num_trials>

where <problem_name> is the benchmark name and –n_trials specifies the number of hyperparameter optimization rounds (default is 1000).

To train all models sequentially, run:
python train_all_pb.py

Note: training all models can take a significant amount of time.

**2.Prediction**

To load the optimal parameters, fit the model, and compute cross-validation and test losses, run:
python prediction.py <problem_name>

This reproduces the evaluation results reported in the paper.

**3.	Summary**


To summarize results across problems in a single table, run:
python summarize_results.py

Precomputed results:
The “results” folder contains optimized hyperparameters and model predictions for each problem.
These can be used to run steps 2 (Prediction) and 3 (Summary) directly without retraining.


problem_name should be one of:
* burgers_pde	
* darcy_pde_2d
* elliptic_pde
* seismic_res5
* seismic_res7	
* seismic_res10	
* seismic_res14
* InverseScattering	
* Calderon



