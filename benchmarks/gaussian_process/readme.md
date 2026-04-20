# Classic Operator Learning Gaussian Process Benchmarks 

Authored by Matthieu Darcy.

This folder implements the Gaussian Process Operator Learning framework described in the paper ["Kernel Methods are Competitive for Operator Learning”](https://arxiv.org/abs/2304.13202). It provides scripts to train, evaluate, and benchmark the model on the problems considered in the paper.

## Data extraction

Before running any experiment, extract the dataset:

```bash
unzip data.zip
```

## Problem Name Options

The `<problem_name>` options are 

* `elliptic_pde`
* `burgers_pde`
* `darcy_pde_2d`
* `inverse_scattering`	
* `Calderon`
* `seismic_res5`
* `seismic_res7`	
* `seismic_res10`	
* `seismic_res14`

## 1.Training

To train the model and perform hyperparameter tuning, run:

```bash
python train_pb.py <problem_name> –-n_trials <num_trials>
```

where `<problem_name>` is the benchmark name and `–n_trials` specifies the number of hyperparameter optimization rounds (default is 1000).

To train all models sequentially, run:

```bash
python train_all_pb.py
```

Note: training all models can take a significant amount of time.

## 2.Prediction

To load the optimal parameters, fit the model, and compute cross-validation and test losses, run:

```bash
python prediction.py <problem_name>
```

To also run timings on CPU, use

```bash 
JAX_PLATFORMS=cpu python prediction.py <problem_name> --time
```

or on GPU with 

```bash 
JAX_PLATFORMS=cuda python prediction.py <problem_name> --time
```

This reproduces the evaluation results reported in the paper.

## 3.	Summary

To summarize results across problems in a single table, run:

```bash
python summarize_results.py
```

## Precomputed results:

The `results/` folder contains optimized hyperparameters and model predictions for each problem. These can be used to run steps 2 (Prediction) and 3 (Summary) directly without retraining.
