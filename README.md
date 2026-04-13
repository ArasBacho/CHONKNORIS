# Operator Learning at Machine Precision

**CHONKNORIS** — Cholesky Newton–Kantorovich Neural Operator Residual Iterative System

[![arXiv](https://img.shields.io/badge/arXiv-2511.19980-b31b1b.svg)](https://arxiv.org/abs/2511.19980)

---

## Authors

[Aras Bacho](mailto:bacho@caltech.edu)<sup>★,1</sup>,
Aleksei G. Sorokin<sup>★,2</sup>,
Xianjin Yang<sup>★,1</sup>,
Théo Bourdais<sup>1</sup>,
Edoardo Calvello<sup>1</sup>,
Matthieu Darcy<sup>1</sup>,
Alexander Hsu<sup>3</sup>,
Bamdad Hosseini<sup>3</sup>,
Houman Owhadi<sup>1</sup>

<sup>1</sup> California Institute of Technology &nbsp;|&nbsp;
<sup>2</sup> Illinois Institute of Technology &nbsp;|&nbsp;
<sup>3</sup> University of Washington &nbsp;|&nbsp;
<sup>★</sup> Equal contribution

---

## Abstract

Neural operator learning methods have garnered significant attention in scientific computing for their ability to approximate infinite-dimensional operators. However, increasing their complexity often fails to substantially improve their accuracy, leaving them on par with much simpler approaches such as kernel methods and more traditional reduced-order models. In this article, we set out to address this shortcoming and introduce **CHONKNORIS**, an operator learning paradigm that can achieve machine precision. CHONKNORIS draws on numerical analysis: many nonlinear forward and inverse PDE problems are solvable by Newton-type methods. Rather than regressing the solution operator itself, our method regresses the Cholesky factors of the elliptic operator associated with Tikhonov-regularized Newton–Kantorovich updates. The resulting unrolled iteration yields a neural architecture whose machine-precision behavior follows from achieving a contractive map, requiring far lower accuracy than end-to-end approximation of the solution operator.

We benchmark CHONKNORIS on a range of nonlinear forward and inverse problems, including a nonlinear elliptic equation, Burgers' equation, a nonlinear Darcy flow problem, the Calderón problem, an inverse wave scattering problem, and a problem from seismic imaging. Additionally, we introduce **FONKNORIS** (Foundation Newton–Kantorovich Neural Operator Residual Iterative System), a foundation model variant that aggregates multiple pre-trained CHONKNORIS experts to emulate the solution map of novel nonlinear PDEs such as the Klein–Gordon and Sine–Gordon equations.

---

## Repository Structure

```
CHONKNORIS/
├── chonknoris/                          # Core Python package
│   ├── datasets.py                      # Dataset classes (including Cholesky factor datasets)
│   ├── gp.py                            # Gaussian Process models (GPyTorch-based)
│   ├── gp_custom.py                     # Custom GP with cross-validation training
│   ├── nn.py                            # MLP, DeepONet, FNO wrappers (Lightning)
│   ├── plots.py                         # Plotting utilities
│   └── util.py                          # Train/val split, metrics, kernels
│
├── Sparse_Cholesky/                     # Sparse Cholesky factorization
│   ├── cholesky.py                      # Sparse Cholesky implementation
│   ├── ordering.py                      # Maximin ordering for sparsity patterns
│   └── maxheap.py                       # Max-heap utility for ordering
│
├── Forward_Problems/                    # Forward PDE experiments
│   ├── Semilinear_Elliptic_Equation_1D.ipynb
│   ├── Burgers_Equation_1D.ipynb
│   └── darcy_2d.ipynb
│
├── Inverse_Problems/                    # Inverse PDE experiments
│   ├── Calderon_problem.ipynb
│   ├── inverse_scattering.ipynb
│   └── fwi/                             # Full Waveform Inversion (seismic imaging)
│       ├── fwi.ipynb
│       ├── acoustic_forward_solver.py
│       └── gauss_newton_solver.py
│
├── Foundational_Model_FONKNORIS/        # FONKNORIS foundation model
│   └── FONKNORIS_Foundational_Model.ipynb
│
├── benchmarks/                          # Baseline comparisons
│   └── gaussian_process/                # GP operator learning baseline
│
├── paper_plots/                         # Scripts to reproduce paper figures
│   ├── forward_problems/
│   └── inverse_problems/
│
└── Paper/                               # Paper PDF
    └── 2511.19980v1.pdf
```

---

## Installation

### 1. Create the conda environment

```bash
conda env create -f env.yml
conda activate chonknoris
```

### 2. Install the `chonknoris` package

```bash
pip install -e .
```

### 3. Verify

```python
import chonknoris
import torch
print(torch.cuda.is_available())  # should be True for GPU experiments
```

> **Note:** All experiments require a CUDA-capable GPU and use `torch.float64` (double precision) throughout.

---

## Experiments

Each experiment is a self-contained Jupyter notebook. Run notebooks from their own directory so that relative imports resolve correctly.

### Forward Problems

| Notebook | Problem | Key method |
|---|---|---|
| `Forward_Problems/Semilinear_Elliptic_Equation_1D.ipynb` | Nonlinear elliptic PDE (1D) | CHONKNORIS with sparse Cholesky |
| `Forward_Problems/Burgers_Equation_1D.ipynb` | Burgers' equation | CHONKNORIS with sparse Cholesky |
| `Forward_Problems/darcy_2d.ipynb` | Nonlinear Darcy flow (2D) | CHONKNORIS + GP initial guess |

### Inverse Problems

| Notebook | Problem | Key method |
|---|---|---|
| `Inverse_Problems/Calderon_problem.ipynb` | Calderón's inverse problem | CHONKNORIS |
| `Inverse_Problems/inverse_scattering.ipynb` | Inverse wave scattering | CHONKNORIS |
| `Inverse_Problems/fwi/fwi.ipynb` | Seismic imaging (Full Waveform Inversion) | CHONKNORIS + Gauss–Newton |

### Foundation Model

| Notebook | Problem | Key method |
|---|---|---|
| `Foundational_Model_FONKNORIS/FONKNORIS_Foundational_Model.ipynb` | Klein–Gordon, Sine–Gordon (unseen PDEs) | FONKNORIS |

### Baselines (GP operator learning)

```bash
cd benchmarks/gaussian_process
python train_pb.py <problem_name> --n_trials 1000
python prediction.py <problem_name>
python summarize_results.py
```

where `<problem_name>` is one of: `burgers_pde`, `darcy_pde_2d`, `elliptic_pde`, `InverseScattering`, `Calderon`, `seismic_res5`, `seismic_res7`, `seismic_res10`, `seismic_res14`.

Pre-computed results are available in `benchmarks/gaussian_process/results/`.

### Paper Plots

```bash
# After running experiments, place output .pt files in paper_plots/*/in/
jupyter notebook paper_plots/forward_problems/forward_problems.ipynb
jupyter notebook paper_plots/inverse_problems/inverse_problems.ipynb
```

---

## Citation

```bibtex
@article{bacho2025chonknoris,
  title   = {Operator Learning at Machine Precision},
  author  = {Bacho, Aras and Sorokin, Aleksei G. and Yang, Xianjin and Bourdais, Th{\'e}o
             and Calvello, Edoardo and Darcy, Matthieu and Hsu, Alexander
             and Hosseini, Bamdad and Owhadi, Houman},
  journal = {arXiv preprint arXiv:2511.19980},
  year    = {2025}
}
```
