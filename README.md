# CHONKNORIS: Operator Learning at Machine Precision

Python code for the paper ["Operator learning at machine precision"](https://arxiv.org/abs/2511.19980) implementing the CHONKNORIS method (Cholesky Newton--Kantorovich Neural Operator Residual Iterative System).

```bibtex
@article{bacho.CHONKNORIS,
  title               = {Operator learning at machine precision},
  author              = {Aras Bacho and Aleksei G. Sorokin and Xianjin Yang and Th\'{e}o Bourdais and Edoardo Calvello and Matthieu Darcy and Alexander Hsu and Bamdad Hosseini and Houman Owhadi},
  year                = {2025},
  doi                 = {10.48550/arXiv.2511.19980},
}
```

## Installation 

Run the following command to install all packages necessary to run files in this repo. 

``` 
pip install -e .
```

## Main Directory Structure

- `Forward_Problems`
  - `burgers_1d`
  - `darcy_2d`
  - `elliptic_1d`
- `Foundational_Model_FONKNORIS`
- `Inverse_Problems` 
  - `Calderon`
  - `full_waveform_inversion`
  - `inverse_scattering`
- `paper_plots`
  - `forward_problems`
  - `inverse_problems`
- `benchmarks`
  - `gaussian_process`
  - `FNO_TNO`: Fourier Neural Operators (FNOs) and Transformer Neural Operators (TNOs)
- `chonknoris`: utilities for running problems 

## Accessing Pre-generated Datasets and Models

Various datasets and models are stored when running the different problems. As some of these computations are quite expensive, pre-generated datasets and models are available upon requests to the lead author of the problem, see above.
