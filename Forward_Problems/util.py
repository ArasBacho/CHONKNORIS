import torch 
from math import pi
import numpy as np 

"""
https://github.com/yangx0e/HierarchicalOperatorLearning/blob/aleksei/debug_darcy.py#L554
https://github.com/PKU-CMEGroup/InverseProblems.jl/blob/master/Fluid/Darcy-2D.ipynb
"""

def build_kl_modes_2d(
    #Nx: int,      # grid resolution => we'll have Nx x Nx points in [0,1]^2
    X_2d,
    Y_2d,
    tau: float,   # parameter in  (π² (l1² + l2²) + τ²)^(-d)
    d: float,     # exponent in    ( ... )^(-d)
    Ntheta: int,  # how many modes to keep
    lmax: int=20
):
    """
    Build the top Nθ KL modes (λ, ψ) for log(a(x)):

        λ_{(l1,l2)} = [ π²(l1² + l2²) + τ² ]^(-d),
        ψ_{(l1,l2)}(x1,x2) =
          √2·cos(π·l1·x1)                if l1>0, l2=0
          √2·cos(π·l2·x2)                if l1=0, l2>0
          2·cos(π·l1·x1)·cos(π·l2·x2)    if l1>0, l2>0
    excluding (l1,l2)=(0,0).

    Returns:
      lam  (torch.Tensor, shape (Nθ,)): the top eigenvalues, sorted descending
      psi  (torch.Tensor, shape (Nθ, Nx, Nx)): the corresponding eigenfunctions
    """
    # 1) Enumerate l1, l2 in [0..lmax], except (0,0)
    lvals = torch.arange(lmax+1)
    # shape => (lmax+1,); we'll make a meshgrid:
    L1_2d, L2_2d = torch.meshgrid(lvals, lvals, indexing='ij')  # each => (lmax+1, lmax+1)

    # Flatten to 1D:
    l1_flat = L1_2d.flatten()  # shape ( (lmax+1)^2, )
    l2_flat = L2_2d.flatten()

    # Filter out (l1=0, l2=0)
    mask_nonzero = ~((l1_flat==0) & (l2_flat==0))
    l1_non0 = l1_flat[mask_nonzero]  # shape (Ncand,)
    l2_non0 = l2_flat[mask_nonzero]

    # 2) Compute lam = [π²(l1² + l2²) + τ²]^(-d)
    lam_vals = ((pi**2)*(l1_non0**2 + l2_non0**2) + tau**2)**(-d)  # shape (Ncand,)

    # 3) Sort by lam descending; pick top Ntheta
    lam_sorted, idx_sort = torch.sort(lam_vals, descending=True)
    lam_top     = lam_sorted[:Ntheta]             # shape (Ntheta,)
    idx_top     = idx_sort[:Ntheta]              # shape (Ntheta,)
    l1_top      = l1_non0[idx_top]               # shape (Ntheta,)
    l2_top      = l2_non0[idx_top]

    # 4) Build the functions ψ_{(l1,l2)}(x1,x2) on the Nx x Nx grid
    #    We'll do it in a fully vectorized manner:
    #    X, Y => shape (Nx, Nx), containing coordinates in [0,1].
    #xvals = torch.linspace(0, 1, Nx)
    #X_2d, Y_2d = torch.meshgrid(xvals, xvals, indexing='ij')  # shape => (Nx, Nx)

    # We want an array psi_all shape => (Ntheta, Nx, Nx).
    # We'll handle the factor:
    #   factor =  2.0        if l1>0 && l2>0
    #             sqrt(2.0)  if (l1>0,l2=0) or (l1=0,l2>0)
    # We'll broadcast that factor into the Nx x Nx result of cos(...)*cos(...).

    # shape => (Ntheta,)
    bothPos_mask = (l1_top>0) & (l2_top>0)
    oneZero_mask = ~bothPos_mask  # i.e. exactly one is zero, the other > 0

    factor = torch.zeros_like(l1_top)
    factor[bothPos_mask] = 2.0
    factor[oneZero_mask] = np.sqrt(2.0)

    # Now build cos(π l1 x1) => shape(Ntheta,Nx,Nx) by broadcasting:
    #   - l1_top => shape(Ntheta,)
    #   - X_2d   => shape(Nx,Nx) => we'll make it shape(1,Nx,Nx)
    #   => multiply => shape(Ntheta,Nx,Nx)
    Nx = X_2d.size(-1) 

    assert X_2d.shape==(Nx,Nx)
    X_3d = X_2d.unsqueeze(0)  # shape (1,Nx,Nx)
    l1_3d= l1_top.view(-1,1,1) # shape (Ntheta,1,1)
    cos_l1X = torch.cos(pi * l1_3d * X_3d)  # shape(Ntheta,Nx,Nx)

    assert Y_2d.shape==(Nx,Nx)
    Y_3d = Y_2d.unsqueeze(0)  # shape(1,Nx,Nx)
    l2_3d = l2_top.view(-1,1,1)
    cos_l2Y = torch.cos(pi * l2_3d * Y_3d)  # shape(Ntheta,Nx,Nx)

    # Combine: psi_all = factor[:,None,None]* cos_l1X * cos_l2Y
    psi_all = factor.view(-1,1,1) * cos_l1X * cos_l2Y  # shape(Ntheta,Nx,Nx)

    return lam_top, psi_all

def generate_random_a_kl(R, Nx, lam, psi):
    """
    Nx x Nx grid
    lam: shape (Nθ,)
    psi: shape (Nθ, Nx, Nx)

    log a(x) = Σ_{k=1..Nθ} [ θ_k * sqrt(lam[k]) * psi[k](x) ],
    where θ_k ~ Normal(0,1).

    Returns a_field = exp(log_a), shape(Nx,Nx).
    """
    Ntheta = lam.shape[0]
    theta  = torch.randn(R,Ntheta)   # each ~ N(0,1)

    # Vectorized sum over k:
    #   log_a(x) = Σ θ_k sqrt(lam[k]) psi[k](x).
    # We'll do:
    #   log_a = (theta * sqrt(lam)) has shape(Nθ,) => call it coeffs
    #   then a sum over k dimension => result (Nx,Nx)

    coeffs = theta * lam.sqrt()  # shape (Nθ,)
    # Expand psi => shape(Nθ,Nx,Nx), multiply by coeffs[:,None,None], then sum over Nθ
    #log_a = torch.einsum('k,kij->ij', coeffs, psi)  # shape (Nx,Nx)
    log_a = (coeffs[...,None,None]*psi).sum(-3)
    # Alternatively, we could do a manual sum or keep a for-loop in dimension k only,
    # but einsum is fully vectorized.

    # a_field = torch.exp(log_a)
    # return a_field
    return log_a