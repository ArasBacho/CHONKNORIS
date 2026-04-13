from ordering import *

def __col(Theta, s, nugget = 1e-12):
    """
    compute \Theta_{s, s}^{-1}e_j / \sqrt{e_j^T\Theta_{s, s}^{-1}e_j}

    A simple implementation is to invert \Theta_{s, s} directly using the following code

    m = torch.inverse(Theta[s][:, s])
    return m[:, -1] / torch.sqrt(m[-1, -1])

    However, it has some stability issues. When, \Theta_{s, s} is ill-conditioned, torch.inverse(Theta[s, s]) may not
    give us the accurate solution and e_j^T\Theta_{s, s}^{-1}e_j may be negative and torch.sqrt(m[-1, -1]) produces NaN value.

    Thus, instead, we use standard Cholesky directly on \Theta_{s, s}^{-1} and also adding a small nugget to prevent numerical issue
    """
    try:
        m = Theta[s][:, s] + nugget * torch.eye(len(s))
        L = torch.linalg.cholesky(m)
        ej = torch.zeros(len(s))  # Create a tensor of zeros
        ej[-1] = 1
        v = torch.cholesky_solve(ej.unsqueeze(-1), L, upper=False).squeeze(-1)
        # print(v.shape)
        if v[-1] < 0:
            raise ValueError(f"Negative value encountered for square root: {v[-1]}")
        vl = torch.sqrt(v[-1])
        return v / vl
    except torch.linalg.LinAlgError as e:
        # Handle Cholesky decomposition failure
        print("When doing sparse Cholesky decomposition, Cholesky decomposition of the submatrix \Theta_{s, s} " + f"failed: {e}")
        raise  # Optionally re-raise the exception after logging
    except ValueError as e:
        # Handle negative square root or other value issues
        print("When doing sparse Cholesky decomposition, \Theta_{s, s}[-1, -1] is negative. " + f"Value error encountered: {e}")
        raise  # Optionally re-raise the exception


def __cholesky(Theta, sparsity):
    n = Theta.size(0)
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    # Prepare storage for sparse matrix components
    data = torch.zeros(total_nonzeros, dtype=torch.float64)
    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        col_data = __col(Theta, s)
        start, end = indptr[i], indptr[i + 1]

        data[start:end] = col_data
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    sparse_cholesky = torch.sparse_coo_tensor(indices, data, size=(n, n))
    return sparse_cholesky

def non_zeros(n, sparsity):
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        start, end = indptr[i], indptr[i + 1]
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    return indices

# def sparse_cholesky(x, Theta, rho) -> torch.sparse_coo_tensor:
#     """
#     Computes Cholesky with at most s nonzero entries per column.
#     Args:
#         x: points
#         Theta: positive definite matrix
#         rho: a factor controls the sparsity of the cholesky factor
#     Returns:
#         Sparse Cholesky factor L as a torch.sparse_coo_tensor, \Theta^{-1} = LL^T.
#     """
#     indices, lengths = maximin(x)
#     sparsity = sparsity_pattern(x[indices], lengths, rho)
#
#     reordered_Theta = Theta[indices][:, indices]
#     return __cholesky(reordered_Theta, sparsity), indices


def sparse_cholesky(Theta, Perm, sparsity) -> torch.sparse_coo_tensor:
    reordered_Theta = Theta[Perm][:, Perm]
    return __cholesky(reordered_Theta, sparsity)


if __name__ == '__main__':
    import torch
    torch.set_default_dtype(torch.float64)

    A = torch.tensor([[4.0, 1.0, 0.5, 0.0, 0.0],
                      [1.0, 3.0, 0.5, 0.0, 0.0],
                      [0.5, 0.5, 2.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0, 3.0, 0.5],
                      [0.0, 0.0, 0.0, 0.5, 1.0]])
    Theta = A @ A.T  # Ensure positive definiteness

    # Input points (not used in this example, but required for function signature)
    x = torch.linspace(0, 1, 6)[1:][:, None]

    # Sparsity control parameter
    rho = 3

    # Compute sparse Cholesky factor
    # U_sparse, P = sparse_cholesky(x, Theta, rho)

    Perm, lengths = maximin(x)
    sparsity = sparsity_pattern(x[Perm], lengths, rho)
    U_sparse = sparse_cholesky(Theta, Perm, sparsity)

    # Convert sparse tensor to dense for validation
    U_dense = U_sparse.to_dense()

    invPerm = torch.argsort(Perm)
    # Validate L_sparse satisfies Theta^-1 = LL^T
    Theta_inverse_reconstructed = (U_dense @ U_dense.T)[invPerm][:, invPerm]
    Theta_inverse = torch.inverse(Theta)

    # Condition number before preconditioning
    cond_before = torch.linalg.cond(Theta)

    # Precondition Theta
    Theta_preconditioned = U_dense.T @ Theta[Perm][:, Perm] @ U_dense #L_dense @ Theta @ L_dense.T

    # Condition number after preconditioning
    cond_after = torch.linalg.cond(Theta_preconditioned)

    # Output results
    print("Original Theta^-1:\n", Theta_inverse)
    print("\nReconstructed Theta^-1:\n", Theta_inverse_reconstructed)
    print("\nSparse Cholesky Factor (U):\n", U_dense)
    print("\nCondition number before preconditioning:", cond_before.item())
    print("Condition number after preconditioning:", cond_after.item())

    # Check correctness
    reconstruction_error = torch.linalg.norm(Theta_inverse - Theta_inverse_reconstructed, ord='fro')
    print("\nReconstruction error of Theta^-1:", reconstruction_error.item())

    # import torch
    # import matplotlib.pyplot as plt
    # from cholesky import *
    # from ordering import *
    #
    # torch.set_default_dtype(torch.float64)
    #
    # # Generate grid points for Gaussian kernel
    # N = 50  # Number of points
    # x = torch.linspace(0, 1, N)[:, None]
    #
    #
    # # Gaussian kernel function
    # def gaussian_kernel(x1, x2, sigma=0.1):
    #     return torch.exp(-((x1 - x2.T) ** 2) / (2 * sigma ** 2))
    #
    #
    # # Generate Theta (Gram matrix from Gaussian kernel)
    # sigma = 0.1  # Length scale of the kernel
    # Theta = gaussian_kernel(x, x, sigma=sigma)
    #
    # # Range of rho values (log scale)
    # rho_values = torch.linspace(1, 100, 20)  # Rho values from 0.1 to ~30
    # print(rho_values)
    # # rho_values = torch.logspace(0.0, 2.0, steps=20)
    #
    # errors = []
    #
    # Perm, lengths = maximin(x)
    #
    # for rho in rho_values:
    #     ########################################################
    #     # Compute permutation and sparsity pattern
    #     sparsity = sparsity_pattern(x[Perm], lengths, rho)
    #     # Compute sparse Cholesky decomposition
    #     U_sparse = sparse_cholesky(Theta, Perm, sparsity)
    #     # Convert sparse tensor to dense (if needed)
    #     U_dense = U_sparse.to_dense()
    #     # Reconstruct Theta^-1
    #     invPerm = torch.argsort(Perm)
    #     Theta_inverse_reconstructed = (U_dense @ U_dense.T)[invPerm][:, invPerm]
    #     Theta_inverse = torch.inverse(Theta)
    #
    #     # Compute reconstruction error (Frobenius norm)
    #     reconstruction_error = torch.linalg.norm(Theta_inverse - Theta_inverse_reconstructed, ord='fro')
    #     errors.append(reconstruction_error.item())
    #
    # # Plot results
    # plt.figure()
    # print(rho_values)
    # plt.plot(rho_values.numpy(), errors, marker='o', label='Reconstruction Error')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.xlabel(r'$\rho$ (log scale)')
    # plt.ylabel('Reconstruction Error (Frobenius Norm)')
    # plt.title(r'Reconstruction Error vs Sparsity Parameter $\rho$')
    # plt.legend()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.show()
