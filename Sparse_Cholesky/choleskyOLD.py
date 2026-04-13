from ordering import *

def __col(Theta, s):
    m = torch.inverse(Theta[s][:, s])
    return m[:, -1] / torch.sqrt(m[-1, -1])

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
    rho = 10

    # Compute sparse Cholesky factor
    # U_sparse, P = sparse_cholesky(x, Theta, rho)
    N = x.shape[0]
    Perm, lengths = maximin(x)
    sparsity = sparsity_pattern(x[Perm], lengths, rho)
    nonzeros_indices = non_zeros(N, sparsity)

    U_sparse = sparse_cholesky(Theta, Perm, sparsity)
    mtx = U_sparse.coalesce().values()

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