import torch
import torch.nn as nn

# Gaussian Process class with learnable nugget
class GaussianProcess(nn.Module):
    def __init__(self, d, length_scale=1.0, nugget_init=1e-6, nu=3 / 2):
        """
        :param d:               # The input dimension of GP
        :param length_scale:    # The initial lengthscale of the GP
        :param nugget_init:     # The nugget for regularize the covariance matrix for positivity
        :param nu:              # The regularity of the GP
        """
        super(GaussianProcess, self).__init__()
        self.length_scale = nn.Parameter(torch.ones(d) * length_scale)
        # self.length_scale = nn.Parameter(torch.ones(1) * length_scale)
        # self.nugget = nn.Parameter(torch.tensor(nugget_init))
        self.nugget = torch.tensor(nugget_init)
        self.nu = nu

    def matern_kernel(self, X1, X2):
        # Compute pairwise Euclidean distance between points in scaled X1 and X2
        X1_scaled = X1 * self.length_scale
        X2_scaled = X2 * self.length_scale
        pairwise_dists = torch.cdist(X1_scaled, X2_scaled, p=2)

        if self.nu == 3 / 2:
            scale = torch.sqrt(torch.tensor(3.0)) * pairwise_dists
            K = (1 + scale) * torch.exp(-scale)
        else:
            raise ValueError("Only nu=3/2 is supported.")
        return K

    def fit(self, X_train, y_train):
        # Kernel matrices with learnable nugget
        K_train = self.matern_kernel(X_train, X_train) + torch.eye(X_train.shape[0], device=X_train.device) * self.nugget
        L = torch.linalg.cholesky(K_train)
        alpha_train = torch.cholesky_solve(y_train, L)
        self.alpha_train = alpha_train
        self.X_train = X_train

    def forward(self, X_val):
        K_val_train = self.matern_kernel(X_val, self.X_train)
        y_pred_val = K_val_train @ self.alpha_train
        return y_pred_val