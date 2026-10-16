import logging
from typing import Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class TICAModule(nn.Module):
    """Time-lagged Independent Component Analysis (TICA) Module.

    Calculates TICA linear projections that maximize time-autocorrelation at a specified lag time tau.
    Solves the generalized eigenvalue problem: C(tau) W = C(0) W Lambda
    where C(0) is the instantaneous covariance matrix and C(tau) is the time-lagged cross-covariance matrix.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        lag: int = 1,
        epsilon: float = 1e-6,
        center: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.lag = int(lag)
        self.epsilon = float(epsilon)
        self.center = bool(center)

        # Register persistent buffers
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("components", torch.zeros((self.input_dim, self.latent_dim), dtype=torch.float32))
        self.register_buffer("eigenvalues", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("timescales", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("cov_0", torch.zeros((self.input_dim, self.input_dim), dtype=torch.float32))

    def fit(self, X: torch.Tensor) -> "TICAModule":
        """Fits TICA parameters on sequential trajectory data X.

        Args:
            X: 2D Tensor of shape (N, D) representing sequential frames.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.to(dtype=torch.float32)

        N, D = X.shape
        if D != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {D}")
        if N <= self.lag:
            raise ValueError(f"Trajectory length N={N} must be greater than lag time tau={self.lag}")

        # Time-lagged pairs
        X0 = X[:-self.lag]
        Xtau = X[self.lag:]
        n_pairs = X0.shape[0]

        # Empirical mean
        if self.center:
            mean = torch.mean(X, dim=0)
        else:
            mean = torch.zeros(D, dtype=torch.float32, device=X.device)

        X0_centered = X0 - mean
        Xtau_centered = Xtau - mean

        # Symmetrized instantaneous covariance C(0)
        C0 = 0.5 * (X0_centered.T @ X0_centered + Xtau_centered.T @ Xtau_centered) / n_pairs

        # Symmetrized time-lagged cross-covariance C(tau)
        Ctau = 0.5 * (X0_centered.T @ Xtau_centered + Xtau_centered.T @ X0_centered) / n_pairs

        # Solve generalized eigenvalue problem: C(tau) W = C(0) W Lambda
        # 1. SVD / Eigendecomposition of C(0) for whitening
        U, S, Vh = torch.linalg.svd(C0)
        S_clamped = torch.clamp(S, min=self.epsilon)
        K = U @ torch.diag(1.0 / torch.sqrt(S_clamped))

        # 2. Whitened cross-covariance matrix C_whitened = K.T @ C(tau) @ K
        C_whitened = K.T @ Ctau @ K
        C_whitened = 0.5 * (C_whitened + C_whitened.T)  # Ensure exact symmetry

        # 3. Eigendecomposition of symmetric whitened matrix
        eigvals, V = torch.linalg.eigh(C_whitened)

        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigvals, descending=True)
        sorted_eigvals = eigvals[idx]
        sorted_V = V[:, idx]

        # 5. Extract top latent_dim components
        k = min(self.latent_dim, D)
        top_eigvals = sorted_eigvals[:k]
        W = K @ sorted_V[:, :k]

        # Pad with zeros if latent_dim > D
        if k < self.latent_dim:
            W_padded = torch.zeros((D, self.latent_dim), dtype=torch.float32, device=X.device)
            W_padded[:, :k] = W
            W = W_padded

            eigvals_padded = torch.zeros(self.latent_dim, dtype=torch.float32, device=X.device)
            eigvals_padded[:k] = top_eigvals
            top_eigvals = eigvals_padded

        # Compute implied timescales: t_i = -tau / log(|lambda_i|)
        abs_eigvals = torch.abs(top_eigvals)
        clamped_abs = torch.clamp(abs_eigvals, min=1e-8, max=1.0 - 1e-8)
        timescales = -float(self.lag) / torch.log(clamped_abs)

        # Store persistent buffers
        self.mean.copy_(mean)
        self.components.copy_(W)
        self.eigenvalues.copy_(top_eigvals)
        self.timescales.copy_(timescales)
        self.cov_0.copy_(C0)
        self.is_fitted.copy_(torch.tensor(True, dtype=torch.bool, device=X.device))

        _log.info(
            f"TICAModule fitted successfully. Top eigenvalues: "
            f"{[round(float(v), 4) for v in top_eigvals[:min(5, k)]]}"
        )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input data x onto TICA components.

        Args:
            x: Tensor of shape (..., D).

        Returns:
            Latent projections of shape (..., K).
        """
        if not self.is_fitted:
            raise RuntimeError("TICAModule must be fitted before forward pass.")

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")

        x_float = x.to(dtype=self.components.dtype, device=self.components.device)
        x_centered = x_float - self.mean
        return torch.matmul(x_centered, self.components)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Linear reconstruction from TICA latent projections back to input space.

        Reconstruction matrix A = C(0) W.
        Reconstruction: hat{x} = z A.T + mean

        Args:
            z: Tensor of shape (..., K).

        Returns:
            Reconstructed data of shape (..., D).
        """
        if not self.is_fitted:
            raise RuntimeError("TICAModule must be fitted before inverse pass.")

        z_float = z.to(dtype=self.components.dtype, device=self.components.device)
        # Linear reconstruction matrix A = C(0) W
        A = torch.matmul(self.cov_0, self.components)
        x_rec = torch.matmul(z_float, A.T) + self.mean
        return x_rec
