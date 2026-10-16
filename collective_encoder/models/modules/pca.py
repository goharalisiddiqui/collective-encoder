import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class PCAModule(nn.Module):
    """PyTorch module implementing Principal Component Analysis (PCA).

    Supports exact SVD-based fitting, low-rank linear projection from data space
    to latent principal component space, and inverse linear reconstruction.
    All fitted parameters are registered as persistent buffers to ensure seamless
    checkpoint serialization and loading.

    Args:
        input_dim: Number of input features (data dimension D).
        latent_dim: Number of principal components to extract (latent dimension K).
        center: Whether to subtract empirical feature means before SVD.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.center = bool(center)

        if self.latent_dim > self.input_dim:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) cannot exceed input_dim ({self.input_dim})"
            )

        # Persistent buffers saved in PyTorch checkpoints
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer(
            "components", torch.zeros((self.input_dim, self.latent_dim), dtype=torch.float32)
        )
        self.register_buffer(
            "explained_variance", torch.zeros(self.latent_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "explained_variance_ratio", torch.zeros(self.latent_dim, dtype=torch.float32)
        )

    def fit(self, X: torch.Tensor) -> "PCAModule":
        """Fits PCA on the dataset tensor X (N, D) via SVD.

        Args:
            X: Input data tensor of shape (N, D).
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input tensor (N, D), got shape {X.shape}")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {X.shape[1]}"
            )

        N = X.shape[0]
        if N < self.latent_dim:
            raise ValueError(
                f"Number of samples ({N}) must be at least latent_dim ({self.latent_dim})"
            )

        X_float = X.to(dtype=torch.float32, device=self.components.device)

        if self.center:
            mean_vec = torch.mean(X_float, dim=0)
            X_c = X_float - mean_vec
        else:
            mean_vec = torch.zeros(self.input_dim, device=X_float.device)
            X_c = X_float

        # SVD: X_c = U @ diag(S) @ Vh
        # Right singular vectors (columns of V) correspond to principal directions
        _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
        V = Vh.mH

        W = V[:, : self.latent_dim]
        var_all = (S**2) / max(1, N - 1)
        exp_var = var_all[: self.latent_dim]
        total_var = torch.sum(var_all)
        exp_var_ratio = exp_var / (total_var + 1e-12)

        self.mean.copy_(mean_vec)
        self.components.copy_(W)
        self.explained_variance.copy_(exp_var)
        self.explained_variance_ratio.copy_(exp_var_ratio)
        self.is_fitted.copy_(torch.tensor(True, dtype=torch.bool))

        _log.info(
            "Fitted PCA: input_dim=%d -> latent_dim=%d | Total Explained Variance Ratio: %.4f",
            self.input_dim,
            self.latent_dim,
            float(torch.sum(exp_var_ratio)),
        )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input data x into principal component latent space.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Latent projection tensor of shape (..., latent_dim).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "PCAModule is not fitted yet. Fit the module on training data before calling forward."
            )
        x_float = x.to(dtype=torch.float32, device=self.components.device)
        if self.center:
            x_float = x_float - self.mean
        return torch.matmul(x_float, self.components)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs data from latent principal component representation.

        Args:
            z: Latent tensor of shape (..., latent_dim).

        Returns:
            Reconstructed data tensor of shape (..., input_dim).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "PCAModule is not fitted yet. Fit the module before calling inverse."
            )
        z_float = z.to(dtype=torch.float32, device=self.components.device)
        x_rec = torch.matmul(z_float, self.components.t())
        if self.center:
            x_rec = x_rec + self.mean
        return x_rec
