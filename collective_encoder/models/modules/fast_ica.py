import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class FastICAModule(nn.Module):
    r"""PyTorch module implementing the FastICA algorithm.

    Theoretical basis:
    Hyvärinen, A. (1999). "Fast and robust fixed-point algorithms for independent component analysis."
    IEEE Transactions on Neural Networks, 10(3), 626-634.

    Algorithm:
    1. Centering: Subtract empirical mean \mu = E[x].
    2. Whitening: Compute whitening projection K via SVD: X_white = (X - \mu) @ K.
    3. Parallel Fixed-Point Iteration:
       - Update unmixing vectors with contrast non-linearity g(u) (default: 'logcosh'):
         W^+ = (1/N) X_white^T g(X_white W) - (1/N) W diag(\sum g'(X_white W))
       - Symmetric orthogonalization: W \leftarrow (W W^T)^{-1/2} W via SVD.
       - Iterate until convergence max_i |1 - |(W_{new}^T W)_{i, i}|| < tol or max_iter.
    4. Total Linear Mapping:
       - Unmixing matrix W_{ICA} = K @ W \in R^{D x K}.
       - Mixing matrix A = pinv(W_{ICA}) \in R^{K x D}.
       - Forward projection: z = (x - \mu) @ W_{ICA}.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        fun: str = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-4,
        whiten: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.fun = str(fun).lower()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.whiten = bool(whiten)
        self.random_state = random_state

        if self.latent_dim > self.input_dim:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) cannot exceed input_dim ({self.input_dim})"
            )

        if self.fun not in ["logcosh", "exp", "cube"]:
            raise ValueError(
                f"Unknown contrast function '{self.fun}'. Supported: 'logcosh', 'exp', 'cube'."
            )

        # Persistent buffers saved in PyTorch checkpoints
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer(
            "unmixing_matrix",
            torch.zeros((self.input_dim, self.latent_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "mixing_matrix",
            torch.zeros((self.latent_dim, self.input_dim), dtype=torch.float32),
        )

    def _g(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates contrast non-linearity g(u) and its derivative g'(u)."""
        if self.fun == "logcosh":
            g = torch.tanh(u)
            g_prime = 1.0 - g**2
        elif self.fun == "exp":
            exp_term = torch.exp(-0.5 * (u**2))
            g = u * exp_term
            g_prime = (1.0 - u**2) * exp_term
        elif self.fun == "cube":
            g = u**3
            g_prime = 3.0 * (u**2)
        else:
            raise ValueError(f"Unknown fun: {self.fun}")
        return g, g_prime

    def fit(self, X: torch.Tensor) -> "FastICAModule":
        """Fits FastICA on dataset tensor X (N, D).

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

        X_float = X.to(dtype=torch.float32, device=self.unmixing_matrix.device)

        # 1. Centering
        mean_vec = torch.mean(X_float, dim=0)
        X_c = X_float - mean_vec

        # 2. Whitening via SVD: X_c = U S Vh
        # Covariance = (1/N) X_c^T X_c = V (S^2 / N) V^T
        if self.whiten:
            _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
            V = Vh.mH
            # Standard deviation scaling per component
            scale = S[: self.latent_dim] / torch.sqrt(torch.tensor(max(1, N - 1), dtype=torch.float32, device=X_float.device))
            scale = torch.clamp(scale, min=1e-12)
            # Whitening matrix K: (input_dim, latent_dim)
            K = V[:, : self.latent_dim] / scale.unsqueeze(0)
            X_white = torch.matmul(X_c, K)
        else:
            K = torch.eye(self.input_dim, self.latent_dim, device=X_float.device)
            X_white = X_c[:, : self.latent_dim]

        # 3. Parallel FastICA Fixed-Point Iteration on whitened space
        k = self.latent_dim
        gen = torch.Generator(device=X_float.device)
        if self.random_state is not None:
            gen.manual_seed(int(self.random_state))

        # Initial random orthogonal unmixing matrix W in whitened space (k, k)
        W_init = torch.randn(k, k, generator=gen, device=X_float.device, dtype=torch.float32)
        q, _ = torch.linalg.qr(W_init)
        W = q

        converged = False
        iteration = 0

        for it in range(self.max_iter):
            iteration = it + 1
            # Y: (N, k) = X_white @ W
            Y = torch.matmul(X_white, W)
            g_val, g_prime = self._g(Y)

            # W_plus = (1/N) X_white^T @ g(Y) - (1/N) W @ diag(mean(g'(Y)))
            # Term 1: (k, k)
            term1 = torch.matmul(X_white.t(), g_val) / float(N)
            # Term 2: (k, k)
            diag_mean_gp = torch.mean(g_prime, dim=0)  # (k,)
            term2 = W * diag_mean_gp.unsqueeze(0)

            W_plus = term1 - term2

            # Symmetric orthogonalization: W_new = (W_plus @ W_plus^T)^{-1/2} @ W_plus
            # Using SVD: W_plus = U S Vh -> W_new = U @ Vh
            U_w, _, Vh_w = torch.linalg.svd(W_plus, full_matrices=False)
            W_new = torch.matmul(U_w, Vh_w)

            # Check convergence: max_i |1 - |(W_new^T @ W)_{i, i}|| < tol
            lim = torch.max(torch.abs(torch.abs(torch.diag(torch.matmul(W_new.t(), W))) - 1.0))
            W = W_new

            if float(lim) < self.tol:
                converged = True
                break

        # 4. Total unmixing matrix W_ICA: (input_dim, latent_dim)
        W_ica = torch.matmul(K, W)  # (input_dim, k)

        # Mixing matrix A = pinv(W_ica) = (k, input_dim)
        A = torch.linalg.pinv(W_ica)

        self.mean.copy_(mean_vec)
        self.unmixing_matrix.copy_(W_ica)
        self.mixing_matrix.copy_(A)
        self.is_fitted.copy_(torch.tensor(True, dtype=torch.bool))

        status_str = "converged" if converged else "reached max iterations"
        _log.info(
            "Fitted FastICA: input_dim=%d -> latent_dim=%d (%s in %d iterations)",
            self.input_dim,
            self.latent_dim,
            status_str,
            iteration,
        )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input data x into independent component latent space.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Latent independent components tensor of shape (..., latent_dim).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FastICAModule is not fitted yet. Fit the module on training data before calling forward."
            )
        x_float = x.to(dtype=torch.float32, device=self.unmixing_matrix.device)
        x_c = x_float - self.mean
        return torch.matmul(x_c, self.unmixing_matrix)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs data from latent independent components.

        Args:
            z: Latent tensor of shape (..., latent_dim).

        Returns:
            Reconstructed data tensor of shape (..., input_dim).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FastICAModule is not fitted yet. Fit the module before calling inverse."
            )
        z_float = z.to(dtype=torch.float32, device=self.mixing_matrix.device)
        x_rec = torch.matmul(z_float, self.mixing_matrix)
        return x_rec + self.mean
