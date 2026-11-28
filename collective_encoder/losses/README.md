# Losses

This module defines loss functions used by neural network models during training.

## Interface

All loss functions must inherit from `CELossBase` defined in `base.py`.

### `CELossBase`

Inherits from both `torch.nn.Module` and `CEModule`. Subclasses must implement `forward`:

```python
def forward(
    self,
    inp: torch.Tensor,
    latent: torch.Tensor,
    output: torch.Tensor,
    labels: torch.Tensor,
    meta: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```
It returns a tuple of:
1. The primary scalar loss tensor (`torch.Tensor`) used for backpropagation.
2. A dictionary of auxiliary loss components or diagnostics (`Dict[str, torch.Tensor]`) for logging.

## Registration & Resolution

Losses are registered under the `"losses"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("losses", CELossBase)`
- **Resolution**: `get_module("losses", loss_type)` from `collective_encoder.common.registry` (replacing the legacy `resolver.py`).
- **Adding a new Loss**: Subclass `CELossBase` and register it in `_BUILTINS` inside `losses/__init__.py` or via:
  ```python
  from collective_encoder.common.registry import register
  register("losses", "MY_LOSS", "my_package.losses", "MyLoss")
  ```

## Available Losses

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `CELossMSE` | `CELossMSE` | Standard Mean Squared Error reconstruction loss. |
| `CELossMSEDict` | `CELossMSEDict` | Dictionary-based MSE loss for multi-head outputs (e.g., bond lengths, angles, dihedrals in `BGE`). |
| `CELossNLL` | `CELossNLL` | Gaussian Negative Log-Likelihood reconstruction loss. |
| `CELossLatent` | `CELossLatent` | Supervises the latent space directly against target labels. |
| `CELossSteric` | `CELossSteric` | Physics-informed loss penalizing steric clashes (interatomic distances below a threshold). |
| `CELossBondDeviation` | `CELossBondDeviation` | Physics-informed loss penalizing non-physical covalent bond length deviations. |
| `CELossKLDUniformGaussian` | `CELossKLDUniformGaussian` | Standard VAE Kullback-Leibler divergence against an isotropic Gaussian prior. |
| `CELossKLDGaussianMixture` | `CELossKLDGaussianMixture` | KLD against a learnable or fixed Gaussian Mixture prior. |
| `CELossKLDFlow` | `CELossKLDFlow` | KLD with a normalizing flow prior. |
