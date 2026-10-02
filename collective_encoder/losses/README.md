# Losses

This module defines loss functions used by the neural networks during training.

## Interface

All loss functions must inherit from `CELossBase` defined in `base.py`.

### `CELossBase`
Inherits from both `torch.nn.Module` and `CEModule`.
Subclasses must implement the `forward` method, which is the standard PyTorch mechanism for defining the forward pass computation.

The `forward` method signature looks like this:
```python
def forward(self,
            inp: torch.Tensor,
            latent: torch.Tensor,
            output: torch.Tensor,
            labels: torch.Tensor,
            meta: Dict[str, torch.Tensor],
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```
It returns a tuple:
1. The total scalar loss value (`torch.Tensor`) that will be used for backpropagation.
2. A dictionary of additional loss components or metrics (`Dict[str, torch.Tensor]`) which will be logged but not backpropagated automatically unless included in the scalar loss.

## KLD and Regularization
There are specialized implementations for computing Kullback-Leibler Divergence (KLD) needed by Variational Autoencoders (VAEs).
- KLD computation modules (`kld_uniform_gaussian.py`, `kld_gaussian_mixture.py`, `kld_flow.py`)
- KLD schedulers inside the `kld_schedulers` subfolder for annealing the beta parameter during training.

## Available Losses
- `MSELoss` / `MSELossDict`: Standard Mean Squared Error reconstruction losses.
- `NLLLoss`: Negative Log-Likelihood loss.
- `BondDeviationLoss`: A physics-informed loss to penalize non-physical bond lengths.
- `StericLoss`: Penalizes atoms being too close to each other.
- `LatentMSELoss`: Forces the encoded latent space to match external targets/labels.
