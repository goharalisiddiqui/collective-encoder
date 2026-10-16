# Metrics

This module provides metric calculation classes used to track and evaluate network performance during and after training (e.g., in validation loops or testing phases). Note that unlike `losses`, these metrics are not used for backpropagation.

## Interface

All metric classes must inherit from `CEMetricBase` found in `base.py`.

### `CEMetricBase`
Inherits from `CEModule` and provides an abstract `calculate` method. The signature is identical to the forward pass of a loss function:
```python
def calculate(self,
            inp: torch.Tensor,
            latent: torch.Tensor,
            output: torch.Tensor,
            labels: torch.Tensor,
            meta: Dict[str, torch.Tensor],
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```
It returns a tuple of the primary metric value and a dictionary of secondary metrics (which can be empty).

## Available Metrics

- `MAEMetric` / `MAEMetricDict`: Mean Absolute Error. Useful for intuitive human-readable reconstruction errors (as opposed to MSE).
- `KLDMetric`: A metric identical to the KLD loss that can be tracked without affecting the optimization graph if KLD is only being monitored.
