# Metrics

This module provides observational evaluation metric classes used to track model performance during validation and testing without contributing gradients during backpropagation.

## Interface

All metric classes must inherit from `CEMetricBase` found in `base.py`.

### `CEMetricBase`

Inherits from `CEModule` and defines an abstract `calculate` method with the same signature as `CELossBase.forward`:

```python
def calculate(
    self,
    inp: torch.Tensor,
    latent: torch.Tensor,
    output: torch.Tensor,
    labels: torch.Tensor,
    meta: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```
It returns a tuple of the primary scalar metric value and a dictionary of secondary metrics.

## Registration & Resolution

Metrics are registered under the `"metrics"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("metrics", CEMetricBase)`
- **Resolution**: `get_module("metrics", metric_type)` from `collective_encoder.common.registry` (replacing the legacy `resolver.py`).
- **Adding a new Metric**: Subclass `CEMetricBase` and register it in `_BUILTINS` inside `metrics/__init__.py` or via:
  ```python
  from collective_encoder.common.registry import register
  register("metrics", "MY_METRIC", "my_package.metrics", "MyMetric")
  ```

## Available Metrics

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `MAE` | `CEMetricMAE` | Mean Absolute Error between inputs and reconstructed outputs. |
| `MAE_DICT` | `CEMetricMAEDict` | Component-wise Mean Absolute Error for dictionary outputs (e.g., `BGE` models). |
| `KLD` | `CEMetricKLD` | Tracks Kullback-Leibler divergence as a validation/test metric. |
