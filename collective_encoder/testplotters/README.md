# Test Plotters

This module provides tools for plotting, visualizing, and quantitatively evaluating model outputs during the testing phase. Test plotters are attached to a model during evaluation, accumulate batch outputs step-by-step, and generate plots and summary metrics when testing finishes.

## Interface

All test plotters must inherit from `BaseTestPlotter` in `base.py`.

### `BaseTestPlotter`

Inherits from `CEModule`. Subclasses implement:
- `add_batch(self, inp, latent, output, labels, meta)` (or `collect_data` / `plot`): Accumulates batch predictions, latent representations, and ground-truth labels during the test loop.
- `finish(self)` / `plot(self)`: Executed at the end of testing to compute metrics and write plots to `self.output_dir`.

## Registration & Resolution

Test plotters are registered under the `"testplotters"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("testplotters", BaseTestPlotter)`
- **Resolution**: `get_module("testplotters", plotter_type)` from `collective_encoder.common.registry` (replacing the legacy `resolver.py`).
- **Adding a new TestPlotter**: Subclass `BaseTestPlotter` and register it in `_BUILTINS` inside `testplotters/__init__.py` or via:
  ```python
  from collective_encoder.common.registry import register
  register("testplotters", "MY_PLOTTER", "my_package.testplotters", "MyPlotter")
  ```

## Available Plotters

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `SimplePlotter` | `SimplePlotter` | Generic 1D and 2D scatter plots of latent variables and reconstructions vs. ground-truth labels. |
| `ALA2plotter` | `ALA2plotter` | Alanine dipeptide ($\phi/\psi$ Ramachandran) specialized evaluation plotter. |
| `LatentCorrelationsPlotter` | `LatentCorrelationsPlotter` | Computes and plots correlation matrices between latent dimensions and physical labels. |
| `DisentanglementPlotter` | `DisentanglementPlotter` | Evaluates a suite of formal disentanglement metrics on the learned latent space. |
| `DisentanglementBetaMetric` | `DisentanglementBetaMetric` | $\beta$-VAE disentanglement score. |
| `DisentanglementFactorMetric` | `DisentanglementFactorMetric` | FactorVAE disentanglement score. |
| `DisentanglementDCIMetric` | `DisentanglementDCIMetric` | DCI (Disentanglement, Completeness, Informativeness) metric. |
| `DisentanglementMIGMetric` | `DisentanglementMIGMetric` | Mutual Information Gap (MIG) metric. |
| `DisentanglementModularityMetric` | `DisentanglementModularityMetric` | Modularity and Explicitness disentanglement score. |
| `DisentanglementSAPMetric` | `DisentanglementSAPMetric` | Separated Attribute Predictability (SAP) score. |
