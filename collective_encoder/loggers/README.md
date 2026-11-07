# Loggers

This module provides logging utilities integrated with PyTorch Lightning. While standard loggers (like TensorBoard or Weights & Biases) are primarily used, custom loggers in this package extend functionality for local analysis.

## Available Loggers

- **`CSVPlotLogger`**: Inherits from PyTorch Lightning's `CSVLogger`. In addition to logging metrics to a CSV file during training, it automatically generates and saves line plots (`.png`) of training vs. validation metrics. This provides quick, offline visual feedback without needing to spin up a TensorBoard server.
