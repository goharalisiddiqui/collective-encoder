# Data Analysers

This module contains classes and utilities for performing exploratory data analysis on datasets before or during training. Data analysers receive prepared datasets from the datamodule, compute summary statistics or visualizations (e.g., 1D/2D scatter plots, correlation matrices, dihedral angle distributions), and save the outputs to disk.

## Interface

All data analysers must inherit from `BaseDataAnalyser` defined in `base.py`.

### `BaseDataAnalyser`

`BaseDataAnalyser` inherits from `CEModule`, sets up the output directory (`data_analysis_<identifier>`), and validates that the incoming dataset type (`datamodule_args['dataset_type']`) is listed in the analyser's `_COMPATIBLE_DATASET_TYPES` attribute.

**Key Method to Implement:**
- `write_data(self, data, label="")`: Receives a dataset split (`data`) and an optional split label (e.g., `"train"`, `"val"`, `"test"`, `"predict"`) and writes the analysis plots/files to `self.output_dir`.

## Registration & Resolution

Data analysers are registered under the `"dataanalysers"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("dataanalysers", BaseDataAnalyser)`
- **Resolution**: `get_module("dataanalysers", analyser_type)` from `collective_encoder.common.registry` (no separate `resolver.py` file is used).
- **Adding a new analyser**: Subclass `BaseDataAnalyser` and register it either in `_BUILTINS` inside `dataanalysers/__init__.py` or externally via:
  ```python
  from collective_encoder.common.registry import register
  register("dataanalysers", "MY_ANALYSER", "my_package.analysers", "MyAnalyser")
  ```

## Available Analysers

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `LABELS` | `LabelsAnalyser` | Extracts target labels from datasets and plots 1D sequences, 2D scatter plots (`extra_2d`), and correlation matrices (`correlation`). |
| `DATAPOINTS` | `DatapointsAnalyser` | Extracts raw datapoint features instead of target labels and runs the `LabelsAnalyser` plotting pipeline. |
| `GRAPH_DATAPOINTS` | `GraphDatapointsAnalyser` | Extracts selective attributes from graph datasets via `labels_selection_map` and plots them. |
| `LABELS_DIHEDRAL` | `LabelsDihedralAnalyser` | Specializes `LabelsAnalyser` with circular radian axes ($[-\pi, \pi]$) for dihedral angles. |
| `DATAPOINTS_DIHEDRAL` | `DatapointsDihedralAnalyser` | Specializes `DatapointsAnalyser` with circular radian axes ($[-\pi, \pi]$). |
| `GRAPH_DATAPOINTS_DIHEDRAL` | `GraphDatapointsDihedralAnalyser` | Converts cosine/sine graph features into dihedral angles in $[-\pi, \pi]$ and plots them. |
