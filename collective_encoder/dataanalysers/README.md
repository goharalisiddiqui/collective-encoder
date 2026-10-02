# Data Analysers

This module contains classes and utilities for performing exploratory data analysis on the datasets before or during training. The primary purpose of Data Analysers is to read data provided by the datamodule, perform various analytical tasks (e.g., plotting distributions, visualizing graphs, examining dihedrals), and output the results to disk.

## Interface

All data analysers must inherit from the `BaseDataAnalyser` class defined in `base.py`.

### `BaseDataAnalyser`

The base class inherits from `CEModule` and sets up the output directory structure for analysis. It validates that the incoming dataset type is compatible with the analyser via the `_COMPATIBLE_DATASET_TYPES` class attribute.

**Key Method to Implement:**
- `write_data(self, data, label="")`: This abstract method must be implemented by all subclasses. It receives the data (and an optional label to distinguish subsets like train/val/test) and performs the core analysis and disk writing operations (e.g., generating and saving plots).

## Available Analysers

- `DatapointsAnalyser`: Visualizes basic datapoint statistics and distributions.
- `DihedralsAnalyser`: Specifically built for analyzing dihedral angles in molecular datasets.
- `GraphsAnalyser`: Built for examining graph-based data.
- `LabelsAnalyser`: Focuses on analyzing target labels or scalar values associated with the dataset.
