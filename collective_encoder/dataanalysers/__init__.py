"""
Plotting utilities for data analysis and visualization.

This module provides various classes derived from `BaseDataAnalyser` to perform
exploratory data analysis. The analysers receive data from the datamodules,
compute relevant statistics or visualizations (e.g., histograms, graph structures,
dihedral angle distributions), and save the results to disk. Subclasses must
implement the `write_data` method defined in `base.py`.
"""
