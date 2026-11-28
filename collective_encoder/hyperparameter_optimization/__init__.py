"""
Hyperparameter Optimization package for collective_encoder.

This module provides an Optuna-based framework for automated hyperparameter
tuning. It includes tools for defining search spaces from configs, wrapping
the training loop as an objective, running studies, and visualizing the results.
"""

from collective_encoder.hyperparameter_optimization.resolver import ConfigResolver
from collective_encoder.hyperparameter_optimization.search_space import SearchSpace
from collective_encoder.hyperparameter_optimization.objective import OptunaObjective
from collective_encoder.hyperparameter_optimization.runner import OptunaStudyRunner
from collective_encoder.hyperparameter_optimization.visualizer import (
    export_study_visualizations,
    launch_dashboard,
)

__all__ = [
    "ConfigResolver",
    "SearchSpace",
    "OptunaObjective",
    "OptunaStudyRunner",
    "export_study_visualizations",
    "launch_dashboard",
]
