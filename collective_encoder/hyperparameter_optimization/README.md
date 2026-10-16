# Hyperparameter Optimization

This module integrates Optuna for automated hyperparameter tuning within the Collective Encoder framework.

## Components

- **`SearchSpace`**: Responsible for parsing YAML configurations and defining the parameter distributions (e.g., categorical, float uniform) that Optuna will sample from.
- **`OptunaObjective`**: Wraps the framework's training loop (`Trainer` class or similar) so it can be evaluated as an objective function by Optuna. It takes a trial, constructs a model config, runs training, and returns the target metric (e.g., validation loss).
- **`OptunaStudyRunner`**: Manages the Optuna study lifecycle, handling creation, database storage, running optimization loops, and saving results.
- **`ConfigResolver`**: A utility class to substitute trial parameters into the base YAML configuration dynamically.
- **`Visualizer`**: Provides functions to generate standard Optuna plots (optimization history, parameter importances) and export them to disk or launch interactive dashboards.
