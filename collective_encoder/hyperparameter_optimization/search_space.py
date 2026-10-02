"""Search space parser for Optuna hyperparameter studies."""

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple
import optuna


class SearchSpace:
    """
    Parses a declarative dictionary or YAML configuration into Optuna trial suggestions.

    Parameters
    ----------
    space_config : dict
        A dictionary defining hyperparameter ranges or choices.
    """

    def __init__(self, space_config: Dict[str, Any]):
        self.space_config = space_config or {}

    def is_grid_compatible(self) -> bool:
        """
        Checks if all parameters in the search space can form a discrete grid.

        Returns
        -------
        bool
            True if all search space dimensions are discrete/categorical.
        """
        if not self.space_config:
            return False
        for param_name, spec in self.space_config.items():
            if isinstance(spec, list):
                continue
            if isinstance(spec, dict):
                param_type = spec.get("type", "").lower()
                if param_type in ("categorical", "choice", "grid") and ("choices" in spec or "values" in spec):
                    continue
                if param_type in ("int", "integer") and "step" in spec:
                    continue
                return False
            # Fixed value is compatible
        return True

    def get_grid_search_space(self) -> Dict[str, List[Any]]:
        """
        Extracts discrete choice lists for Optuna GridSampler.

        Returns
        -------
        dict
            Dictionary mapping parameter names to lists of discrete choices.

        Raises
        ------
        ValueError
            If a parameter specification cannot be converted into a discrete list.
        """
        grid = {}
        for param_name, spec in self.space_config.items():
            if isinstance(spec, list):
                grid[param_name] = spec
            elif isinstance(spec, dict):
                param_type = spec.get("type", "").lower()
                if param_type in ("categorical", "choice", "grid"):
                    grid[param_name] = list(spec.get("choices", spec.get("values", [])))
                elif param_type in ("int", "integer"):
                    low = int(spec["low"])
                    high = int(spec["high"])
                    step = int(spec.get("step", 1))
                    grid[param_name] = list(range(low, high + 1, step))
                elif param_type in ("float", "real") and "values" in spec:
                    grid[param_name] = list(spec["values"])
                else:
                    raise ValueError(f"Parameter '{param_name}' with spec {spec} cannot be converted to a discrete grid.")
            else:
                grid[param_name] = [spec]
        return grid

    def total_combinations(self) -> int:
        """
        Computes the total number of Cartesian grid combinations.

        Returns
        -------
        int
            Total number of possible hyperparameter combinations, or 0 if not grid-compatible.
        """
        if not self.is_grid_compatible():
            return 0
        grid = self.get_grid_search_space()
        if not grid:
            return 0
        total = 1
        for choices in grid.values():
            total *= max(len(choices), 1)
        return total

    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Samples a full set of hyperparameters for the given Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial to suggest values for.

        Returns
        -------
        dict
            Dictionary of suggested hyperparameter values.

        Raises
        ------
        ValueError
            If an unsupported parameter type is encountered in the configuration.
        """
        samples = {}
        for param_name, spec in self.space_config.items():
            if isinstance(spec, list):
                # Direct list of discrete choices -> categorical
                samples[param_name] = trial.suggest_categorical(param_name, spec)
                continue

            if not isinstance(spec, dict):
                # Fixed scalar value
                samples[param_name] = spec
                continue

            param_type = spec.get("type", "float").lower()

            if param_type in ("categorical", "choice", "grid"):
                choices = spec.get("choices", spec.get("values", []))
                samples[param_name] = trial.suggest_categorical(param_name, choices)

            elif param_type in ("int", "integer"):
                low = int(spec["low"])
                high = int(spec["high"])
                step = int(spec.get("step", 1))
                log = bool(spec.get("log", False))
                samples[param_name] = trial.suggest_int(
                    param_name, low=low, high=high, step=step, log=log
                )

            elif param_type in ("float", "real"):
                low = float(spec["low"])
                high = float(spec["high"])
                step = float(spec["step"]) if "step" in spec else None
                log = bool(spec.get("log", False))
                samples[param_name] = trial.suggest_float(
                    param_name, low=low, high=high, step=step, log=log
                )

            elif param_type == "loguniform":
                low = float(spec["low"])
                high = float(spec["high"])
                samples[param_name] = trial.suggest_float(
                    param_name, low=low, high=high, log=True
                )

            elif param_type == "uniform":
                low = float(spec["low"])
                high = float(spec["high"])
                samples[param_name] = trial.suggest_float(
                    param_name, low=low, high=high, log=False
                )

            else:
                raise ValueError(f"Unsupported parameter type: '{param_type}' for '{param_name}'")

        return samples
