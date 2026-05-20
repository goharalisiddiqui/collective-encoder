import logging
from abc import ABC
import os
from typing import List, Any, Dict, Union

import numpy as np

from collective_encoder.common.config_check import validate_required_fields

class CEModule(ABC):
    """
    Mixin base class for collective encoder modules.

    Provides structured logging via Python's :mod:`logging` module and
    gslibs-based input validation helpers shared across the project.

    Args:
        verbose (bool): When ``False`` suppress INFO-level log output.
    """
    _IDENTIFIER: str = None
    _REQUIRED_ARGS: List[str] = []
    _OPTIONAL_ARGS: Dict[str, Any] = {}

    def __init__(self, args: Dict[str, Union[float, int, str]] = None, **kwargs):
        if args is None:
            args = {}
        self.verbose_level = kwargs.get("verbose", "INFO")
        self.verbose = self.verbose_level in [True, "INFO", "DEBUG"]
        root_logger_name = kwargs.get("root_logger_name", "collective_encoder")
        self.run_dir = kwargs.get("run_dir", None)
        self.run_args = kwargs
        self.args = args

        # Use _ce_log to avoid shadowing pl.LightningModule's self.logger property
        # when CEModule is mixed into Lightning classes.
        self._ce_log = logging.getLogger(root_logger_name + "." + self.__class__.__name__)
        if self.verbose:
            self._ce_log.info("=" * 80)
            self._ce_log.info("[Initializing module: %s]", self.__class__.__name__)
            self._ce_log.info("=" * 80)
            self.ce_log_dict("Initialization args", args, indent=2)

        if self._REQUIRED_ARGS is not None:
            res = validate_required_fields(args, fields=self._REQUIRED_ARGS)
            if isinstance(res, list):
                self.raise_error(f"Missing required args: {res} in {self.__class__.__name__} initialization.")

        for key, default_value in self._OPTIONAL_ARGS.items():
            if key not in args:
                args[key] = default_value
        for key in args:
            self.__setattr__(key, args[key])
    
    def creater_results_dir(self):
        results_dir = os.path.join(self.run_dir, f"{self.__class__.__name__}_results")
        os.makedirs(results_dir, exist_ok=True) 
        self.results_dir = results_dir
    
    def create_results_file(self):
        if not hasattr(self, "results_dir"):
            self.creater_results_dir()
        self.results_file = os.path.join(self.results_dir, f"results.txt")
    
    def log_result_msg(self, res: str) -> None:
        self.log_info(res)
        if not hasattr(self, "results_file"):
            self.create_results_file()
        with open(self.results_file, "a") as f:
            f.write(res + "\n")
    
    def save_npy(self, arr: np.ndarray, label: str) -> None:
        if not hasattr(self, "results_dir"):
            self.creater_results_dir()
        npy_path = os.path.join(self.results_dir, f"{label.lower().replace(' ', '_')}.npy")
        np.save(npy_path, arr)
        self.log_result_msg(f"@@ Saved {label} to {npy_path}")
    
    def log_result(self, res: Union[str, float, int, list, np.ndarray, Dict], 
                   label: str = None, save_bin: bool = True) -> None:
        if isinstance(res, str):
            self.log_result_msg(f"{label}: {res}" if label else res)
        elif isinstance(res, (float, int)):
            self.log_result_msg(f"{label}: {res}" if label else str(res))
        elif isinstance(res, list):
            self.log_result_msg("-"*40)
            res = "\n".join(str(r) for r in res)
            if not label:
                label = "Unnmed List"
            self.log_result_msg(f"{label} (size: {len(res)}):\n{res}")
            self.log_result_msg("-"*40)
        elif isinstance(res, np.ndarray):
            self.log_result_msg("-"*40)
            res_str = np.array2string(res,
                                      max_line_width=1000,
                                      precision=4,
                                      threshold=1000,
                                      suppress_small=True)
            if not label:
                label = "Unnamed Array"
            self.log_result_msg(f"{label} (shape: {res.shape}):\n{res_str}")
            if save_bin:
                self.save_npy(res, label)
            self.log_result_msg("-"*40)
        elif isinstance(res, dict):
            self.log_result_msg("-"*40)
            res_str = "\n".join(f"{k}: {v}" for k, v in res.items())
            if not label:
                label = "Unnamed Dict"
            self.log_result_msg(f"{label} (size: {len(res)}):\n{res_str}")
            self.log_result_msg("-"*40)
        else:
            self.log_result_msg(f"{label}: {res}" if label else str(res))
        
            
    def get_args(self) -> Dict[str, Any]:
        """Return the dict of arguments used to initialize this module."""
        return self.args

    def get_run_args(self) -> Dict[str, Any]:
        """Return the dict of arguments used to initialize this module."""
        return self.run_args

    def warn(self, message: str) -> None:
        """Emit a WARNING-level log message."""
        self._ce_log.warning(message)
    
    def log_error(self, message: str) -> None:
        """Emit an ERROR-level log message."""
        self._ce_log.error(message)
    
    def log_info(self, message: str) -> None:
        if self.verbose:
            self._ce_log.info(message)
    
    def log_debug(self, message: str) -> None:
        self._ce_log.debug(message)
    
    def log_exception(self, message: str, exc: Exception) -> None:
        """Emit an ERROR-level log message with exception info."""
        self._ce_log.error(message, exc_info=exc)
    
    def log_warn(self, message: str) -> None:
        """Emit a WARNING-level log message."""
        self._ce_log.warning(message)

    def log_msg(self, message: str) -> None:
        """Emit an INFO-level log message (no-op when ``verbose=False``)."""
        self.log_info(message)

    def ce_log_list(self, message: str, values: List[Any]) -> None:
        """Emit an INFO-level log message followed by each item in *values*."""
        if self.verbose:
            self._ce_log.info("%s:", message)
            for v in values:
                self._ce_log.info("    %s", v)
            self._ce_log.info("End - %s", message)
    
    def raise_error(self, message: str, error_type: type = ValueError) -> None:
        """Emit an ERROR-level log message and raise a ValueError."""
        self._ce_log.exception(message, exc_info=error_type(message))
        raise error_type(message)

    def ce_log_dict(self, message: str, data: Dict[str, Any], indent: int = 2) -> None:
        """Emit *message* followed by each key-value pair in *data*, indented.

        Nested dicts are printed with an extra level of indentation so the
        structure is immediately visible in the log output.

        Args:
            message: Header line printed before the dict contents.
            data: The dict to log.
            indent: Number of spaces per nesting level (default 2).
        """
        if not self.verbose:
            return

        def _emit(d: Dict[str, Any], level: int) -> None:
            pad = " " * (indent * level)
            for k, v in d.items():
                if isinstance(v, dict):
                    self._ce_log.info("%s%s:", pad, k)
                    _emit(v, level + 1)
                else:
                    self._ce_log.info("%s%s = %s", pad, k, v)

        self._ce_log.info(message)
        _emit(data, level=1)
