from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from collective_encoder.common.plotter import BasePlotter

class BaseTestPlotter(BasePlotter, ABC):
    """
    Abstract base class for all test plotters.

    Test plotters accumulate data (inputs, latents, outputs, labels) during
    evaluation batches and generate visualizations and metrics at the end.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary for the plotter.
    kwargs : dict
        Additional keyword arguments (e.g. for `CEModule`).
    """

    _IDENTIFIER = ""
    _OPTIONAL_ARGS = BasePlotter._OPTIONAL_ARGS | {
        'data_selection': None,  
        'labels_selection': None,
        'latents_selection': None,  
        'pred_selection': None,
        'meta_selection': None,
    }
    _PLOTTER_TYPE = "test_plotter"
    
    def __init__(self, args: dict = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.metrics_dict: Dict[str, float] = {}


    def set_metric(self, name: str, value: Any) -> None:
        """Records a scalar metric evaluated by this test plotter.

        Parameters
        ----------
        name : str
            Metric identifier name.
        value : float or torch.Tensor or numpy.number
            Scalar numeric value.
        """
        if value is not None:
            try:
                val_float = float(value.item() if hasattr(value, 'item') else value)
                self.metrics_dict[name] = val_float
            except Exception as e:
                self.log_warn(f"Failed to record metric '{name}': {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Returns all recorded scalar metrics from this test plotter."""
        return self.metrics_dict.copy()
        
    ############################################################################
    # Abstract Methods to be implemented by subclasses
    ############################################################################
    
    @abstractmethod
    def collection_list(self) -> List[str]:
        """
        Returns a list of data types to collect during testing.
        Can include any combination of "data", "latent", "pred", "labels", and "meta".
        """
        pass
    
    @abstractmethod
    def plot(self, data, latent, pred, labels, meta) -> None:
        """
        Generates plots based on the collected data.
        This method is called at the end of testing after all batches have been processed.
        Subclasses must implement this method to define how to plot the collected data.
        
        Arguments are 'data', 'latent', 'pred', 'labels', and 'meta', which correspond 
        to the collected data types specified in `collection_list`.
        
        Arguments:
            data: Collected input data.
            latent: Collected latent representations.
            pred: Collected predictions.
            labels: Collected labels.
            meta: Collected metadata.
        """
        pass

    ############################################################################
    # Batch Data Collection
    ############################################################################

    def _collect_data(self, data, name):
        if isinstance(data, dict):
            if not hasattr(self, f"collected_{name}"):
                converted = {
                    k: self._convert_data(v)
                    for k, v in data.items()
                }
                setattr(self, f"collected_{name}", converted)
            else:
                for k in data.keys():
                    data_np = self._convert_data(data[k])
                    if k in getattr(self, f"collected_{name}"):
                        getattr(self, f"collected_{name}")[k] = np.concatenate(
                            (getattr(self, f"collected_{name}")[k], data_np), axis=0)
                    else:
                        getattr(self, f"collected_{name}")[k] = data_np
        else:
            data_np = self._convert_data(data)
            if not hasattr(self, f"collected_{name}"):
                setattr(self, f"collected_{name}", data_np)
            else:
                setattr(self, f"collected_{name}", np.concatenate(
                    (getattr(self, f"collected_{name}"), data_np), axis=0))
    
    def add_batch(self, data, latent, pred, labels, meta):
        collection_list = self.collection_list()
        if "data" in collection_list:
            self._collect_data(data, "data")
        if "latent" in collection_list:
            self._collect_data(latent, "latent")
        if "pred" in collection_list:
            self._collect_data(pred, "pred")
        if "labels" in collection_list:
            self._collect_data(labels, "labels")
        if "meta" in collection_list:
            self._collect_data(meta, "meta")
    
    def finish(self) -> None:
        self.plot(
            data=getattr(self, "collected_data", None),
            latent=getattr(self, "collected_latent", None),
            pred=getattr(self, "collected_pred", None),
            labels=getattr(self, "collected_labels", None),
            meta=getattr(self, "collected_meta", None)
        )
