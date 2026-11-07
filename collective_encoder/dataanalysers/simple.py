from typing import Any, Dict, Tuple
import numpy as np
from matplotlib import pyplot as plt

from .base import BaseDataAnalyser

class SimpleDataAnalyser(BaseDataAnalyser):
    """
    Data analyser for extracting and plotting label data.

    This analyser fetches properties designated as labels in datasets
    and can create 1D time-series style plots, 2D scatter plots,
    and correlation matrices based on provided configurations.
    """

    _IDENTIFIER = "SIMPLE"
    _COMPATIBLE_DATASET_TYPES = ["DISTANCES", "POSITIONS"]
    
    def after_2dscatter(self, 
                        plot : Dict[str, Any],
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]], # x, y, xerr, yerr, labels
                        fig: plt.Figure,
                        axes: np.ndarray
                        ) -> None:
        plot.update({
            "tag":  self.label,
        })
        
    def parse_data(self, data) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse the data and labels from the dataset.

        Parameters
        ----------
        data : list
            Data points from the dataset.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Parsed datapoints and labels as dictionaries.
        """
        
        return data.get_data()
        
    def write_data(self, data, label="") -> None:
        """
        Execute the plotting logic after all batches have been collected.

        Parameters
        ----------
        data : list
            Data points from the dataset.
        """
        self.label = label
        datapoints, labelpoints = self.parse_data(data)
        try:
            datapoints = self._parse_selection(self.data_selection, datapoints, "data")
            labels = self._parse_selection(self.labels_selection, labelpoints, "labels")
        except Exception as e:
            self.log_exception(f"Error occurred while parsing selections: {e}")
            return

        # All names must be unique
        all_names = list(labels.keys()) + list(datapoints.keys())
        if len(all_names) == 0:
            self.log_exception("No labels, data found after parsing selections. Nothing to plot.")
            return

        if len(all_names) != len(set(all_names)):
            raise ValueError(f"Duplicate names found in labels and data keys. "
                             f"All names must be unique. Found names: {all_names}")
        self.vals = {**labels, **datapoints}
        
        self.plot_vals()