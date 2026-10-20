from typing import Dict, Tuple
import numpy as np
import torch

from collective_encoder.dataanalysers.simple import SimpleDataAnalyser

class GraphDataAnalyser(SimpleDataAnalyser):
    """
    Data analyser for extracting and plotting specific features from graph datasets.

    Relies on a ``data_selection_map`` to selectively pick attributes
    or properties out of complex graph data structures before passing
    them to standard plotting routines.
    """

    _IDENTIFIER = "GRAPH"
    _COMPATIBLE_DATASET_TYPES = ["BONDGRAPH"]
    
    
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
        all_keys = set()
        for graph in data:
            all_keys.update(graph.keys())
            
        data_dict = {}
        labels_dict = {}
        
        for key in all_keys:
            tensors = [graph[key] for graph in data if key in graph and graph[key] is not None]
            
            if tensors:
                cat_tensor = torch.stack(tensors, dim=0)
                if key.startswith('y_') or key == 'y':
                    labels_dict[key] = cat_tensor
                else:
                    data_dict[key] = cat_tensor

        return data_dict, labels_dict
