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
            tensors = [
                graph[key]
                for graph in data
                if key in graph and isinstance(graph[key], torch.Tensor)
            ]

            if not tensors:
                continue

            first_shape = tensors[0].shape
            same_shape = all(t.shape == first_shape for t in tensors)

            if same_shape:
                cat_tensor = torch.stack(tensors, dim=0)
                if cat_tensor.ndim == 3 and cat_tensor.shape[1] == 1:
                    cat_tensor = cat_tensor.squeeze(1)
            elif all(t.ndim == 1 for t in tensors):
                cat_tensor = torch.nn.utils.rnn.pad_sequence(
                    [t.to(dtype=torch.float64) for t in tensors],
                    batch_first=True,
                    padding_value=float("nan"),
                )
            else:
                # Variable-size 2-D graph topology tensors (x, edge_index, edge_attr, pos)
                continue

            if key.startswith("y_") or key in ("y", "species_id"):
                labels_dict[key] = cat_tensor
            else:
                data_dict[key] = cat_tensor

        return data_dict, labels_dict
