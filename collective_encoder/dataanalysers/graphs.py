import os
import numpy as np
from tqdm import tqdm

from .datapoints import DatapointsAnalyser
from .base import BaseDataAnalyser

from collective_encoder.testplotters.labels_selector import label_selector

class GraphDatapointsAnalyser(DatapointsAnalyser):
    """
    Data analyser for extracting dihedral angles from graph datasets.
    """

    _IDENTIFIER = "GRAPH_DATAPOINTS"
    _COMPATIBLE_DATASET_TYPES = ["GRAPH"]
    _REQUIRED_ARGS = BaseDataAnalyser._REQUIRED_ARGS + [
        'labels_selection_map',
    ]

    def _extract_labels(self, data):
        req_labels = list(set([a[0] for a in self.labels_selection_map.values()]))
        present_labels = data[0].to_dict().keys()
        for label in req_labels:
            if label not in present_labels:
                self.raise_error(f"Required label '{label}' not found in data. "
                    f"Available labels: {present_labels}. "
                    f"Check that the dataset is correctly configured for dihedral extraction.")
                
        labels = {}
        for label in req_labels:
            labels[label] = []
        for d in tqdm(data, desc="Extracting labels"):
            for key in req_labels:
                labels[key].append(getattr(d, key).cpu().numpy())
        for key in labels.keys():
            labels[key] = np.array(labels[key])
        
        dihedrals = label_selector(labels, self.labels_selection_map)
        
        return dihedrals