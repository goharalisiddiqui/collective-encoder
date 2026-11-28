from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

from collective_encoder.dataanalysers.simple import SimpleDataAnalyser
from collective_encoder.datasets.base_graph import BaseGraphDataset


def _stack_tensor_list(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of per-graph tensors into a 2-D tensor of shape ``(n_graphs, D)``.
    """
    first_shape = tensors[0].shape
    same_shape = all(t.shape == first_shape for t in tensors)
    if same_shape:
        cat_tensor = torch.stack(tensors, dim=0)
        if cat_tensor.ndim == 3 and cat_tensor.shape[1] == 1:
            cat_tensor = cat_tensor.squeeze(1)
        return cat_tensor

    flat_tensors = [t.reshape(-1).to(dtype=torch.float64) for t in tensors]
    return torch.nn.utils.rnn.pad_sequence(
        flat_tensors,
        batch_first=True,
        padding_value=float("nan"),
    )


class GraphDataAnalyser(SimpleDataAnalyser):
    """
    Data analyser for extracting and plotting specific features from graph datasets.

    Uses :meth:`~collective_encoder.datasets.base_graph.BaseGraphDataset.get_data`
    to obtain ``(graphs, labels)`` in a single pass and prepares ``data_dict`` and
    ``labels_dict`` for plotting.
    """

    _IDENTIFIER = "GRAPH"
    _COMPATIBLE_DATASET_TYPES = ["BONDGRAPH"]

    def parse_data(self, data: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse the data and labels from a graph dataset (or ``(graphs, labels)`` tuple / graph list).

        Parameters
        ----------
        data : BaseGraphDataset or tuple or list
            Graph dataset instance, ``(graphs, labels)`` tuple, or list of PyG ``Data`` objects.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Parsed datapoints and labels as dictionaries of stacked tensors.
        """
        if isinstance(data, BaseGraphDataset):
            graphs, raw_labels = data.get_data()
            self.log_info(f"Parsed {len(graphs)} graphs from dataset")
        elif isinstance(data, tuple) and len(data) == 2:
            graphs, raw_labels = data
        else:
            graphs = list(data)
            raw_labels = None
            if len(graphs) > 0:
                if hasattr(graphs[0], "y") and graphs[0].y is not None:
                    raw_labels = [g.y for g in graphs]
                elif any(k.startswith("y") for k in graphs[0].keys()):
                    raw_labels = {
                        k: [g[k] for g in graphs if k in g.keys()]
                        for k in graphs[0].keys()
                        if k.startswith("y")
                    }

        data_dict: Dict[str, torch.Tensor] = {}
        labels_dict: Dict[str, torch.Tensor] = {}

        if len(graphs) == 0:
            return data_dict, labels_dict

        # 1. Process labels extracted by get_data()
        if isinstance(raw_labels, dict):
            for key, tensor_list in tqdm(
                raw_labels.items(),
                desc=f"Parsing graph labels ({getattr(self, 'label', '') or 'data'})",
                disable=not self.verbose,
                leave=False,
                dynamic_ncols=True,
            ):
                valid_tensors = [t for t in tensor_list if isinstance(t, torch.Tensor)]
                if not valid_tensors:
                    continue
                labels_dict[key] = _stack_tensor_list(valid_tensors)
        elif isinstance(raw_labels, list) and len(raw_labels) > 0:
            if isinstance(raw_labels[0], torch.Tensor):
                labels_dict["y"] = _stack_tensor_list(raw_labels)

        # Mask zero-padded topology entries with NaN for accurate distribution analysis
        for b_key in ("y_bonds", "bonds"):
            if b_key in labels_dict:
                b_t = labels_dict[b_key].to(dtype=torch.float64)
                labels_dict[b_key] = torch.where(b_t == 0.0, float("nan"), b_t)
        for a_key in ("y_angles", "angles"):
            if a_key in labels_dict:
                a_t = labels_dict[a_key].to(dtype=torch.float64)
                labels_dict[a_key] = torch.where(a_t == 0.0, float("nan"), a_t)
        for c_key, s_key in (("y_cos_torsions", "y_sin_torsions"), ("cos_torsions", "sin_torsions")):
            if c_key in labels_dict and s_key in labels_dict:
                c_t = labels_dict[c_key].to(dtype=torch.float64)
                s_t = labels_dict[s_key].to(dtype=torch.float64)
                pad_mask = (c_t == 0.0) & (s_t == 0.0)
                labels_dict[c_key] = torch.where(pad_mask, float("nan"), c_t)
                labels_dict[s_key] = torch.where(pad_mask, float("nan"), s_t)

        # 2. Process non-label graph attributes from the in-memory graphs list
        sample_keys = [
            k for k in graphs[0].keys() if not k.startswith("y_") and k != "y"
        ]
        for key in sample_keys:
            tensors = [
                g[key]
                for g in graphs
                if key in g and isinstance(g[key], torch.Tensor)
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
                continue

            if key == "species_id":
                labels_dict[key] = cat_tensor
            else:
                data_dict[key] = cat_tensor

        return data_dict, labels_dict
