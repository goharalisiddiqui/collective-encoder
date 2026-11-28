import os
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List, Tuple, Dict

import numpy as np

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from .base import BaseDataset


def _extract_graph_label(graph: Data, label_prefix: str):
    """Extract label tensor or dictionary of ``y_*`` label tensors from a single PyG Data object."""
    if hasattr(graph, "y") and graph.y is not None:
        return ("single", graph.y)
    y_keys = [k for k in graph.keys() if k.startswith(label_prefix)]
    if y_keys:
        return ("dict", {k: graph[k] for k in y_keys})
    return ("none", None)


def _graph_to_ipc_dict(graph: Data) -> Dict[str, object]:
    """
    Convert a PyG ``Data`` object into a plain dict of NumPy arrays for
    file-descriptor-free multiprocessing IPC.

    PyTorch's multiprocessing tensor reducer opens a shared-memory file
    descriptor per ``torch.Tensor``, which exhausts ``RLIMIT_NOFILE`` and
    deadlocks ``multiprocessing.Pool`` when returning thousands of graphs.
    Serializing as NumPy arrays transfers raw bytes over the pipe with zero
    shared-memory file descriptors.
    """
    return {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in graph.to_dict().items()
    }


def _ipc_dict_to_graph(g_dict: Dict[str, object]) -> Data:
    """Reconstruct a PyG ``Data`` object in the parent process from a NumPy dict."""
    return Data(
        **{
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in g_dict.items()
        }
    )


def _extract_ipc_label(g_dict: Dict[str, object], label_prefix: str):
    """Extract label entries from an IPC dict as NumPy arrays."""
    if "y" in g_dict and g_dict["y"] is not None:
        return ("single", g_dict["y"])
    y_keys = [k for k in g_dict.keys() if k.startswith(label_prefix)]
    if y_keys:
        return ("dict", {k: g_dict[k] for k in y_keys})
    return ("none", None)


def _restore_label_tensors(lbl_entry):
    """Convert NumPy arrays in an extracted label entry back to ``torch.Tensor``."""
    if lbl_entry is None:
        return None
    mode, val = lbl_entry
    if mode == "single":
        return (
            mode,
            torch.from_numpy(val) if isinstance(val, np.ndarray) else val,
        )
    if mode == "dict":
        return (
            mode,
            {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in val.items()
            },
        )
    return lbl_entry


def _load_graphs_worker(args):
    """Worker: loads a chunk of precomputed graph .pt files and optionally extracts their labels."""
    worker_id, indices, graphs_dir, label_prefix, extract_labels, verbose = args
    loaded = []
    for idx in tqdm(
        indices,
        position=worker_id,
        desc=f"Worker {worker_id}",
        leave=False,
        disable=not verbose,
        dynamic_ncols=True,
    ):
        graph_path = os.path.join(graphs_dir, f"graph_{idx}.pt")
        graph = torch.load(graph_path, weights_only=False)
        g_dict = _graph_to_ipc_dict(graph)
        lbl = _extract_ipc_label(g_dict, label_prefix) if extract_labels else None
        loaded.append((idx, g_dict, lbl))
    return loaded

class BaseGraphDataset(BaseDataset, Dataset, ABC):
    """
    Abstract base class for graph datasets.
    """

    _IDENTIFIER = "BASEGRAPH"

    _OPTIONAL_ARGS = {
        'precompute_graphs': True,
        'parallel': True,
    }
    
    _GRAPH_DATA_KEYS = ["x", "edge_index", "edge_attr"]
    _GRAPH_LABEL_PREFIX = "y"
    _GRAPH_NORM_KEYS = ["x", "edge_attr"]
    
    def _post_init(self):
        if self.precompute_graphs:
            self.log_msg("Precomputing graph representations for all structures...")
            self._precompute_graphs()
        # Log dataset info
        sample = self.get(0)
        self._num_nodes = sample.x.shape[0]
        self._num_node_feat = sample.x.shape[1]
        self._num_edges = sample.edge_index.shape[1]
        self._num_edge_feat = sample.edge_attr.shape[1]
        self.log_msg(f"Loaded {self._IDENTIFIER} dataset with {self.len()} graphs") 
        self.log_msg(f"Number of nodes: {self.num_nodes}") 
        self.log_msg(f"Node feature size: {self.num_node_features}") 
        self.log_msg(f"Number of edges: {self.num_edges}") 
        self.log_msg(f"Edge feature size: {self.num_edge_features}") 
    
    def __len__(self):
        return len(self.structures)
    
    @abstractmethod
    def _precompute_graphs(self):
        """
        Precompute graph representations for all structures in the dataset.
        This method should be implemented by subclasses to define how graphs are computed and saved.
        """
        raise NotImplementedError("Subclasses must implement _precompute_graphs method")
    
    @abstractmethod
    def _compute_graph(self, idx: int) -> Data:
        """
        Compute the graph representation for the structure at the given index.

        Parameters
        ----------
        idx : int
            Index of the structure to compute the graph for.

        Returns
        -------
        Data
            The computed graph representation as a PyG Data object.
        """
        raise NotImplementedError("Subclasses must implement _compute_graph method")

    def __getitem__(self, idx) -> Data:
        if self.precompute_graphs:
            graph_path = os.path.join(self.precompute_graphs_dir, f"graph_{idx}.pt")
            graph = torch.load(graph_path, weights_only=False)
        else:
            graph = self._compute_graph(idx)
        return graph
    
    def len(self) -> int: # for PyG Dataset compatibility
        return self.__len__()

    def get(self, idx: int) -> Data:  # for PyG Dataset compatibility
        return self.__getitem__(idx)

    def get_datapoint_shape(self) -> Dict[str, Tuple]:
        """
        Return dictionary of data point tensor shapes.

        Returns
        -------
        dict
            Mapping of feature keys (like 'x', 'edge_index') to their tensor shapes.
        """
        sample = self.get(0)
        shapes = {}
        for att in sample.keys():
            if att in self._GRAPH_DATA_KEYS or \
              att.startswith(self._GRAPH_LABEL_PREFIX + "_"):
                shapes[att] = tuple(sample[att].shape)
        return shapes

    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def num_node_features(self) -> int:
        return self._num_node_feat

    @property
    def num_edges(self) -> int:
        return self._num_edges
    
    @property
    def num_edge_features(self) -> int:
        return self._num_edge_feat

    def _load_all_graphs(self, extract_labels: bool = False):
        """
        Load all graphs (and optionally extract their labels in parallel) using
        multiprocessing when precomputed and ``self.parallel`` is True, showing
        a ``tqdm`` progress bar.

        Parameters
        ----------
        extract_labels : bool, optional
            When ``True``, extracts ``y`` / ``y_*`` labels inside each worker while
            loading the graph and returns ``(graphs, labels)``.

        Returns
        -------
        list of Data or tuple of (list of Data, Any)
            List of graphs when ``extract_labels=False``, or ``(graphs, labels)``
            when ``extract_labels=True``.
        """
        n_total = self.len()
        if n_total == 0:
            return ([], None) if extract_labels else []

        graphs: List[Data] = [None] * n_total
        raw_lbl_entries: List[Tuple[str, object]] = [None] * n_total if extract_labels else []

        if self.precompute_graphs and self.parallel and n_total >= 8:
            n_workers = min(16, os.cpu_count() or 1, n_total)
            all_indices = list(range(n_total))
            chunks = [all_indices[i::n_workers] for i in range(n_workers)]
            chunks = [c for c in chunks if c]
            args = [
                (
                    i,
                    chunk,
                    self.precompute_graphs_dir,
                    self._GRAPH_LABEL_PREFIX,
                    extract_labels,
                    self.verbose,
                )
                for i, chunk in enumerate(chunks)
            ]
            with Pool(processes=len(args)) as pool:
                for chunk_res in tqdm(
                    pool.imap(_load_graphs_worker, args),
                    total=len(args),
                    position=len(args),
                    desc="Loading graphs",
                    leave=True,
                    disable=not self.verbose,
                    dynamic_ncols=True,
                ):
                    for idx, g_dict, lbl in chunk_res:
                        graphs[idx] = _ipc_dict_to_graph(g_dict)
                        if extract_labels:
                            raw_lbl_entries[idx] = _restore_label_tensors(lbl)
        else:
            for idx in tqdm(
                range(n_total),
                desc="Loading graphs",
                disable=not self.verbose,
                dynamic_ncols=True,
            ):
                g = self.get(idx)
                graphs[idx] = g
                if extract_labels:
                    raw_lbl_entries[idx] = _extract_graph_label(
                        g, self._GRAPH_LABEL_PREFIX
                    )

        if not extract_labels:
            return graphs

        first_mode, first_val = raw_lbl_entries[0]
        if first_mode == "single":
            labels = [entry[1] for entry in raw_lbl_entries]
        elif first_mode == "dict":
            keys = list(first_val.keys())
            labels = {
                k: [entry[1][k] for entry in raw_lbl_entries if entry[1] is not None and k in entry[1]]
                for k in keys
            }
        else:
            labels = None
        
        self.log_debug(f"Loaded {len(graphs)} graphs with labels keys: {list(labels.keys()) if isinstance(labels, dict) else 'single' if labels is not None else 'none'}")

        return graphs, labels

    def get_data(self):
        """
        Get all graphs and their labels.

        Returns
        -------
        tuple
            Tuple of (graphs_list, labels).
        """
        return self._load_all_graphs(extract_labels=True)
    
    def get_norm_data(self) -> np.ndarray:
        """
        Return array of data to be normalized.

        Returns
        -------
        numpy.ndarray
            Flattened array containing node and edge features for normalizer fitting.
        """
        graphs = self._load_all_graphs()
        data_to_normalize = []
        for data in graphs:
            node_feats = data.x.numpy()
            edge_feats = data.edge_attr.numpy()
            data_to_normalize.append(np.hstack([node_feats.mean(axis=0), edge_feats.mean(axis=0)]))
        data_to_normalize = np.vstack(data_to_normalize)
        return data_to_normalize
