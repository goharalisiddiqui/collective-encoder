import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from collective_encoder import TENSOR_PRECISION
from collective_encoder.common.registry import get_module
from collective_encoder.datamodules.base import BaseDataModule
from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datareaders.base import BaseDataReader
from collective_encoder.datasets.resolver import get_dataset_cls_dl


_DR_META_KEYS = {
    "datareader_type",
    "type",
    "datareader_args",
    "labeler_type",
    "labeler_args",
    "name",
    "species_name",
    "species_id",
    "weight",
    "max_datapoints",
}


class HeterogeneousCoordinatesDataModule(CoordinatesDataModule):
    """
    PyTorch Lightning DataModule for multi-species molecular coordinate datasets.

    Inherits from :class:`~collective_encoder.datamodules.coordinates.CoordinatesDataModule`
    and extends coordinate-based dataset preparation to heterogeneous collections of
    trajectories (different molecules/species). Reads coordinates and labels from
    multiple :class:`~collective_encoder.datareaders.base.BaseDataReader` instances,
    aligns label schemas into a unified target representation with ``species_id``
    appended, tags each structure with its per-molecule bond list and species metadata,
    and mixes frames across species into balanced or proportional splits.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Supported keys:
        - ``datareaders`` (list of dict or BaseDataReader, required): Specifications
          for each species/trajectory source.
        - ``dataset_type`` (str, required): Dataset type identifier (e.g. ``"BONDGRAPH"``).
        - ``train_size`` (int): Number of training samples across all species.
        - ``batch_size`` (int): Batch size for training.
        - ``validation_size`` (int, optional): Number of validation samples. Default is 0.
        - ``val_batch_size`` (int, optional): Batch size for validation. Default is 0.
        - ``test_size`` (int, optional): Number of test samples. Default is 0.
        - ``test_batch_size`` (int, optional): Batch size for testing. Default is 0.
        - ``predict_size`` (int, optional): Number of prediction samples. Default is 0.
        - ``predict_batch_size`` (int, optional): Batch size for prediction. Default is 0.
        - ``test_full_dataset`` (bool, optional): Whether to use all trajectory frames for testing.
        - ``sequential`` (bool, optional): Whether to slice splits sequentially or randomly.
        - ``max_datapoints`` (int, optional): Maximum total samples to read across all sources.
        - ``labeler_type`` (str, optional): Default labeler type used when not specified per datareader.
        - ``labeler_args`` (dict, optional): Default labeler arguments.
        - ``mixing_strategy`` (str, optional): Strategy for dividing split counts across species
          (``"balanced"`` or ``"proportional"``). Default is ``"balanced"``.
        - ``norm_type`` (str, optional): Type of normalization to apply ('standard' or 'minmax').
        - ``num_workers`` (int, optional): Number of DataLoader worker processes. Default is 1.
    kwargs : dict
        Additional keyword arguments forwarded to :class:`BaseDataModule`.
    """

    _IDENTIFIER = "HETEROGENEOUS_COORDINATES"
    _COMPATIBLE_DATAREADERS = ["COMPRESSED_TRAJECTORY", "XTC", "MD17"]
    _COMPATIBLE_DATASETS = [
        "BONDGRAPH",
        "BONDGRAPH_LATENT",
        "ATOMISTIC_DARTS",
    ]

    _REQUIRED_ARGS = BaseDataModule._REQUIRED_ARGS + [
        "datareaders",
        "dataset_type",
    ]

    _OPTIONAL_ARGS = CoordinatesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "datareader_type": "COMPRESSED_TRAJECTORY",
        "mixing_strategy": "balanced",
        "max_frames": None,
    })

    def __init__(self, args: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize the heterogeneous coordinates DataModule.

        Parameters
        ----------
        args : dict, optional
            Configuration dictionary.
        **kwargs
            Runtime keyword arguments forwarded to :class:`BaseDataModule`.
        """
        self.save_hyperparameters()
        BaseDataModule.__init__(self, args, **kwargs)

        if self.dataset_args is None:
            self.dataset_args = {}
        if self.labeler_args is None:
            self.labeler_args = {}
        if self.max_datapoints is None and self.max_frames is not None:
            self.max_datapoints = self.max_frames

        self._initialize_datareaders()

        seq_len = getattr(self, "sequence_length", 1) or 1
        total_datapoints = sum(
            self._get_dr_max_frames(i) // seq_len
            for i in range(len(self.datareaders))
        )
        if self.max_datapoints is None:
            self.max_datapoints = total_datapoints
        else:
            self.max_datapoints = min(self.max_datapoints, total_datapoints)

        if self.test_size > 0 and self.test_full_dataset:
            self.raise_error("test_size and test_full_dataset are mutually exclusive")

        req_size = (
            self.train_size
            + self.validation_size
            + self.predict_size
            + (0 if self.test_full_dataset else self.test_size)
        )
        if self.max_datapoints < req_size:
            self.raise_error(
                f"Not enough datapoints ({self.max_datapoints}) across trajectories "
                f"for the requested dataset sizes (train: {self.train_size}, "
                f"val: {self.validation_size}, test: {self.test_size}, "
                f"predict: {self.predict_size})"
            )

        if self.test_full_dataset:
            self.test_size = self.max_datapoints

        self._create_datasets()
        self._analyze_data()

    def _get_dr_max_frames(self, idx: int) -> int:
        """
        Return the maximum frame budget for datareader ``idx``.

        Parameters
        ----------
        idx : int
            Datareader index.

        Returns
        -------
        int
            Maximum available frames for datareader ``idx``.
        """
        total = self.datareaders[idx].get_total_frames()
        cap = self.dr_max_datapoints[idx]
        if cap is not None:
            return min(int(cap), total)
        return total

    def _initialize_datareaders(self) -> None:
        """
        Instantiate datareaders and record species metadata and labeler configurations.
        """
        self.datareaders: List[BaseDataReader] = []
        self.species_names: List[str] = []
        self.species_ids: List[int] = []
        self.dr_labelers: List[Tuple[Optional[str], Dict[str, Any]]] = []
        self.dr_max_datapoints: List[Optional[int]] = []

        raw_readers = getattr(self, "datareaders_configs", None)
        if raw_readers is None:
            raw_readers = self.args.get("datareaders", [])
        self.datareaders_configs = raw_readers

        for idx, dr_cfg in enumerate(raw_readers):
            if isinstance(dr_cfg, BaseDataReader):
                dr_inst = dr_cfg
                dr_name = getattr(dr_inst, "name", f"species_{idx}")
                dr_sid = int(getattr(dr_inst, "species_id", idx))
                lbl_type = getattr(dr_inst, "labeler_type", self.labeler_type)
                lbl_args = getattr(dr_inst, "labeler_args", self.labeler_args or {})
                dr_max_dp = getattr(dr_inst, "max_datapoints", None)
            elif isinstance(dr_cfg, dict):
                dr_type = dr_cfg.get(
                    "datareader_type",
                    dr_cfg.get("type", self.datareader_type or "COMPRESSED_TRAJECTORY"),
                )
                self.check_datareader_compatibility(dr_type)
                if "datareader_args" in dr_cfg:
                    dr_args = dict(dr_cfg["datareader_args"])
                else:
                    dr_args = {
                        k: v for k, v in dr_cfg.items() if k not in _DR_META_KEYS
                    }
                dr_name = dr_cfg.get(
                    "species_name", dr_cfg.get("name", f"species_{idx}")
                )
                dr_sid = int(dr_cfg.get("species_id", idx))

                if "labeler_type" in dr_cfg:
                    lbl_type = dr_cfg["labeler_type"]
                    lbl_args = dict(dr_cfg.get("labeler_args") or {})
                else:
                    lbl_type = self.labeler_type
                    lbl_args = dict(self.labeler_args or {})

                dr_max_dp = dr_cfg.get("max_datapoints", None)
                dr_cls = get_module("datareaders", dr_type)
                dr_inst = dr_cls(args=dr_args, **self.run_args)
            else:
                self.raise_error(
                    f"Invalid datareader specification: {type(dr_cfg)}. Expected dict or BaseDataReader."
                )

            self.datareaders.append(dr_inst)
            self.species_names.append(dr_name)
            self.species_ids.append(dr_sid)
            self.dr_labelers.append((lbl_type, lbl_args))
            self.dr_max_datapoints.append(dr_max_dp)
            self.log_info(
                f"Initialized datareader {idx} ({dr_name}, species_id={dr_sid}): "
                f"{dr_inst.get_total_frames()} frames, labeler={lbl_type}"
            )

        # Keep first datareader as self.datareader for CoordinatesDataModule backward compatibility
        if self.datareaders:
            self.datareader = self.datareaders[0]
            self.atomic_numbers = self.datareader.get_atomic_numbers()
            self.element_symbols = self.datareader.get_element_symbols()
            self.bonds = self.datareader.get_bonds()

    def _allocate_counts(self, total_count: int) -> List[int]:
        """
        Divide a requested split count across datareaders based on mixing strategy.

        Parameters
        ----------
        total_count : int
            Total number of samples requested for the split.

        Returns
        -------
        list of int
            Sample count allocated to each datareader.
        """
        k = len(self.datareaders)
        if k == 0 or total_count <= 0:
            return [0] * k

        if self.mixing_strategy == "proportional":
            weights = []
            for idx in range(k):
                cfg = self.datareaders_configs[idx]
                w = (
                    cfg.get("weight", 1.0)
                    if isinstance(cfg, dict)
                    else getattr(cfg, "weight", 1.0)
                )
                weights.append(float(w) * self._get_dr_max_frames(idx))
            sum_w = sum(weights)
            if sum_w == 0:
                weights = [1.0] * k
                sum_w = float(k)
            counts = [int(np.round((w / sum_w) * total_count)) for w in weights]
            diff = total_count - sum(counts)
            for i in range(abs(diff)):
                counts[i % k] += 1 if diff > 0 else -1
            return counts

        # Default: balanced allocation
        base = total_count // k
        remainder = total_count % k
        return [base + (1 if i < remainder else 0) for i in range(k)]

    def _calculate_indices(self) -> None:
        """
        Calculate train, validation, test, and prediction frame/chunk indices for each datareader.
        """
        self.split_indices_per_dr: List[Dict[str, np.ndarray]] = []
        seq_len = getattr(self, "sequence_length", 1) or 1

        train_counts = self._allocate_counts(self.train_size)
        val_counts = self._allocate_counts(self.validation_size)
        test_counts = (
            [self._get_dr_max_frames(i) // seq_len for i in range(len(self.datareaders))]
            if self.test_full_dataset
            else self._allocate_counts(self.test_size)
        )
        predict_counts = self._allocate_counts(self.predict_size)

        for i in range(len(self.datareaders)):
            n_frames = self._get_dr_max_frames(i)
            c_train = train_counts[i]
            c_val = val_counts[i]
            c_test = test_counts[i]
            c_pred = predict_counts[i]

            req_i = c_train + c_val + (0 if self.test_full_dataset else c_test) + c_pred
            max_start = n_frames - seq_len + 1
            if max_start <= 0:
                self.raise_error(
                    f"Datareader {i} ({self.species_names[i]}) has only {n_frames} frames, "
                    f"which is less than sequence_length={seq_len}."
                )

            non_overlap_starts = list(
                range(0, (n_frames // seq_len) * seq_len, seq_len)
            )
            if len(non_overlap_starts) >= req_i:
                candidate_starts = non_overlap_starts
            elif max_start >= req_i:
                candidate_starts = list(range(max_start))
            else:
                self.raise_error(
                    f"Datareader {i} ({self.species_names[i]}) has only {n_frames} frames, "
                    f"not enough for {req_i} sequence chunks of length {seq_len}."
                )

            if self.sequential:
                start = (
                    random.randint(0, len(candidate_starts) - req_i)
                    if len(candidate_starts) > req_i
                    else 0
                )
                selected = candidate_starts[start : start + req_i]
            else:
                selected = random.sample(candidate_starts, req_i)

            idx_train = np.array(selected[:c_train], dtype=int)
            idx_val = np.array(selected[c_train : c_train + c_val], dtype=int)
            if self.test_full_dataset:
                idx_test = np.array(candidate_starts, dtype=int)
            else:
                idx_test = np.array(
                    selected[c_train + c_val : c_train + c_val + c_test], dtype=int
                )
            idx_pred = np.array(
                selected[
                    c_train + c_val + (0 if self.test_full_dataset else c_test) : req_i
                ],
                dtype=int,
            )

            self.split_indices_per_dr.append({
                "train": idx_train,
                "val": idx_val,
                "test": idx_test,
                "predict": idx_pred,
            })

    def _align_labels(
        self,
        raw_labels: Optional[Union[Dict[str, torch.Tensor], List[Any]]],
        dr_idx: int,
        all_label_names: List[str],
        n_frames: int,
        key_max_dims: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Align a datareader's computed dictionary labels to the unified multi-species
        schema using zero-padding across varying feature dimensions.

        Parameters
        ----------
        raw_labels : dict of str to torch.Tensor or list or None
            Raw label dictionary from the datareader for a split.
        dr_idx : int
            Index of the datareader.
        all_label_names : list of str
            Unified list of non-dummy label dictionary keys across all datareaders.
        n_frames : int
            Number of frames in the read split.
        key_max_dims : dict of str to int, optional
            Maximum feature dimension per label key across all species.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary mapping each unified label key (plus ``"species_id"``) to a
            zero-padded 2-D tensor of shape ``(n_frames, max_dim_key)``.
        """
        species_id = float(self.species_ids[dr_idx])
        key_max_dims = key_max_dims or {k: 1 for k in all_label_names}
        aligned: Dict[str, torch.Tensor] = {}

        if isinstance(raw_labels, dict):
            for k in all_label_names:
                w = max(1, int(key_max_dims.get(k, 1)))
                padded = torch.zeros((n_frames, w), dtype=TENSOR_PRECISION)
                if k in raw_labels and n_frames > 0:
                    src = torch.as_tensor(raw_labels[k], dtype=TENSOR_PRECISION).reshape(
                        n_frames, -1
                    )
                    c = min(src.shape[1], w)
                    if c > 0:
                        padded[:, :c] = src[:, :c]
                aligned[k] = padded
        else:
            dr = self.datareaders[dr_idx]
            dr_lnames: List[str] = []
            try:
                dr_lnames = [
                    n for n in dr.get_label_names() if n not in ("dummy", "dummylabeler")
                ]
            except (AttributeError, Exception):
                pass
            has_raw = raw_labels is not None and len(raw_labels) > 0
            for k in all_label_names:
                w = max(1, int(key_max_dims.get(k, 1)))
                padded = torch.zeros((n_frames, w), dtype=TENSOR_PRECISION)
                if has_raw and k in dr_lnames:
                    col_idx = dr_lnames.index(k)
                    for f_idx in range(min(n_frames, len(raw_labels))):
                        val_arr = np.asarray(raw_labels[f_idx]).reshape(-1)
                        if col_idx < len(val_arr):
                            padded[f_idx, 0] = float(val_arr[col_idx])
                aligned[k] = padded

        aligned["species_id"] = torch.full(
            (n_frames, 1), species_id, dtype=TENSOR_PRECISION
        )
        return aligned

    def _combine_split(
        self,
        structures_per_dr: List[List[Any]],
        labels_per_dr: List[List[Dict[str, torch.Tensor]]],
        split_idx: int,
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        """
        Combine and interleave structures and zero-padded dictionary labels from all
        datareaders for a single split.

        Subclasses (e.g. heterogeneous trajectory data modules) may override this method
        to preserve chunk contiguity per sequence while shuffling across species.

        Parameters
        ----------
        structures_per_dr : list of list of ase.Atoms
            Per-datareader structures for each split.
        labels_per_dr : list of list of dict of str to torch.Tensor
            Per-datareader aligned label dictionaries for each split.
        split_idx : int
            Split index (0=train, 1=val, 2=test, 3=predict).

        Returns
        -------
        tuple of (list of ase.Atoms, dict of str to torch.Tensor)
            Combined structures and aligned dictionary of label tensors.
        """
        combined_structs: List[Any] = []
        for i in range(len(self.datareaders)):
            combined_structs.extend(structures_per_dr[i][split_idx])

        all_keys: List[str] = []
        for i in range(len(self.datareaders)):
            for k in labels_per_dr[i][split_idx].keys():
                if k not in all_keys:
                    all_keys.append(k)

        combined_lbls: Dict[str, torch.Tensor] = {}
        for k in all_keys:
            parts = [
                labels_per_dr[i][split_idx][k]
                for i in range(len(self.datareaders))
                if k in labels_per_dr[i][split_idx]
                and labels_per_dr[i][split_idx][k].shape[0] > 0
            ]
            if parts:
                combined_lbls[k] = torch.cat(parts, dim=0)
            else:
                w = 1
                for i in range(len(self.datareaders)):
                    if (
                        k in labels_per_dr[i][split_idx]
                        and labels_per_dr[i][split_idx][k].ndim == 2
                    ):
                        w = labels_per_dr[i][split_idx][k].shape[1]
                        break
                combined_lbls[k] = torch.zeros((0, w), dtype=TENSOR_PRECISION)

        if not self.sequential and len(combined_structs) > 0:
            perm = np.random.permutation(len(combined_structs))
            perm_t = torch.as_tensor(perm, dtype=torch.long)
            combined_structs = [combined_structs[p] for p in perm]
            combined_lbls = {k: v[perm_t] for k, v in combined_lbls.items()}

        return combined_structs, combined_lbls

    def _create_datasets(self) -> None:
        """
        Read coordinates from all datareaders, align labels, tag structures, and instantiate splits.
        """
        self.log_info("Calculating heterogeneous split indices...")
        self._calculate_indices()
        self.log_info("Dataset split indices calculated.")

        self.log_msg(
            f"Heterogeneous dataset sizes: Train: {self.train_size}, "
            f"Validation: {self.validation_size}, Test: {self.test_size}, "
            f"Predict: {self.predict_size}"
        )

        trajs_per_dr: List[List[List[Any]]] = []
        raw_labels_per_dr: List[List[Any]] = []

        for i, dr in enumerate(self.datareaders):
            lbl_type, lbl_args = self.dr_labelers[i]
            split_indices = [
                self.split_indices_per_dr[i]["train"],
                self.split_indices_per_dr[i]["val"],
                self.split_indices_per_dr[i]["test"],
                self.split_indices_per_dr[i]["predict"],
            ]
            self.log_info(
                f"Reading trajectory for species {i} ({self.species_names[i]}, id={self.species_ids[i]})..."
            )
            trajs_i, labels_i = dr.read_trajectory(
                indices=split_indices,
                labeler_type=lbl_type,
                labeler_args=lbl_args,
            )
            dr_bonds = dr.get_bonds()
            sid = self.species_ids[i]
            sname = self.species_names[i]
            for s_split in trajs_i:
                for struct in s_split:
                    struct.info["bonds"] = dr_bonds
                    struct.info["species_id"] = sid
                    struct.info["species_name"] = sname

            trajs_per_dr.append(trajs_i)
            raw_labels_per_dr.append(labels_i)

        # Build unified label list across all species
        all_label_names: List[str] = []
        for dr in self.datareaders:
            try:
                lnames = dr.get_label_names()
                for name in lnames:
                    if (
                        name not in all_label_names
                        and name not in ("dummy", "dummylabeler", "species_id")
                    ):
                        all_label_names.append(name)
            except (AttributeError, Exception):
                pass

        if len(all_label_names) > 0:
            self.label_list = all_label_names + ["species_id"]
        else:
            self.label_list = ["species_id"]

        self.log_info(f"Unified label columns: {self.label_list}")

        # Determine maximum width per label key across all species and splits
        key_max_dims: Dict[str, int] = {k: 1 for k in all_label_names}
        for i in range(len(self.datareaders)):
            if raw_labels_per_dr[i] is None:
                continue
            for split_idx in range(4):
                split_lbls = raw_labels_per_dr[i][split_idx]
                if isinstance(split_lbls, dict):
                    for k in all_label_names:
                        if k in split_lbls:
                            t = torch.as_tensor(split_lbls[k])
                            width = int(t.shape[1]) if t.ndim >= 2 else 1
                            key_max_dims[k] = max(key_max_dims[k], width)

        # Align labels per datareader
        aligned_labels_per_dr: List[List[Dict[str, torch.Tensor]]] = []
        for i in range(len(self.datareaders)):
            dr_aligned = []
            for split_idx in range(4):
                n_f = len(trajs_per_dr[i][split_idx])
                raw_split_lbls = (
                    raw_labels_per_dr[i][split_idx]
                    if raw_labels_per_dr[i] is not None
                    else None
                )
                dr_aligned.append(
                    self._align_labels(
                        raw_split_lbls, i, all_label_names, n_f, key_max_dims
                    )
                )
            aligned_labels_per_dr.append(dr_aligned)

        # Resolve dataset class and DataLoader class
        dataset_class, dataset_args, self.dl_cls = get_dataset_cls_dl(
            self.dataset_type, self.dataset_args, self.datareaders[0]
        )

        # Apply processor wrappers if any
        for dr in self.datareaders:
            dps = dr.get_processors()
            for dp in dps:
                dataset_class = dp.get_dataset_wrapper(dataset_class, dataset_args)
                self.log_info(
                    f"Applied processor {dp.__class__.__name__} to dataset class."
                )
            if len(dps) > 0:
                break

        # Combine splits across species
        train_s, train_l = self._combine_split(trajs_per_dr, aligned_labels_per_dr, 0)
        val_s, val_l = self._combine_split(trajs_per_dr, aligned_labels_per_dr, 1)
        test_s, test_l = self._combine_split(trajs_per_dr, aligned_labels_per_dr, 2)
        pred_s, pred_l = self._combine_split(trajs_per_dr, aligned_labels_per_dr, 3)

        if self.train_size > 0 and len(train_s) > 0:
            self.train_data = dataset_class(
                structures=train_s,
                labels=train_l,
                args=dataset_args,
                **{**self.run_args, "tag": "train"},
            )
        else:
            self.train_data = []

        if self.validation_size > 0 and len(val_s) > 0:
            self.val_data = dataset_class(
                structures=val_s,
                labels=val_l,
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "val"},
            )
        else:
            self.val_data = []

        if self.test_size > 0 and len(test_s) > 0:
            self.test_data = dataset_class(
                structures=test_s,
                labels=test_l,
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "test"},
            )
        else:
            self.test_data = []

        if self.predict_size > 0 and len(pred_s) > 0:
            self.predict_data = dataset_class(
                structures=pred_s,
                labels=pred_l,
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "predict"},
            )
        else:
            self.predict_data = []

        self.num_frames = (
            len(self.train_data)
            + len(self.val_data)
            + len(self.test_data)
            + len(self.predict_data)
        )
        self.datapoint_shape = (
            self.train_data.get_datapoint_shape()
            if len(self.train_data) > 0
            else (
                self.predict_data.get_datapoint_shape()
                if len(self.predict_data) > 0
                else ()
            )
        )

        self.log_msg(
            f"Loaded heterogeneous dataset with {self.num_frames} total samples -> "
            f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, "
            f"Test: {len(self.test_data)}, Predict: {len(self.predict_data)}"
        )
        self._check_batch_sizes()

    # ----------------------------------------------------------------------
    # Multi-Species Topology & Metadata Accessors
    # ----------------------------------------------------------------------

    def get_species_names(self) -> List[str]:
        """
        Return the list of species identifiers/names.

        Returns
        -------
        list of str
            Names assigned to each datareader species.
        """
        return list(self.species_names)

    def get_species_ids(self) -> List[int]:
        """
        Return the list of integer species IDs.

        Returns
        -------
        list of int
            Integer IDs assigned to each datareader species.
        """
        return list(self.species_ids)

    def get_num_species(self) -> int:
        """
        Return the total number of distinct species sources.

        Returns
        -------
        int
            Count of configured datareaders.
        """
        return len(self.species_names)

    def _resolve_species_idx(self, species: Union[int, str]) -> int:
        """
        Resolve a species identifier (name, species_id, or index) into a datareader list index.

        Parameters
        ----------
        species : int or str
            Species name or integer index/ID.

        Returns
        -------
        int
            Valid integer index into ``self.datareaders``.
        """
        if isinstance(species, int):
            if species in self.species_ids:
                return self.species_ids.index(species)
            if 0 <= species < len(self.datareaders):
                return species
            self.raise_error(
                f"Species index/ID {species} out of range [0, {len(self.datareaders)})."
            )
        if isinstance(species, str):
            if species in self.species_names:
                return self.species_names.index(species)
            self.raise_error(
                f"Unknown species name '{species}'. Available: {self.species_names}"
            )
        self.raise_error(f"Expected int or str for species, got {type(species).__name__}")

    def get_atns(
        self, species: Optional[Union[int, str]] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get atomic numbers for a specific species or all species.

        Parameters
        ----------
        species : int or str, optional
            Species name or index. If ``None``, returns a list of arrays across all species.

        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            Atomic numbers array(s).
        """
        if species is not None:
            idx = self._resolve_species_idx(species)
            return self.datareaders[idx].get_atomic_numbers()
        return [dr.get_atomic_numbers() for dr in self.datareaders]

    def get_bond_indices(
        self, species: Optional[Union[int, str]] = None
    ) -> Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
        """
        Get bond index pairs for a specific species or all species.

        Parameters
        ----------
        species : int or str, optional
            Species name or index. If ``None``, returns bond lists for all species.

        Returns
        -------
        list of tuple or list of list of tuple
            Bond index pairs.
        """
        if species is not None:
            idx = self._resolve_species_idx(species)
            return self.datareaders[idx].get_bonds()
        return [dr.get_bonds() for dr in self.datareaders]

    def get_element_symbols(
        self, species: Optional[Union[int, str]] = None
    ) -> Union[List[str], List[List[str]]]:
        """
        Get chemical element symbols for a specific species or all species.

        Parameters
        ----------
        species : int or str, optional
            Species name or index. If ``None``, returns symbols for all species.

        Returns
        -------
        list of str or list of list of str
            Element symbol lists.
        """
        if species is not None:
            idx = self._resolve_species_idx(species)
            return self.datareaders[idx].get_element_symbols()
        return [dr.get_element_symbols() for dr in self.datareaders]

    def get_fake_systems(self) -> List[Any]:
        """
        Get fake metatomic systems for testing purposes across species.

        Returns
        -------
        list of metatomic.torch.System
            Fake systems created from the initial frames of available datareaders.
        """
        from metatomic.torch import System

        systems = []
        for dr in self.datareaders:
            if hasattr(dr, "mol_traj") and len(dr.mol_traj) > 0:
                at_types = dr.get_atomic_numbers()
                mol_traj = dr.mol_traj
                systems.append(
                    System(
                        types=torch.tensor(at_types, dtype=torch.long),
                        positions=torch.tensor(
                            mol_traj[0].positions, dtype=TENSOR_PRECISION
                        ),
                        cell=torch.tensor(
                            mol_traj[0].get_cell(), dtype=TENSOR_PRECISION
                        ),
                        pbc=torch.tensor([True, True, True]),
                    )
                )
        return systems


# Support both spellings
HetrogenousCoordinatesDataModule = HeterogeneousCoordinatesDataModule
