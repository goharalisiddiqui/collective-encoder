import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple, Union

from collective_encoder.common.registry import get_module
import gslibs.validation as gsv
from gslibs.utils.plumed import plumed_outfile_reader

from collective_encoder.datalabelers.base import BaseLabeler
from collective_encoder.datareaders.base import BaseDataReader


class PlumedOutputReader(BaseDataReader):
    """
    Data reader for PLUMED output files.

    Reads collective variable data from PLUMED output files using the
    plumed_outputfile_reader utility function. This reader is designed
    for analyzing collective variables and order parameters computed
    during molecular dynamics simulations.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary containing ``plumed_file`` and optional
        ``columns``, ``ignore_list``, and ``column_match``.
    plumed_file : str, optional
        Path to the PLUMED output file (can also be passed via ``args``).
    columns : list of str, optional
        Specific columns to read from the file. If None, reads all available columns.
    ignore_list : list of str, optional
        List of columns to ignore when reading. Default includes '#!' and 'FIELDS' tokens.
    column_match : str, optional
        String pattern to match column names. Only columns containing this pattern will be read.
    kwargs : dict
        Additional arguments passed to the parent class.
    """
    _IDENTIFIER = "PLUMED_OUTPUT"
    _COMPATIBLE_LABELERS = BaseDataReader._COMPATIBLE_LABELERS + ["COLUMN_SELECTOR"]
    _REQUIRED_ARGS = ["plumed_file"]
    _OPTIONAL_ARGS = {
        "columns": None,
        "ignore_list": None,
        "column_match": None,
    }

    def __init__(self,
                 args: Optional[Dict[str, Any]] = None,
                 plumed_file: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 ignore_list: Optional[List[str]] = None,
                 column_match: Optional[str] = None,
                 **kwargs):
        merged_args = dict(args) if args is not None else {}
        if plumed_file is not None:
            merged_args["plumed_file"] = plumed_file
        if columns is not None:
            merged_args["columns"] = columns
        if ignore_list is not None:
            merged_args["ignore_list"] = ignore_list
        if column_match is not None:
            merged_args["column_match"] = column_match

        super().__init__(args=merged_args, **kwargs)

        self.ignore_list = self.ignore_list or []

        # Check if file exists and validate column selection arguments
        gsv.check_exists(plumed_file=self.plumed_file)
        gsv.check_mutually_exclusive(
            columns=self.columns,
            column_match=self.column_match,
            require_one=False,
        )
        self.log_msg(f"Loading PLUMED output from file {self.plumed_file}")

        # Read the data using the utility functions
        try:
            self.data = plumed_outfile_reader(
                plumed_outfile=self.plumed_file,
                columns=self.columns,
                ignore_list=self.ignore_list,
                column_match=self.column_match,
                vstime=False
            )

        except Exception as e:
            raise ValueError(f"Failed to read PLUMED output file {self.plumed_file}: {e}")

        self.log_msg(f"Successfully loaded PLUMED data with shape {self.data.shape}")
        self.log_msg(f"Available columns: {list(self.data.columns)}")

        # Set up data properties
        self._setup_data_properties()

    def _setup_data_properties(self):
        """
        Set up properties (like label list, number of features, and frame count)
        from the loaded dataframe.
        """
        # Set up label information based on columns
        self.label_list = list(self.data.columns)
        self.n_features = len(self.label_list)
        self.n_frames = len(self.data)

    def get_total_frames(self) -> int:
        """
        Get the total number of frames (data points) in the PLUMED output.

        Returns
        -------
        int
            Number of frames (rows) in the dataset.
        """
        return self.n_frames

    def read(
        self,
        labeler_type: str = "DUMMY",
        labeler_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Read PLUMED output data using the specified labeler.
        """
        return self.read_trajectory(
            indices=kwargs.get("indices", []),
            labeler_type=labeler_type,
            labeler_args=labeler_args or {},
        )

    def _initialize_labeler(
        self,
        labeler_type: str,
        labeler_args: Dict[str, Any],
    ) -> BaseLabeler:
        """
        Instantiate the labeler for this PLUMED reader.
        """
        labeler_cls = get_module("datalabelers", labeler_type)
        return labeler_cls(
            dataframe=self.data,
            args=labeler_args or {},
            **self.get_run_args(),
        )

    def read_trajectory(self,
                       indices: List[List[int]],
                       labeler_type: str = 'DUMMY',
                       labeler_args: Optional[Dict[str, Union[str, float, List[int]]]] = None,
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[List[int], ...]]:
        """
        Read the PLUMED collective variable data for requested index splits.

        Parameters
        ----------
        indices : list of list of int
            List of index sequences to read (e.g. for train, val, test splits).
        labeler_type : str, optional
            Type of labeler to use, by default 'DUMMY'.
        labeler_args : dict, optional
            Arguments for the labeler.

        Returns
        -------
        tuple
            A tuple containing:
            - List of numpy arrays with collective variable data.
            - List of numpy arrays with corresponding labels computed by the labeler.
            - Tuple of empty lists (indicating no partial frame failures per split).
        """
        self.log_msg("Reading PLUMED collective variable data...")

        if labeler_type in (None, "Dummy"):
            labeler_type = "DUMMY"

        labeler = self.initialize_labeler(
            labeler_type=labeler_type,
            labeler_args=labeler_args or {},
        )
        self.label_list = labeler.get_label_names()

        trajs, labels = [], []

        for seq_indices in indices:
            # Extract data for this sequence
            seq_data = self.data.iloc[seq_indices]

            # Exclude any columns used in labels
            seq_data = seq_data.drop(columns=self.label_list, errors='ignore')

            # Convert to numpy array
            traj_data = seq_data.values  # Shape: (n_frames, n_features)

            if len(seq_indices) == 0:
                label_data = np.empty((0, len(self.label_list)))
            else:
                label_data = labeler.compute(seq_indices)

            trajs.append(traj_data)
            labels.append(label_data)

        self.log_msg(f"Finished reading PLUMED data. "
                    f"Read {len(trajs)} sequences with shapes "
                    f"{[t.shape for t in trajs]} and "
                    f"label shapes {[l.shape for l in labels]}.")

        return trajs, labels, tuple([] for _ in indices)