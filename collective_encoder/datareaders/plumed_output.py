import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union

from collective_encoder.labels.resolver import get_labeler
import gslibs.validation as gsv
from gslibs.utils.plumed import plumed_outfile_reader

from collective_encoder.datareaders.base import BaseDataReader


class PlumedOutputReader(BaseDataReader):
    """
    Data reader for PLUMED output files.

    Reads collective variable data from PLUMED output files using the
    plumed_outputfile_reader utility function. This reader is designed
    for analyzing collective variables and order parameters computed
    during molecular dynamics simulations.

    Args:
        plumed_file (str): Path to the PLUMED output file
        columns (List[str], optional): Specific columns to read from the file.
            If None, reads all available columns.
        ignore_list (List[str], optional): List of columns to ignore when reading.
            Default includes '#!' and 'FIELDS' tokens.
        column_match (str, optional): String pattern to match column names.
            Only columns containing this pattern will be read.
        **kwargs: Additional arguments passed to the parent class.
    """
    _IDENTIFIER = "PLUMEDCOLVARFILE"

    def __init__(self,
                 plumed_file: str,
                 columns: Optional[List[str]] = None,
                 ignore_list: Optional[List[str]] = None,
                 column_match: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        # Store parameters
        self.plumed_file = plumed_file
        self.columns = columns
        self.ignore_list = ignore_list or []
        self.column_match = column_match

        # Check if file exists
        gsv.check_exists(plumed_file=plumed_file)
        gsv.check_mutually_exclusive(columns=columns, 
                                     column_match=column_match, 
                                     require_one=True)
        self.log_msg(f"Loading PLUMED output from file {plumed_file}")

        # Read the data using the utility functions
        try:
            self.data = plumed_outfile_reader(
                plumed_outfile=plumed_file,
                columns=columns,
                ignore_list=ignore_list,
                column_match=column_match,
                vstime=False
            )

        except Exception as e:
            raise ValueError(f"Failed to read PLUMED output file {plumed_file}: {e}")

        self.log_msg(f"Successfully loaded PLUMED data with shape {self.data.shape}")
        self.log_msg(f"Available columns: {list(self.data.columns)}")

        # Set up data properties
        self._setup_data_properties()

    def _setup_data_properties(self):
        """Set up properties from the loaded data."""
        # Set up label information based on columns
        self.label_list = list(self.data.columns)
        self.n_features = len(self.label_list)
        self.n_frames = len(self.data)

    def get_total_frames(self) -> int:
        """
        Get the total number of frames (data points) in the PLUMED output.

        Returns:
            int: Number of frames/data points
        """
        return self.n_frames

    def read_trajectory(self,
                       indices: List[List[int]],
                       labeler_type : str = 'Dummy',
                       labeler_args : Dict[str, Union[str, float, List[int]]] = {},
                    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Read the PLUMED collective variable data.

        Args:
            indices (List[List[int]]): List of index sequences to read.
            labeler_type (str, optional): Type of labeler to use.
            labeler_args (Dict[str, Union[str, float, List[int]]], optional): Arguments for the labeler.

        Returns:
            Tuple containing:
                - List of numpy arrays with collective variable data
                - List of numpy arrays with corresponding labels computed by the labeler
        """
        self.log_msg("Reading PLUMED collective variable data...")
        
        # Get the labels from labeler
        labeler_cls = get_labeler(labeler_type)
        labeler = labeler_cls(
            dataframe=self.data,
            args=labeler_args,
        )
        self.label_list = labeler.get_names()
        
        trajs, labels = [], []

        for seq_indices in indices:
            # Extract data for this sequence
            seq_data = self.data.iloc[seq_indices]

            # Exclude any columns used in labels
            seq_data = seq_data.drop(columns=self.label_list, errors='ignore')

            # Convert to numpy array
            traj_data = seq_data.values  # Shape: (n_frames, n_features)

            # For compatibility, also provide as list of lists
            label_data = labeler.compute(seq_indices)

            trajs.append(traj_data)
            labels.append(label_data)

        self.log_msg(f"Finished reading PLUMED data. "
                    f"Read {len(trajs)} sequences with shapes "
                    f"{[t.shape for t in trajs]} and "
                    f"label shapes {[l.shape for l in labels]}.")

        return trajs, labels