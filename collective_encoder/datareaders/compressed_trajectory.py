import os

from glob import glob
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import ase
import MDAnalysis as mda

import gslibs.validation as gsv
from .trajectory import TrajectoryReaderBase

class CompressedTrajectoryReader(TrajectoryReaderBase):
    """
    Read compressed MD trajectories (e.g. XTC, DCD) via MDAnalysis.

    Loads a molecular topology together with one or more trajectory
    files and converts each requested frame into an ASE ``Atoms``
    object. Labeling is performed per-frame using a
    :class:`~collective_encoder.datalabelers.base.FrameLabeler`.

    Supports optional multi-process reading: the trajectory is copied to
    independent worker processes (each with its own Universe) to parallelise
    I/O and label computation.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``topology_file`` (str): Path to the topology file (e.g., ``.tpr``, ``.pdb``).
        - ``trajectory_file`` (str, optional): Path to a single trajectory file.
        - ``trajectory_files`` (list of str, optional): List of trajectory files to concatenate.
        - ``coord_glob`` (str, optional): Glob pattern matching one or more trajectory files.
        - ``selection`` (str, optional): MDAnalysis atom selection string (default: ``'all'``).
        - ``type_to_elements`` (dict, optional): Mapping of atom types to element numbers
          when the topology lacks element information.
        - ``parallel`` (bool, optional): Whether to use multiprocessing for reading (default: ``True``).
        - ``xtcfile`` (str, optional): Deprecated. Use ``trajectory_file``.
        - ``xtcfiles`` (list of str, optional): Deprecated. Use ``trajectory_files``.
    kwargs : dict
        Additional arguments forwarded to :class:`~collective_encoder.datareaders.base.BaseDataReader`.

    Note
    ----
    Exactly one of ``trajectory_file``, ``trajectory_files``, or ``coord_glob`` must be provided.
    """
    _IDENTIFIER = "COMPRESSED_TRAJECTORY"
    _REQUIRED_ARGS = TrajectoryReaderBase._REQUIRED_ARGS + [
        'topology_file',
    ]
    _OPTIONAL_ARGS = TrajectoryReaderBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'trajectory_file': None,
        'trajectory_files': None,
        'xtcfile': None,  # For backward compatibility
        'xtcfiles': None, # For backward compatibility
        'coord_glob': None,
        'topology_format': 'TPR',
    })

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        super().__init__(args=args, **kwargs)

        # Handle backward compatibility
        if self.xtcfile and not self.trajectory_file:
            self.trajectory_file = self.xtcfile
        if self.xtcfiles and not self.trajectory_files:
            self.trajectory_files = self.xtcfiles

        # Check files
        gsv.check_exists(topology_file=self.topology_file)
        self.log_msg(f"Loading topology from file {self.topology_file}")
        gsv.check_mutually_exclusive(trajectory_file=self.trajectory_file,
                                      coord_glob=self.coord_glob,
                                      trajectory_files=self.trajectory_files,
                                      require_one=True)
        if self.coord_glob:
            self.log_msg(f"Loading trajectory files matching glob pattern: {self.coord_glob}")
            traj_files = glob(self.coord_glob)
            if not traj_files:
                self.raise_error(f"No files found for pattern {self.coord_glob}")
            self.log_msg(f"Found {len(traj_files)} files")
            u = mda.Universe(self.topology_file, *traj_files,
                             topology_format=self.topology_format)
        elif self.trajectory_files:
            self.log_msg("Loading trajectory from multiple files: ")
            files = '\n\t - '.join(self.trajectory_files)
            self.log_msg(f"- {files}")
            for xf in self.trajectory_files:
                if not os.path.exists(xf):
                    self.raise_error(f"File {xf} not found")
            u = mda.Universe(self.topology_file, *self.trajectory_files,
                             topology_format=self.topology_format)
        else:
            self.log_msg(f"Loading trajectory from file: {self.trajectory_file}")
            if not os.path.exists(self.trajectory_file):
                self.raise_error(f"File {self.trajectory_file} not found")
            u = mda.Universe(self.topology_file, self.trajectory_file,
                             topology_format=self.topology_format)

        self.u = u
        self.select_atoms()
        self.extract_topology_info()

    def get_total_frames(self) -> int:
        """
        Get the number of frames in the trajectory.

        Returns
        -------
        int
            Number of frames.
        """
        return len(self.u.trajectory)

    def get_total_frames(self):
        total = len(self.u.trajectory)
        if hasattr(self, 'processors'):
            for processor in self.processors:
                total = processor.adjust_total_frames(total)
        return total
