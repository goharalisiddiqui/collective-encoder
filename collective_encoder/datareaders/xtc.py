import os

from glob import glob
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import ase
import MDAnalysis as mda

import gslibs.validation as gsv
from .trajectory import TrajectoryReaderBase

class XTCReader(TrajectoryReaderBase):
    """
    Read GROMACS XTC/TPR trajectories via MDAnalysis.

    Loads a molecular topology (``.tpr``) together with one or more trajectory
    files (``.xtc``) and converts each requested frame into an ASE ``Atoms``
    object.  Labeling is performed per-frame using a
    :class:`~collective_encoder.datalabelers.base.FrameLabeler`.

    Supports optional multi-process reading: the trajectory is copied to
    independent worker processes (each with its own Universe) to parallelise
    I/O and label computation.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``topology_file`` (str): Path to the topology file (e.g., ``.tpr``).
        - ``xtcfile`` (str, optional): Path to a single ``.xtc`` trajectory file.
        - ``xtcfiles`` (list of str, optional): List of ``.xtc`` files to concatenate.
        - ``coord_glob`` (str, optional): Glob pattern matching one or more ``.xtc`` files.
        - ``selection`` (str, optional): MDAnalysis atom selection string (default: ``'all'``).
        - ``type_to_elements`` (dict, optional): Mapping of atom types to element numbers
          when the topology lacks element information.
        - ``parallel`` (bool, optional): Whether to use multiprocessing for reading (default: ``True``).
    kwargs : dict
        Additional arguments forwarded to :class:`~collective_encoder.datareaders.base.BaseDataReader`.

    Note
    ----
    Exactly one of ``xtcfile``, ``xtcfiles``, or ``coord_glob`` must be provided.
    """
    _IDENTIFIER = "XTC"
    _REQUIRED_ARGS = TrajectoryReaderBase._REQUIRED_ARGS + [
        'topology_file',
    ]
    _OPTIONAL_ARGS = TrajectoryReaderBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'xtcfile': None,
        'xtcfiles': None,
        'coord_glob': None,
        'topology_format': 'TPR',
    })
    
    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        super().__init__(args=args, **kwargs)
        
        # Check files
        gsv.check_exists(topology_file=self.topology_file)
        self.log_msg(f"Loading topology from file {self.topology_file}")
        gsv.check_mutually_exclusive(xtcfile=self.xtcfile, 
                                      coord_glob=self.coord_glob, 
                                      xtcfiles=self.xtcfiles,
                                      require_one=True)
        if self.coord_glob:
            self.log_msg(f"Loading trajectory files matching glob pattern: {self.coord_glob}") 
            xtcfiles = glob(self.coord_glob)
            if not xtcfiles:
                self.raise_error(f"No files found for pattern {self.coord_glob}")
            self.log_msg(f"Found {len(xtcfiles)} files") 
            u = mda.Universe(self.topology_file, *xtcfiles, 
                             topology_format=self.topology_format)
        elif self.xtcfiles:
            self.log_msg("Loading trajectory from multiple files: ")
            files = '\n\t - '.join(self.xtcfiles)
            self.log_msg(f"- {files}")
            for xf in self.xtcfiles:
                if not os.path.exists(xf):
                    self.raise_error(f"File {xf} not found")
            u = mda.Universe(self.topology_file, *xtcfiles,
                             topology_format=self.topology_format)
        else:
            self.log_msg(f"Loading trajectory from file: {self.xtcfile}") 
            if not os.path.exists(self.xtcfile):
                self.raise_error(f"File {self.xtcfile} not found")
            u = mda.Universe(self.topology_file, self.xtcfile,
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
    