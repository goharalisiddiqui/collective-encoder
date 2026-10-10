import os
import urllib.request
from typing import Any, Dict, List, Optional
import numpy as np
import MDAnalysis as mda

import gslibs.validation as gsv
from collective_encoder.datareaders.trajectory import TrajectoryReaderBase

class ExternalDatasetReaderBase(TrajectoryReaderBase):
    """
    Base class for downloading and reading array-based trajectory datasets.

    This class handles downloading `.npz` files, extracting atomic numbers and
    coordinates, constructing an in-memory MDAnalysis Universe, and exposing
    pre-computed labels via `import_labels`.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``molecule_name`` (str): The name of the molecule to load.
        - ``data_dir`` (str, optional): The directory where data should be downloaded.
          Defaults to ``./data/<dataset_name>``.
        - ``import_labels`` (list of str, optional): List of keys in the `.npz` file
          to extract as pre-computed labels (e.g., ``["E", "F"]``).
        - ``parallel`` (bool, optional): Whether to use multiprocessing.
    kwargs : dict
        Additional arguments forwarded to `TrajectoryReaderBase`.
    """
    _IDENTIFIER = None  # To be defined by subclasses
    _REQUIRED_ARGS = TrajectoryReaderBase._REQUIRED_ARGS + ['molecule_name']
    _OPTIONAL_ARGS = TrajectoryReaderBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'data_dir': None,
        'import_labels': [],
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        # We need to bypass the topology checking in the TrajectoryReaderBase,
        # so we don't call super().__init__ directly until we set up the Universe.
        # But CEModule requires initialization.
        super(TrajectoryReaderBase, self).__init__(args=args, **kwargs)

        self.processors = []
        processor_configs = self.args.get('processors', [])
        from collective_encoder.datareaders.processors.resolver import get_processor
        for config in processor_configs:
            processor_cls = get_processor(config['type'])
            processor_args = config.get('args', {})
            self.processors.append(processor_cls(**processor_args))
            self.log_msg(f"Initialized processor: {config['type']} with args {processor_args}")

        self._validate_molecule()
        self._setup_data_dir()
        self._download_and_load()
        self.select_atoms()
        self._extract_topology_info()

    def _validate_molecule(self):
        """
        Validate the molecule name. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _validate_molecule")

    def _get_download_url(self) -> str:
        """
        Get the download URL for the requested molecule. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_download_url")

    def _setup_data_dir(self):
        """
        Set up the local directory for caching dataset files.
        """
        dataset_name = self.get_identifier().lower()
        if self.data_dir is None:
            # Fall back to a local data directory if run_dir isn't available
            base_dir = getattr(self, 'run_dir', './')
            self.data_dir = os.path.join(base_dir, 'data', dataset_name)

        # Use safe_create_dir from CEModule
        self.data_dir = self.safe_create_dir(self.data_dir)

    def _download_and_load(self):
        """
        Download the dataset if it doesn't exist, load it into numpy,
        and construct the in-memory MDAnalysis Universe.
        """
        url = self._get_download_url()
        filename = url.split('/')[-1]
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            self.log_msg(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                self.log_msg(f"Download complete: {filepath}")
            except Exception as e:
                self.raise_error(f"Failed to download dataset from {url}: {e}")
        else:
            self.log_msg(f"Using cached dataset: {filepath}")

        # Load NPZ file
        try:
            data = np.load(filepath)
            self.raw_data = {k: data[k] for k in data.files}
        except Exception as e:
            self.raise_error(f"Failed to load .npz file {filepath}: {e}")

        self._construct_universe()
        self._extract_imported_labels()

    def _construct_universe(self):
        """
        Construct an empty MDAnalysis Universe and populate it with coordinates
        and atomic numbers from the dataset.
        Assumes `self.raw_data` has 'R' (coordinates) and 'z' (atomic numbers).
        """
        if 'R' not in self.raw_data or 'z' not in self.raw_data:
            self.raise_error("Dataset must contain 'R' (coordinates) and 'z' (atomic numbers).")

        coords = self.raw_data['R']  # Shape: (n_frames, n_atoms, 3)
        atomic_numbers = self.raw_data['z']  # Shape: (n_atoms,)

        n_frames = coords.shape[0]
        n_atoms = coords.shape[1]

        self.log_msg(f"Constructing in-memory Universe with {n_frames} frames and {n_atoms} atoms.")

        # Create empty Universe
        u = mda.Universe.empty(n_atoms,
                               n_residues=1,
                               atom_resindex=[0]*n_atoms,
                               residue_segindex=[0],
                               trajectory=True)

        # Add topology attributes
        # Map atomic numbers to elements using ASE
        from ase.data import chemical_symbols
        elements = [chemical_symbols[z] for z in atomic_numbers]

        u.add_TopologyAttr('name', elements)
        u.add_TopologyAttr('type', elements)
        u.add_TopologyAttr('element', elements)
        u.add_TopologyAttr('resname', ['UNK'])
        u.add_TopologyAttr('resid', [1])
        u.add_TopologyAttr('segid', ['SYSTEM'])

        # Load coordinates into memory reader
        # MDAnalysis MemoryReader expects a shape of (n_frames, n_atoms, 3)
        u.load_new(coords, format="MemoryReader")

        self.u = u

    def _extract_imported_labels(self):
        """
        Extract pre-computed labels from the raw data.
        """
        self.imported_labels = {}
        for label_key in self.import_labels:
            if label_key in self.raw_data:
                self.imported_labels[label_key] = self.raw_data[label_key]
                self.log_msg(f"Imported pre-computed label: {label_key} with shape {self.raw_data[label_key].shape}")
            else:
                self.log_warn(f"Requested import_label '{label_key}' not found in dataset.")

    def get_total_frames(self) -> int:
        """
        Get the total number of frames considering processor adjustments.
        """
        total = len(self.u.trajectory)
        if hasattr(self, 'processors'):
            for processor in self.processors:
                total = processor.adjust_total_frames(total)
        return total
