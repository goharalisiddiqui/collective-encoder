import os
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import MDAnalysis as mda

from collective_encoder.datalabelers.base import BaseLabeler
from collective_encoder.datareaders.trajectory import TrajectoryReaderBase
from collective_encoder.common.registry import get_module

class ExternalDatasetReaderBase(TrajectoryReaderBase, ABC):
    """
    Base class for downloading and reading array-based trajectory datasets.

    This class handles downloading datasets (including archives like `.tar.bz2`),
    extracting atomic numbers and coordinates, constructing an in-memory MDAnalysis
    Universe, and exposing pre-computed labels via
    :class:`~collective_encoder.datalabelers.from_external.FromExternalLabeler`.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``molecule_name`` (str): The name of the molecule to load.
        - ``data_dir`` (str, optional): The directory where data should be downloaded.
          Defaults to ``./data/<dataset_name>``.
        - ``parallel`` (bool, optional): Whether to use multiprocessing.
    kwargs : dict
        Additional arguments forwarded to `TrajectoryReaderBase`.
    """
    _COMPATIBLE_LABELERS = TrajectoryReaderBase._COMPATIBLE_LABELERS + [
        "FROM_EXTERNAL"
    ]
    _IDENTIFIER = None  # To be defined by subclasses
    _REQUIRED_ARGS = TrajectoryReaderBase._REQUIRED_ARGS + ['molecule_name']
    _OPTIONAL_ARGS = TrajectoryReaderBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'data_dir': None,
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args=args, **kwargs)

        self._validate_molecule()
        self._setup_data_dir()
        self._download_and_load()
        self.select_atoms()
        self._extract_topology_info()

    @abstractmethod
    def _validate_molecule(self):
        """
        Validate the molecule name. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _validate_molecule")

    @abstractmethod
    def _get_download_url(self) -> str:
        """
        Get the download URL for the requested molecule or archive. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_download_url")

    def _get_target_filepath(self) -> str:
        """
        Return the local path to the target .npz data file.
        Subclasses can override this to point inside extracted archive directories.
        """
        url = self._get_download_url()
        filename = url.split('/')[-1]
        return os.path.join(self.data_dir, filename)

    def _setup_data_dir(self):
        """
        Set up the local directory for caching dataset files.
        Uses a stable directory without generating numbered run dirs.
        """
        self._create_dir("md17_data", 
                         var_name="data_dir", 
                         use_existing=True, 
                         in_exec_dir=True)

    def _download_file(self, url: str, destination: str):
        """
        Download a file over HTTP using a custom User-Agent and streaming chunks.
        """
        self.log_msg(f"Downloading {os.path.basename(destination)} from {url}...")
        tmp_destination = destination + ".tmp"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (collective-encoder)"}
            )
            with urllib.request.urlopen(req) as resp, open(tmp_destination, "wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            os.replace(tmp_destination, destination)
            self.log_msg(f"Download complete: {destination}")
        except Exception as e:
            if os.path.exists(tmp_destination):
                os.remove(tmp_destination)
            self.raise_error(f"Failed to download dataset from {url}: {e}")

    def _extract_archive(self, archive_path: str, extract_dir: str):
        """
        Extract an archive (.tar.bz2, .tar.gz, .tar, .zip) into extract_dir.
        """
        self.log_msg(f"Extracting {archive_path} to {extract_dir}...")
        try:
            if archive_path.endswith((".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar")):
                with tarfile.open(archive_path, "r:*") as tar:
                    if hasattr(tarfile, 'data_filter'):
                        tar.extractall(path=extract_dir, filter='data')
                    else:
                        tar.extractall(path=extract_dir)
            elif archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(path=extract_dir)
            else:
                self.raise_error(f"Unsupported archive format: {archive_path}")
            self.log_msg("Extraction complete.")
        except Exception as e:
            self.raise_error(f"Failed to extract archive {archive_path}: {e}")

    def _download_and_load(self):
        """
        Ensure the dataset file exists (downloading and uncompressing once if necessary),
        load it into numpy, and construct the in-memory MDAnalysis Universe.
        """
        filepath = self._get_target_filepath()

        if not os.path.exists(filepath):
            url = self._get_download_url()
            url_filename = url.split('/')[-1]
            archive_path = os.path.join(self.data_dir, url_filename)
            is_archive = url_filename.endswith((".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar", ".zip"))

            if is_archive:
                # If archive doesn't exist, download it
                if not os.path.exists(archive_path):
                    self._download_file(url, archive_path)
                # Extract archive
                self._extract_archive(archive_path, self.data_dir)
            else:
                # Direct file download
                self._download_file(url, filepath)

            # Re-check target filepath after extraction
            filepath = self._get_target_filepath()
            if not os.path.exists(filepath):
                self.raise_error(f"Target data file '{filepath}' not found after downloading/extracting from {url}.")
        else:
            self.log_msg(f"Using cached dataset: {filepath}")

        # Load NPZ file
        try:
            data = np.load(filepath)
            self.raw_data = {k: data[k] for k in data.files}
        except Exception as e:
            self.raise_error(f"Failed to load .npz file {filepath}: {e}")

        self._construct_universe()

    def _construct_universe(self):
        """
        Construct an empty MDAnalysis Universe and populate it with coordinates
        and atomic numbers from the dataset.
        Looks for coordinates in 'coords' (or 'R') and atomic numbers in 'nuclear_charges' (or 'z').
        """
        coords = None
        for k in ('coords', 'R'):
            if k in self.raw_data:
                coords = self.raw_data[k]
                break
        if coords is None:
            self.raise_error("Dataset must contain coordinates under 'coords' or 'R'.")

        atomic_numbers = None
        for k in ('nuclear_charges', 'z'):
            if k in self.raw_data:
                atomic_numbers = self.raw_data[k]
                break
        if atomic_numbers is None:
            self.raise_error("Dataset must contain atomic numbers under 'nuclear_charges' or 'z'.")

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
        u.add_TopologyAttr('atomnum', atomic_numbers)
        u.add_TopologyAttr('id', list(range(1, n_atoms + 1)))
        u.add_TopologyAttr('resname', ['UNK'])
        u.add_TopologyAttr('resid', [1])
        u.add_TopologyAttr('segid', ['SYSTEM'])
        
        # Load coordinates into memory reader
        # MDAnalysis MemoryReader expects a shape of (n_frames, n_atoms, 3)
        u.load_new(coords, format="MEMORY")

        u.atoms.guess_bonds()
        
        self.u = u

    def get_total_frames(self) -> int:
        """
        Get the total number of frames considering processor adjustments.
        """
        total = len(self.u.trajectory)
        if hasattr(self, 'processors'):
            for processor in self.processors:
                total = processor.adjust_total_frames(total)
        return total

    def _initialize_labeler(self,
                          labeler_type: str,
                          labeler_args: dict) -> BaseLabeler:
        """
        Instantiate the labeler for this external dataset reader, passing the dataset.
        """
        labeler_cls = get_module("datalabelers", labeler_type)
        run_args = self.get_run_args().copy()
        run_args.pop("dataset", None)
        return labeler_cls(dataset=self.raw_data, 
                           args=labeler_args, 
                           **run_args)
