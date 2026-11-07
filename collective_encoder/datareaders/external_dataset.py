import os
import random
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import ase
import numpy as np
import MDAnalysis as mda
from tqdm import tqdm

from collective_encoder.datalabelers.base import BaseLabeler
from collective_encoder.datareaders.trajectory import TrajectoryReaderBase, _read_and_label
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

    def _get_target_filepaths(self) -> List[str]:
        """
        Return the local paths to all configured target .npz data files.
        """
        return [self._get_target_filepath()]

    def _setup_data_dir(self):
        """
        Set up the local directory for caching dataset files.
        Uses a stable directory without generating numbered run dirs.
        """
        if self.data_dir is None:
            self._create_dir("md17_data", 
                             var_name="data_dir", 
                             use_existing=True, 
                             in_exec_dir=True)
        else:
            self._create_dir(self.data_dir, 
                             var_name="data_dir", 
                             use_existing=True, 
                             base_type="absolute")
            

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
        Ensure all dataset files exist (downloading and uncompressing once if necessary),
        load them into numpy, and construct the in-memory MDAnalysis Universe(s).
        """
        filepaths = self._get_target_filepaths()
        if any(not os.path.exists(fp) for fp in filepaths):
            url = self._get_download_url()
            url_filename = url.split('/')[-1]
            archive_path = os.path.join(self.data_dir, url_filename)
            is_archive = url_filename.endswith((".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar", ".zip"))

            if is_archive:
                if not os.path.exists(archive_path):
                    self._download_file(url, archive_path)
                self._extract_archive(archive_path, self.data_dir)
            else:
                for fp in filepaths:
                    if not os.path.exists(fp):
                        self._download_file(url, fp)

            filepaths = self._get_target_filepaths()
            for fp in filepaths:
                if not os.path.exists(fp):
                    self.raise_error(f"Target data file '{fp}' not found after downloading/extracting from {url}.")

        mol_names = getattr(self, "molecule_names", None)
        if mol_names is None:
            raw_name = getattr(self, "molecule_name", "molecule")
            mol_names = [raw_name] if isinstance(raw_name, str) else list(raw_name)

        self.molecules_data: List[Dict[str, Any]] = []
        self.raw_data_list: List[Dict[str, Any]] = []
        self.universes: List[mda.Universe] = []

        for mol_name, filepath in zip(mol_names, filepaths):
            self.log_msg(f"Using cached dataset for '{mol_name}': {filepath}")
            try:
                data = np.load(filepath)
                raw_data = {k: data[k] for k in data.files}
            except Exception as e:
                self.raise_error(f"Failed to load .npz file {filepath}: {e}")

            u = self._construct_universe_from_raw(raw_data, mol_name=mol_name)
            self.raw_data_list.append(raw_data)
            self.universes.append(u)
            self.molecules_data.append({
                "name": mol_name,
                "filepath": filepath,
                "raw_data": raw_data,
                "u": u,
                "n_frames": len(u.trajectory),
            })

        # Keep single-molecule attributes pointing to the first molecule for compatibility
        self.raw_data = self.raw_data_list[0]
        self.u = self.universes[0]

    def _construct_universe_from_raw(
        self, raw_data: Dict[str, Any], mol_name: str = ""
    ) -> mda.Universe:
        """
        Construct an empty MDAnalysis Universe and populate it with coordinates
        and atomic numbers from ``raw_data``.
        Looks for coordinates in 'coords' (or 'R') and atomic numbers in 'nuclear_charges' (or 'z').
        """
        coords = None
        for k in ('coords', 'R'):
            if k in raw_data:
                coords = raw_data[k]
                break
        if coords is None:
            self.raise_error("Dataset must contain coordinates under 'coords' or 'R'.")

        atomic_numbers = None
        for k in ('nuclear_charges', 'z'):
            if k in raw_data:
                atomic_numbers = raw_data[k]
                break
        if atomic_numbers is None:
            self.raise_error("Dataset must contain atomic numbers under 'nuclear_charges' or 'z'.")

        n_frames = coords.shape[0]
        n_atoms = coords.shape[1]

        label = f" for '{mol_name}'" if mol_name else ""
        self.log_msg(f"Constructing in-memory Universe{label} with {n_frames} frames and {n_atoms} atoms.")

        u = mda.Universe.empty(n_atoms,
                               n_residues=1,
                               atom_resindex=[0]*n_atoms,
                               residue_segindex=[0],
                               trajectory=True)

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

        u.load_new(coords, format="MEMORY")
        u.atoms.guess_bonds()
        return u

    def _construct_universe(self):
        """
        Backward-compatible helper to construct ``self.u`` from ``self.raw_data``.
        """
        self.u = self._construct_universe_from_raw(self.raw_data)

    def select_atoms(self):
        """
        Select atoms from each molecule's MDAnalysis universe based on ``self.selection``.
        """
        for m in self.molecules_data:
            self.u = m["u"]
            super().select_atoms()
            m["mol"] = self.mol
        self.u = self.universes[0]
        self.mol = self.molecules_data[0]["mol"]

    def _extract_topology_info(self):
        """
        Extract atomic numbers, elements, atom IDs, and bonds for each loaded molecule.
        """
        for m in self.molecules_data:
            self.u = m["u"]
            self.mol = m["mol"]
            super()._extract_topology_info()
            m["at_elements"] = self.at_elements
            m["atns"] = self.atns
            m["atm_ids"] = self.atm_ids
            m["bonds"] = self.bonds

        self.u = self.universes[0]
        self.mol = self.molecules_data[0]["mol"]

        if len(self.molecules_data) == 1:
            m0 = self.molecules_data[0]
            self.at_elements = m0["at_elements"]
            self.atns = m0["atns"]
            self.atm_ids = m0["atm_ids"]
            self.bonds = m0["bonds"]
        else:
            self.at_elements = [m["at_elements"] for m in self.molecules_data]
            self.atns = [m["atns"] for m in self.molecules_data]
            self.atm_ids = [m["atm_ids"] for m in self.molecules_data]
            self.bonds = [m["bonds"] for m in self.molecules_data]

    def get_total_frames(self) -> int:
        """
        Return the total number of frames across all loaded molecules.
        """
        return sum(m["n_frames"] for m in self.molecules_data)

    def _initialize_labeler(self,
                          labeler_type: str,
                          labeler_args: dict,
                          raw_data: Optional[Dict[str, Any]] = None,
                          universe: Optional[mda.Universe] = None) -> BaseLabeler:
        """
        Instantiate the labeler for this external dataset reader, passing the dataset.
        """
        labeler_cls = get_module("datalabelers", labeler_type)
        run_args = self.get_run_args().copy()
        run_args.pop("dataset", None)
        run_args.pop("universe", None)
        ds = raw_data if raw_data is not None else self.raw_data
        u = universe if universe is not None else self.u
        return labeler_cls(dataset=ds,
                           universe=u,
                           args=labeler_args,
                           **run_args)

    def _get_processor_sequence_span(self) -> int:
        """
        Return the frame span required by configured processors for a single start index.
        """
        probe = [0]
        for processor in self.processors:
            probe = processor.prepare_seq(probe)
        if not probe:
            return 1
        return max(probe) - min(probe) + 1

    def read_trajectory(
        self,
        indices: List[List[int]],
        labeler_type: str = 'Dummy',
        labeler_args: Dict[str, Union[str, float, List[int]]] = {},
    ) -> Tuple[Tuple[List[ase.Atoms]], Tuple[List[List[float]]]]:
        """
        Read the requested frames and compute labels across one or multiple molecules.
        When multiple molecules are configured, each sequence/sample in ``indices``
        randomly selects one of the loaded molecules and reads its consecutive frames
        from that molecule's trajectory.
        """
        if labeler_type is None:
            labeler_type_resolved = "DUMMY"
        else:
            labeler_type_resolved = labeler_type
        if labeler_type_resolved not in self._COMPATIBLE_LABELERS:
            raise ValueError(
                f"Labeler type '{labeler_type_resolved}' is not compatible with this data reader. "
                f"Compatible labelers: {self._COMPATIBLE_LABELERS}"
            )

        mol_labelers = [
            self._initialize_labeler(
                labeler_type=labeler_type_resolved,
                labeler_args=labeler_args,
                raw_data=m["raw_data"],
                universe=m["u"],
            )
            for m in self.molecules_data
        ]
        self.label_list = mol_labelers[0].get_label_names()

        self.log_msg("Reading trajectories...")
        seq_span = self._get_processor_sequence_span()
        n_molecules = len(self.molecules_data)

        trajs, labels = (), ()
        for seq_idx, raw_seq in enumerate(
            tqdm(
                indices,
                position=0,
                disable=not getattr(self, "verbose", True),
                leave=True,
                desc="Processing sequences",
                dynamic_ncols=True,
            )
        ):
            raw_seq_list = list(raw_seq)
            if len(raw_seq_list) == 0:
                trajs += ([],)
                labels += ([],)
                continue

            split_traj: List[ase.Atoms] = []
            split_label: List[Any] = []
            split_failed: List[int] = []
            expected_indices_flat: List[int] = []

            for raw_idx in raw_seq_list:
                mol_idx = 0 if n_molecules == 1 else random.randrange(n_molecules)
                mol_info = self.molecules_data[mol_idx]
                max_valid_start = max(1, mol_info["n_frames"] - seq_span + 1)
                local_start = int(raw_idx) % max_valid_start

                expanded_local = [local_start]
                for processor in self.processors:
                    expanded_local = processor.prepare_seq(expanded_local)
                expected_indices_flat.extend(expanded_local)

                worker_args = (
                    0,
                    mol_info["u"],
                    self.selection,
                    mol_info["atns"],
                    mol_info["at_elements"],
                    expanded_local,
                    mol_labelers[mol_idx],
                )
                sample_traj, sample_label, sample_failed = _read_and_label(worker_args)
                for struct in sample_traj:
                    struct.info["bonds"] = mol_info["bonds"]
                    struct.info["molecule_name"] = mol_info["name"]
                    struct.info["molecule_idx"] = mol_idx

                split_traj.extend(sample_traj)
                split_label.extend(sample_label)
                split_failed.extend(sample_failed)

            self._validate_traj_length(
                indices=expected_indices_flat,
                read_indices=split_traj,
                failed_indices=split_failed,
                idx=seq_idx,
            )

            for processor in self.processors:
                split_traj, split_label = processor.postprocess_seq(
                    split_traj, split_label
                )

            trajs += (split_traj,)
            labels += (split_label,)

        self.log_msg("Finished reading trajectories.")
        return trajs, labels
