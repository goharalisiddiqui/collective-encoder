import os
from typing import Any, Dict
from collective_encoder.datareaders.external_dataset import ExternalDatasetReaderBase

class MD17Reader(ExternalDatasetReaderBase):
    """
    Data reader for the revised MD17 (rMD17) dataset.

    Downloads and reads molecular trajectories from the revised MD17 database
    hosted on Materials Cloud (record pfffs-fff86). Supported molecules include
    aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene,
    paracetamol, salicylic_acid (or salicylic), toluene, and uracil.

    Data arrays available in ``self.raw_data``:
    - ``coords``: Conformation coordinates of shape ``(n_frames, n_atoms, 3)``.
    - ``nuclear_charges``: Atomic numbers of shape ``(n_atoms,)``.
    - ``energies``: Recalculated total energies in kcal/mol of shape ``(n_frames,)``.
    - ``forces``: Recalculated Cartesian forces in kcal/mol/Å of shape ``(n_frames, n_atoms, 3)``.
    - ``old_indices``: Original conformation indices.
    - ``old_energies``: Original MD17 energies.
    - ``old_forces``: Original MD17 forces.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``molecule_name`` (str): The name of the MD17 molecule.
        - ``data_dir`` (str, optional): The directory where data should be downloaded/cached.
        - ``parallel`` (bool, optional): Whether to use multiprocessing.
    kwargs : dict
        Additional arguments forwarded to `ExternalDatasetReaderBase`.
    """
    _IDENTIFIER = "MD17"

    _ARCHIVE_URL = "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2"

    _MOLECULES = {
        "aspirin": "aspirin",
        "azobenzene": "azobenzene",
        "benzene": "benzene",
        "ethanol": "ethanol",
        "malonaldehyde": "malonaldehyde",
        "naphthalene": "naphthalene",
        "paracetamol": "paracetamol",
        "salicylic": "salicylic",
        "salicylic_acid": "salicylic",
        "toluene": "toluene",
        "uracil": "uracil",
    }

    def _validate_molecule(self):
        """
        Check if the requested molecule exists in the revised MD17 dataset.
        """
        molecule = self.molecule_name.lower()
        if molecule not in self._MOLECULES:
            self.raise_error(f"Invalid molecule '{self.molecule_name}' for MD17. "
                             f"Available molecules: {sorted(list(self._MOLECULES.keys()))}")
        self.molecule_name = molecule
        self._molecule_file_key = self._MOLECULES[molecule]

    def _get_download_url(self) -> str:
        """
        Return the download URL for the revised MD17 archive.
        """
        return self._ARCHIVE_URL

    def _get_target_filepath(self) -> str:
        """
        Locate the .npz file for the configured molecule.
        Checks candidate locations within data_dir.
        """
        filename = f"rmd17_{self._molecule_file_key}.npz"
        candidates = [
            os.path.join(self.data_dir, "rmd17", "npz_data", filename),
            os.path.join(self.data_dir, "npz_data", filename),
            os.path.join(self.data_dir, filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return os.path.join(self.data_dir, "rmd17", "npz_data", filename)
