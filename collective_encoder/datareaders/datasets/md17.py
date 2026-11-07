import os
from typing import Any, Dict, List, Optional
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
        - ``molecule_name`` (str or list of str): The name (or list of names) of the MD17 molecule(s).
        - ``data_dir`` (str, optional): The directory where data should be downloaded/cached.
        - ``parallel`` (bool, optional): Whether to use multiprocessing.
    kwargs : dict
        Additional arguments forwarded to `ExternalDatasetReaderBase`.
    """
    _IDENTIFIER = "MD17"
    
    _REQUIRED_ARGS = ExternalDatasetReaderBase._REQUIRED_ARGS + [
        'molecule_name'
    ]

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
        Check if the requested molecule(s) exist in the revised MD17 dataset.
        Supports either a single molecule name string or a list of molecule names.
        """
        if isinstance(self.molecule_name, str):
            raw_molecules = [self.molecule_name]
        elif isinstance(self.molecule_name, (list, tuple)) and len(self.molecule_name) > 0:
            raw_molecules = list(self.molecule_name)
        else:
            self.raise_error(
                "'molecule_name' must be a non-empty string or list of strings."
            )

        validated_names: List[str] = []
        file_keys: List[str] = []
        for mol in raw_molecules:
            if not isinstance(mol, str):
                self.raise_error(
                    f"Each molecule name must be a string, got {type(mol).__name__}."
                )
            molecule = mol.lower()
            if molecule not in self._MOLECULES:
                self.raise_error(
                    f"Invalid molecule '{mol}' for MD17. "
                    f"Available molecules: {sorted(list(self._MOLECULES.keys()))}"
                )
            validated_names.append(molecule)
            file_keys.append(self._MOLECULES[molecule])

        self.molecule_names = validated_names
        self._molecule_file_keys = file_keys
        self.molecule_name = (
            validated_names[0] if len(validated_names) == 1 else validated_names
        )
        self._molecule_file_key = file_keys[0]

    def _get_download_url(self) -> str:
        """
        Return the download URL for the revised MD17 archive.
        """
        return self._ARCHIVE_URL

    def _get_target_filepath(self, molecule_key: Optional[str] = None) -> str:
        """
        Locate the .npz file for the configured molecule.
        Checks candidate locations within data_dir.
        """
        key = molecule_key if molecule_key is not None else self._molecule_file_key
        filename = f"rmd17_{key}.npz"
        candidates = [
            os.path.join(self.data_dir, "rmd17", "npz_data", filename),
            os.path.join(self.data_dir, "npz_data", filename),
            os.path.join(self.data_dir, filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return os.path.join(self.data_dir, "rmd17", "npz_data", filename)

    def _get_target_filepaths(self) -> List[str]:
        """
        Locate the .npz files for all configured molecules.
        """
        return [self._get_target_filepath(k) for k in self._molecule_file_keys]
