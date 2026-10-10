from typing import Any, Dict
from collective_encoder.datareaders.external_dataset import ExternalDatasetReaderBase

class MD17Reader(ExternalDatasetReaderBase):
    """
    Data reader for the MD17 dataset.

    Downloads and reads molecular trajectories from the MD17 database.
    Supported molecules include aspirin, benzene, ethanol, malonaldehyde,
    naphthalene, salicylic_acid, toluene, and uracil.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Valid keys include:
        - ``molecule_name`` (str): The name of the MD17 molecule.
        - ``data_dir`` (str, optional): The directory where data should be downloaded.
        - ``import_labels`` (list of str, optional): Pre-computed labels to extract
          (e.g., ``["E", "F"]`` for energies and forces).
        - ``parallel`` (bool, optional): Whether to use multiprocessing.
    kwargs : dict
        Additional arguments forwarded to `ExternalDatasetReaderBase`.
    """
    _IDENTIFIER = "MD17"

    # Official URL mappings for MD17 .npz files
    _URLS = {
        "benzene": "http://www.quantum-machine.org/gdml/data/npz/md17_benzene2017.npz",
        "uracil": "http://www.quantum-machine.org/gdml/data/npz/md17_uracil.npz",
        "naphthalene": "http://www.quantum-machine.org/gdml/data/npz/md17_naphthalene.npz",
        "aspirin": "http://www.quantum-machine.org/gdml/data/npz/md17_aspirin.npz",
        "salicylic_acid": "http://www.quantum-machine.org/gdml/data/npz/md17_salicylic.npz",
        "malonaldehyde": "http://www.quantum-machine.org/gdml/data/npz/md17_malonaldehyde.npz",
        "ethanol": "http://www.quantum-machine.org/gdml/data/npz/md17_ethanol.npz",
        "toluene": "http://www.quantum-machine.org/gdml/data/npz/md17_toluene.npz",
    }

    def _validate_molecule(self):
        """
        Check if the requested molecule exists in the MD17 dataset.
        """
        molecule = self.molecule_name.lower()
        if molecule not in self._URLS:
            self.raise_error(f"Invalid molecule '{self.molecule_name}' for MD17. "
                             f"Available molecules: {list(self._URLS.keys())}")
        self.molecule_name = molecule

    def _get_download_url(self) -> str:
        """
        Return the download URL for the configured molecule.
        """
        return self._URLS[self.molecule_name]
