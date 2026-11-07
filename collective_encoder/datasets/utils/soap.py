from typing import Dict
import logging
import numpy as np

_log = logging.getLogger(__name__)

def soap_modifications(dataset_args: Dict, datareader) -> Dict:
    """Resolve MDAnalysis atom selections to integer indices for SOAP datasets."""
    if dataset_args.get("atoms_selections") is None:
        return dataset_args

    u = datareader.univ
    mol = datareader.mol
    selected_indices = []

    for selection in dataset_args["atoms_selections"]:
        sel_atoms = u.select_atoms(selection)
        if sel_atoms.n_atoms != 1:
            _log.warning(
                "[SOAP] Selection '%s' does not select exactly one atom, selected %d atoms",
                selection, sel_atoms.n_atoms,
            )
        n_types = len({at.type for at in sel_atoms})
        if n_types > 1:
            _log.warning("[SOAP] Selection '%s' selects more than one atom type.", selection)
        _log.info("Selected %d atoms of total %d types", sel_atoms.n_atoms, n_types)
        for at in sel_atoms:
            mol_index = np.where(mol.atoms.indices == at.index)[0]
            if len(mol_index) == 0:
                raise ValueError(
                    f"Atom {at.index} in selection '{selection}' not found in "
                    "selected molecule atoms"
                )
            selected_indices.append(int(mol_index[0]))

    dataset_args["selected_atoms"] = dataset_args.get("selected_atoms", []) + selected_indices
    del dataset_args["atoms_selections"]
    return dataset_args