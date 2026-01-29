from typing import Dict

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

def get_dataset_cls_dl(dataset_type: str, dataset_args, datareader) \
        -> tuple[type[Dataset], type[Dict], type[DataLoader]]:
    if dataset_type == 'DEFAULT':
        from datasets.positions import PositionsDataset as dataset_class
        dl_cls = DataLoader
    elif dataset_type == 'DISTANCES':
        from datasets.distances import DistancesDataset as dataset_class
        dataset_args['atm_ids'] = datareader.atm_ids
        dl_cls = DataLoader
    elif dataset_type == 'GRAPH':
        from datasets.bondgraph import BondGraphDataset as dataset_class
        dataset_args['bond_indices'] = datareader.bonds
        dl_cls = GeoDataLoader
    elif dataset_type == 'SOAP':
        from datasets.soap import SOAPDataset as dataset_class
        dataset_args = soap_modifications(dataset_args, datareader)
        dl_cls = DataLoader
    elif dataset_type == 'SOAP_PS':
        from datasets.soap_ps import SoapPowerSpectrumDataset as dataset_class
        dataset_args = soap_modifications(dataset_args, datareader)
        dl_cls = DataLoader
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset_class, dataset_args, dl_cls

def soap_modifications(dataset_args: Dict, datareader) -> Dict:
    # To enable selecting specific atoms for SOAP descriptors using MDAnalysis selections
    if dataset_args.get('atoms_selections', None) is not None:
        u = datareader.univ
        mol = datareader.mol
        selected_indices = []
        for selection in dataset_args['atoms_selections']:
            sel_atoms = u.select_atoms(selection)
            if sel_atoms.n_atoms != 1:
                print(f"[SOAP] WARNING! Selection {selection} does not select exactly one atom, selected {sel_atoms.n_atoms} atoms")
            n_types = len(set([at.type for at in sel_atoms]))
            if n_types > 1:
                print(f"[SOAP] WARNING! Selection {selection} selects more than one atom type.)")
            print(f"Selected {sel_atoms.n_atoms} atoms of total {n_types} types")
            for at in sel_atoms:
                mol_index = np.where(mol.atoms.indices == at.index)[0]
                if len(mol_index) == 0:
                    raise ValueError(f"Atom {at.index} in selection {selection} not found in selected molecule atoms")
                selected_indices.append(int(mol_index[0]))
        dataset_args['selected_atoms'] = dataset_args.get('selected_atoms', []) + selected_indices
        dataset_args.pop('atoms_selections')
        
    return dataset_args