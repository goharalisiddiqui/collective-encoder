from typing import List, Optional, Dict, Union

import ase
import numpy as np

import torch
from torch.utils.data import Dataset

from collective_encoder.collective_encoder.datasets.base import BaseDataset
import featomic.torch
import metatensor.torch as mts

import metatensor
import metatomic.torch as mta
import warnings


class MetatomicSoapPowerSpectrumDataset(torch.nn.Module):
    def __init__(self, spex, 
                 selected_atoms: List[int],
                 included_types: List[int]
                 ):
        super().__init__()

        self.spex = spex
        self.selected_atoms = selected_atoms
        # self.selected_atoms = self.register_buffer(
        #     "selected_atoms",
        #     torch.tensor(selected_atoms, dtype=torch.int32) if selected_atoms is not None else None
        # )
        self.register_buffer(
            "included_types",
            torch.tensor(included_types, dtype=torch.int32)
        )

    def get_atomic_types(self):
        return [a for a in range(0, 119)]  # all elements

    def get_interaction_range(self):
        return torch.inf

    def get_length_unit(self):
        return "angstrom"

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> torch.Tensor:
        
        # FIXME: Cannot compile this as TorchScript
        # if selected_atoms is not None:
        #     # Check consistency
        #     for label in selected_atoms:
        #         if int(label["atom"]) not in self.selected_atoms:
        #             raise ValueError(
        #                 f"selected_atoms contains atom index {label['atom'].item()}, "
        #                 f"which is not in the predefined selected_atoms {self.selected_atoms}"
        #             )

        if selected_atoms is None:
            warnings.warn(
                "No selected_atoms provided to MetatomicSoapPowerSpectrumDataset, "
                "using predefined selected_atoms which can be incorrect."
            )
            selected_cores: mts.Labels = mts.Labels(
                "atom", torch.tensor(self.selected_atoms).reshape(-1, 1)
            )
        else:
            selected_cores: mts.Labels = selected_atoms
        
        # computes the spherical expansion
        spex = self.spex(
            systems, selected_samples=selected_cores
        )

        # then manipulate the tensormap to remove some of the sparsity
        spex = spex.keys_to_properties("neighbor_1_type")
        spex = spex.keys_to_properties("neighbor_2_type")
        spex = spex.keys_to_samples("center_type")

        atom_desc = []
        selected_atom_indices: torch.Tensor = selected_cores.values
        if selected_atom_indices.shape[1] > 1:
            selected_atom_indices = selected_atom_indices[:, 1]
        selected_atom_indices = selected_atom_indices.flatten()
        selected_atom_indices_list: List[int] = selected_atom_indices.to(torch.int64).tolist()
        for atom in selected_atom_indices_list:
            desc_block = []
            sel_map = mts.slice(
                            spex,
                            axis="samples",
                            selection=mts.Labels(
                                "atom", torch.tensor([atom]).reshape(-1, 1)
                            ),
                        )
            sel_map = mts.slice(
                            sel_map,
                            axis="properties",
                            selection=mts.Labels(
                                ["neighbor_1_type", "neighbor_2_type"], 
                                torch.stack([
                                    self.included_types,
                                    self.included_types
                                ], dim=-1)
                            ),
                        )
            for block in sel_map.blocks():
                desc = block.values
                desc_block.append(desc)
            atom_desc.append(torch.concatenate(desc_block, dim=-2))
        descriptors = torch.concatenate([d.unsqueeze(1) for d in atom_desc], dim=1)

        # flatten the descriptor dimensions
        # descriptors = descriptors.reshape(descriptors.shape[0], -1) #FIXME: flatten for now

        # sum the atoms dimension
        descriptors = descriptors.sum(dim=1)


        return descriptors

class SoapPowerSpectrumDataset(Dataset, BaseDataset):
    '''
    Docstring for SoapPowerSpectrumDataset
    '''
    
    _IDENTIFIER = "SOAP_POWER_SPECTRUM"
    _REQUIRED_ARGS = ['selected_atoms', 'cutoff']
    _OPTIONAL_ARGS = {
        'max_angular': 6,
        'smoothing_width': 1.5,
        'gaussian_width': 1.0,
        'n_radial': 4,
        'excluded_types': [1],  # Exclude hydrogen by default
    }

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[float],
        dataset_args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        Dataset.__init__(self)
        BaseDataset.__init__(self, dataset_args=dataset_args, **kwargs)
        if any([atom >= len(structures[0]) for atom in self.selected_atoms]):
            raise ValueError(f"Selected atom indices {self.selected_atoms} "
                             f"are out of bounds for structure with "
                             f"{len(structures[0])} atoms.")

        # initialize and store the featomic calculator inside the class
        self.spex = featomic.torch.SoapPowerSpectrum(
            **{
                "cutoff": {
                    "radius": self.cutoff,
                    "smoothing": {"type": "ShiftedCosine", "width": self.smoothing_width},
                },
                "density": {"type": "Gaussian", "width": self.gaussian_width},
                "basis": {
                    "type": "TensorProduct",
                    "max_angular": self.max_angular,
                    "radial": {"type": "Gto", "max_radial": self.n_radial},
                },
            }
        )
        self.at_types = structures[0].get_atomic_numbers()
        self.included_types = list(set(self.at_types) - set(self.excluded_types))


        # Precompute all the systems
        systems = featomic.torch.systems_to_torch(structures)
        selected_atoms_label = mts.Labels(
            "atom", torch.tensor(self.selected_atoms).reshape(-1, 1)
        )
        descriptors = self.spex(
            systems, selected_samples=selected_atoms_label
        )

        descriptors = descriptors.keys_to_properties("neighbor_1_type")
        descriptors = descriptors.keys_to_properties("neighbor_2_type")
        descriptors = descriptors.keys_to_samples("center_type")

        # We want to return a tensor of shape (n_structures, n_selected_atoms, *descriptor_dimensions)
        
        atom_desc = []
        for atom in self.selected_atoms:
            desc_block = []
            sel_map = mts.slice(
                            descriptors,
                            axis="samples",
                            selection=mts.Labels(
                                "atom", torch.tensor([atom]).reshape(-1, 1)
                            ),
                        )
            sel_map = mts.slice(
                            sel_map,
                            axis="properties",
                            selection=mts.Labels(
                                ["neighbor_1_type", "neighbor_2_type"], 
                                torch.stack([
                                    torch.tensor(self.included_types),
                                    torch.tensor(self.included_types)
                                ], dim=-1)
                            ),
                        )
            for block in sel_map.blocks():
                desc = block.values
                desc_block.append(desc)
            atom_desc.append(torch.concatenate(desc_block, dim=-2))
        descriptors = torch.concatenate([d.unsqueeze(1) for d in atom_desc], dim=1)

        # descriptors shape: (n_structures, n_selected_atoms, sum of (2l+1) over l in angular_list), radial components * no of species
        self.descriptors = descriptors
        self.labels = [torch.tensor(d) for d in labels]
        print(f"[{type(self).__name__}]: Loaded {self.descriptors.shape[0]} data points with {self.descriptors.shape[1]} selected atoms.")

        # flatten the last two dimensions
        # self.descriptors = self.descriptors.reshape(self.descriptors.shape[0], -1)
        
        # sum the atoms dimension
        self.descriptors = self.descriptors.sum(dim=1)
        
        self.num_inputs = self.descriptors.shape[-1]
        print(f"[{type(self).__name__}]: Each data point has input dimension {self.num_inputs}.")
        print("="*80)
        
    def __len__(self):
        return self.descriptors.shape[0]

    def __getitem__(self, index):
        return self.descriptors[index], self.labels[index]
    
    def get_data(self):
        return self.descriptors, self.labels

    def get_metatomic_dataprocessor(self):
        return MetatomicSoapPowerSpectrumDataset(self.spex, 
                                                 self.selected_atoms,
                                                 included_types=self.included_types)