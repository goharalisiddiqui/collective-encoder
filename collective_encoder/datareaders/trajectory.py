from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union
import os
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import ase

import MDAnalysis.transformations as trans
from MDAnalysis.exceptions import NoDataError

from collective_encoder.datareaders.base import BaseDataReader
from collective_encoder.datalabelers.resolver import get_labeler
from collective_encoder.datareaders.processors.resolver import get_processor


def _read_and_label_parallel(args):
    """Worker function: reads a chunk of frame sequences from a copied Universe.

    Receives a pre-copied Universe (picklable MemoryReader), re-applies trajectory
    transforms and creates a fresh labeler so no shared state is needed.
    Takes a single packed tuple so it is compatible with pool.imap.
    """
    worker_id, u_copy, selection, atns, at_elements, seqs, labeler_type, labeler_args, run_args = args

    from MDAnalysis.exceptions import NoDataError
    from collective_encoder.datalabelers.resolver import get_labeler

    mol = u_copy.select_atoms(selection)
    verbose = run_args.get('verbose', True)

    if worker_id > 0:
        run_args['verbose'] = False  # Only the main process shows progress bars and logs
    labeler_cls = get_labeler(labeler_type)
    labeler = labeler_cls(universe=u_copy,
                          args=labeler_args,
                          **run_args,
                          )

    # Cache residue/atom-name info once — constant across all frames
    try:
        residues = np.array([str(r.residue.resname) for r in mol.atoms])
    except NoDataError:
        residues = np.array(['UNK'] * mol.n_atoms)
    try:
        resids = np.array([r.residue.resid for r in mol.atoms])
    except NoDataError:
        resids = np.array([0] * mol.n_atoms)
    try:
        atomnames = np.array([str(a.name) for a in mol.atoms])
    except NoDataError:
        atomnames = np.array(at_elements)

    # Re-apply the same transformations to the copied Universe since they are not shared.
    transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
    u_copy.trajectory.add_transformations(*transforms)

    mol_traj, labels, failed_indices = [], [], []
    for idx in tqdm(seqs,
                    position=worker_id,
                    desc=f"Worker {worker_id}",
                    leave=False,
                    disable=not verbose,
                    dynamic_ncols=True):
        try:
            u_copy.trajectory[idx]
        except OSError:
            failed_indices.append(idx)
            continue
        structure = ase.Atoms(numbers=atns,
                                positions=mol.atoms.positions.copy(),
                                cell=mol.dimensions[:3].copy())
        if not np.all(mol.dimensions[:3] == 0):
            structure.set_pbc([True, True, True])
        structure.set_array('residuenames',   residues)
        structure.set_array('residuenumbers', resids)
        structure.set_array('atomtypes',      atomnames)
        labels.append(labeler.compute())
        mol_traj.append(structure)

    return mol_traj, labels, failed_indices

class TrajectoryReaderBase(BaseDataReader, ABC):
    """
    Abstract base class for trajectory readers.

    Provides common MDAnalysis functionality for selecting atoms,
    unwrapping trajectories, and extracting topological data (elements,
    atomic numbers, bonds) for downstream models.
    """
    _OPTIONAL_ARGS = {
        'selection': "all",
        'type_to_elements': None,
        'parallel': True,
    }

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.processors = []
        processor_configs = self.args.get('processors', [])
        for config in processor_configs:
            processor_cls = get_processor(config['type'])
            processor_args = config.get('args', {})
            self.processors.append(processor_cls(**processor_args))
            self.log_msg(f"Initialized processor: {config['type']} with args {processor_args}")

    def read_trajectory(self,
                        indices: List[List[int]],
                        labeler_type : str = 'Dummy',
                        labeler_args : Dict[str, Union[str, float, List[int]]] = {},
                        ) -> Tuple[Tuple[List[ase.Atoms]], Tuple[List[List[float]]], Tuple[List[int]]]:
        """
        Read the requested frames and compute labels using multi-processing.

        Parameters
        ----------
        indices : list of list of int
            List of index sequences to read (e.g. for train, val, test splits).
        labeler_type : str, optional
            Type of labeler to use, by default 'Dummy'.
        labeler_args : dict, optional
            Arguments for the labeler.

        Returns
        -------
        tuple
            A tuple containing:
            - Tuple of lists of `ase.Atoms` (one list per split).
            - Tuple of lists of labels (one list per split).
            - Tuple of lists of failed frame indices (one list per split).
        """
        labeler_cls = get_labeler(labeler_type)
        labeler = labeler_cls(
            universe=self.u,
            args=labeler_args,
        )
        self.label_list = labeler.get_label_names()

        self.log_msg(f"Reading trajectories...")

        # Apply processors to prepare sequences
        prepared_indices = []
        for seq in indices:
            processed_seq = seq
            for processor in self.processors:
                processed_seq = processor.prepare_seq(processed_seq)
            prepared_indices.append(processed_seq)

        trajs, labels, all_failed = (), (), ()
        for index_list in tqdm(prepared_indices,
                               position=0,
                               disable=not getattr(self, 'verbose', True),
                               leave=True,
                               desc="Processing sequences",
                               dynamic_ncols=True):

            if not getattr(self, 'parallel', True) or len(index_list) < 8:  # Threshold for parallel processing
                # Sequential read
                args = (0, self.u.copy(), self.selection, self.atns, self.at_elements, index_list,
                        labeler_type, labeler_args, getattr(self, 'run_args', {}))
                traj, label, failed = _read_and_label_parallel(args)
            else:
                # Parallel read
                n_workers = min(16, os.cpu_count() or 1, max(1, len(index_list)))
                chunks = [index_list[i::n_workers] for i in range(n_workers)]

                args = [
                    (i+1, self.u.copy(), self.selection, self.atns, self.at_elements,
                    chunk, labeler_type, labeler_args, getattr(self, 'run_args', {}))
                    for i, chunk in enumerate(chunks)
                ]

                with Pool(processes=len(args)) as pool:
                    chunk_results = pool.map(_read_and_label_parallel, args)

                # Reassemble results
                n_seqs = len(index_list)
                traj   = [None] * n_seqs
                label  = [None] * n_seqs
                failed = []
                for worker_idx, result in enumerate(chunk_results):
                    failed.extend(result[2])
                    for local_idx, (read_traj, read_label) in enumerate(zip(result[0], result[1])):
                        original_idx = worker_idx + local_idx * n_workers
                        traj[original_idx]  = read_traj
                        label[original_idx] = read_label

            # Apply post-processors
            for processor in self.processors:
                traj, label = processor.postprocess_seq(traj, label)

            trajs      += (traj,)
            labels     += (label,)
            all_failed += (failed,)

        self.log_msg(f"Finished reading trajectories.")
        return trajs, labels, all_failed


    def get_atomic_numbers(self) -> List[int]:
        """
        Get the extracted atomic numbers.

        Returns
        -------
        list of int
            List of atomic numbers for selected atoms.
        """
        return self.atns
    
    def get_element_symbols(self) -> List[str]:
        """
        Get the extracted element symbols.

        Returns
        -------
        list of str
            List of element symbols for selected atoms.
        """
        return self.at_elements
    
    def get_atom_ids(self) -> List[int]:
        """
        Get the original atom IDs.

        Returns
        -------
        list of int
            List of 1-indexed atom IDs from the topology.
        """
        return self.atm_ids
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """
        Get the bond connectivity for selected atoms.

        Returns
        -------
        list of tuple
            List of (atom_i, atom_j) index pairs.
        """
        return self.bonds
    
    def select_atoms(self):
        """
        Select atoms from the MDAnalysis universe based on `self.selection`.

        Raises
        ------
        ValueError
            If the selection string is invalid or matches zero atoms.
        """
        try:
            mol = self.u.select_atoms(self.selection)
        except Exception as e:
            raise ValueError(f"Selection {self.selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {self.selection} does not match any atoms in the trajectory")
        self.mol = mol
    
    def mda_add_default_transforms(self, universe, mol):
        """
        Add default transformations (unwrapping, centering) to the universe.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            The MDAnalysis universe.
        mol : MDAnalysis.AtomGroup
            The selected atom group used for centering.

        Returns
        -------
        MDAnalysis.Universe
            The modified universe object.
        """
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        universe.trajectory.add_transformations(*transforms)
        return universe
    
    def extract_topology_info(self):
        """
        Extract atomic numbers, elements, atom ids, and bonds from the MDAnalysis atom group.
        """
        self._extract_elements()
        self._extract_atomic_numbers()
        self._extract_atom_ids()
        self._extract_bonds()
        
        self.ce_log_dict("Extracted topology: ", {
            "Number of atoms": len(self.atns),
            "Number of bonds": len(self.bonds),
            "Element Counts": {
                a: self.at_elements.count(a) for a in set(self.at_elements)
            }
        })
            
    def _extract_elements(self):
        """
        Extract atomic elements from the MDAnalysis atom group.
        Falls back to `type_to_elements` if MDAnalysis cannot deduce them.
        """
        try:
            at_elements = [at.element for at in self.mol]
        except NoDataError:
            if self.type_to_elements is None:
                raise ValueError("Atom elements not found in trajectory. Please provide type_to_elements mapping.")
            at_types = self.mol.types
            at_elements = [self.type_to_elements[at] for at in at_types]
            # Add elements information back to the universe
            try:
                if hasattr(self.u.atoms, "types"):
                    all_elements = [self.type_to_elements.get(t, t) for t in self.u.atoms.types]
                    self.u.add_TopologyAttr("elements", all_elements)
            except Exception as e:
                self.log_warn(f"Could not add elements information to the universe: {e}",)
        self.at_elements = at_elements

    def _extract_atomic_numbers(self):
        """
        Get atomic numbers derived from elements.
        """
        from ase.data import atomic_numbers
        at_numbers = [atomic_numbers[el] for el in self.at_elements]
        self.atns = at_numbers
    
    def _extract_atom_ids(self):
        """
        Get 1-based atom IDs from the MDAnalysis atom group.
        """
        atm_ids = [int(at.id) + 1 for at in self.mol.atoms]
        self.atm_ids = atm_ids
    
    def _extract_bonds(self):
        """
        Get bonds from the MDAnalysis atom group and remap indices relative to the selection.
        """
        bonds = self.mol.get_connections('bonds', outside=False).indices
        for i in range(len(bonds)): # remap to mol atoms indices (without hydrogens)
            bonds[i] = (np.where(self.mol.atoms.indices == bonds[i][0])[0][0], np.where(self.mol.atoms.indices == bonds[i][1])[0][0])
        self.bonds = bonds