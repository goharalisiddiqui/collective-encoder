import os
from multiprocessing import Pool

from glob import glob
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm

import numpy as np
import ase

import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.exceptions import NoDataError

import gslibs.validation as gsv
from .trajectory import TrajectoryReaderBase
from collective_encoder.datalabelers.resolver import get_labeler

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


class XTCReader(TrajectoryReaderBase):
    """Read GROMACS XTC/TPR trajectories via MDAnalysis.

    Loads a molecular topology (``.tpr``) together with one or more trajectory
    files (``.xtc``) and converts each requested frame into an ASE ``Atoms``
    object.  Labeling is performed per-frame using a
    :class:`~collective_encoder.datalabelers.base.FrameLabeler`.

    Supports optional multi-process reading: the trajectory is copied to
    independent worker processes (each with its own Universe) to parallelise
    I/O and label computation.

    Args:
        tprfile: Path to the GROMACS ``.tpr`` topology file.
        xtcfile: Path to a single ``.xtc`` trajectory file.
        xtcfiles: List of ``.xtc`` files to concatenate.
        coord_glob: Glob pattern matching one or more ``.xtc`` files.
        selection: MDAnalysis atom selection string (default: ``'all'``).
        type_to_elements: Optional mapping of atom types to element numbers
            when the topology lacks element information.
        parallel: Whether to use multiprocessing for reading (default: ``True``).
        **kwargs: Forwarded to :class:`~collective_encoder.datareaders.base.BaseDataReader`.

    Note:
        Exactly one of ``xtcfile``, ``xtcfiles``, or ``coord_glob`` must be
        provided.
    """
    _IDENTIFIER = "XTC"
    _REQUIRED_ARGS = ['tprfile']
    _OPTIONAL_ARGS = {
        'xtcfile': None,
        'xtcfiles': None,
        'coord_glob': None,
        'selection': "all",
        'type_to_elements': None,
        'parallel': True,
    }
    
    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        super().__init__(args=args, **kwargs)
        
        # Check files
        gsv.check_exists(tprfile=self.tprfile)
        self.log_msg(f"Loading topology from file {self.tprfile}")
        gsv.check_mutually_exclusive(xtcfile=self.xtcfile, 
                                      coord_glob=self.coord_glob, 
                                      xtcfiles=self.xtcfiles, 
                                      require_one=True)
        if self.coord_glob:
            self.log_msg(f"Loading trajectory files matching glob pattern: {self.coord_glob}") 
            xtcfiles = glob(self.coord_glob)
            if not xtcfiles:
                self.raise_error(f"No files found for pattern {self.coord_glob}")
            self.log_msg(f"Found {len(xtcfiles)} files") 
            u = mda.Universe(self.tprfile, *xtcfiles)
        elif self.xtcfiles:
            self.log_msg("Loading trajectory from multiple files: ")
            self.log_msg(f"- {('\n\t - '.join(self.xtcfiles))}")
            for xf in self.xtcfiles:
                if not os.path.exists(xf):
                    self.raise_error(f"File {xf} not found")
            u = mda.Universe(self.tprfile, *xtcfiles)
        else:
            self.log_msg(f"Loading trajectory from file: {self.xtcfile}") 
            if not os.path.exists(self.xtcfile):
                self.raise_error(f"File {self.xtcfile} not found")
            u = mda.Universe(self.tprfile, self.xtcfile)

        # Select the atoms
        self.mol = self.mda_select_atoms(u, self.selection)
        self.u = u
        self._selection = self.selection
        self.parallel = self.parallel
        
        # Extract the atomic numbers
        self.atns, self.at_elements = self.mda_get_atomic_numbers_and_elements(
                                                self.mol, self.type_to_elements)

        # Get the atom numbers in the trajectory
        self.atm_ids = self.mda_get_atom_ids(self.mol)
        
        # Extract the bonds information
        self.bonds = self.mda_get_bonds(self.mol)
    
    def read_trajectory(self, 
                        indices: List[List[int]],
                        labeler_type : str = 'Dummy',
                        labeler_args : Dict[str, Union[str, float, List[int]]] = {},
                        ) -> Tuple[Tuple[List[ase.Atoms]], Tuple[List[List[float]]]]:
    
        labeler_cls = get_labeler(labeler_type)
        labeler = labeler_cls(
            universe=self.u,
            args=labeler_args,
        )
        self.label_list = labeler.get_label_names()
        
        self.no_residues, self.no_resids, self.no_atomnames = False, False, False

        self.log_msg(f"Reading trajectories...")
        
        # Expand/transform indices for each seq (e.g. chunk expansion in subclasses)
        prepared_indices = [self._prepare_seq(seq) for seq in indices]
        
        trajs, labels, all_failed = (), (), ()
        for index_list in tqdm(prepared_indices,
                               position=0,
                               disable=not self.verbose,
                               leave=True,
                               desc="Processing sequences",
                               dynamic_ncols=True):

            if not self.parallel or len(index_list) < 10:  # Threshold for parallel processing
                # Apply transforms if not already applied
                args = (0, self.u.copy(), self._selection, self.atns, self.at_elements, index_list,
                        labeler_type, labeler_args, self.run_args)
                traj, label, failed = _read_and_label_parallel(args)
            else:
                # Distribute sequences across up to 8 workers using interleaved chunks
                # so each worker gets one Universe copy and processes its chunk sequentially.
                n_workers = min(16, os.cpu_count() or 1, max(1, len(index_list)))
                chunks = [index_list[i::n_workers] for i in range(n_workers)]
                chunks = [c for c in chunks]

                # Pack worker_id and verbose so imap can use a single-argument callable.
                # worker bars occupy positions 0..n_workers-1; outer bar sits below them.
                args = [
                    (i+1, self.u.copy(), self._selection, self.atns, self.at_elements,
                    chunk, labeler_type, labeler_args, self.run_args)
                    for i, chunk in enumerate(chunks)
                ]

                with Pool(processes=len(args)) as pool:
                    chunk_results = pool.map(_read_and_label_parallel, args)

                # Reassemble in original order (interleaved chunks → interleaved results)
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

            traj, label = self._postprocess_seq(traj, label)
            trajs      += (traj,)
            labels     += (label,)
            all_failed += (failed,)

        self.log_msg(f"Finished reading trajectories.")
        return trajs, labels, all_failed

    def _prepare_seq(self, seq):
        """Transform a seq of indices before passing to the parallel worker.
        Override in subclasses to expand or remap indices (e.g. chunk expansion).
        """
        return seq

    def _postprocess_seq(self, mol_traj, labels):
        """Post-process a single seq's (mol_traj, labels) after parallel reading.
        Override in subclasses to apply transformations such as coarse-graining.
        """
        return mol_traj, labels

    def get_total_frames(self):
        return len(self.u.trajectory)
    