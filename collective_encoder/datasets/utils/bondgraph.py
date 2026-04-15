from typing import List, Tuple, Dict, Optional

import numpy as np
import ase

import torch
from torch import Tensor

from torch_geometric.data import Data

from rdkit import Chem

def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    '''
    Compute dihedral angle defined by four points p0-p1-p2-p3 in radians.
    
    Parameters
    ----------
    p0, p1, p2, p3 : np.ndarray
        3D coordinates of the four points defining the dihedral angle.
    
    Returns
    -------
    float
        Dihedral angle in radians in the range [-pi, pi].
    
    '''
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    # Normalize b1 so that projection works correctly
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.arctan2(y, x))

def bond_type_one_hot(kind: str) -> List[float]:
    '''
    Return one-hot encoding for bond type. Supported types: single, double, triple, aromatic, virtual.
    
    Parameters
    ----------
    kind : str
        Bond type as a string. Must be one of: "single", "double", "triple", "aromatic", "virtual". 
    
    Returns
    -------
    List[float]
        One-hot encoded list of length 5 corresponding to the bond type.
    '''
    # single, double, triple, aromatic, virtual
    mapping = {"single": 0, "double": 1, "triple": 2, "aromatic": 3, "virtual": 4}
    vec = [0.0] * 5
    vec[mapping[kind]] = 1.0
    return vec

def bond_node_features(atoms: ase.Atoms, bonds: List[Tuple[int, int, float]]) -> Tensor:
    '''
    Compute node features for bond graph. Each node corresponds to a bond (i,j) and has features:
    [Z_i, Z_j, bond_length]
    
    Parameters    
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing the structure.
    bonds : List[Tuple[int, int, float]]
        List of bonds defined as tuples (i, j, distance) where i,j are atom indices and distance is the bond length.    
        
    Returns
    -------
    Tensor
        Node feature tensor of shape (num_bonds, 3) where each row is [Z_i, Z_j, bond_length].
    '''
    Z = atoms.get_atomic_numbers()
    feats = []
    for i, j, d in bonds:
        Zi, Zj = Z[i], Z[j]
        # mi = atomic_masses[Zi]
        # mj = atomic_masses[Zj]
        feats.append([Zi, Zj, d])
    return torch.tensor(feats)

def angle_and_torsion_edges(atoms: ase.Atoms, bonds: List[Tuple[int, int, float]]):
    '''
    Compute angle and torsion edges for bond graph. Edges connect bonds that share an atom (angle edges) or are separated by one bond (torsion edges).
    Edge features:
    - For angle edges: [1.0, 0.0, angle_value] where angle_value is the bond angle in radians.
    - For torsion edges: [0.0, 1.0, torsion_value] where torsion_value is the dihedral angle in radians.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing the structure.
    bonds : List[Tuple[int, int, float]]
        List of bonds defined as tuples (i, j, distance) where i,j are atom indices and distance is the bond length.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        Edge index tensor, edge attribute tensor, and atom indices for angle/torsion edges.
    '''
    # Map bond (i,j) with i<j to bond index
    bond_index_map: Dict[Tuple[int, int], int] = {}
    for idx, (i, j, _) in enumerate(bonds):
        bond_index_map[(i, j)] = idx
        bond_index_map[(j, i)] = idx  # allow reverse lookup

    pos = atoms.get_positions()
    # Build adjacency: atom -> bonds containing it
    # atom_to_bonds[atom_index] = list of bond indices
    atom_to_bonds: Dict[int, List[int]] = {}
    for b_idx, (i, j, _) in enumerate(bonds):
        atom_to_bonds.setdefault(i, []).append(b_idx)
        atom_to_bonds.setdefault(j, []).append(b_idx)#

    edge_index_src = []
    edge_index_dst = []
    edge_attr = []
    angle_atoms = []
    torsion_atoms = []

    # Angle edges (adjacent bonds sharing one atom)
    for shared_atom, bond_list in atom_to_bonds.items():
        m = len(bond_list)
        if m < 2: # need at least two bonds to form an angle
            continue
        # For every unordered pair of bonds sharing this atom
        for a_i in range(m):
            bi = bond_list[a_i]
            ai1, ai2, _ = bonds[bi]
            other_i = ai1 if ai2 == shared_atom else ai2 
            for a_j in range(a_i + 1, m):
                bj = bond_list[a_j]
                aj1, aj2, _ = bonds[bj]
                other_j = aj1 if aj2 == shared_atom else aj2
                # Compute angle at shared_atom: other_i - shared - other_j
                v1 = pos[other_i] - pos[shared_atom]
                v2 = pos[other_j] - pos[shared_atom]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 == 0 or n2 == 0:
                    continue
                cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angle = float(np.arccos(cosang))
                # Add both directions
                for s, t in [(bi, bj), (bj, bi)]:
                    edge_index_src.append(s)
                    edge_index_dst.append(t)
                    edge_attr.append([1.0, 0.0, angle])
                    angle_atoms.append([other_i, shared_atom, other_j])

    # Torsion edges (bonds separated by exactly one bond: i-j and k-l with j-k bond existing)
    # For torsion: for bond path (i,j)-(j,k)-(k,l) create edge between (i,j) and (k,l)
    # We'll iterate over central bond (j,k) and pairs of incident bonds on each side.
    # Build atom adjacency for quick bond existence test
    bonded_neighbors: Dict[int, List[int]] = {a: [] for a in range(len(atoms))}
    for i, j, _ in bonds:
        bonded_neighbors[i].append(j)
        bonded_neighbors[j].append(i)
    # Iterate central bond
    for (j, k, _) in bonds:
        lefts = [i for i in bonded_neighbors[j] if i != k]
        rights = [l for l in bonded_neighbors[k] if l != j]
        if not lefts or not rights:
            continue
        for i in lefts:
            # ensure bond (i,j) exists
            if (i, j) not in bond_index_map:
                continue
            b_left = bond_index_map[(i, j)]
            for l in rights:
                if l in (i, j, k):
                    continue
                if (k, l) not in bond_index_map:
                    continue
                b_right = bond_index_map[(k, l)]
                # dihedral i-j-k-l
                angle = dihedral(pos[i], pos[j], pos[k], pos[l])
                # angle = 0.5 + math.cos(angle - 1.25) # Following https://doi.org/10.1073/pnas.1600917113
                for s, t in [(b_left, b_right), (b_right, b_left)]:
                    edge_index_src.append(s)
                    edge_index_dst.append(t)
                    edge_attr.append([0.0, 1.0, angle])
                    torsion_atoms.append([i, j, k, l])

    if len(edge_index_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, 3))
    else:
        edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr)

    return edge_index, edge_attr_t, bond_index_map, angle_atoms, torsion_atoms

def assemble_bonds(atoms: ase.Atoms, bond_indices: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    """Compute bond lengths for the predefined bond index list."""
    pos = atoms.get_positions()
    bonds: List[Tuple[int, int, float]] = []
    for (i, j) in bond_indices:
        dist = float(np.linalg.norm(pos[i] - pos[j]))
        bonds.append((i, j, dist))
    return bonds

def build_template_graph(
        atomic_numbers: List[int],
        bond_indices: List[Tuple[int, int]],
        k: int = 3,
        residue_names: Optional[List[str]] = None,
        atom_names: Optional[List[str]] = None,
        bond_lengths: Optional[List[float]] = None,
        bond_types: Optional[List[str]] = None,
    ) -> Data:
        """Return an atom-level template molecular graph constant across all frames.

        Node features (concatenated):
            - Atomic number (1 scalar)
            - Chirality one-hot (RDKit chiral tags) or zero if RDKit unavailable
            - Degree (1 scalar)
            - Ring count (# rings atom participates in, scalar; RDKit else 0)
            - Implicit valence (scalar; RDKit else 0)
            - Formal charge (scalar; RDKit else 0)
            - Total bonded hydrogens (scalar; RDKit else 0)
            - Hybridization one-hot (RDKit enum; else zeros)
            - Aromatic flag (1 scalar)
            - In 5-ring flag (1 scalar)
            - In 6-ring flag (1 scalar)
            - Residue name one-hot (if provided)
            - Atom name one-hot (if provided)

        Edge features (concatenated):
            - Bond type one-hot: [single, double, triple, aromatic, virtual]
            - Covalent bond length (1 scalar, Å)

        Additional edges: for each pair of atoms with shortest path length <= k (k-hop neighbors), add edge if not direct bond (marked as virtual).

        Parameters
        ----------
        k : int
            Hop distance for adding virtual edges (>=1). k=1 => only direct bonds.
        residue_names, atom_names : Optional[List[str]]
            Lists of strings length n_atoms giving residue & atom names used for one-hot encoding.

        Returns
        -------
        Data
            PyG Data object with atom-level features.
        """
        Z = atomic_numbers
        n = len(Z)

        # Build base adjacency from provided bond indices
        bond_set = set()
        for (i, j) in bond_indices:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            bond_set.add((a, b))

        ring_count = [0] * n
        chiral_tags = [0] * n
        implicit_valence = [0] * n
        formal_charge = [0] * n
        num_hs = [0] * n
        aromatic = [0] * n
        in5 = [0] * n
        in6 = [0] * n
        hybridization_list = [0] * n

        chiral_enum = []  # collect present tags
        hybrid_enum = []

        # Build RDKit molecule and add bonds to it
        mol = Chem.RWMol()
        idx_rd_map = [0] * n  # map from input atom index to RDKit atom index
        for i, z in enumerate(Z):
            a = Chem.Atom(int(z))
            rd_idx = mol.AddAtom(a)
            idx_rd_map[i] = rd_idx
        # Add bonds (assume single; attempt aromatic detection later)
        for (i, j) in bond_set:
            try:
                mol.AddBond(idx_rd_map[i], idx_rd_map[j], Chem.BondType.SINGLE)
            except Exception:
                pass

        rmol = mol.GetMol()
        Chem.SanitizeMol(rmol, catchErrors=True)
        Chem.AssignAtomChiralTagsFromStructure(rmol)
        
        # Analyze rings for ring count and in5/in6 flags
        for ring in rmol.GetRingInfo().AtomRings():
            size = len(ring)
            for idx in ring:
                ring_count[idx] += 1
                if size == 5:
                    in5[idx] = 1
                if size == 6:
                    in6[idx] = 1

        # Extract atom features from RDKits
        for i in range(n):
            a = rmol.GetAtomWithIdx(idx_rd_map[i])
            ct = int(a.GetChiralTag())
            chiral_tags[i] = ct
            hv = int(a.GetHybridization())
            hybridization_list[i] = hv
            implicit_valence[i] = a.GetValence(Chem.ValenceType.IMPLICIT)
            formal_charge[i] = a.GetFormalCharge()
            num_hs[i] = a.GetTotalNumHs()
            aromatic[i] = 1 if a.GetIsAromatic() else 0
            chiral_enum.append(ct)
            hybrid_enum.append(hv)
        chiral_enum = sorted(set(chiral_enum)) or [0]
        hybrid_enum = sorted(set(hybrid_enum)) or [0]

        chiral_index = {c: i for i, c in enumerate(chiral_enum)}
        hybrid_index = {h: i for i, h in enumerate(hybrid_enum)}

        # Residue name one-hot encodings
        if residue_names is not None:
            assert len(residue_names) == n, "residue_names length mismatch"
            uniq_res = sorted(set(residue_names))
            res_index = {r: i for i, r in enumerate(uniq_res)}
        else:
            uniq_res = []
            res_index = {}

        # Atom name one-hot encodings
        if atom_names is not None:
            assert len(atom_names) == n, "atom_names length mismatch"
            uniq_atoms = sorted(set(atom_names))
            atom_index = {r: i for i, r in enumerate(uniq_atoms)}
        else:
            uniq_atoms = []
            atom_index = {}

        chiral_dim = len(chiral_enum)
        hybrid_dim = len(hybrid_enum)
        res_dim = len(uniq_res)
        atomname_dim = len(uniq_atoms)

        # Build node feature matrix
        node_features = []
        for i in range(n):
            vec: List[float] = []
            # Atomic number
            vec.append(float(Z[i]))
            # Chirality one-hot
            chi_one = [0.0] * chiral_dim
            chi_one[chiral_index[chiral_tags[i]]] = 1.0
            vec.extend(chi_one)
            # Degree (graph degree from bond_set)
            deg = sum(1 for a, b in bond_set if a == i or b == i)
            vec.append(float(deg))
            # Ring count
            vec.append(float(ring_count[i]))
            # Implicit valence
            vec.append(float(implicit_valence[i]))
            # Formal charge
            vec.append(float(formal_charge[i]))
            # Number of bonded hydrogens
            vec.append(float(num_hs[i]))
            # Hybridization one-hot
            hy_one = [0.0] * hybrid_dim
            hy_one[hybrid_index[hybridization_list[i]]] = 1.0
            vec.extend(hy_one)
            # Aromatic flag
            vec.append(float(aromatic[i]))
            # In 5 / 6 ring flags
            vec.append(float(in5[i]))
            vec.append(float(in6[i]))
            # Residue name one-hot
            if res_dim:
                res_one = [0.0] * res_dim
                res_one[res_index[residue_names[i]]] = 1.0
                vec.extend(res_one)
            # Atom name one-hot
            if atomname_dim:
                at_one = [0.0] * atomname_dim
                at_one[atom_index[atom_names[i]]] = 1.0
                vec.extend(at_one)
            node_features.append(vec)
        x = torch.tensor(node_features)

        edge_src = []
        edge_dst = []
        edge_attr_list: List[Tensor] = []

        if bond_lengths is not None:
            assert len(bond_lengths) == len(bond_indices), "bond_lengths length mismatch"
            bond_len_map = {(min(i, j), max(i, j)): d for (i, j), d in zip(bond_indices, bond_lengths)}
            bond_lengths_fn = lambda i, j: bond_len_map.get((min(i, j), max(i, j)), 0.0)
        else:
            bond_lengths_fn = lambda i, j: float(Chem.GetPeriodicTable().GetRcovalent(int(Z[i])) \
                + Chem.GetPeriodicTable().GetRcovalent(int(Z[j])))
        
        if bond_types is not None:
            assert len(bond_types) == len(bond_indices), "bond_types length mismatch"
            bond_type_map = {(min(i, j), max(i, j)): t for (i, j), t in zip(bond_indices, bond_types)}
            bond_type_fn = lambda i, j: bond_type_map.get((min(i, j), max(i, j)), "single")
        else:
            bond_type_fn = lambda i, j: "single"

        # Add direct bond edges (for distance, we use sum of covalent radii)
        for (i, j) in bond_set:
            dist = bond_lengths_fn(i, j)
            bt = torch.tensor(bond_type_one_hot(bond_type_fn(i, j)))
            feat = torch.cat([bt, torch.tensor([dist])], dim=0)
            for s, t in ((i, j), (j, i)):
                edge_src.append(s)
                edge_dst.append(t)
                edge_attr_list.append(feat)

        # Compute shortest-path distances up to k using BFS from each node
        if k > 1:
            adjacency = {i: set() for i in range(n)}
            for (i, j) in bond_set:
                adjacency[i].add(j)
                adjacency[j].add(i)
            for start in range(n):
                visited = {start: 0}
                frontier = [start]
                while frontier:
                    new_frontier = []
                    for u in frontier:
                        for v in adjacency[u]:
                            if v not in visited:
                                visited[v] = visited[u] + 1
                                if visited[v] < k:
                                    new_frontier.append(v)
                    frontier = new_frontier
                for target, hops in visited.items():
                    if target == start or hops == 0:
                        continue
                    a, b = (start, target)
                    if (min(a, b), max(a, b)) in bond_set:
                        continue  # already direct edge
                    # add virtual edge both directions
                    dist = bond_lengths_fn(a, b)
                    bt = torch.tensor(bond_type_one_hot("virtual"))
                    feat = torch.cat([bt, torch.tensor([dist])], dim=0)
                    edge_src.append(a)
                    edge_dst.append(b)
                    edge_attr_list.append(feat)
                    edge_src.append(b)
                    edge_dst.append(a)
                    edge_attr_list.append(feat)

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.stack(edge_attr_list, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 5 + 1))

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            template=True,
            k_hop=k,
        )
        return data
