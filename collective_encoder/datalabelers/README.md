# Data Labelers

This module provides tools for computing target "labels" or auxiliary physical properties for molecular structures and trajectories. These labels can be used for supervised learning, latent space evaluation, or physics-informed loss calculations (e.g., Structure Factors, Steinhardt order parameters, Dihedrals, Coordination numbers).

## Interface

All data labelers must inherit from `BaseLabeler` defined in `base.py`.

### `BaseLabeler`

`BaseLabeler` inherits from `CEModule` and defines the core label computation contract:
- `get_label_names(self) -> List[str]`: Returns the list of column names produced by the labeler.
- `compute(self, indices: Optional[List[int]] = None) -> np.ndarray`: Computes labels either for the current frame (`indices=None`, used by `FrameLabeler`) or for a batch of row/frame indices (`indices=[...]`, used by `BatchLabeler`).
- `reinitialize(self, **args) -> None`: Optional hook to reset state when the underlying trajectory/DataFrame changes.

Depending on how labels are computed, concrete labelers typically inherit from:
- `FrameLabeler` (`frame_base.py`): Per-frame computation on a positioned MDAnalysis `Universe`.
- `BatchLabeler` (`batch_base.py`): Batch index-based computation over tabular data or arrays.

## Registration & Resolution

Data labelers are registered under the `"datalabelers"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("datalabelers", BaseLabeler)`
- **Resolution**: `get_module("datalabelers", labeler_type)` from `collective_encoder.common.registry` (replacing the legacy `resolver.py`).
- **Adding a new labeler**: Subclass `BaseLabeler` (or `FrameLabeler` / `BatchLabeler`) and register it in `_BUILTINS` inside `datalabelers/__init__.py` or via:
  ```python
  from collective_encoder.common.registry import register
  register("datalabelers", "MY_LABELER", "my_package.labelers", "MyLabeler")
  ```

## Available Labelers

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `DUMMY` | `DummyLabeler` | No-op pass-through labeler returning zeros for unsupervised pipelines. |
| `COORDINATION` | `CoordinationCountLabeler` | Computes smooth atomic coordination numbers. |
| `DISTANCE` | `DistanceValueLabeler` | Computes pairwise distances between specified atom pairs. |
| `DIHEDRAL` | `DihedralValueLabeler` | Computes dihedral angles (and optional $\cos/\sin$ components) for quadruplets of atoms. |
| `COLUMN_SELECTOR` | `ColumnSelectorLabeler` | Extracts specified named columns from a tabular `pandas.DataFrame` (e.g., PLUMED output). |
| `STRUCTURE_FACTOR` | `StaticStructureFactorLabeler` | Computes static structure factors $S(q)$. |
| `DEBYE_STRUCTURE_FACTOR` | `DebyeStructureFactorLabeler` | Computes isotropic Debye scattering structure factors. |
| `STEINHARDT_ORDER_PARAMETER` | `SteinhardtOrderParameterLabeler` | Computes Steinhardt bond-orientational order parameters ($Q_l$, $W_l$). |
| `CONCAT` | `ConcatLabeler` | Concatenates the outputs of multiple labelers into a single label array. |
| `FROM_EXTERNAL` | `FromExternalLabeler` | Extracts pre-computed label arrays from external dataset mappings (e.g., `.npz` or Hugging Face datasets). |
