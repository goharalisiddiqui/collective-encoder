from typing import Any, Dict

from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.common.registry import get_module


class ColvarsDataModule(CoordinatesDataModule):
    """
    PyTorch Lightning DataModule for collective variable data from PLUMED output files.

    Inherits the full train/val/test splitting, batching, and normalisation
    logic from :class:`~collective_encoder.datamodules.coordinates.CoordinatesDataModule`.
    The only customisation needed is to initialise the datareader without
    attempting to read coordinate-specific metadata (atomic numbers, bonds,
    element symbols) that PLUMED output files do not contain.

    Compatible pipeline:
        ``PLUMED_OUTPUT`` datareader → ``COLVAR`` dataset → any flat-input network

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary for ``ColvarsDataModule``.
    kwargs : dict
        Additional arguments forwarded to ``CoordinatesDataModule``.
    """

    _IDENTIFIER = "COLVAR"
    _COMPATIBLE_DATAREADERS = ["PLUMED_OUTPUT"]
    _COMPATIBLE_DATASETS = ["COLVAR"]
    _COMPATIBLE_LABELERS = ["COLUMN_SELECTOR", "DUMMY", "CONCAT"]
    _REQUIRED_ARGS = CoordinatesDataModule._REQUIRED_ARGS.copy()
    _OPTIONAL_ARGS = CoordinatesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "sequential": True,
    })

    def __init__(
        self,
        args: Dict[str, Any] = None,
        **kwargs,
    ):
        merged_args = dict(args) if args is not None else {}
        known_keys = set(self._REQUIRED_ARGS) | set(self._OPTIONAL_ARGS.keys())
        for k in list(kwargs.keys()):
            if k in known_keys:
                merged_args.setdefault(k, kwargs.pop(k))
        super().__init__(
            args=merged_args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Override coordinate-specific initialisation
    # ------------------------------------------------------------------

    def _initialize_datareader(self) -> None:
        """
        Initialise the PLUMED datareader without reading atomic metadata.
        """
        datareader_cls = get_module("datareaders", self.datareader_type)
        self.datareader = datareader_cls(args=self.datareader_args, **self.run_args)
        # COLVAR files have no atomic numbers, bonds, or element symbols.
        # These attributes are intentionally left unset; methods that access
        # them raise AttributeError as expected (see overrides below).

    # ------------------------------------------------------------------
    # Coordinate-specific methods — not applicable to COLVAR data
    # ------------------------------------------------------------------

    def get_atns(self):
        """
        Raise an error as atomic numbers are not applicable.
        """
        raise AttributeError("Atomic numbers are not applicable for COLVAR data.")

    def get_bond_indices(self):
        """
        Raise an error as bond indices are not applicable.
        """
        raise AttributeError("Bond indices are not applicable for COLVAR data.")

    def get_element_symbols(self):
        """
        Raise an error as element symbols are not applicable.
        """
        raise AttributeError("Element symbols are not applicable for COLVAR data.")

    def get_fake_systems(self):
        """
        Raise an error as fake systems are not implemented.
        """
        raise NotImplementedError("Fake systems are not implemented for COLVAR data.")
