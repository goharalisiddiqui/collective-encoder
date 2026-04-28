from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datareaders.resolver import get_datareader


class ColvarsDataModule(CoordinatesDataModule):
    """PyTorch Lightning DataModule for collective variable data from PLUMED output files.

    Inherits the full train/val/test splitting, batching, and normalisation
    logic from :class:`~collective_encoder.datamodules.coordinates.CoordinatesDataModule`.
    The only customisation needed is to initialise the datareader without
    attempting to read coordinate-specific metadata (atomic numbers, bonds,
    element symbols) that PLUMED output files do not contain.

    Compatible pipeline:
        ``PLUMED_OUTPUT`` datareader → ``COLVAR`` dataset → any flat-input network
    """

    _IDENTIFIER = "COLVAR"
    _COMPATIBLE_DATAREADERS = ["PLUMED_OUTPUT"]
    _COMPATIBLE_DATASETS = ["COLVAR"]
    _COMPATIBLE_LABELERS = ["COLUMN_SELECTOR", "DUMMY"]

    def __init__(
        self,
        datareader_type: str,
        dataset_type: str,
        sequential: bool = True,
        **kwargs,
    ):
        super().__init__(
            datareader_type=datareader_type,
            dataset_type=dataset_type,
            sequential=sequential,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Override coordinate-specific initialisation
    # ------------------------------------------------------------------

    def _initialize_datareader(self) -> None:
        """Initialise the PLUMED datareader without reading atomic metadata."""
        datareader_cls = get_datareader(self.hparams.datareader_type)
        self.datareader = datareader_cls(**self.hparams.datareader_args)
        # COLVAR files have no atomic numbers, bonds, or element symbols.
        # These attributes are intentionally left unset; methods that access
        # them raise AttributeError as expected (see overrides below).

    # ------------------------------------------------------------------
    # Coordinate-specific methods — not applicable to COLVAR data
    # ------------------------------------------------------------------

    def get_atns(self):
        raise AttributeError("Atomic numbers are not applicable for COLVAR data.")

    def get_bond_indices(self):
        raise AttributeError("Bond indices are not applicable for COLVAR data.")

    def get_element_symbols(self):
        raise AttributeError("Element symbols are not applicable for COLVAR data.")

    def get_fake_systems(self):
        raise NotImplementedError("Fake systems are not implemented for COLVAR data.")
