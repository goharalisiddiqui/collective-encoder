from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datareaders.resolver import get_datareader

class ColvarsDataModule(CoordinatesDataModule):
    """
    PyTorch Lightning DataModule for COLVAR data.
    Handles loading and processing of collective variable data from COLVAR files.

    Args:
        
    """

    # Compatible datareaders and datasets for COLVAR data
    _IDENTIFIER = "COLVAR"
    _COMPATIBLE_DATAREADERS = ["PLUMEDCOLVARFILE"]
    _COMPATIBLE_DATASETS = ["COLVAR"]
    _COMPATIBLE_LABELERS = ["COLUMN_SELECTOR"]
    
    def __init__(self, 
                 datareader_type: str,
                 dataset_type: str,
                 sequential: bool = True,
                 **kwargs,
                 ):
        super().__init__(
            datareader_type=datareader_type,
            dataset_type=dataset_type,
            sequential=sequential,
            **kwargs
        )

    def _read_data(self):
        # Initialize the trajectory reader
        datareader_cls = get_datareader(self.hparams.datareader_type)
        self.datareader = datareader_cls(**self.hparams.datareader_args)

    # Coordinate-specific methods
    def get_atns(self):
        """Get atomic numbers."""
        raise AttributeError("Atomic numbers are not applicable for COLVAR data.")

    def get_bond_indices(self):
        """Get bond indices."""
        raise AttributeError("Bond indices are not applicable for COLVAR data.")

    def get_element_symbols(self):
        """Get element symbols."""
        raise AttributeError("Element symbols are not applicable for COLVAR data.")
    
    def get_fake_systems(self):
        """Get fake systems for testing purposes."""
        #FIXME: Implement fake systems for COLVAR data if needed
        raise NotImplementedError("Fake systems are not implemented for COLVAR data.")