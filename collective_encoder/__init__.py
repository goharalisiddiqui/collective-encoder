"""
Collective Encoder: A machine learning framework for molecular dynamics.

This package provides autoencoder-based architectures for creating surrogate models
that predict dynamics of molecular systems as time series data.
"""

__version__ = "0.1.0"

# Lazy imports to avoid heavy dependencies during package import
def __getattr__(name):
    if name == "trainer":
        from collective_encoder.trainer import train
        return train
    elif name == "nets":
        import collective_encoder.nets as nets
        return nets
    elif name == "dataloaders":
        import collective_encoder.datamodules as datamodules
        return datamodules
    elif name == "datasets":
        import collective_encoder.datasets as datasets
        return datasets
    elif name == "datareaders":
        import collective_encoder.datareaders as datareaders
        return datareaders
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "trainer",
    "nets",
    "datamodules",
    "datasets",
    "datareaders",
]