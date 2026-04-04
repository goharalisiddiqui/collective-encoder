"""Data readers for processing raw molecular data files."""

from collective_encoder.datareaders.trajectory import TrajectoryReaderBase
from collective_encoder.datareaders.xtc import XTCReader
from collective_encoder.datareaders.xtc_chunks import XTCChunksReader
from collective_encoder.datareaders.xtc_chunks_cg import XTCChunksCGReader
from collective_encoder.datareaders.plumed_output import PlumedOutputReader
from collective_encoder.datareaders.resolver import get_datareader

__all__ = [
    "TrajectoryReaderBase",
    "XTCReader",
    "XTCChunksReader",
    "XTCChunksCGReader",
    "PlumedOutputReader",
    "get_datareader"
]