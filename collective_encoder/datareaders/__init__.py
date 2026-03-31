"""Data readers for processing raw molecular data files."""

from collective_encoder.datareaders.base import TrajectoryReaderBase
from collective_encoder.datareaders.xtc import XTCReader
from collective_encoder.datareaders.xtc_chunks import XTCChunksReader
from collective_encoder.datareaders.xtc_chunks_cg import XTCChunksCGReader
from collective_encoder.datareaders.resolver import get_datareader

__all__ = [
    "TrajectoryReaderBase",
    "XTCReader",
    "XTCChunksReader",
    "XTCChunksCGReader",
    "get_datareader"
]