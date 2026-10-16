from typing import List
from collective_encoder.datareaders.processors.base import BaseProcessor

class ChunkProcessor(BaseProcessor):
    """
    Expands sequence start indices into frame ranges of a specified length.
    """
    def __init__(self, sequence_length: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Expand start indices into consecutive frame ranges of length sequence_length.
        """
        return [j for i in seq for j in range(i, i + self.sequence_length)]

    def adjust_total_frames(self, total_frames: int) -> int:
        """
        Adjust the total frames because a full sequence requires multiple frames.
        """
        return total_frames - self.sequence_length
