from abc import ABC, abstractmethod
from typing import List, Dict, Any

from collective_encoder.common.module import CEModule
from collective_encoder.datalabelers.base import BaseLabeler
from collective_encoder.common.registry import get_module

class BaseDataReader(CEModule, ABC):
    """
    Abstract base class for data readers.
    """
    _COMPATIBLE_LABELERS = ["CONCAT", "DUMMY"]

    _OPTIONAL_ARGS = {
        'processors': [],
        "non_readability_tolerance": 0.02,  # Allow up to 2% of frames to be unreadable before raising an error
    }
    
    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self._add_processors()
    
    def get_processors(self) -> List[Any]:
        """
        Get the list of initialized data processors.

        Returns
        -------
        list
            List of initialized data processor instances.
        """
        return self.processors
    
    def _add_processors(self):
        processors = []
        for config in self.processors:
            processor_cls = get_module("dataprocessors", config['type'])
            processor_args = config.get('args', {})
            processors.append(
                processor_cls(
                    args=processor_args, 
                    **self.get_run_args()
                    )
                )
            self.log_msg(f"Initialized processor: {config['type']} with args {processor_args}")
        self.processors = processors
    
    def _validate_traj_length(self, 
        indices: List[int], 
        read_indices: List[Any],
        failed_indices: List[int],
        idx: int = None,
        ):
        
        """Validate that trajectory has expected number of frames.

        Logs a WARNING with the exact failed frame indices when frames are dropped,
        so downstream analysis tools can correlate the shortened trajectory.
        """
        split = ['train', 'val', 'test', 'predict'][idx] if idx <= 4 else \
                    f'Sequence {idx}' if idx is not None \
                        else "Unknown"
        if failed_indices:
            self.warn(
                f"[{split}] {len(failed_indices)} frame(s) could not be read "
                f"(OSError) and were skipped. Failed frame indices: {failed_indices}"
            )
        threshold = len(indices) * (1 - self.non_readability_tolerance)
        if len(read_indices) < threshold:
            self.raise_error(
                f"[{split}] Trajectory length {len(read_indices)} is less than the minimum expected "
                f"{threshold:.0f} frames (tolerance={self.non_readability_tolerance:.1%}). "
                f"Failed indices: {failed_indices}",
                ValueError
            )

    @abstractmethod
    def read(self,
            labeler_type: str,
            labeler_args: dict,
            **kwargs):
        """
        Abstract method to read the trajectory data and compute labels.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        **kwargs
            Additional keyword arguments for reading the trajectory.
        """
        raise NotImplementedError("Subclasses must implement read method")

    
    def get_total_datapoints(self) -> int:
        """
        Get the total number of datapoints the datareader can generate.

        Returns
        -------
        int
            The total number of datapoints or records available.
        """
        total = self.get_total_frames()
        for processor in self.get_processors():
            total = processor.adjust_total_frames(total)
        return total

    @abstractmethod
    def get_total_frames(self):
        """
        Method to get the total number of frames in the trajectory.

        Returns
        -------
        int
            The total number of frames or records available.
        """
        raise NotImplementedError("Subclasses must implement get_total_frames method")
    
    def get_label_names(self) -> List[str]:
        """
        Get the names of the collective variables (labels).

        Returns
        -------
        list of str
            List of column names for the labels.

        Raises
        ------
        AttributeError
            If `label_list` has not been set (usually done during reading).
        """
        
        if not hasattr(self, 'label_list'):
            raise AttributeError(f"{type(self).__name__} does not have "
                                "'label_list' attribute. Ensure that the "
                                "trajectory has been read and labels have been "
                                "computed before calling get_label_names.")
        return self.label_list
    
    def initialize_labeler(self, labeler_type: str, labeler_args: dict) -> BaseLabeler:
        """
        Initialize the labeler based on the provided type and arguments.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        """
        if labeler_type is None:
            labeler_type = "DUMMY" 
        if labeler_type not in self._COMPATIBLE_LABELERS:
            raise ValueError(f"Labeler type '{labeler_type}' is not compatible with this data reader. "
                             f"Compatible labelers: {self._COMPATIBLE_LABELERS}")
        
        return self._initialize_labeler(labeler_type, labeler_args)
    
    @abstractmethod
    def _initialize_labeler(self, labeler_type: str, labeler_args: dict) -> BaseLabeler:
        """
        Internal method to initialize the labeler. Subclasses should implement this.

        Parameters
        ----------
        labeler_type : str
            The type of labeler to use for computing labels.
        labeler_args : dict
            Arguments to pass to the labeler for label computation.
        """
        raise NotImplementedError("Subclasses must implement _initialize_labeler method")