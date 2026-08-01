from abc import ABC, abstractmethod
import os

from collective_encoder.common.module import CEModule

class BaseDataAnalyser(CEModule, ABC):
    '''
    Base class for data analysers.
    Used to perform data analysis on datasets and write plots or other analysis results to disk.
    
    Subclasses should implement the `write_data` method to handle specific data analysis tasks.
    
    Args:
        output_dir: Directory where output plots will be written.
        args: Optional configuration dict.  Recognised keys:

            - ``input_chunk_length`` (int): Length of the input window in
              sequence models; used to colour-code time-series by split.
            - ``output_chunk_length`` (int): Length of the prediction window.
            - ``n_seq_per_sample`` (int): Number of output sequences per
              input sample (default: ``1``).
        **kwargs: Forwarded to :class:`BaseDataAnalyser` and
            :class:`~collective_encoder.common.module.CEModule`.
            
    '''
    _IDENTIFIER = "BASE_DATA_ANALYSER"
    _REQUIRED_ARGS = ['datamodule_args']
    
    def __init__(self,
                 args: dict = None,
                 **kwargs):
        super().__init__(args=args, **kwargs)
        output_dir = self.safe_create_dir(
          os.path.join(self.run_dir, 
                       f"data_analysis_{self._IDENTIFIER.lower()}")
        )
        self.output_dir = output_dir
        
        ds_type = self.datamodule_args.get('dataset_type', None)
        if ds_type not in self._COMPATIBLE_DATASET_TYPES:
            self.raise_error(f"Dataset type '{ds_type}' is not compatible."
                f"Expected one of: "
                f"{self._COMPATIBLE_DATASET_TYPES}")
        self.ds_type = ds_type

    @abstractmethod
    def write_data(self, data, label = ""):
        '''
        Abstract method to write data analysis results.
        '''
        raise NotImplementedError("Subclasses must implement write_data method")
        
