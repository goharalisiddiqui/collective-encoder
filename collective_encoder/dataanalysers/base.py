from abc import ABC, abstractmethod
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from collective_encoder.common.module import CEModule

class BaseDataAnalyser(CEModule, ABC):
    '''
    Docstring for BaseDataAnalyser
    '''
    
    def __init__(self,
                 output_dir: str,
                 args: dict = None,
                 **kwargs):
        self.output_dir = output_dir
        if args is None:
            args = {}
        super().__init__(args=args, **kwargs)
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def write_data(self, data, label = ""):
        '''
        Abstract method to write data analysis results.
        '''
        raise NotImplementedError("Subclasses must implement write_data method")
        
