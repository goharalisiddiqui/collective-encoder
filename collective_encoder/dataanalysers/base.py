from abc import ABC, abstractmethod
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from collective_encoder.common.module import CEModule

class BaseDataAnalyser(CEModule, ABC):
    def __init__(self, output_dir, data_args = {}):
        self.output_dir = output_dir
        self.data_args = data_args
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def write_data(self, data, label = ""):
        '''
        Abstract method to write data analysis results.
        '''
        raise NotImplementedError("Subclasses must implement write_data method")
        
