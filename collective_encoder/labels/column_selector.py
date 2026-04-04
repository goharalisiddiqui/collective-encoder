import pandas as pd
import numpy as np
from typing import Dict, List, Union

from .base import BaseLabeler

class ColumnSelectorLabeler(BaseLabeler):
    '''
    Docstring for ColumnSelectorLabeler
    '''
    _IDENTIFIER = "COLUMN_SELECTOR"
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 columns: List[str],
                 args: Dict[str, Union[str, List[str]]]
                 ) -> None:
        self.dataframe = dataframe
        self.columns = columns
        
        df_columns = set(dataframe.columns)
        missing_columns = [col for col in columns if col not in df_columns]
        if missing_columns:
            raise ValueError(f"[{type(self).__name__}] The following specified "
                             f"columns are not present in the "
                             f"dataframe: {missing_columns}")
        
    def get_names(self) -> List[str]:
        return self.columns

    def compute(self, indices: List[int]) -> np.ndarray:
        # Select the specified columns and the given indices
        selected_data = self.dataframe.loc[indices, self.columns]
        return selected_data.values  # Return as numpy array
        
