import pandas as pd
import numpy as np
from typing import Dict, List, Union

from .base import BatchLabeler


class ColumnSelectorLabeler(BatchLabeler):
    """Extract specific columns from a PLUMED/tabular DataFrame as labels.

    Selects one or more named columns from a ``pd.DataFrame`` and returns
    the corresponding rows as a NumPy array.  Intended for use with
    :class:`~collective_encoder.datareaders.plumed_output.PlumedOutputReader`.

    Args:
        dataframe: The full collective-variable DataFrame loaded from the
            PLUMED output file.
        args: Configuration dict with key ``columns`` (required) — a list of
            column names to use as labels (e.g. ``['phi', 'psi']``).
        **kwargs: Additional keyword arguments forwarded to the base class.

    Raises:
        ValueError: If any column in ``columns`` is not present in ``dataframe``.
    """

    _IDENTIFIER = "COLUMN_SELECTOR"
    _REQUIRED_ARGS = ["columns"]
    _OPTIONAL_ARGS = {}

    def __init__(self,
                 dataframe: pd.DataFrame,
                 args: Dict[str, Union[str, List[str]]],
                 **kwargs,
                 ) -> None:
        super().__init__(args=args, **kwargs)

        self.dataframe = dataframe
        if len(self.columns) == 0:
            self.raise_error("'columns' must be provided in args")
        df_columns = set(dataframe.columns)
        missing_columns = [col for col in self.columns if col not in df_columns]
        if missing_columns:
            self.raise_error(f"The following specified "
                             f"columns are not present in the "
                             f"dataframe: {missing_columns}")
        
    def get_label_names(self) -> List[str]:
        return self.columns

    def compute(self, indices: List[int]) -> np.ndarray:
        # Select the specified columns and the given indices
        selected_data = self.dataframe.loc[indices, self.columns]
        return selected_data.values  # Return as numpy array
        
