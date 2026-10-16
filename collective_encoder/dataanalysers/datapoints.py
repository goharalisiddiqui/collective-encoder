from .labels import LabelsAnalyser

class DatapointsAnalyser(LabelsAnalyser):
    """
    Data analyser for extracting and plotting raw datapoints.

    This analyser fetches raw features (like node features from graphs
    or coordinates from sequence data) instead of explicit target labels
    and passes them to the plotting routines.
    """

    _IDENTIFIER = "DATAPOINTS"

    def _get_label(self, datapoint):
        """
        Extract the input features from a datapoint instead of labels.

        Parameters
        ----------
        datapoint : Any
            The input data point.

        Returns
        -------
        Any
            The feature representation (e.g., node features 'x' for graphs,
            or sequence inputs).
        """
        if self.ds_type == "GRAPH":
            return datapoint.x
        elif self.ds_type == "DISTANCES":
            return datapoint[0]
        else:
            self.raise_error(f"Unknown dataset type '{self.ds_type}'. ")
