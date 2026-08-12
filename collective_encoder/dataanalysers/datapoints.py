from .labels import LabelsAnalyser

class DatapointsAnalyser(LabelsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles.
    """

    _IDENTIFIER = "DATAPOINTS"

    def _get_label(self, datapoint):
        if self.ds_type == "GRAPH":
            return datapoint.x
        elif self.ds_type == "DISTANCES":
            return datapoint[0]
        else:
            self.raise_error(f"Unknown dataset type '{self.ds_type}'. ")
