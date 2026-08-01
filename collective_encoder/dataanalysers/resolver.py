import importlib

_REGISTRY: dict = {
    "LABELS":     ("collective_encoder.dataanalysers.labels",    "LabelsAnalyser"),
    "DATAPOINTS":     ("collective_encoder.dataanalysers.datapoints",    "DatapointsAnalyser"),
    "GRAPH_DATAPOINTS": ("collective_encoder.dataanalysers.graphs", "GraphDatapointsAnalyser"),
    "LABELS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "LabelsDihedralAnalyser"),
    "DATAPOINTS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "DatapointsDihedralAnalyser"),
    "GRAPH_DATAPOINTS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "GraphDatapointsDihedralAnalyser"),
}


def get_dataanalyser(dataanalyser_name: str):
    """Return the data analyser class for *dataanalyser_name*.

    Raises:
        ValueError: If *dataanalyser_name* is not registered.
    """
    if dataanalyser_name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataanalyser name: '{dataanalyser_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[dataanalyser_name]
    return getattr(importlib.import_module(module_path), class_name)