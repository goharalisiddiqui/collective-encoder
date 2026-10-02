import importlib

_REGISTRY: dict = {
    "DUMMY":           (".dummy",            "DummyLabeler"),
    "COORDINATION":    (".coordination",     "CoordinationCountLabeler"),
    "DISTANCE":        (".distance",         "DistanceValueLabeler"),
    "DIHEDRAL":        (".dihedral",         "DihedralValueLabeler"),
    "COLUMN_SELECTOR":         (".column_selector",  "ColumnSelectorLabeler"),
    "STRUCTURE_FACTOR":        (".structure_factor", "StaticStructureFactorLabeler"),
    "DEBYE_STRUCTURE_FACTOR":  (".debye",            "DebyeStructureFactorLabeler"),
    "STEINHARDT_ORDER_PARAMETER": (".steinhardt",    "SteinhardtOrderParameterLabeler"),
    "CONCAT":                  (".concat",           "ConcatLabeler"),
}

_PACKAGE = "collective_encoder.datalabelers"


def get_labeler(labeler_type: str):
    """
    Return the labeler class mapped to a given string identifier.

    Parameters
    ----------
    labeler_type : str
        The registered string identifier for the target labeler. If None, defaults to "DUMMY".

    Returns
    -------
    type
        The corresponding labeler class.

    Raises
    ------
    ValueError
        If `labeler_type` is not found in the registry.
    """
    if labeler_type is None:
        labeler_type = "DUMMY"
    if labeler_type not in _REGISTRY:
        raise ValueError(
            f"Unknown labeler type: '{labeler_type}'. "
            f"Available: {sorted(set(_REGISTRY) - {'Dummy'})}"
        )
    module_rel, class_name = _REGISTRY[labeler_type]
    module = importlib.import_module(module_rel, package=_PACKAGE)
    return getattr(module, class_name)