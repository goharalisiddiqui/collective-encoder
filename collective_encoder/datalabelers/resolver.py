import importlib

_REGISTRY: dict = {
    "DUMMY":           (".dummy",            "DummyLabeler"),
    "COORDINATION":    (".coordination",     "CoordinationCountLabeler"),
    "DISTANCE":        (".distance",         "DistanceValueLabeler"),
    "DIHEDRAL":        (".dihedral",         "DihedralValueLabeler"),
    "COLUMN_SELECTOR": (".column_selector",  "ColumnSelectorLabeler"),
}

_PACKAGE = "collective_encoder.datalabelers"


def get_labeler(labeler_type: str):
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