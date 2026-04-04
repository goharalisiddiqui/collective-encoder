def get_labeler(labeler_type: str):
    if labeler_type == 'Dummy' or labeler_type is None:
        from .dummy import DummyLabeler as labeler_class
    elif labeler_type == 'COORDINATION':
        from .coordination import CoordinationCountLabeler as labeler_class
    elif labeler_type == 'DISTANCE':
        from .distance import DistanceValueLabeler as labeler_class
    elif labeler_type == 'DIHEDRAL':
        from .dihedral import DihedralValueLabeler as labeler_class
    else:
        raise ValueError("Unknown datareader type: " + labeler_type)
    return labeler_class