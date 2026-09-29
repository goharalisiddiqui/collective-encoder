import importlib

_REGISTRY: dict = {
    "MAE":     ("collective_encoder.metrics.mae",    "CEMetricMAE"),
    "MAE_DICT": ("collective_encoder.metrics.mae_dict", "CEMetricMAEDict"),
    "KLD":     ("collective_encoder.metrics.kld",    "CEMetricKLD"),
}


def get_metric_cls(metric_name: str):
    """
    Return the metric class mapped to a given string identifier.

    Parameters
    ----------
    metric_name : str
        The registered string identifier for the target metric.

    Returns
    -------
    type
        The corresponding metric class.

    Raises
    ------
    ValueError
        If `metric_name` is not found in the registry.
    """
    if metric_name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric name: '{metric_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[metric_name]
    return getattr(importlib.import_module(module_path), class_name)