import importlib

_REGISTRY: dict = {
    "MAE":     ("collective_encoder.metrics.mae",    "CEMetricMAE"),
    "MAE_DICT": ("collective_encoder.metrics.mae_dict", "CEMetricMAEDict"),
    "KLD":     ("collective_encoder.metrics.kld",    "CEMetricKLD"),
}


def get_metric_cls(metric_name: str):
    """Return the metric class for *metric_name*.

    Raises:
        ValueError: If *metric_name* is not registered.
    """
    if metric_name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric name: '{metric_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[metric_name]
    return getattr(importlib.import_module(module_path), class_name)