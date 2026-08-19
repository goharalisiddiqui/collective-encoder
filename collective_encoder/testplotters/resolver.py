import importlib

_REGISTRY: dict = {
    "SimplePlotter": ("collective_encoder.testplotters.simple", "SimplePlotter"),
    "ALA2plotter": ("collective_encoder.testplotters.ala2", "ALA2plotter"),
    "DisentanglementBetaMetric": ("collective_encoder.testplotters.disentanglement_metrics.beta", "DisentanglementBetaMetric"),
    "DisentanglementFactorMetric": ("collective_encoder.testplotters.disentanglement_metrics.factor", "DisentanglementFactorMetric"),
    "DisentanglementDCIMetric": ("collective_encoder.testplotters.disentanglement_metrics.dci", "DisentanglementDCIMetric"),
    "DisentanglementMIGMetric": ("collective_encoder.testplotters.disentanglement_metrics.mig", "DisentanglementMIGMetric"),
    "DisentanglementModularityMetric": ("collective_encoder.testplotters.disentanglement_metrics.modularity", "DisentanglementModularityMetric"),
    "DisentanglementSAPMetric": ("collective_encoder.testplotters.disentanglement_metrics.sap", "DisentanglementSAPMetric"),
    "LatentCorrelationsPlotter": ("collective_encoder.testplotters.latent_correlations", "LatentCorrelationsPlotter"),
    "DisentanglementPlotter": ("collective_encoder.testplotters.disentanglement", "DisentanglementPlotter"),
}

def get_testplotter(model_name: str):
    """Return the neural network class for *model_name*.

    Raises:
        ValueError: If *model_name* is not registered.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[model_name]
    return getattr(importlib.import_module(module_path), class_name)
