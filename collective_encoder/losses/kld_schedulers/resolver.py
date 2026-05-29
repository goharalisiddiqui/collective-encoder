import importlib

_REGISTRY = {
        'Fixed': ('collective_encoder.losses.kld_schedulers.fixed', 'KLDFixedScheduler'),
        'Linear': ('collective_encoder.losses.kld_schedulers.linear', 'KLDLinearScheduler'),
        'Auto': ('collective_encoder.losses.kld_schedulers.auto', 'KLDAutoScheduler'),
    }

def KLDResolver(kld_max_type, kld_max_scheduler_args, **kwargs):
    
    if kld_max_type not in _REGISTRY:
        raise ValueError(f"Invalid kld_max_type: {kld_max_type}. "
                        f"Must be one of {list(_REGISTRY.keys())}")
    module_name, class_name = _REGISTRY[kld_max_type]
    module = importlib.import_module(module_name)
    kld_scheduler_class = getattr(module, class_name)
    return kld_scheduler_class(args=kld_max_scheduler_args, **kwargs)