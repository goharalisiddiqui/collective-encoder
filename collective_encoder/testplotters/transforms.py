import logging

import numpy as np

def transform_lv2std(args, vals):
    if len(args) != 1:
        logging.exception(f"lv2std transformation requires exactly 1 argument: logvar. Found: {args}.")
        return None
    if args[0] not in vals:
        logging.exception(f"Log variance label '{args[0]}' not found in collected data for lv2std transformation. Available labels: {list(vals.keys())}.")
        return None
    logvar = vals[args[0]]
    if not isinstance(logvar, np.ndarray):
        logging.exception(f"Log variance label '{args[0]}' must be a numpy array for lv2std transformation. Found type: {type(logvar)}.")
        return None
    std = np.sqrt(np.exp(logvar))
    return std

def add_transformed(tvals, vals):
    if tvals is None:
        return vals
    for name, transform in tvals.items():
        func = transform.split(':')[0]
        args = transform.split(':')[1:]
        if func == 'lv2std':
            if name in vals:
                logging.warning(f"Transformed value '{name}' already exists in collected data. Overwriting with new transformation.")
                continue
            vals[name] = transform_lv2std(args, vals)
        else:
            logging.exception(f"Unknown transformation function '{func}' for transformed value '{name}'. Skipping transformation.")
    return vals