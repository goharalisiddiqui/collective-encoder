import logging

import numpy as np

def transform_lv2std(args, vals):
    """
    Transforms log-variance values to standard deviation.

    Parameters
    ----------
    args : list of str
        List containing the label key for the log-variance.
    vals : dict of str to numpy.ndarray
        Dictionary of available data.

    Returns
    -------
    numpy.ndarray or None
        The computed standard deviation array.
    """
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
    """
    Applies registered transformations to dynamically generate new plotting variables.

    Parameters
    ----------
    tvals : dict of str to str, optional
        Mapping of new variable name to transformation string (e.g. ``'std': 'lv2std:logvar_latent'``).
    vals : dict of str to numpy.ndarray
        Dictionary of available data to augment.

    Returns
    -------
    dict
        The updated dictionary containing the new transformed variables.
    """
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