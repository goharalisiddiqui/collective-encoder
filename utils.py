import numpy as np

def compute_mfpt_matrix(vals : np.ndarray, minima : np.ndarray, lag: int = 1):
    '''
    Compute and save Mean First Passage Time (MFPT) matrix between the given minima.
    Step 1: Construct an MSM from the trajectory data (phi, psi) on state values defined by the given minima using PyEMMA.
    Step 2: Compute the MFPT matrix between the states defined by the minima.

    Parameters:
    vals (np.ndarray): Trajectory data of shape (n_samples, n_features).
    minima (np.ndarray): Array of minima of shape (n_states, n_features).
    lag (int): Lag time for the MSM.
    '''
    assert vals.shape[1] == minima.shape[1], "Dimensionality of vals and minima must match."
    assert minima.shape[0] >= 2, "At least two minima are required to compute MFPT."
    # Step 1: Assign trajectory points to states based on nearest minima
    traj_data = vals

    # Assign each point to the nearest minimum (state assignment)
    distances = np.linalg.norm(traj_data[:, None, :] - minima[None, :, :], axis=-1)
    state_assignments = np.argmin(distances, axis=-1)
    
    n_states = len(minima)
    lag_time = lag

    # Build transition count matrix
    transition_counts = np.zeros((n_states, n_states))
    
    for t in range(len(state_assignments) - lag_time):
        i = state_assignments[t]
        j = state_assignments[t + lag_time]
        transition_counts[i, j] += 1
    
    # Convert to transition probability matrix
    row_sums = transition_counts.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums[:, np.newaxis]
    
    # Step 2: Compute MFPT matrix
    # MFPT from state i to state j is computed by solving linear system
    mfpt_matrix = np.zeros((n_states, n_states))
    
    for j in range(n_states):
        # For each target state j, solve for mean hitting times
        # Set up system: (I - P + e_j e_j^T) * tau = 1
        # where e_j is unit vector for state j
        
        A = np.eye(n_states) - transition_matrix
        A[j, :] = 0  # Replace j-th row
        A[j, j] = 1  # Make it absorbing
        
        b = np.ones(n_states)
        b[j] = 0  # No time needed to reach j from j
        
        try:
            tau = np.linalg.solve(A, b)
            mfpt_matrix[:, j] = tau
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            tau = np.linalg.pinv(A) @ b
            mfpt_matrix[:, j] = tau
    
    # Set diagonal to 0 (no time to reach same state)
    np.fill_diagonal(mfpt_matrix, 0)
    
    # Handle infinite/very large values
    mfpt_matrix = np.where(np.isfinite(mfpt_matrix), mfpt_matrix, np.inf)
    
    return transition_counts, transition_matrix, mfpt_matrix

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}
    
    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def parse_slice(st):
    """
    Parse a string into a slice object.
    The string should be in the format "start:stop:step".
    """
    if st == "":
        return slice(None)
    if ":" not in st:
        return slice(int(st), None, None)
    parts = st.split(":")
    if len(parts) == 1:
        return slice(int(parts[0]), None, None)
    elif len(parts) == 2:
        return slice(int(parts[0]), int(parts[1]), None)
    elif len(parts) == 3:
        return slice(int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError(f"Invalid slice format: {st}")
    
    