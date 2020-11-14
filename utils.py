import numpy as np

def rand_prob(prob):
    """
    Returns:
        True, with probability prob. Otherwise, returns False
    """

    return np.random.random() < prob
