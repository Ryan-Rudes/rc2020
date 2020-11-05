import numpy as np

def rand_prob(prob):
    """
    Returns:
        True, with probability prob. Otherwise, returns False
    """

    return np.random.random() < prob

def pos_to_coords(position, width):
    """Converts an integer positional representation into an (x, y) coordinate pair"""

    return position % width, position // width
