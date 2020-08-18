
"""
This module contains the skew tent map function.

"""

def skewtent_onestep(value, threshold):
    """
    Computes a single step of iteration through the skew-tent map given an
    input (previous) value and a threshold. Returns the next value as output.
    This function is called by _iterate_skewtent for iterating repeatedly.

    Parameters
    ----------
    value : scalar, float64
        Input value to the skew-tent map.
    threshold : scalar, float64
        Threshold value of the skew-tent map.

    Returns
    -------
    Output value as float64 from the skew-tent map.
    Computed conditionally as follows:
        If value < threshold, then output is value / threshold
        Else, output is (1 - value)/(1 - threshold)

    """
    if value < threshold:
        return value / threshold
    return (1 - value) / (1 - threshold)
