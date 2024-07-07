import numpy as np

def rayleigh_vector(phases):
    """
    Calculate the normalized Rayleigh vector magnitude and direction for a set of phase angles.
    Formula:
    R = sqrt((1/n) * (sum(sin(phases))**2 + sum(cos(phases))**2))

    :param phases: NumPy array of phase angles in radians.
    :return: A tuple containing the normalized Rayleigh vector magnitude and direction.
    """
    n = len(phases)
    sum_sin = np.nansum(np.sin(phases))
    sum_cos = np.nansum(np.cos(phases))

    R = np.sqrt(sum_sin**2 + sum_cos**2) / n
    mean_angle = np.arctan2(sum_sin, sum_cos)

    return R, mean_angle
