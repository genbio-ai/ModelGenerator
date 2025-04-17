

import numpy as np
from scipy.spatial.transform import Rotation


def angle_3p(a, b, c):
    """
    Calculate the angle between three points in a 2D space.

    Args:
        a (list or array-like): The coordinates of the first point.
        b (list or array-like): The coordinates of the second point.
        c (list or array-like): The coordinates of the third point.

    Returns:
        float: The angle in degrees (0, 180) between the vectors
               from point a to point b and point b to point c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    cos_theta = np.clip(dot_product / (norm_ab * norm_bc + 1e-4), -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees

