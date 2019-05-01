import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from numpy.linalg import norm


def getReferencePoint(contour, fingers_indexes, center_of_mass):

    middle_index = fingers_indexes[2]
    middle_point = np.asarray(contour[middle_index])
    
    semi_contour = contour[fingers_indexes[0]:fingers_indexes[-1]]

    center_of_mass = np.asarray(center_of_mass)

    d_point_rect = [ np.abs(norm(np.cross(middle_point-center_of_mass, center_of_mass-p))/norm(middle_point-center_of_mass)) for p in semi_contour ]

    r_partial_index = np.argmin(d_point_rect)

    r_index = fingers_indexes[0] + r_partial_index
    r_point = contour[r_index]

    return r_point, r_index

    # coordinates of reference point are calculated as the point of intersection between the line
    # passing by center of mass and middle finger point and the end of the image (y)
