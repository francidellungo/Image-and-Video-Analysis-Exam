import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from numpy.linalg import norm
from extractionShapeFeatures import *
from Exercise2.findFingerLAB getComplementaryValleyPoints, getMediumFingerPoint

def getReferencePoint(contour, fingers_indexes, center_of_mass):
    """
        The coordinates of reference point (xr, yr) are calculated as the point 
        of intersection between the line passing by the center of mass and middle finger point
        and the image contour in the opposite side.
    """

    middle_index = fingers_indexes[2]
    # print(middle_index)
    middle_point = contour[middle_index]
    
    semi_contour = contour[fingers_indexes[0]:fingers_indexes[-1]]

    center_of_mass = np.asarray(center_of_mass)

    # point to rect distance between middle point - centroid rect and all points of semi contour
    d_point_rect = [ np.abs(norm(np.cross(middle_point - center_of_mass, center_of_mass - p))/norm(middle_point - center_of_mass)) for p in semi_contour ]

    # get index of point in semicontour that has minimal point to middle point - centroid rect
    r_partial_index = np.argmin(d_point_rect)

    # adjust index of r_point adding 0 finger index ( semi-contour index )
    r_index = fingers_indexes[0] + r_partial_index

    # return point and index
    r_point = contour[r_index]

    return r_point, r_index


def calculateMediumPoints(contour, valley_indexes, fingers_indexes):

    # valley_indexes and valley_points have 5 points updating after complementary search
    comp_valley_indexes, valley_indexes = getComplementaryValleyPoints(len(contour), valley_indexes, fingers_indexes)

    medium_points = getMediumFingerPoint(contour[valley_indexes], contour[comp_valley_indexes])

    return medium_points, valley_indexes, comp_valley_indexes
    

def updateContour(contour, valley_indexes, fingers_indexes, r_index):

    length = len(contour)

    updated_contour = np.concatenate((contour[r_index::], contour[:r_index:]))

    # print('ATTENZIONE: ', updated_contour , contour)

    # print(len(contour), len(updated_contour))
    
    valley_indexes  = [ (index + length - r_index)%length  for index in valley_indexes]
    fingers_indexes = [ (index + length - r_index)%length  for index in fingers_indexes]

    return updated_contour[0], updated_contour, valley_indexes, fingers_indexes