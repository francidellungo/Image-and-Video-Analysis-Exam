import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from numpy.linalg import norm
from extractionShapeFeatures import *


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


def getComplementaryValleyPoints(cnt_length, valley_indexes, fingers_indexes):
    
    valley_indexes.insert(0, valley_indexes[0])

    complementary_valley_indexes = [( 2 * p_index - v_index ) % cnt_length for v_index, p_index in zip(valley_indexes[::-1], fingers_indexes[::-1])]
            
    app = complementary_valley_indexes[-1]
    complementary_valley_indexes[-1] = valley_indexes[0]
    valley_indexes[0] = app
    complementary_valley_indexes = complementary_valley_indexes[::-1]

    all_valley = []
    all_valley.append(valley_indexes[0])
    for x, o in zip(valley_indexes[1:], complementary_valley_indexes[:-1]):
        app = [x, o]
        app.sort(key = lambda x: x, reverse= True)
        all_valley.append(app[0])
        all_valley.append(app[1])

    all_valley.append(complementary_valley_indexes[-1])

    valley_indexes = all_valley[0::2]
    complementary_valley_indexes = all_valley[1::2]

    return complementary_valley_indexes, valley_indexes


def getMediumFingerPoint(valley_points, complementary_valley_points):
    medium_points = []

    for v_point, c_v_point in zip(valley_points, complementary_valley_points):
        app = [v_point, c_v_point]
        medium_points.append(np.mean(app, axis = 0).tolist())

    return medium_points

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