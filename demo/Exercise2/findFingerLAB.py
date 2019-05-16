

import imageio as iio
import cv2
import math
import numpy as np
import os

from skimage import filters
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.morphology import diamond, square, dilation, erosion, skeletonize, skeletonize_3d, thin, rectangle

from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
import copy

from Exercise1.preprocessingLAB import preprocessingHSV


def getOneHandContour(img_binary):
    """ 
        GOAL:   
            the funciton calculates the contours of image,
            and take only the hand contour (the one that has the biggest area)

        PARAMS:
            (input)
                - img_binary: 
                    original binary image

            (output)
                - bigger_area_contour:
                    result hand contour
    """
    
    # COMPLETE WITH SOME CODE

    return bigger_area_contour




def getHand(img_bgr):
    """ 
        GOAL:   
            the function calculates on grey scale input image
            a mask of the hand, and returns the mask with also ellipse 
            and some important params of ellipse.

        PARAMS:
            (input)
                - img_grey: 
                    the original grey scale image

            (output)
                - hand_mask: 
                    the mask of the hand
                - contour:
                    contour of the mask
                - prop: 
                    array with center of mass, length of major and 
                    minor axes of ellipse, and orientation of ellipse.
                - ellipse_mask: 
                    same image of hand_mask with drawn on the ellipse
                    and the axes
    """

    # COMPLETE WITH SOME CODE
    
    return hand_mask, contour, center_of_mass, None 




def getImportantDefect(cnt, hull):
    """ 
    GOAL:   
            the function returns the 4 element of defects
            that are between the fingers.

    PARAMS:
            (input)
            - cnt: 
                    contourn of the hand mask
            - hull: 
                    convex hull of the contourn

            (output)
            - defects: 
                    only the 4 defects between the fingers
    """

    # COMPLETE WITH SOME CODE

    return defects



def getFingerCoordinates(cnt, img_binary):
    """ 
    GOAL:   
            the function returns drawing images with the 
            fingers and valleys point, and the two lists of points

    PARAMS:
            (input)
            - cnt: 
                    contourn of the hand mask
            - img_binary: 
                    binary mask of the hand

            (output)
            - drawing: 
                    images of the hand mask with drawn on it fingers 
                    and valley points.
            - finger_points:
                    list of finger points
            - valley_points:
                    list of valley points

    """

    # COMPLETE WITH SOME CODE

    return finger_points, valley_points, fingers_indexes, valley_indexes

