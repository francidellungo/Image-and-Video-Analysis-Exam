

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
                - contour:
                    result hand contour
    """

    # # find contours of processed image
    # contours_bin, hierarchy_bin = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # area_max = 0
    # index_area_max = -1

    # # select biggest contour
    # for i in range(len(contours_bin)):

    #     # get area of ith contour
    #     area_i = cv2.contourArea(contours_bin[i])

    #     # check if dimension of area is bigger than the biggest
    #     if area_max < area_i:
    #         index_area_max = i
    #         area_max = area_i

    return contours_bin[index_area_max]




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

    # # apply gaussian mixtures and get binary image
    # img_binary = preprocessingHSV(img_bgr)

    # # create an empty black image
    # hand_mask = np.zeros((img_binary.shape[0], img_binary.shape[1]), np.uint8)
    
    # contour = getOneHandContour(img_binary)

    # # create polylines from contour points, and draw if filled in image
    # cv2.fillPoly(hand_mask, pts =[contour], color=(255))

    # # get a copy of image in order to work with
    # img_binary = hand_mask.copy()

    # # create a binary mask, where img is white(255) put 1, else let 0
    # img_binary[img_binary == 255] = 1
    
    # # skimage regionprops need labeled mask and binary mask
    # # and return properties object with pixels property like
    # # centroid, ellipse around pixels, ecc...
    # properties = regionprops(img_binary, coordinates='xy')

    # # get center of mass of pixels (also called centroid)
    # center_of_mass = properties[0].centroid[::-1]
    
    return hand_mask, contour, center_of_mass, None # ellipse_mask
    




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

        # # find convexityDefects (a deviation of the hand shape from his hull is a convexity defect.)
        # defects = cv2.convexityDefects(cnt, hull)

        # # first defect become the last one (in case of the first one corrisponds to the one between little and ring fingers)
        # defects = np.concatenate((defects[1::],[defects[0]]))

        # # find segments at maximum distance from the relative depth points (d), these correspond to the segments between the fingers (fingertips)        
        # defects = [[list(elem[0][:]), i] for i, elem in enumerate(defects)]

        # # sort in descending order elements of defects list. Sorting is based on the distance between the farthest point and the convex hull 
        # defects.sort(key = lambda x: x[0][3], reverse= True)
        
        # # consider only the 4 segments that have maximum distance from their defects points (they correspond to the spaces between the 5 fingers calculated at the fingertips)
        # defects = defects[:4]

        # defects.sort(key = lambda x: x[1], reverse= True)
        
        # defects = [ elem[0] for elem in defects]

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

        # # Find the convex hull related to contours of the binary image
        # hull = cv2.convexHull(cnt, clockwise=True, returnPoints = False)

        # # obtain the 4 important defects from contourn and hull of hand image
        # defects = getImportantDefect(cnt, hull)


        # # initialize empty lists
        # all_fingers_indexes = []
        # valley_points = []
        # valley_indexes = []

        # font = cv2.FONT_HERSHEY_SIMPLEX
        
        # for i in range(len(defects)):
        #         # get indexes of important points from defects
        #         s,e,f,d = defects[i]

        #         # update coordinates of valley points, that are points between each pair of fingers
        #         valley_points.append(list(cnt[f][0]))
        #         valley_indexes.append(f)

        #         # get points
        #         start = tuple(cnt[s][0])
        #         end = tuple(cnt[e][0])
        #         far = tuple(cnt[f][0])

        #         all_fingers_indexes.append(e)
        #         all_fingers_indexes.append(s)



        # # find one representative point for each fingertips (if a fingertips had two points the final point is calculated as the middlepoint)
        # fingers_indexes = findFingerIndexesSimple(len(cnt), all_fingers_indexes)

        # # print('contour: ',len(cnt))
        # # print('fin_idx: ',fingers_indexes)
        # finger_points = cnt[fingers_indexes]

        return finger_points, valley_points, fingers_indexes, valley_indexes



