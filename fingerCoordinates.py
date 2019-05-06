import imageio as iio
import cv2
import math
import numpy as np
from skimage import filters
from skimage.measure import regionprops
from skimage.morphology import diamond, dilation, erosion, square
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import os
import random as rng


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

        # find convexityDefects (a deviation of the hand shape from his hull is a convexity defect.)
        defects = cv2.convexityDefects(cnt, hull)


        # first defect become the last one (in case of the first one corrisponds to the one between little and ring fingers)
        defects = np.concatenate((defects[1::],[defects[0]]))

        # find segments at maximum distance from the relative depth points (d), these correspond to the segments between the fingers (fingertips)        
        defects = [[list(elem[0][:]), i] for i, elem in enumerate(defects)]


        # sort in descending order elements of defects list. Sorting is based on the distance between the farthest point and the convex hull 
        defects.sort(key = lambda x: x[0][3], reverse= True)
        

        # consider only the 4 segments that have maximum distance from their defects points (they correspond to the spaces between the 5 fingers calculated at the fingertips)
        defects = defects[:4]

        defects.sort(key = lambda x: x[1], reverse= True)
        
        defects = [ elem[0] for elem in defects]

        return defects


def findFingerIndexesSimple(lenght, all_fingers_indexes):

        finger_indexes = []
        finger_indexes.append(all_fingers_indexes.pop(0))

        while len(all_fingers_indexes) > 1:
                a = all_fingers_indexes.pop(0)
                b = all_fingers_indexes.pop(0)
                if a < b:
                        if lenght - b > a:
                                c = lenght - int((a + (lenght - b))/2 )
                        else:
                                c = int((a + (lenght - b))/2 )
                else:
                        c = int((a+b)/2)
                finger_indexes.append(c)
        
        finger_indexes.append(all_fingers_indexes.pop())

        return finger_indexes


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

        # Find the convex hull related to contours of the binary image
        hull = cv2.convexHull(cnt, clockwise=True, returnPoints = False)
        
        # obtain the 4 important defects from contourn and hull of hand image
        defects = getImportantDefect(cnt, hull)


        # initialize empty lists
        all_fingers_indexes = []
        valley_points = []
        valley_indexes = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(len(defects)):
                # get indexes of important points from defects
                s,e,f,d = defects[i]

                # update coordinates of valley points, that are points between each pair of fingers
                valley_points.append(list(cnt[f][0]))
                valley_indexes.append(f)

                # get points
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                all_fingers_indexes.append(e)
                all_fingers_indexes.append(s)



        # find one representative point for each fingertips (if a fingertips had two points the final point is calculated as the middlepoint)
        fingers_indexes = findFingerIndexesSimple(len(cnt), all_fingers_indexes)

        finger_points = cnt[fingers_indexes]

        return finger_points, valley_points, fingers_indexes, valley_indexes

