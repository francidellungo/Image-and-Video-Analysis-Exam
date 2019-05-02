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


# def calculateDistance(p1,p2):  
#         """ 
#         GOAL:   
#                 the function returns distance between point
#                 (x1, y1) and (x2, y2)

#         PARAMS:
#                 (input)
#                 - (x1, y1): 
#                         first point P1
#                 - (x2, y2): 
#                         second point P2

#                 (output)
#                 - dist:
#                         euclidean distance between P1 and P2

#         """
#         x1,y1 = p1
#         x2,y2 = p2
#         dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        
#         return dist


# def findFingerIndexesDistance(all_fingers_point):
#         finger_points = []

#         while(all_fingers_point):

#                 # remove from list the first element, the most left element
#                 xyi = all_fingers_point.pop(0)
#                 min_dist = float('inf')
#                 min_dist_index = None

#                 # find min distance between selected point and all the others
#                 for i in range(len(all_fingers_point)):
#                         dist = calculateDistance(xyi, all_fingers_point[i])
#                         if dist < min_dist:
#                                 min_dist = dist
#                                 min_dist_index = i

#                 if min_dist_index is not None:
#                         # calculate the length of the convexityDefects segment of which the selected point is an endpoint
#                         index = xyi[3]
#                         start = cnt[defects[index][0]][0]
#                         dist_start = calculateDistance(xyi, start)
#                         end = cnt[defects[index][1]][0]
#                         dist_end = calculateDistance(xyi, end)
#                         if min_dist < max(dist_start, dist_end)/2:
#                                 # calculate midpoint of the two points that are placed on the same finger -> this is the representative of the finger considered
#                                 finger_points.append([(xyi[0]+all_fingers_point[min_dist_index][0])/2, (xyi[1]+all_fingers_point[min_dist_index][1])/2] )
#                                 # the point at the min distance from the first one considered is removed from the app_fingers_point list
#                                 #  (it has been used to create the middle point)
#                                 all_fingers_point.pop(min_dist_index)
#                         else:
#                                 # the representative point is only the one selected
#                                 finger_points.append(xyi[:2])

#                 else:
#                         # no other points in app_fingers_point list
#                         finger_points.append(xyi[:2])


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

        # create an empty black image
        drawing = np.zeros((img_binary.shape[0], img_binary.shape[1], 3), np.uint8)

        # Find the convex hull related to contours of the binary image
        hull = cv2.convexHull(cnt, clockwise=True, returnPoints = False)

        # draw contours
        cv2.drawContours(drawing, cnt, -1, (0, 255, 0), 1, 8)   # green color
        
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

                # draw on image lines between fingertips and valley points 
                cv2.line(drawing,start,end,[0,255,0],2)
                # cv2.putText(drawing, str(i), (int((start[0]+end[0])/2), int((start[1]+end[1])/2)), font, 2,(255,0,0),2,cv2.LINE_AA)
                # cv2.circle(drawing,far,5,[0,0,255],-1)

                all_fingers_indexes.append(e)
                all_fingers_indexes.append(s)



        # find one representative point for each fingertips (if a fingertips had two points the final point is calculated as the middlepoint)
        fingers_indexes = findFingerIndexesSimple(len(cnt), all_fingers_indexes)

        finger_points = cnt[fingers_indexes]

        print(finger_points)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # draw final finger representative points
        for finger in finger_points:
                xy = tuple([ int(x) for x in finger[0] ])
                print('final xy fingers', xy)
                cv2.circle(drawing,xy,5,[255,0,0],-1)

        return drawing, finger_points, valley_points, fingers_indexes, valley_indexes

