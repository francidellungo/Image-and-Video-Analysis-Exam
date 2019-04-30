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


def getFingerCoordinates(contours, img_binary):
        cnt = contours[0]
        # Find the convex hull related to contours of the binary image
        hull = cv2.convexHull(cnt, returnPoints = False)
        
        # create an empty black image
        drawing = np.zeros((img_binary.shape[0], img_binary.shape[1], 3), np.uint8)

        # draw contours
        color_contours = (0, 255, 0) # green - color for contours
        cv2.drawContours(drawing, contours, 0, color_contours, 1, 8)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        # find convexityDefects (a deviation of the hand shape from his hull is a convexity defect.)
        defects = cv2.convexityDefects(cnt,hull)
        print('n elements defects: ',len(defects), defects[0][0])

        # find segments at maximum distance from the relative depth points (d), these correspond to the segments between the fingers (fingertips)
        # print(defects)
        defects_new = [list(defects[i][0][:]) for i in range(len(defects))]
        # sort in descending order elements of defects list. Sorting is based on the distance between the farthest point and the convex hull 
        defects_new.sort(key = lambda x: x[3], reverse= True)

        # consider only the 4 segments that have maximum distance from their defects points (they correspond to the spaces between the 5 fingers calculated at the fingertips)
        defects_new = defects_new[:4]
        print('defects new:', defects_new)
        
        xyi_fingers_point = []
        xy_valleys = []

        
        for i in range(len(defects_new)):
                s,e,f,d = defects_new[i]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                # update coordinates of valley points, that are points between each pair of fingers
                xy_valleys.append(list(cnt[f][0]))
                # draw lines between fingertips and valley points 
                cv2.line(drawing,start,end,[0,255,0],2)
                cv2.circle(drawing,far,5,[0,0,255],-1)

                # update coordinates of points related to each fingertips (each fingertip has still 1 or 2 points)
                xy_tmp = list(cnt[s][0])
                xy_tmp.append(i)
                xyi_fingers_point.append(xy_tmp)
                xy_tmp = list(cnt[e][0])
                xy_tmp.append(i)
                xyi_fingers_point.append(xy_tmp)


        print('valleys: ', xy_valleys)
        print('xyi fingers point : ', xyi_fingers_point)

        # find one representative point for each fingertips (if a fingertips had two points the final point is calculated as the middlepoint)
        xy_finger_final = []
        while(xyi_fingers_point):
                xyi = xyi_fingers_point.pop(0)
                min_dist = float('inf')
                min_dist_index = None
                # find min distance between selected point and all the others
                for i in range(len(xyi_fingers_point)):
                        dist = calculateDistance(xyi[0], xyi[1], xyi_fingers_point[i][0], xyi_fingers_point[i][1])
                        if dist < min_dist:
                                min_dist = dist
                                min_dist_index = i

                if min_dist_index is not None:
                        # calculate the length of the convexityDefects segment of which the selected point is an endpoint
                        index = xyi[2]
                        start = cnt[defects_new[index][0]][0]
                        dist_start = calculateDistance(xyi[0], xyi[1], start[0], start[1])
                        end = cnt[defects_new[index][1]][0]
                        dist_end = calculateDistance(xyi[0], xyi[1], end[0], end[1])
                        if min_dist < max(dist_start, dist_end)/2:
                                # calculate midpoint of the two points that are placed on the same finger -> this is the representative of the finger considered
                                xy_finger_final.append([(xyi[0]+xyi_fingers_point[min_dist_index][0])/2, (xyi[1]+xyi_fingers_point[min_dist_index][1])/2] )
                                # the point at the min distance from the first one considered is removed from the xyi_fingers_point list
                                #  (it has been used to create the middle point)
                                xyi_fingers_point.pop(min_dist_index)
                        else:
                                # the representative point is only the one selected
                                xy_finger_final.append(xyi[:2])

                else:
                        # no other points in xyi_fingers_point list
                        xy_finger_final.append(xyi[:2])

        # draw final finger representative points
        for i in range(len(xy_finger_final)):
                xy = xy_finger_final[i]
                xy = tuple([ int(x) for x in xy ])
                print('final xy fingers', xy)
                cv2.circle(drawing,xy,5,[255,0,0],-1)

        # show all
        cv2.imshow('img',drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def calculateDistance(x1,y1,x2,y2):  
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist

