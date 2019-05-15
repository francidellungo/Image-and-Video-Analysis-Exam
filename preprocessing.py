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

def otsu_grid(img_grey, grid):
    """ 
        GOAL:   
            the funciton create grid x grid cells,
            and apply otsu on each of this cells.
            Then put all sub-mask in one.

        PARAMS:
            (input)
                - img_grey: 
                    original grey scale image
                - grid: 
                    number of cell for side

            (output)
                - img_otsu:
                    result mask image
    """
    img_otsu = img_grey.copy()
    sublen = int(len(img_grey)/grid)
    sublon = int(len(img_grey[0])/grid)

    for i in range(grid):
        for j in range(grid):
            # threshold and image of one single cell
            th, im = cv2.threshold(np.array([ np.array(row[j*int(len(img_grey[0])/grid):(j+1)*int(len(img_grey[0])/grid)]) for row in img_grey[i*int(len(img_grey)/grid):(i+1)*int(len(img_grey)/grid)]]), 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # print(type(im))

            # unify image cell in one unique image, cell by cell
            
            # for k in range(len(im)):
            #     for y in range(len(im[0])):
            #         img_otsu[i*int(len(img_grey)/grid)+k][j*int(len(img_grey[0])/grid)+y] = im[k][y]

            img_otsu[np.ix_(range(i*sublen, (i+1)*sublen), range(j*sublon, (j+1)*sublon))] = im

    return img_otsu
    
def preprocessingHSV(img_bgr):
    """ 
        GOAL:   
            the funciton make the preprocessing applying
            some filter, dilation, erosion and closure.

        PARAMS:
            (input)
                - img_grey: 
                    original grey scale image

            (output)
                - img_binary:
                    result binary image
    """

    img = cv2.resize(img_bgr, None, fx=0.5, fy=0.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    shape = b.shape

    # Smooth the three color channels one by one
    h = cv2.medianBlur(h,5)
    s = cv2.medianBlur(s,5)
    v = cv2.medianBlur(v,5)

    num_clusters = 2
    # Warning: X is 3xNum_pixels. To fit the kmeans model X.T should be used
    X = np.array([h.reshape(-1), s.reshape(-1), v.reshape(-1)])
    gmm=GaussianMixture(n_components=num_clusters,
                    covariance_type='full',
                    init_params='kmeans',
                    max_iter=300, n_init=4, random_state=10)
    gmm.fit(X.T)

    Y = gmm.predict(X.T)

    mask_img = copy.deepcopy(h.reshape(-1))

    unique, counts = np.unique(Y, return_counts=True)
    dic = dict(zip(unique, counts))
    
    if dic[0] > dic[1]:
        mask_img[ Y==0 ] = 0 
        mask_img[ Y==1 ] = 1
    else:
        mask_img[ Y==0 ] = 1
        mask_img[ Y==1 ] = 0

    mask_img = mask_img.reshape(shape)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
    img_bin = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

    return img_bin


def preprocessingGREY(img_grey):
    """ 
        GOAL:   
            the funciton make the preprocessing applying
            some filter, dilation, erosion and closure.

        PARAMS:
            (input)
                - img_grey: 
                    original grey scale image

            (output)
                - img_binary:
                    result binary image
    """

    # apply median blurr to delete sale and pepper noise
    img_median = cv2.medianBlur(img_grey,3)
    
    # grid = 2 means:  2 x 2 cells
    grid = 2   

    # get mask image
    img_binary = otsu_grid(img_median, grid)

    # dilation with rectangle of dimension 6 x 16 
    kernel = rectangle(16, 6)
    img_dilated = dilation(img_binary, kernel)

    # erosion with square of dimension 9 x 9 
    kernel = square(9)
    img_erosed = erosion(img_dilated, kernel)

    # closure with elliptical kernel of dimension 16 x 16
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31, 31))
    img_closed = cv2.morphologyEx(img_erosed, cv2.MORPH_CLOSE, kernel)

    return img_closed


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

    # find contours of processed image
    contours_bin, hierarchy_bin = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # create hull array for convex hull points
    # hull = []
    
    # # calculate points for each contour
    # for i in range(len(contours_bin)):
    #     # creating convex hull object for each contour
    #     hull.append(cv2.convexHull(contours_bin[i], clockwise=True))

    # # create an empty black image
    # drawing = np.zeros((1200, 1600, 3), np.uint8)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # # draw contours and hull points
    # for i in range(len(contours_bin)):
    #         color_contours = (0, 255, 0) # green - color for contours
    #         color = (255, 0, 0) # blue - color for convex hull
    #         # draw ith contour
    #         cv2.drawContours(drawing, contours_bin, i, color_contours, 1, 8, hierarchy_bin)
    #         # draw ith convex hull object
    #         cv2.drawContours(drawing, hull, i, color, 1, 8)
    #         defects = cv2.convexityDefects(contours_bin[i], hull[i])
    #         for j, (point1, point2) in enumerate(zip(hull[i][0::2], hull[i][1::2])):
    #             print(point1, point2)
    #             point1, point2 = point1[0], point2[0]
    #             print(point1, point2)
    #             x = int((point1[0] + point2[0] ) / 2)
    #             y = int((point1[1] + point2[1] ) / 2)
    #             print(x, y)
    #             cv2.putText(drawing, str(j), (x, y), font, 0.5,(255,0,0),2,cv2.LINE_AA)

    # defects = cv2.convexityDefects(contours_bin,cv2.convexHull(contours_bin[0], clockwise=True))

    # for i in range(defects.shape[0]):
    #     s,e,f,d = defects[i,0]
    #     start = tuple(contours_bin[s][0])
    #     end = tuple(contours_bin[e][0])
    #     far = tuple(contours_bin[f][0])
    #     cv2.line(drawing,start,end,[0,255,0],2)
    #     cv2.circle(drawing,far,5,[0,0,255],-1)

    # cv2.imshow('img',drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # we have more than one contours in some image
    # than we need to consider only hand contour, it is 
    # the biggest one
    area_max = 0
    index_area_max = -1

    # select biggest contour
    for i in range(len(contours_bin)):

        # get area of ith contour
        area_i = cv2.contourArea(contours_bin[i])

        # check if dimension of area is bigger than the biggest
        if area_max < area_i:
            index_area_max = i
            area_max = area_i

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

    ## get binary image after preprocessing
    # img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # img_binary = preprocessingGREY(img_grey)
    
    img_binary = preprocessingHSV(img_bgr)

    # create an empty black image
    hand_mask = np.zeros((img_binary.shape[0], img_binary.shape[1]), np.uint8)
    
    contour = getOneHandContour(img_binary)

    # create polylines from contour points, and draw if filled in image
    cv2.fillPoly(hand_mask, pts =[contour], color=(255))

    # get a copy of image in order to work with
    img_binary = hand_mask.copy()

    # create a binary mask, where img is white(255) put 1, else let 0
    img_binary[img_binary == 255] = 1

    
    # create a labeled mask of image where image is 1
    # labeled_foreground = (img_binary > 0).astype(int)
    
    # skimage regionprops need labeled mask and binary mask
    # and return properties object with pixels property like
    # centroid, ellipse around pixels, ecc...
    properties = regionprops(img_binary, coordinates='xy')

    # get center of mass of pixels (also called centroid)
    center_of_mass = properties[0].centroid[::-1]

    # # get integer values of center of mass
    # x0, y0 = center_of_mass
    # y0, x0 = int(y0), int(x0)

    # # exists also weighted center of mass
    # weighted_center_of_mass = properties[0].weighted_centroid

    # # get major and minor axis of ellips
    # major_axis_length = properties[0].major_axis_length
    # minor_axis_length = properties[0].minor_axis_length

    # # get orientation of ellipse (in degree from x axis) 
    # orientation = properties[0].orientation
    
    # # print('centroid coord: ' , center_of_mass)
    # # print('maj ax lenght:  ' , major_axis_length)
    # # print('min ax lenght:  ' , minor_axis_length)
    # # print('orientation :   ' , orientation)

    # prop = [center_of_mass, major_axis_length, minor_axis_length, orientation]

    # # create image from binary image and labeled mask
    # ellipse_mask = label2rgb(labeled_foreground, img_binary, colors=['red', 'white'], alpha=0.2)
    # ellipse_mask = 255 * ellipse_mask
    
    # # draw in that image the center of mass and weighted center of mass
    # cv2.circle(ellipse_mask,(x0, y0), 5, (0,0,255), -1)
    # cv2.circle(ellipse_mask,(int(weighted_center_of_mass[1]), int(weighted_center_of_mass[0])), 5, (0,255,255), -1)

    # # calculate and draw major axis of ellipse
    # x1 = int(x0 + math.cos(orientation) * 0.5 * major_axis_length)
    # y1 = int(y0 - math.sin(orientation) * 0.5 * major_axis_length)
    # cv2.line(ellipse_mask,(x0, y0), (x1, y1), (255,0,0) , 3) 

    # # calculate and draw minor axis of ellipse
    # x2 = int(x0 - math.sin(orientation) * 0.5 * minor_axis_length)
    # y2 = int(y0 - math.cos(orientation) * 0.5 * minor_axis_length)
    # cv2.line(ellipse_mask,(x0, y0), (x2, y2), (255,0,0) , 3) 

    # # set width and height for ellipse
    # width = int(major_axis_length/2)
    # height = int(minor_axis_length/2)
    
    # # draw ellipse 
    # cv2.ellipse(ellipse_mask,
    #             (x0,y0),
    #             (height, width), 
    #             int(90 - orientation*360/(2*np.pi)), 
    #             startAngle=0, 
    #             endAngle=360, 
    #             color=255, 
    #             thickness=2)
    
    return hand_mask, contour, center_of_mass, None # ellipse_mask
    
