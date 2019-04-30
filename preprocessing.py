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


def otsu_grid(img_grey, grid):
    """ 

        GOAL:   the funciton create grid x grid cells,
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

    for i in range(grid):
        for j in range(grid):
            # threshold and image of one single cell
            th, im = cv2.threshold(np.array([ np.array(row[j*int(len(img_grey[0])/grid):(j+1)*int(len(img_grey[0])/grid)]) for row in img_grey[i*int(len(img_grey)/grid):(i+1)*int(len(img_grey)/grid)]]), 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # unify image cell in one unique image, cell by cell
            for k in range(len(im)):
                for y in range(len(im[0])):
                    img_otsu[i*int(len(img_grey)/grid)+k][j*int(len(img_grey[0])/grid)+y] = im[k][y]

    return img_otsu
    

def preprocessing(img_grey):
    """ 

        GOAL:   the funciton make the preprocessing applying
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
    img_binary = dilation(img_binary, kernel)

    # erosion with square of dimension 9 x 9 
    kernel = square(9)
    img_binary = erosion(img_binary, kernel)

    # closure with elliptical kernel of dimension 16 x 16
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16, 16))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    return img_binary


def getOneHandContour(img_binary):
    """ 

        GOAL:   the funciton calculates the contours of image,
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
    contours_bin, hierarchy_bin = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def getHand(img_grey):
    """ 

        GOAL:   the funciton calculate on grey scale input image
                a mask of the hand, and return the mask with also ellipse 
                and some important params of ellipse.

        PARAMS:
            (input)
                - img_grey: 
                    the original gery scale image

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

    # get binary image after preprocessing
    img_binary = preprocessing(img_grey)

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
    labeled_foreground = (img_binary > 0).astype(int)

    # skimage regionprops need labeled mask and binary mask
    # and return properties object with pixels property like
    # centroid, ellipse around pixels, ecc...
    properties = regionprops(labeled_foreground, img_binary, coordinates='xy')

    # get center of mass of pixels (also called centroid)
    center_of_mass = properties[0].centroid

    # get integer values of center of mass
    y0, x0 = center_of_mass
    y0, x0 = int(y0), int(x0)

    # exists also weighted center of mass
    weighted_center_of_mass = properties[0].weighted_centroid

    # get major and minor axis of ellips
    major_axis_length = properties[0].major_axis_length
    minor_axis_length = properties[0].minor_axis_length

    # get orientation of ellipse (in degree from x axis) 
    orientation = properties[0].orientation
    
    print('\n \n')
    print('centroid coord: ' , center_of_mass)
    print('maj ax lenght:  ' , major_axis_length)
    print('min ax lenght:  ' , minor_axis_length)
    print('orientation :   ' , orientation)

    prop = [center_of_mass, major_axis_length, minor_axis_length, orientation]

    # create image from binary image and labeled mask
    ellipse_mask = label2rgb(labeled_foreground, img_binary, colors=['red', 'white'], alpha=0.2)
    ellipse_mask = 255 * ellipse_mask
    
    # draw in that image the center of mass and weighted center of mass
    cv2.circle(ellipse_mask,(x0, y0), 5, (0,0,255), -1)
    cv2.circle(ellipse_mask,(int(weighted_center_of_mass[1]), int(weighted_center_of_mass[0])), 5, (0,255,255), -1)

    # calculate and draw major axis of ellipse
    x1 = int(x0 + math.cos(orientation) * 0.5 * major_axis_length)
    y1 = int(y0 - math.sin(orientation) * 0.5 * major_axis_length)
    cv2.line(ellipse_mask,(x0, y0), (x1, y1), (255,0,0) , 3) 

    # calculate and draw minor axis of ellipse
    x2 = int(x0 - math.sin(orientation) * 0.5 * minor_axis_length)
    y2 = int(y0 - math.cos(orientation) * 0.5 * minor_axis_length)
    cv2.line(ellipse_mask,(x0, y0), (x2, y2), (255,0,0) , 3) 

    # set width and height for ellipse
    width = int(major_axis_length/2)
    height = int(minor_axis_length/2)
    
    # draw ellipse 
    cv2.ellipse(ellipse_mask,
                (x0,y0),
                (height, width), 
                int(90 - orientation*360/(2*np.pi)), 
                startAngle=0, 
                endAngle=360, 
                color=255, 
                thickness=2)
    
    return hand_mask, contour, prop, ellipse_mask
    
