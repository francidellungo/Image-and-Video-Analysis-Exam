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

path_in = './images/'
path_out = './a_masks/'
path_ell = './a_ellipses/'

paths = os.listdir(path_in)

colours = [
            (0, 0, 255),        # blue
            (0,128,0),	        # green
            (173,255,47),       # greenyellow	
            (255, 160, 0),      # orange
            (0,238,0),	        # green2
            (250, 246, 0),      # yellow
            (0,205,0),          # green3
            (255, 186, 230),    # pink
            (255, 69, 230),     # purple
            (0,255,0),	        # green1	 
            (0,139,0),          # green4	
]


def preprocessing(name_img):

    img_natural = cv2.imread(path_in + name_img, cv2.IMREAD_GRAYSCALE)

    img_median = cv2.medianBlur(img_natural,3)

    img_end = img_median.copy()
    grid = 2
    for i in range(grid):
        for j in range(grid):
            th, im = cv2.threshold(np.array([ np.array(row[j*int(len(img_median[0])/grid):(j+1)*int(len(img_median[0])/grid)]) for row in img_median[i*int(len(img_median)/grid):(i+1)*int(len(img_median)/grid)]]), 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            for k in range(len(im)):
                for y in range(len(im[0])):
                    img_end[i*int(len(img_median)/grid)+k][j*int(len(img_median[0])/grid)+y] = im[k][y]

    (threshold_value, img_binary) = cv2.threshold(img_end, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    

    # kernel = diamond(10)
    kernel = rectangle(16, 6)
    img_dilation = dilation(img_binary, kernel)
    kernel = square(9)
    img_erosion = erosion(img_dilation, kernel)

    # img_gaussian_blur = cv2.GaussianBlur(img_erosion,(5,5),0)
    img_binary = img_erosion.copy()
    # Elliptical Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16, 16))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    # find center
    # contours_end, hierarchy_end = cv2.findContours(img_end, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty black image
    drawing_end = np.zeros((img_end.shape[0], img_end.shape[1]), np.uint8)
    # draw contours and hull points
    # for i in range(len(contours_end)):
    #     print("contour ", i, " color: ", colours[i%len(colours)], " area: ", cv2.contourArea(contours_end[i]))
    #     color_contours_end = colours[i%len(colours)] #(0, 255, 0) # green - color for contours
    #     color = (255, 0, 0) # blue - color for convex hull
    #     # draw ith contour
        
    #     cv2.drawContours(drawing_end, contours_end, i, color_contours_end, 1, 8, hierarchy_end)
        
    #     # draw ith convex hull object
    #     # cv2.drawContours(drawing, hull, i, color, 1, 8)
    
    # find center
    contours_bin, hierarchy_bin = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty black image
    # draw contours and hull points
    area_max = 0
    index_area_max = -1
    for i in range(len(contours_bin)):
        print("contour ", i, " color: ", colours[i%len(colours)], " area: ", cv2.contourArea(contours_bin[i]))
        color_contours_bin = colours[i%len(colours)] #(0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        area_i = cv2.contourArea(contours_bin[i])
        if area_max < area_i:
            index_area_max = i
            area_max = area_i

        # cv2.drawContours(drawing_end, contours_bin, i, color_contours_bin, 1, 8, hierarchy_bin)
        
        # draw ith convex hull object
        # cv2.drawContours(drawing, hull, i, color, 1, 8)

    contours_bin = contours_bin[index_area_max]

    cv2.fillPoly(drawing_end, pts =[contours_bin], color=(255))
    
    #cv2.drawContours(drawing_end, contours_bin, -1, 255, 1)
    
    cv2.imwrite(path_out + name_img, drawing_end)

    img_binary = drawing_end.copy()

    img_binary[img_binary == 255] = 1
    
    labeled_foreground = (img_binary > 0).astype(int)
    # print(labeled_foreground.shape, img_binary.shape)

    properties = regionprops(labeled_foreground, img_binary, coordinates='xy')

    center_of_mass = properties[0].centroid

    weighted_center_of_mass = properties[0].weighted_centroid

    major_axis_lenght = properties[0].major_axis_length
    orientation = properties[0].orientation
    
    print('\n \n')
    print('centroid coord:',center_of_mass)
    print('maj ax lenght: ', major_axis_lenght)
    print('min ax lenght: ', properties[0].minor_axis_length)
    print('orientation : ', orientation)

#   ----------------------------------------------------------------------------    #


    # draw img with center of mass / centroid 
    colorized = label2rgb(labeled_foreground, img_binary, colors=['red', 'white'], alpha=0.2)
    colorized = 255*colorized
    
    # cv2.imwrite(path_color_out + name_img, colorized)

    # ax.plot(weighted_center_of_mass[1], weighted_center_of_mass[0], marker = 'o', color= 'purple'
    # )
    cv2.circle(colorized,(int(weighted_center_of_mass[1]), int(weighted_center_of_mass[0])), 5, (0,255,255), -1)

    # ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
    cv2.circle(colorized,(int(center_of_mass[1]), int(center_of_mass[0])), 5, (0,0,255), -1)

    x2, y2 = [(int(center_of_mass[1]), int(center_of_mass[0]))], [0, center_of_mass[0]*1.9]
    # ax.plot(x2, y2, marker = 'o', color= 'green')

    y0, x0 = properties[0].centroid
    y0, x0 = int(y0), int(x0)
    orientation = properties[0].orientation
    x1 = int(x0 + math.cos(orientation) * 0.5 * properties[0].major_axis_length)
    y1 = int(y0 - math.sin(orientation) * 0.5 * properties[0].major_axis_length)
    # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5, color = 'red')
    cv2.line(colorized,(x0, y0), (x1, y1), (255,0,0) , 3) 

    x2 = int(x0 - math.sin(orientation) * 0.5 * properties[0].minor_axis_length)
    y2 = int(y0 - math.cos(orientation) * 0.5 * properties[0].minor_axis_length)
    # ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5, color = 'red')
    cv2.line(colorized,(x0, y0), (x2, y2), (255,0,0) , 3) 

    mean = [ x0,y0 ]
    width = int(properties[0].major_axis_length/2)
    height = int(properties[0].minor_axis_length/2)
    
    cv2.ellipse(colorized,
                (x0,y0),
                (height, width), 
                int(90 - orientation*360/(2*np.pi)), 
                startAngle=0, 
                endAngle=360, 
                color=255, 
                thickness=2)
    
    # , int(90 - orientation*360/(2*np.pi)),0, 360, color=tuple(255),  thickness=1)
    # ell = mpl.patches.Ellipse(xy=mean, width=height, height=width, angle = 90 - orientation*360/(2*np.pi), color = 'red', edgecolor='r', fc='None', lw=2)

    # ax.add_patch(ell)
    # ax.set_aspect('equal')
    # ax.autoscale()

    cv2.imwrite(path_ell + name_img, colorized)

    # plt.savefig(path_convx_out + name_img)
    # plt.show(path_convx_out + name_img)
    

def main():

    for path in paths:
        preprocessing(path)

# preprocessing('img_dataset2.png')



if __name__== "__main__":
  main()