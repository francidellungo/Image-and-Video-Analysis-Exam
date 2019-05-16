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

def preprocessingHSV(img_bgr):
    """ 
        GOAL:   
            the funciton make the preprocessing applying
            gaussian mixture and an open with.

        PARAMS:
            (input)
                - img_bgr: 
                    original color scale image

            (output)
                - img_binary:
                    result binary image
    """

    # do something...
    img = cv2.resize(img_bgr, None, fx=0.5, fy=0.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    shape = h.shape

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


img_bgr = cv2.imread('./hands/dataset/001_1.JPG')
img_bin = preprocessingHSV(img_bgr)
plt.imshow(img_bin)
plt.show()