import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import copy
import os
path = './eroded/'
paths = os.listdir(path)

for name_img in paths:

    img = cv2.imread(path+name_img, -1)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(img)

    # Smooth the three color channels one by one
    b = cv2.medianBlur(b,5)
    g = cv2.medianBlur(g,5)
    r = cv2.medianBlur(r,5)

    num_clusters = 2
    # Warning: X is 3xNum_pixels. To fit the kmeans model X.T should be used
    X = np.array([b.reshape(-1), g.reshape(-1), r.reshape(-1)])
    gmm=GaussianMixture(n_components=num_clusters,
                    covariance_type='full',
                    init_params='kmeans',
                    max_iter=300, n_init=4, random_state=10)
    gmm.fit(X.T)
    # extract the cluster ID of each pixel
    Y = gmm.predict(X.T)

    b_remap = copy.deepcopy(b.reshape(-1))
    g_remap = copy.deepcopy(g.reshape(-1))
    r_remap = copy.deepcopy(r.reshape(-1))
    for k in range(num_clusters):
        b_remap[ Y==k ] = gmm.means_[k,0]
        g_remap[ Y==k ] = gmm.means_[k,1]
        r_remap[ Y==k ] = gmm.means_[k,2]

    img_remap = cv2.merge( (b_remap.reshape(r.shape),
                            g_remap.reshape(r.shape),
                            r_remap.reshape(r.shape)) )

    img_remap = cv2.cvtColor(img_remap, cv2.COLOR_HSV2RGB)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))

    img_OPEN = cv2.morphologyEx(img_remap, cv2.MORPH_OPEN, kernel)

    plt.subplot(1,2,1), plt.imshow(img_remap)
    plt.title('base'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2), plt.imshow(img_OPEN)
    plt.title('img_OPEN'), plt.xticks([]), plt.yticks([])
    plt.show()
