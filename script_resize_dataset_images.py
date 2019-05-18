import cv2
import numpy as np
import os

downscale_factor = 3.0
path = './'
files = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.JPG' in file:
            files.append(os.path.join(r, file))

for f in files:
    print('Processing %s'%(f))
    img = cv2.imread(f, -1)
    img_downsampled = cv2.resize(img,(np.uint16(img.shape[1]/downscale_factor),
                            np.uint16(img.shape[0]/downscale_factor) ) )
    cv2.imwrite(f, img_downsampled)
    
