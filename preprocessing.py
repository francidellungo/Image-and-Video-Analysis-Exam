import imageio as iio
import cv2
import math
import numpy as np
from skimage import filters
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.color import label2rgb

# get one image
image1 = cv2.imread('./images/hand4.jpg', cv2.IMREAD_GRAYSCALE)
(threshold_value, image) = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# find center of mass
# threshold_value = filters.threshold_otsu(image)
print(threshold_value)
labeled_foreground = (image > threshold_value).astype(int)
properties = regionprops(labeled_foreground, image, coordinates='xy')
center_of_mass = properties[0].centroid
# weighted_center_of_mass = properties[0].weighted_centroid
major_axis_lenght = properties[0].major_axis_length
orientation = properties[0].orientation
print('centroid coord:',center_of_mass)
print('maj ax lenght: ', major_axis_lenght)
print('min ax lenght: ', properties[0].minor_axis_length)
print('orientation : ', orientation)


# draw img with center of mass / centroid 
colorized = label2rgb(labeled_foreground, image, colors=['black', 'red'], alpha=0.1)
fig, ax = plt.subplots()
ax.imshow(colorized)
# Note the inverted coordinates because plt uses (x, y) while NumPy uses (row, column)
ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
x2, y2 = [center_of_mass[1], center_of_mass[1]], [0, center_of_mass[0]*1.9]
ax.plot(x2, y2, marker = 'o', color= 'green')

# draw major axis 
x0, y0 = center_of_mass[1], center_of_mass[0]
x1 = x0 + math.cos(orientation) * 0.5 * properties[0].major_axis_length
y1 = y0 - math.sin(orientation) * 0.5 * properties[0].major_axis_length
ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5, color = 'red')

# draw minor axis 
x2 = x0 - math.sin(orientation) * 0.5 * properties[0].minor_axis_length
y2 = y0 - math.cos(orientation) * 0.5 * properties[0].minor_axis_length
ax.plot((x0, x2), (y0, y2), '-r', linewidth=2., color = 'yellow')

# cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# find center
im, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt_union = contours[0]
for i, cnt in enumerate(contours):
    if i > 0:
        cnt_union = np.concatenate((cnt_union, cnt), axis=0)
ellipse = cv2.fitEllipse(cnt_union)
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt_union)
print(x,y, MA, ma, angle)

# (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)


mean = [ x,y ]
width = MA
height = ma
ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle = angle, color = 'green', edgecolor='r', fc='None', lw=2)

ax.add_patch(ell)
ax.set_aspect('equal')
ax.autoscale()


x1 = x + math.cos(angle) * 0.5 * MA
y1 = y - math.sin(angle) * 0.5 * MA
ax.plot((x, x1), (y, y1), '-r', linewidth=2.5, color = 'green')

x2 = x - math.sin(angle) * 0.5 * ma
y2 = y - math.cos(angle) * 0.5 * ma
ax.plot((x, x2), (y, y2), '-r', linewidth=2.5, color = 'green')

plt.show()