import cv2
import numpy as np
import os
from math import ceil

# print(cv2.__version__)

def draw(img, contour, cnt_color, point_lists, point_list_colors = [255,255,255], text_colors = [255,255,255]):
	
	if img is None:
		shape = getShape(contour, 150, 50)
		img = np.zeros((shape[0], shape[1], 3), np.uint8)

	if len(contour) > 0:
		cv2.drawContours(img, contour, -1, cnt_color if cnt_color else (0, 255, 0), 1, 8)   # green color

	if len(point_lists) > 0:
		for i, point in enumerate(point_lists):
			xy = tuple([ int(x) for x in point[0] ])
			cv2.circle(img,xy,5, point_list_colors,-1)

	if text_colors:
		font = cv2.FONT_HERSHEY_SIMPLEX
		for i, point in enumerate(point_lists):
			xy = tuple([ int(x) for x in point[0] ])
			cv2.putText(img, 'r', xy, font, 1, text_colors ,2,cv2.LINE_AA)
	
	return img


def rotateHand(shape, contour, angle, centre_of_mass, fingers_indexes, valley_indexes):
	""" 
		GOAL:   
			the function rotate the contour of the hand mask, the coordinates of center of mass, 
			finger points and valley points of the given angle 

		PARAMS:
			(input)
				- shape: 
					shape of the hand mask 
				- contour:
					contour of the hand mask
				- angle:
					angle in rad of which the image has to be rotated
				- center_of_mass:
					(x,y) coordinates of the center of mass
				- fingers_indexes:
					contour indexes of the 5 fingers (clockwise order starting from little finger)
				- valley_indexes:
					contour indexes of the valleys (same order as fingers indexes)


			(output)
				- hand_mask_rotated:
					rotated hand mask
				- finger_points_rotated:
					new x,y coordinates of each point after rotation
				- valley_points_rotated:
					new x,y coordinates of valley points after rotation
				- np.array(contour_rotated):
					new contour array with rotated components
				- np.add(centre_of_mass, (0, 50):
					updated center of mass coordinates
	"""

	# create an empty black image
	xm, ym = centre_of_mass
	
	angle = 90 - angle
	if angle < -90 or angle > 90:
		angle = 180 + angle
	angle = np.deg2rad(angle)

	# print(np.asanyarray(contour))

	contour_rotated = [ [[ int(xm + (point[0][0]-xm)*np.cos(angle) - (point[0][1]-ym)*np.sin(angle)) , int(ym + 50 +(point[0][0]-xm)*np.sin(angle) + (point[0][1]-ym) * np.cos(angle))]] for point in contour ]

	# print(np.asanyarray(contour_rotated))

	hand_mask_rotated = np.zeros(getShape(contour_rotated, 150, 50), np.uint8)

	# create polylines from contour points, and draw if filled in image
	cv2.fillPoly(hand_mask_rotated, pts = np.array([contour_rotated], dtype=np.int32), color=(255))
	# print(list(fingers_indexes))
	# print(np.array(list(valley_indexes)))
	# print(np.array(list(fingers_indexes)).astype(int))

	finger_points_rotated =  [ contour_rotated[i] for i in fingers_indexes]
	valley_points_rotated = [ contour_rotated[i] for i in valley_indexes]

	return hand_mask_rotated, finger_points_rotated, valley_points_rotated, np.array(contour_rotated), np.add(centre_of_mass, (0, 50))


def getShape(contour, w_bias, h_bias):

	x = [ point[0][0] for point in contour]
	y = [ point[0][1] for point in contour]

	width  = int( np.max(x) + w_bias)
	heigth = int( np.max(y) + h_bias)
	
	return tuple([heigth, width])


def countPeoplePhoto(path):
        paths = os.listdir(path)

        d = dict()

        for name_img in paths:
                
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                
                if person_idx in d:
                        d[person_idx].append(img_idx)
                else:
                        d[person_idx] = [ img_idx ]
                
        # print(d.values())

        d1 = np.array([ len(elem) for elem in list(d.values())])
        if d1.max() > d1.min():
                print('people has different number of photos, check it!')
                n_person, n_imgs = None, None
        else:
                # print('same number of photos')
                n_person, n_imgs = len(d.values()), len(list(d.values())[0])
        
        return n_person, n_imgs

def line(x0, y0, x1, y1):
	"""Bresenham's line algorithm:
		it takes as input two points as (x0,y0) and (x1,y1) 
		and returns a list of all the points of the line connecting them"""

	points_in_line = []
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	x, y = x0, y0
	sx = -1 if x0 > x1 else 1
	sy = -1 if y0 > y1 else 1
	if dx > dy:
		err = dx / 2.0
		while x != x1:
			points_in_line.append((x, y))
			err -= dy
			if err < 0:
				y += sy
				err += dx
			x += sx
	else:
		err = dy / 2.0
		while y != y1:
			points_in_line.append((x, y))
			err -= dx
			if err < 0:
				x += sx
				err += dy
			y += sy
	points_in_line.append((x, y))
	return points_in_line


def get_numbers(sequence, number):
	# get N elements from the sequence of elements normally distributed
	length = float(len(sequence))
	lista = []
	for i in range(number):
		lista.append(sequence[int(ceil(i * length / number))])
	return lista



def get_numbers_with_idx(start_idx, end_idx, number):
	# get N elements from the sequence of elements normally distributed, 
	# Here the sequence is given as the start element and the final one (we consider the sequence as complete and sorted)
	sequence = [i for i in range(start_idx, end_idx+1)]
	return get_numbers(sequence, number)

def distance(alist, blist):
    sum_of = 0
    for x, y in zip(alist, blist):
        ans = (x - y)**2
        sum_of += ans
    return (sum_of)**(1/2)
