import cv2
import numpy as np

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
			cv2.putText(img, str(i), xy, font, 2, text_colors ,2,cv2.LINE_AA)
	
	return img


def rotateHand(shape, contour, angle, centre_of_mass, fingers_indexes, valley_indexes):
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
	print(np.array(list(fingers_indexes)).astype(int))

	return hand_mask_rotated, [ contour_rotated[i] for i in fingers_indexes], [ contour_rotated[i] for i in valley_indexes], np.array(contour_rotated)


def getShape(contour, w_bias, h_bias):

	x = [ point[0][0] for point in contour]
	y = [ point[0][1] for point in contour]

	width  = int( np.max(x) + w_bias)
	heigth = int( np.max(y) + h_bias)
	
	return tuple([heigth, width])