
def getReferencePoint(xy_middle_finger, xy_centroid, hand_mask):

    # coordinates of reference point are calculated as the point of intersection between the line
    #  passing by center of mass and middle finger point and the end of the image (y)
    y_reference_point = hand_mask.shape[0]
    x_reference_point = ((y_reference_point-xy_centroid[1])/(xy_middle_finger[1]-xy_centroid[1]))*(xy_middle_finger[0]-xy_centroid[0]) + xy_centroid[0]

    reference_point = [x_reference_point, y_reference_point]
    return reference_point










