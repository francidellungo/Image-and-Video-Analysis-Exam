import math

def extractGeometricalFeatures(finger_points, medium_points):
    """ 
        GOAL:   
            the funciton extract 7 important distances: 
            the first five are calculated as the distance between i-finger_point and the relative medium_point;
            the sixth distance is the one between 3-medium_point (index finger) and 4-medium_point (thumb finger);
            the seventh distance is the one between 3-medium_point (index finger) and 0-medium_point (little finger)

        PARAMS:
            (input)
                - finger_points: 
                    list of finger points (xy coordinates)
                - medium_points:
                    list of medium_points (xy coordinates)

            (output)
                - distances:
                    list of the 7 distances calculated
                - geom_feature:
                    geometrical feature vector
    """

    distances = []
    for i in range(5):
        print(i, finger_points[i][0], medium_points[i])
        dist = math.hypot(finger_points[i][0][0] - medium_points[i][0], finger_points[i][0][1] - medium_points[i][1]) # Linear distance
        distances.append(dist)
    # dist_6 : distance between 3-medium_point (index finger) and 4-medium_point (thumb finger)
    x_med_point_3,  y_med_point_3 = medium_points[3][0], medium_points[3][1]
    x_med_point_4,  y_med_point_4 = medium_points[4][0], medium_points[4][1]

    dist_6 = math.hypot(x_med_point_3 - x_med_point_4, y_med_point_3- y_med_point_4)
    distances.append(dist_6)

    # dist_7 : distance between 3-medium_point (index finger) and 0-medium_point (little finger)
    x_med_point_3,  y_med_point_3 = medium_points[3][0], medium_points[3][1]
    x_med_point_0,  y_med_point_0 = medium_points[0][0], medium_points[0][1]

    dist_7 = math.hypot(x_med_point_3 - x_med_point_0, y_med_point_3- y_med_point_0)
    distances.append(dist_7)
    
    # calculate geometrical feature vector
    geom_feature = constructGeometricalFeatureVector(distances)

    return distances, geom_feature


def constructGeometricalFeatureVector(distances):
    """ 
        GOAL:   
            extract the geometrical feature vector from the distances previously calculated

        PARAMS:
            (input)
                - distances: 
                    list of distances

            (output)
                - geom_feature:
                    list of geometrical feature 
    """
    geom_feature = []
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            geom_feature.append(distances[i]/distances[j])

    return geom_feature