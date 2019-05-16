import numpy as np
import math
from pywt import wavedec

mu = 10**(-10)

def distanceMap(cnt, r_idx):
    """ 
    GOAL:   
        generate a distance map from each points on contour
        and reference point r, in clockwise orientation.

    PARAMS:
        (input)
        - cnt: 
            contour of hand
        - r_idx: 
            index of reference point on the contour

        (output)
        - dp: 
            distance map of each points in clockwise orientation
    """
    # print('distanceMap: ', r_idx)

    # distance_map = []
    # for point in cnt:
    #     d_value = math.sqrt((cnt[r_idx][0][1] - point[0][1])**2 + (cnt[r_idx][0][0] - point[0][0])**2)
    #     distance_map.append(d_value)
    # max_value = np.max(distance_map)
    
    # print(max_value)
    # # normalize distance map 
    # distance_map = [i/max_value for i in distance_map]
    # print(np.max(distance_map))

    return distance_map

def orientationMap(cnt, r_idx):
    """ 
    GOAL:   
        generate a orientation map from each points on contour
        and reference point r, in clockwise orientation.

    PARAMS:
        (input)
        - cnt: 
            contour of hand
        - r_idx: 
            index of reference point on the contour

        (output)
        - orientation_map: 
            orientation map of each points in clockwise orientation
    """
    

    # sigma = 10**-10
    # orientation_map = []
    # for i in range(len(cnt)):
    #     point = cnt[(r_idx + i)%len(cnt)]

    #     if point[0][1] > cnt[r_idx][0][1]:
    #         point[0][1] = cnt[r_idx][0][1]
    #     otan2_value = np.pi/2 + np.arctan2([ cnt[r_idx][0][1] - point[0][1]], [cnt[r_idx][0][0] - point[0][0] + sigma])

    #     orientation_map.append( np.rad2deg(float(otan2_value) ))

    return orientation_map




dp = distanceMap(cnt, r_idx)
op = orientationMap(cnt, r_idx)

plt.plot(range(len(updated_contour)), dp, 'b--', label="distance map")
# plt.scatter(r_based_fingers_indexes, [ dm_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
# plt.scatter(valley_indexes, [ dm_u[idx] for idx in valley_indexes], c='g', label='valley points')
# plt.scatter(comp_valley_indexes, [ dm_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
#plt.savefig(hand_base + dist_path + new_name_img + '_dmap_update.png')
plt.close()

plt.plot(range(len(updated_contour)), op, 'b--', label="orientation map")
# print(r_based_fingers_indexes, dm)
# plt.scatter(r_based_fingers_indexes, [ om_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
# plt.scatter(valley_indexes, [ om_u[idx] for idx in valley_indexes], c='g', label='valley points')
# plt.scatter(comp_valley_indexes, [ om_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
# plt.savefig(hand_base + dist_path + new_name_img + '_omap_update.png')
plt.close()