import numpy as np
import math
from pywt import wavedec

W = [
    +20, # j = 0 little finger
    +10, # j = 1 ring finger
    -10, # j = 2 medium finger
    -30, # j = 3 index finger
    -60  # j = 4 thumb finger
]

K = [
    +70, # j = 0 little finger
    +80, # j = 1 ring finger
    -80, # j = 2 medium finger
    -60, # j = 3 index finger
    -30  # j = 4 thumb finger
]

def getAngle(p, m):
    """ 
    GOAL:   
        the function returns angle between segment from p to m 
        and the vertical axis (y) passed by m point.

    PARAMS:
        (input)
        - p: 
            (x, y) convention.
            point on the contour of hand, it is the 
            reference point of a particular finger.
        - m: 
            (x, y) convention.
            point into the contour of hand, it is 
            middle point between v and c points
            (see reference to more detailes).


        (output)
        - phi: 
            angle between segment from p point to m point
            and the vertical axis (y) passed by m point.
    """

    # returns angle in the 1 and 4 quadrant, then we need to see negative angle
    # Its real part is in [-pi/2, pi/2]
    m = m[0]
    p = p[0]
    print(m, p)
    phi = np.rad2deg(np.arctan((m[1]-p[1])/(m[0]-p[0])))
    return phi


def allignFinger(cnt, m, idx, phi, c_idx, v_idx):
    """ 
    GOAL:   
        the function returns cnt with j-th fingers 
        contour from c_idx to v-idx modified based on 
        phi angle and rispective defined angles W i-th.

    PARAMS:
        (input)
        - cnt: 
            contour of hand
        - idx:
            index of finger (i.e. 0 is little finger)
        - phi: 
            angle between segment from p point to m point
            and the vertical axis (y) passed by m point.
        - c_idx:
            index of c point for the i-th finger
        - v_idx:
            index of v point for the i-th finger
        - m: 
            (x, y) convention.
            point into the contour of hand, it is 
            middle point between v and c points
            (see reference to more detailes).


        (output)
        - cnt: 
            contour of hand with one finger modified points
    """
    m = m[0]

    # calculate psi based on paper method
    psi = K[idx] - phi
    print('psi angle:', psi)
    if psi < -90 or psi > 90:
        psi = 180 + psi
    psi = np.deg2rad(psi)
    
    # number of elements whose new coordinates are calculated as an average mean between old points and new one
    n_smooth_el = int(abs(c_idx-v_idx)/6)
    
    print('c_idx, v_idx:', c_idx, v_idx)
    new_c_index = c_idx + n_smooth_el
    new_v_index = v_idx - n_smooth_el
    print('new_c_idx, new_v_idx:', new_c_index, new_v_index)
    # mooving all points between c_idx and v_idx on angle psi
    new_points = [ [[ m[0] + (point[0][0]-m[0])*np.cos(psi) - (point[0][1]-m[1])*np.sin(psi) , m[1] + (point[0][0]-m[0])*np.sin(psi) + (point[0][1]-m[1]) * np.cos(psi)]] for point in cnt[ c_idx : v_idx ] ]
    
    for i in range(n_smooth_el):
        alpha = i/n_smooth_el
        print(cnt[v_idx+i][0][0], new_points[i][0])
        
        cnt[v_idx-i][0][0] = alpha*new_points[-i][0][0] + (1-alpha) * cnt[v_idx-i][0][0]
        cnt[v_idx-i][0][1] = alpha*new_points[-i][0][1] + (1-alpha) * cnt[v_idx-i][0][1]

        cnt[c_idx+i][0][0] = alpha*new_points[i][0][0] + (1-alpha) * cnt[c_idx+i][0][0]
        cnt[c_idx+i][0][1] = alpha*new_points[i][0][1] + (1-alpha) * cnt[c_idx+i][0][1]

    # changing original contour points with the new found
    print(len(cnt[ c_idx : v_idx ]), len(cnt[ new_c_index : new_v_index ]), len(new_points[n_smooth_el: len(cnt[ c_idx : v_idx ])-n_smooth_el]))
    
    cnt[ new_c_index : new_v_index ] = new_points[n_smooth_el: len(cnt[ c_idx : v_idx ])-n_smooth_el]
    
    return cnt


def fingerRegistration(cnt, center_of_mass, p_list, m_list, c_list, v_list):
    """ 
    GOAL:   
        the function update contour preparing it for 
        distance and orientation map, and after wavelet. 

    PARAMS:
        (input)
        - cnt: 
            contour of hand
        - p_list: 
            list of p points (or indexes of points)
        - m_list: 
            list of p points (or indexes of points)
        - c_list: 
            list of indexes of points
        - v_list: 
            list of indexes of points

        (output)
        - cnt: 
            contour of hand with modified points
    """

    # origin = fingerAngle(p_list[2], [center_of_mass])
    # print(origin)

    # if origin < 0:
    #     origin = 90 + origin
    # else:
    #     origin = - (90 - origin)

    for i, (p, m) in enumerate(zip(p_list, m_list)):
        
        # calculate phi angle
        phi = getAngle(p, m)
        
        # update partial contour on one finger angle update
        cnt = allignFinger(cnt, m, i, phi, c_list[i], v_list[i])
    
    return cnt


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

    dp = []
    for point in cnt:
        # point = cnt[i] # cnt[(r_idx + i)%len(cnt)]
        d_value = math.sqrt((cnt[r_idx][1] - point[1])**2 + (cnt[r_idx][0] - point[0])**2)  
        dp.append(d_value)

    return dp

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
        - op: 
            orientation map of each points in clockwise orientation
    """

    sigma = 10**-10
    op = []
    for i in range(len(cnt)):
        point = cnt[(r_idx + i)%len(cnt)]
        o_value = 90 + np.arctan((cnt[r_idx][0] - point[0])/(cnt[r_idx][1] - point[1] + sigma))
        op.append(o_value)

    return op


def waveletDecomposition(features_map):
    coeffs = wavedec(features_map, 'db5')
    print(coeffs)

