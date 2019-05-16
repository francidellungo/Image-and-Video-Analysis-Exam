import numpy as np
import math
from pywt import wavedec


from Exercise4.shapeFeaturesExtraction import distanceMap, orientationMap


mu = 10**(-10)

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
    # print(m, p)
    phi = np.rad2deg(np.arctan((m[1]-p[1])/(m[0]-p[0] + mu)))
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
    # print('psi angle:', psi)
    if psi < -90 or psi > 90:
        psi = 180 + psi
    psi = np.deg2rad(psi)
    
    # number of elements whose new coordinates are calculated as an average mean between old points and new one
    n_smooth_el = int(((v_idx-c_idx+len(cnt))%len(cnt))/6)
    
    # print('c_idx, v_idx:', c_idx, v_idx)
    c_smooth_index = (c_idx + n_smooth_el)
    v_smooth_index = v_idx - n_smooth_el

    # mooving all points between c_idx and v_idx on angle psi
    new_points = [ [[ m[0] + (point[0][0]-m[0])*np.cos(psi) - (point[0][1]-m[1])*np.sin(psi) , m[1] + (point[0][0]-m[0])*np.sin(psi) + (point[0][1]-m[1]) * np.cos(psi)]] for point in cnt[ c_idx : v_idx ] ]
    
    # print(len(cnt))
    # print(n_smooth_el)
    broken = False
    
    for i in range(n_smooth_el):
        alpha = i/n_smooth_el
        # print(cnt[v_idx+i][0][0], new_points[i][0])

        # print(v_idx-i, c_idx+i)
        try:
        
            cnt[v_idx-i][0][0] = alpha*new_points[-i][0][0] + (1-alpha) * cnt[v_idx-i][0][0]
            cnt[v_idx-i][0][1] = alpha*new_points[-i][0][1] + (1-alpha) * cnt[v_idx-i][0][1]

            cnt[c_idx+i][0][0] = alpha*new_points[i][0][0] + (1-alpha) * cnt[c_idx+i][0][0]
            cnt[c_idx+i][0][1] = alpha*new_points[i][0][1] + (1-alpha) * cnt[c_idx+i][0][1]
        
        except:

            print("Check imgages mask, it seems damaged! ")
            broken = True
            break


    # changing original contour points with the new found
    # print(len(cnt[ c_idx : v_idx ]), len(cnt[ c_smooth_index : v_smooth_index ]), len(new_points[n_smooth_el: len(cnt[ c_idx : v_idx ])-n_smooth_el]))
    
    if not broken:
        cnt[ c_smooth_index : v_smooth_index ] = new_points[n_smooth_el: len(cnt[ c_idx : v_idx ])-n_smooth_el]

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
        - center_of_mass:
            center of mass coordinates
        - p_list: 
            list of finger points coordinates
        - m_list: 
            list of medium points coordinates
        - c_list: 
            list of indexes of complementary valley points of the 5 finger points
        - v_list: 
            list of indexes of valley points of the 5 finger points

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



def waveletDecomposition(features_map):
    """ 
    GOAL:   
        extract coefficients through waveletDecomposition Daubechies 1 (db1)

    PARAMS:
        (input)
        - features_map: 
            feature map (orientation map or distance map)

        (output)
        - coeffs:
            coefficients given by wavedec function
    """
    # coeffs = wavedec(features_map, 'db5')
    # wavelet = families()
    # print(wavelet[0])
    coeffs = wavedec(features_map, 'db1', mode='symmetric', level=5, axis=-1)
    return coeffs


def extractShapeFeatures(cnt, r_idx):
    """ 
    GOAL:   
        extract shape features (50 coefficients for distance_features and orientation_features)

    PARAMS:
        (input)
        - cnt: 
            contour of hand
        - r_idx: 
            index of reference point on the contour

        (output)
        - distance_features:
            first 50 coefficients given by waveletDecomposition applied to distanceMap
        - orientation_features: 
            first 50 coefficients given by waveletDecomposition applied to orientationMap
    """
    dp = distanceMap(cnt, r_idx)
    op = orientationMap(cnt, r_idx)
    
    distance_coeffs = waveletDecomposition(dp)
    orientation_coeffs = waveletDecomposition(op)
    distance_features = distance_coeffs[0][:50]
    orientation_features = orientation_coeffs[0][:50]

    return distance_features, orientation_features, dp, op
