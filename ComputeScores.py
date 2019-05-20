from sklearn.metrics import pairwise_distances, pairwise
import numpy as np
from cv2 import *
import matplotlib.pyplot as plt



def calculateGenuineCentroids(genuine_scores_list):
    indexes = []

    for matrix in genuine_scores_list:

        indexes.append(np.argmin( np.matrix(matrix).mean(1) ) )

    return indexes


def allScores(scores, num_imgs, cod, measure):

    scores = np.array( scores )
    print('scores', scores.shape)
    row_scores = scores.reshape( -1, scores.shape[-1], order='F' )
    print('row', row_scores.shape)
    if cod == 0:
        big_matrix = pairwise_distances( row_scores , metric=measure )

    else:
        big_matrix = 1 - pairwise.chi2_kernel( row_scores ) 

    big_centroids_indexes = allIndexes( num_imgs, big_matrix )

    return row_scores, big_matrix, big_centroids_indexes


def allIndexes(num_imgs, big_matrix):

    big_centroids_indexes = calculateGenuineCentroids([ big_matrix[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))] for s in range(0, big_matrix.shape[0], num_imgs) ])

    big_centroids_indexes = [ index + i*num_imgs for i, index in enumerate(big_centroids_indexes)]

    return big_centroids_indexes


def matrixNormalization(matrix):

    maxi = matrix.max() 
    # print(maxi * np.diag( np.ones(matrix.shape[0])))
    mini = matrix.min() 
    # print('\n\n\n')
    # print('maxi: ', maxi, 'mini: ', mini)
    # print( matrix )

    matrix = (matrix - mini) / (maxi - mini)

    # print( matrix )

    return matrix, mini, maxi


def matrixNormalizationMiniMaxi(matrix, mini, maxi):

    matrix = (matrix - mini) / (maxi - mini)

    return matrix


def measureCalculate(matrix, mask, num_imgs, centroid_indexes):
    tp, fp, tn, fn = 0, 0, 0, 0

    for s in range(0, matrix.shape[0], num_imgs):

        unique_diag, counts_diag = np.unique( matrix[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))], return_counts=True)
        # print('diag 1: ',dict(zip(unique_diag, counts_diag)).get(1, 0), '0: ',dict(zip(unique_diag, counts_diag)).get(0, 0))

        tp = tp + dict(zip(unique_diag, counts_diag)).get(1, 0) - num_imgs
        fn = fn + dict(zip(unique_diag, counts_diag)).get(0, 0)

        
        unique_over, counts_over = np.unique( matrix[np.ix_(range(s, s+num_imgs), range(s+num_imgs, matrix.shape[0]))], return_counts=True)
        # print('other 1: ',dict(zip(unique_over, counts_over)).get(1, 0), '0: ',dict(zip(unique_over, counts_over)).get(0, 0))

        fp = fp + dict(zip(unique_over, counts_over)).get(1, 0)
        tn = tn + dict(zip(unique_over, counts_over)).get(0, 0)

    imposer_matrix = np.matrix(matrix[np.ix_(centroid_indexes, centroid_indexes)])

    m_i, v_i = np.mean( imposer_matrix ), np.std( imposer_matrix ) ** 2
    m_g, v_g = np.mean( matrix ), np.std( matrix ) ** 2

    # print(np.array([tp, fp, tn, fn, m_i, v_i, m_g, v_g]))

    return np.array([tp, fp, tn, fn, m_i, v_i, m_g, v_g])


def getFAR( FP, TN ):
    # ( FAR : False Accept Rate )

    """
    assume you have a biometric evaluation system that assigns all authentication attempts a 'score' between closed interval [0, 1].
    0 means no match at all and 1 means a full match. If the threshold is set to 0, then all the users including the genuine (positive) 
    and the impostors (negative) are authenticated. If you threshold is set to 1 then there is a high risk that no one may be authenticated.
    Therefore, in realtime systems the threshold is kept somewhere between 0 and 1. So, this threshold setting can sometimes may not authenticate the genuine users, 
    which is called FRR (False Reject Rate) but may also authenticate the imposters, which is given by FAR (False Accept Rate).

    Here, FP: False positive, FN: False Negative, FN: True Negative and TP: True Positive

    FAR is calculated as a fraction of negative scores exceeding your threshold.
    FAR = imposter scores exceeding threshold/all imposter scores.
    imposter scores exceeding threshold = FP
    all imposter scores = FP+TN
    FAR = FPR = FP/(FP+TN)

    FRR is calculated as a fraction of positive scores falling below your threshold.
    FRR = genuines scores exceeding threshold/all genuine scores
    genuines scores exceeding threshold = FN
    all genuine scores = TP+FN
    FRR = FNR = FN/(TP+FN)
    """
    FAR = FP/(FP+TN)

    return FAR
    

def getFRR( TP, FN ):
    # ( FRR : False Reject Rate )
    FRR = FN/(TP+FN)

    return FRR


def getTAR( FRR ):

    return 1 - FRR

def get_TAR(TP, FN):
    return 1 - getFRR( TP, FN )


def getDI( array ):
    m_i, ds_i, m_g, ds_g = tuple(array)
    # print(tuple(array))
    return np.abs(m_g - m_i)/np.sqrt((ds_i ** 2 + ds_g ** 2)/2)
    

def threshPerformanceMeasure(matrix_distances, num_imgs, centroid_indexes, scale):

    performance_measure = np.zeros((scale, 8))

    for i, thresh in enumerate(np.linspace(0.0, 1.0, scale)):
            matrix_thresh = np.where(matrix_distances < thresh, 1, 0) 
            matrix_mask = np.zeros((matrix_thresh.shape[0], matrix_thresh.shape[1]))
            
            for s in range(0, matrix_mask.shape[0], num_imgs):
                    matrix_mask[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))] = np.ones((num_imgs, num_imgs))

            performance_measure[i] = measureCalculate(matrix_thresh, matrix_mask, num_imgs, centroid_indexes)

    return performance_measure


def threshPerformanceParams(matrix_distances, num_imgs, centroid_indexes, scale):

    performance_measure = threshPerformanceMeasure(matrix_distances, num_imgs, centroid_indexes, scale)

    # print(performance_measure)

    performance_params = [  [   getFAR(thresh[1], thresh[2]), 
                                getFRR(thresh[0], thresh[3]), 
                                getTAR(getFRR(thresh[0], thresh[3])) , 
                                getDI(thresh[4::])    ]  for thresh in performance_measure]
    
    performance_params = np.array(performance_params)
    where_are_NaNs = np.isnan(performance_params)
    performance_params[where_are_NaNs] = 0

    # print(performance_params)
    
    return performance_params

