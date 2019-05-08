from sklearn.metrics import pairwise_distances, pairwise
import numpy as np
from cv2 import *

thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1]

def calculateScores(scores, cod, measure):
    n_people, n_imgs = len(scores[0]), len(scores)
    thresholds = 0.4

    # poi ciclerò fra tutte le soglie..

    # calculate genuine distances for all people for all features and all distances measures ( ... )
    
    genuine_scores_list = calculateGenuineDistances(scores, cod, measure)
    centroids_indexes = calculateGenuineCentroids(genuine_scores_list)
    imposter_scores_list = calculateImposterDistances(np.array([ scores[centroids_indexes[i]][i] for i in range(len(scores[0]))]), cod, measure )


    # y_geom_genuine_predicted = [ [] for x in range(n_people) ]

    # # find predicted value (1 if images is correctly associated to the )
    # for person in range(n_people):
    #     for i in range(len(geom_L1_dist_new[person])):
    #         y_geom_genuine_predicted[person].append(1) if geom_L1_dist_new[person][i] < thresholds else y_geom_genuine_predicted[person].append(0)
    
    # # y_geom_genuine_predicted = [ 1  for i in range(len(geom_L1_dist_new)) if (geom_L1_dist_new[i] < thresholds) ]
    
    # # print(y_geom_genuine_predicted)

    return genuine_scores_list, imposter_scores_list, centroids_indexes


def calculateGenuineDistances(scores, cod, measure):
    ### calculate distances between hands of the same person ( genuine scores ) ###
    genuine_scores_list = []

    n_people = len(scores[0])
    print('people:', n_people)

    if cod == 0:
        for i in range(n_people):
            genuine_scores_list.append( pairwise_distances(scores[:,i] , metric=measure ) )

    else:
        for i in range(n_people):
            genuine_scores_list.append( pairwise.chi2_kernel( np.array(scores[:,i]) ) )

    """
    unique_list_geom_L1 = []
    for person in range(n_people):
        for i in range(len(geom_L1_dist_new[person])):
            unique_list_geom_L1.append(geom_L1_dist_new[person][i])


    print(' ')
    
    max_value = np.max(unique_list_geom_L1, axis=0)
    geom_L1_dist_new /= max_value
    """


    return genuine_scores_list
    


def calculateImposterDistances(scores, cod, measure ):
    ### calculate distances between one hand of each person ( imposter scores ) ###
    imposter_scores_list = []

    if cod == 0:
        imposter_scores_list.append(pairwise_distances(scores, metric=measure ) )
    else:
        imposter_scores_list.append( pairwise.chi2_kernel( scores ) )

    return imposter_scores_list



def calculateGenuineCentroids(genuine_scores_list):
    indexes = []
    for matrix in genuine_scores_list:
        indexes.append(np.argmin( np.matrix(matrix).mean(1) ) )
    print('len idexes:', len(indexes))

    return indexes

def performanceMeasure(y_real, y_predicted):
    """ 
    GOAL:   
        the function returns number of True Positives (TP), False Positives (FP),
        True Negatives (TN) and False Negatives (FN) elements.

    PARAMS:
        (input)
        - y_real: 
            actual values of y vector. ( 1 means Positive )
        - y_predicted:
            predicted values of y vector.

        (output)
        - TP:
            number of True Positives elements
        - FP:
            number of False Positives elements
        - TN:
            number of True Negatives elements
        - FN:
            number of False Negatives elements
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predicted)): 
        if y_real[i]==y_predicted[i]==1:
           TP += 1
        if y_predicted[i]==1 and y_real[i]!=y_predicted[i]:
           FP += 1
        if y_real[i]==y_predicted[i]==0:
           TN += 1
        if y_predicted[i]==0 and y_real[i]!=y_predicted[i]:
           FN += 1
    
    return TP, FP, TN, FN

def getFAR(FP, TN):
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
    
def getFRR(TP, FN):
    # ( FRR : False Reject Rate )
    FRR = FN/(TP+FN)

    return FRR

