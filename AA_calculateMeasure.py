import numpy as np
import matplotlib.pyplot as plt


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


def getTAR(TP, FN):
    return 1 - getFRR( TP, FN )


def thresholdMeasures(G_score, I_scores, scale):
    # calculate tp and fn from the genuine matrix and fp_prim tn_prim ecc from the imposter matrix, for each threshold value of the selected scale
    # tp, fn = 0, 0 
    # genuine matrix -> tp, fn
    performance = []

    for thresh in np.linspace(0.0, 1.0, scale):
        measures = []

        tp = sum([sum(1 for i in pers if i < thresh) for pers in G_score])
        measures.append(tp)

        fn = len(G_score[0])*len(G_score) - tp
        measures.append(fn)

        for method in I_scores:

            fp = sum(1 for i in method if i < thresh)
            measures.append(fp)
            tn = len(method) - fp
            measures.append(tn)

        performance.append(measures)

    return performance


def calculatePerformanceMeasures(G_gdof, I_gdof, measure, scale, scores_path):
    # print(I_gdof)
    performance_gdof = []
    for f_type in range(len(G_gdof)):
        # print(f_type)
        performance = thresholdMeasures(G_gdof[f_type], [ x[f_type] for x in I_gdof], scale)
        performance_gdof.append(performance)

    np.save( scores_path + 'Measures_gdof_' + measure , performance_gdof)

    return performance_gdof


def calculatePerformanceParams(performance_gdof, measure, scale, scores_path):

    params_gdof = []
    for score in performance_gdof:
        params_s = [ [  getTAR(thresh[0], thresh[1]),
                        getFRR(thresh[0], thresh[1]), 
                        getFAR(thresh[2], thresh[3]), 
                        getFAR(thresh[4], thresh[5]), 
                        getFAR(thresh[6], thresh[7])   ]  for thresh in score]

        params_gdof.append(params_s)

    np.save( scores_path + 'Params_gdof_' + measure , params_gdof)

    return params_gdof





    