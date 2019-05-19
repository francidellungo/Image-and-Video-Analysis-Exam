import numpy as np
from sklearn.metrics import pairwise_distances, pairwise

def getScores(features, cod, measure):
    
    features = np.array( features )
    print('features', features.shape)

    if cod == 0:
        matrix = pairwise_distances( features , metric=measure )

    else:
        matrix = 1 - pairwise.chi2_kernel( features ) 

    scores = sum([ x[i+1:].tolist() for i, x in enumerate(matrix)], [])

    return scores


def ScoresNormalization(scores, norm):
    norm_scores = []
    for f_type, features in enumerate(scores):
        new_ele = []
        for ele in features:
            # print('ele ', ele)
            new_ele.append((ele - norm[f_type]['min'])/(norm[f_type]['max'] - norm[f_type]['min']))
            # print('new_ele ', (ele - norm[f_type]['min'])/(norm[f_type]['max'] - norm[f_type]['min']))
        norm_scores.append(new_ele)

    # print(norm_scores)

    return norm_scores