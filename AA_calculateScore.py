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

