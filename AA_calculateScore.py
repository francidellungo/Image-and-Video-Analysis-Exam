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

def calculateFusion(G_norm, I_norm, scores_path, measure):
    g, d, o = G_norm
    pd = np.multiply(np.array(d), np.array(o))
    G_fusion = np.minimum(np.array(pd), np.array(g))
    G_norm = np.append(G_norm, [G_fusion], axis=0)
    
    I_new = []
    for i, method in enumerate(I_norm):
        g, d, o = method
        sm = np.add(np.array(d), np.array(o))
        I_fusion = np.maximum(sm, np.array(g))
        print(I_fusion)
        print(I_norm[i])
        I_new.append(np.append(I_norm[i], [I_fusion], axis=0).tolist())

    np.save( scores_path + 'G_gdof_' + measure , G_norm)
    np.save( scores_path + 'I_gdof_' + measure , np.array(I_new))

    return G_norm, I_norm