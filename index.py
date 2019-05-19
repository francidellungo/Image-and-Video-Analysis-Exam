from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from AA_saveScores import saveScores, getFeatureVectors
from Utils import *
from geometricalFeaturesExtraction import *
from ComputeScores import *
import matplotlib.pyplot as plt
import copy
import json
from operator import itemgetter

NUM_IMGS = 5

hand_base = './hands/'
path_in = 'dataset/'
path_out = 'masks/'
path_rot = 'rotates/'
path_pts = 'points/'
path_ell = 'ellipses/'

pickle_base = './pickles/'
scores_path = 'scores/'
params_path = 'params/'
norms_path = 'norms/'
row_path = 'rows/'
dist_path = 'dist_map/'
thresholds = pickle_base

path_figs = './figures/'
path_test = './tests/'
hand_path = 'hands/'
pickle_path = 'pickles/'

# paths = os.listdir(path_in)

colours = [
            [ (0, 0, 255)    ,   'blue'        ],
            [ (0,128,0)      ,   'green'       ],
            [ (173,255,47)   ,   'greenyellow' ],
            [ (255, 160, 0)  ,   'orange'      ],
            [ (0,238,0)      ,	'green2'       ],
            [ (250, 246, 0)  ,   'yellow'      ],
            [ (0,205,0)      ,   'green3'      ],
            [ (255, 186, 230),   'pink'        ],
            [ (255, 69, 230) ,   'purple'      ],
            [ (0,255,0)      ,	'green1'       ],	 
            [ (0,139,0)      ,   'green4'      ]	
]


measures = [
            ( 0  ,  ( 'l1'       , 'l1' ) ),
            ( 0  ,  ('euclidean' , 'euc') ),
            ( 0  ,  ('cosine'    , 'cos') ),
            ( 1  ,  ('chi-square', 'chi') )
]


def saveMatrix(scores, measures, pickle_base, norms_path, row_path, num_imgs):

        for cod, (measure, mea) in measures:

                for score, (l, file_name) in scores:
                
                        row_scores, matrix_distances, centroids_indexes = allScores(score, num_imgs, cod, measure)

                        matrix_distances_norm, mini, maxi = matrixNormalization(matrix_distances)

                        np.save(pickle_base + norms_path + file_name + '_' + mea, (centroids_indexes, matrix_distances_norm))
                        np.save(pickle_base + row_path + file_name + '_' + mea, (centroids_indexes, row_scores))
                        np.save(pickle_base + 'mini_maxi/' + file_name + '_' + mea, (mini, maxi))


def saveParams(scores, measures, num_imgs, pickle_base, params_path, norms_path, scale):

        for cod, (measure, mea) in measures:

                # matrixes = []

                for l, file_name in scores:
                        
                        centroids_indexes, matrix_distances_norm = np.load( pickle_base + norms_path + file_name + '_' + mea + '.npy' )
                        
                        performance_params = threshPerformanceParams(matrix_distances_norm, num_imgs, centroids_indexes, scale)

                        # matrixes.append(matrix_distances_norm)

                        np.save(pickle_base + params_path + file_name + '_' + mea, performance_params)
                        

                # fusion = fusionScores( matrixes, cod, measure )

                # np.save(norms_path + file_name + '_' + mea, fusion)
                

def fusionScores(measures, num_imgs, pickle_base, norms_path):

        for cod, (measure, mea) in measures:
                # print('done')
                (_, g) = np.load( pickle_base + norms_path + 'geom_' + mea + '.npy' )               # g_centroids_indexes
                (_, d) = np.load( pickle_base + norms_path + 'distance_' + mea + '.npy' )            # d_centroids_indexes
                (_, o) = np.load( pickle_base + norms_path + 'orientation_' + mea + '.npy' )         # o_centroids_indexes

                pd = np.multiply(d, o)
                sm = np.add(d, o)
                G = np.minimum(pd, g)
                I = np.maximum(sm, g)

                Z = copy.deepcopy(I)

                for s in range(0, Z.shape[0], num_imgs):
                        Z[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))] = G[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))]

                Z, mini, maxi = matrixNormalization(Z)

                # print('normalized matrix: ', Z)

                centroids_indexes = allIndexes( num_imgs, Z )

                np.save(pickle_base + norms_path + 'fusion_' + mea, (centroids_indexes, Z))
                np.save(pickle_base + 'mini_maxi/' + 'fusion_' + mea, (mini, maxi))


def saveFigures(scores_reduct_all, measures, pickle_base, params_path, path_figs, thresholds, scale):

        h = scale
        w = len(scores_reduct_all)

        perf_matrix = np.matrix( np.zeros(shape=( h * w, w * w )))

        # print(perf_matrix.shape)

        # fig_12_a = []
        
        for i, (cod, (measure, mea)) in enumerate(measures):

                for j, (l, file_name )in enumerate(scores_reduct_all):
                        
                        perf_matrix[np.ix_(range( i * h, (i+1) * h), range( j * w, (j+1) * w))] = np.load(pickle_base + params_path + file_name + '_' + mea +'.npy')

                # fig_12_a.append((measure, perf_matrix[i, -1]))

        
        for j, (_, title)  in enumerate(scores_reduct_all):
                saveThreshFigure(h, w, measures, perf_matrix[np.ix_(range( h * w), range( j * w, (j+1) * w))], path_figs, title, thresholds)
        
        for j, (_, title)  in enumerate(scores_reduct_all):
                saveROCScoreFigure(h, w, perf_matrix[np.ix_(range( h * w), range( j * w, (j+1) * w))], title, measures)

        for i, (cod, (measure, mea))  in enumerate(measures):
                saveROCMeasureFigure(h, w, perf_matrix[np.ix_(range( i * h, (i+1) * h), range( w * w ))], measure, scores_reduct_all)

        
def saveThreshFigure(h, w, measures, performance_params_list, path_figs, title, thresholds):

        x = np.linspace(0, 1.0, h)

        # print(len(performance_params_list))
        EERs = dict()

        for i,  (_, (measure, mea))  in enumerate(measures):
                performance_params = performance_params_list[np.ix_(range( i * h, h * (i+1)), range( w ))]

                # print(len(performance_params))
                
                y_FAR = performance_params[:,0]
                # print(len(y_FAR))
                y_FRR = performance_params[:,1]

                idx_EER = np.argmin(np.abs(np.subtract(y_FAR, y_FRR)))
                # print(idx_EER)
                a = np.array(y_FAR[idx_EER][0][0])[0][0]
                EER = (x[idx_EER], a)

                EERs[mea] = EER 
                
                plt.plot(x, y_FAR, 'r', label='FAR')
                plt.plot(x, y_FRR, 'b', label='FRR')
                plt.scatter([EER[0]], [EER[1]], c='g' , label='EER')
                plt.xlabel('Threshold')
                plt.ylabel('Probability')
                plt.title('Finding best Threshold - '+measure)
                plt.legend()
                plt.savefig(path_figs + 'thresh_fusion_' + measure + '.png')
                plt.close()

        print(EERs)

        with open(thresholds + 'thresholds_' + title + '.json', 'w') as f:
                json.dump(EERs, f)


def saveROCScoreFigure(h, w, column_score, title, measures):

        color = [
                'r',
                'g',
                'c',
                'b'
        ]

        

        for i, (_, (measure, _))  in enumerate(measures):
                performance_params = column_score[np.ix_(range( i * h, h * (i+1)), range( w ))]
                # print(len(performance_params))
                y_FAR = performance_params[:,0]
                # print(y_FAR)
                y_TAR = performance_params[:,2]

                plt.plot(y_FAR, y_TAR, color[i], label=measure)
        
        plt.xlabel('FAR')
        plt.ylabel('TAR')
        plt.xscale('log')
        plt.title('ROC - '+title)
        plt.legend()
        plt.savefig( path_figs + 'ROC_'+ title + '.png')
        plt.close()


def saveROCMeasureFigure(h, w, row_score, measure, scores_reduct_all):

        color = [
                'r',
                'g',
                'c',
                'b'
        ]
        # print(len(row_score[:,0]))

        for j, (_, title)  in enumerate(scores_reduct_all):
                performance_params = row_score[np.ix_(range( h ), range( j * w, (j+1) * w ))]
                y_FAR = performance_params[:,0]
                y_TAR = performance_params[:,2]

                plt.plot(y_FAR, y_TAR, color[j], label=title)
        
        plt.xlabel('FAR')
        plt.ylabel('TAR')
        plt.xscale('log')
        plt.title('ROC - '+measure)
        plt.legend()
        plt.savefig( path_figs + 'ROC_'+ measure + '.png')
        plt.close()


def countPeoplePhoto(path):
        paths = os.listdir(path)

        d = dict()

        for name_img in paths:
                
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                
                if person_idx in d:
                        d[person_idx].append(img_idx)
                else:
                        d[person_idx] = [ img_idx ]
                
        # print(d.values())

        d1 = np.array([ len(elem) for elem in list(d.values())])
        if d1.max() > d1.min():
                print('people has different number of photos, check it!')
                n_person, n_imgs = None, None
        else:
                # print('same number of photos')
                n_person, n_imgs = len(d.values()), len(list(d.values())[0])
        
        return n_person, n_imgs


def test(measures, path_test, hand_path, pickle_path , norms_path, row_path, num_imgs):

        print(' ----            TEST            ---- ')
        
        tests = os.listdir(path_test + hand_path + path_in)
        tests.sort()

        dataset = os.listdir(hand_base + path_in)
        dataset = [ image.replace('.JPG', '').split("_")[0] for image in dataset] 
        
        dataset = set(dataset)
        dataset = list(dataset)
        dataset.sort()

        n_people = len(tests)

        # w, h, path_in, path_out, dist_path, hand_base, scores_path 

        saveScores(n_people, 1, path_test + hand_path + path_in, path_test + hand_path, path_test + hand_path + dist_path, hand_base , path_test + pickle_path + scores_path)

        EERs_g = dict()
        EERs_f = dict()
        EERs_d = dict()
        EERs_o = dict()

        with open(thresholds + 'thresholds_geom.json', 'r') as f:
                EERs_g = json.load(f)

        with open(thresholds + 'thresholds_fusion.json', 'r') as f:
                EERs_f = json.load(f)

        with open(thresholds + 'thresholds_distance.json', 'r') as f:
                EERs_d = json.load(f)

        with open(thresholds + 'thresholds_orientation.json', 'r') as f:
                EERs_o = json.load(f)
        
        GG = np.load( path_test + pickle_path + scores_path + 'geom.npy' )
        DD = np.load( path_test + pickle_path + scores_path + 'distance.npy' )            
        OO = np.load( path_test + pickle_path + scores_path + 'orientation.npy' )     

        centr_fusion_success = 0
        centr_fusion_insuccess = 0
        centr_verification_success = 0
        centr_verification_insuccess = 0

        meanshape_fusion_success = 0
        meanshape_fusion_insuccess = 0
        meanshape_verification_success = 0
        meanshape_verification_insuccess = 0


        for i, name_img in enumerate(tests):

                print('\n \n ---> ' ,name_img)
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                # print(person_idx, img_idx)

                for cod, (measure, mea) in measures:
                        
                        (r_g, ci_g)     = np.load( row_path + 'geom_' + mea + '.npy' )                
                        (g_mn, g_mx)    = np.load(pickle_base + 'mini_maxi/' + 'geom_' + mea + '.npy')

                        (r_d, ci_d)     = np.load( row_path + 'distance_' + mea + '.npy' )            
                        (d_mn, d_mx)    = np.load(pickle_base + 'mini_maxi/' + 'distance_' + mea + '.npy')

                        (r_o, ci_o)     = np.load( row_path + 'orientation_' + mea + '.npy' )         
                        (o_mn, o_mx)    = np.load(pickle_base + 'mini_maxi/' + 'orientation_' + mea + '.npy')

                        (_, _) = np.load(pickle_base + norms_path + 'fusion_' + mea + '.npy')
                        (f_mn, f_mx)    = np.load(pickle_base + 'mini_maxi/' + 'fusion_' + mea + '.npy')

                        g, d, o = GG[0][i], DD[0][i], OO[0][i]

                        g_centroid = ci_g[np.ix_(r_g)]
                        d_centroid = ci_d[np.ix_(r_d)]
                        o_centroid = ci_o[np.ix_(r_o)]

                        _, g_big_matrix, _ = allScores(np.array([np.append(g_centroid, [g], axis=0)]), num_imgs, cod, measure)
                        g_norm = matrixNormalizationMiniMaxi(g_big_matrix, g_mn, g_mx)
                        _, d_big_matrix, _ = allScores(np.array([np.append(d_centroid, [d], axis=0)]), num_imgs, cod, measure)
                        d_norm = matrixNormalizationMiniMaxi(d_big_matrix, d_mn, d_mx)
                        _, o_big_matrix, _ = allScores(np.array([np.append(o_centroid, [o], axis=0)]), num_imgs, cod, measure)
                        o_norm = matrixNormalizationMiniMaxi(o_big_matrix, o_mn, o_mx)
                                        
                        g_dist_norm = g_norm[-1]
                        d_dist_norm = d_norm[-1]
                        o_dist_norm = o_norm[-1]

                        pd = np.multiply(d_dist_norm, o_dist_norm)
                        f_dist = np.minimum(pd, g_dist_norm)
                        f_dist_norm = matrixNormalizationMiniMaxi(f_dist, f_mn, f_mx)

                        g_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( g_dist_norm[:-1] ) if y < EERs_g[mea][0]], key=itemgetter(1))]
                        d_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( d_dist_norm[:-1] ) if y < EERs_d[mea][0]], key=itemgetter(1))]
                        o_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( o_dist_norm[:-1] ) if y < EERs_o[mea][0]], key=itemgetter(1))]
                        f_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( f_dist_norm[:-1] ) if y < EERs_f[mea][0]], key=itemgetter(1))] 

                        # print('TEST ' , person_idx , ' geom ' , [dataset[idx] for idx in g_maybe] , ' ', mea)
                        # print('TEST ' , person_idx , ' dMap ' , [dataset[idx] for idx in d_maybe] , ' ', mea)
                        # print('TEST ' , person_idx , ' oMap ' , [dataset[idx] for idx in o_maybe] , ' ', mea)
                        print('TEST ' , person_idx , ' fusion ' , [dataset[idx] for idx in f_maybe] , ' ', mea)

                        if len([dataset[idx] for idx in f_maybe]) > 0:
                                if person_idx == dataset[f_maybe[0]]:
                                        centr_fusion_success = centr_fusion_success + 1
                                else:
                                        centr_fusion_insuccess = centr_fusion_insuccess + 1

                                if person_idx in  [dataset[idx] for idx in f_maybe]:
                                        centr_verification_success = centr_verification_success + 1
                                else:
                                        centr_verification_insuccess = centr_verification_insuccess + 1
                        else:
                                print(person_idx, ' not found in dataset. ')

                        # print(ci_g[0], ci_g[1], ci_g[3])

                        g_meanshape = [ np.mean(np.array(ci_g[ idx*5 : (idx+1)*5 ] ), axis=0 )for idx in range(len(r_g))]
                        # print(g_meanshape)
                        d_meanshape = [ np.mean(np.array(ci_d[ idx*5 : (idx+1)*5 ] ), axis=0 )for idx in range(len(r_g))]
                        o_meanshape = [ np.mean(np.array(ci_o[ idx*5 : (idx+1)*5 ] ), axis=0 )for idx in range(len(r_g))]

                        _, g_big_matrix, _ = allScores(np.array([np.append(g_meanshape, [g], axis=0)]), num_imgs, cod, measure)
                        g_norm = matrixNormalizationMiniMaxi(g_big_matrix, g_mn, g_mx)
                        _, d_big_matrix, _ = allScores(np.array([np.append(d_meanshape, [d], axis=0)]), num_imgs, cod, measure)
                        d_norm = matrixNormalizationMiniMaxi(d_big_matrix, d_mn, d_mx)
                        _, o_big_matrix, _ = allScores(np.array([np.append(o_meanshape, [o], axis=0)]), num_imgs, cod, measure)
                        o_norm = matrixNormalizationMiniMaxi(o_big_matrix, o_mn, o_mx)
                                        
                        g_dist_norm = g_norm[-1]
                        d_dist_norm = d_norm[-1]
                        o_dist_norm = o_norm[-1]

                        pd = np.multiply(d_dist_norm, o_dist_norm)
                        f_dist = np.minimum(pd, g_dist_norm)
                        f_dist_norm = matrixNormalizationMiniMaxi(f_dist, f_mn, f_mx)

                        g_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( g_dist_norm[:-1] ) if y < EERs_g[mea][0]], key=itemgetter(1))]
                        d_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( d_dist_norm[:-1] ) if y < EERs_d[mea][0]], key=itemgetter(1))]
                        o_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( o_dist_norm[:-1] ) if y < EERs_o[mea][0]], key=itemgetter(1))]
                        f_maybe = [x[0] for x in sorted([ [x, y] for x, y in enumerate( f_dist_norm[:-1] ) if y < EERs_f[mea][0]], key=itemgetter(1))] 

                        # print('TEST ' , person_idx , ' geom ' , [dataset[idx] for idx in g_maybe] , ' ', mea)
                        # print('TEST ' , person_idx , ' dMap ' , [dataset[idx] for idx in d_maybe] , ' ', mea)
                        # print('TEST ' , person_idx , ' oMap ' , [dataset[idx] for idx in o_maybe] , ' ', mea)
                        print('TEST ' , person_idx , ' fusion ' , [dataset[idx] for idx in f_maybe] , ' ', mea)

                        if len([dataset[idx] for idx in f_maybe]) > 0:
                                if person_idx == dataset[f_maybe[0]]:
                                        meanshape_fusion_success = meanshape_fusion_success + 1
                                else:
                                        meanshape_fusion_insuccess = meanshape_fusion_insuccess + 1

                                if person_idx in  [dataset[idx] for idx in f_maybe]:
                                        meanshape_verification_success = meanshape_verification_success + 1
                                else:
                                        meanshape_verification_insuccess = meanshape_verification_insuccess + 1
                        else:
                                print(person_idx, ' not found in dataset. ')

        print('TEST identification success   (centroid): ', centr_fusion_success)
        print('TEST identification insuccess (centroid): ', centr_fusion_insuccess)
        print('TEST verification success     (centroid): ', centr_verification_success)
        print('TEST verification insuccess   (centroid): ', centr_verification_insuccess)

        print('TEST identification success   (meanshape): ', meanshape_fusion_success)
        print('TEST identification insuccess (meanshape): ', meanshape_fusion_insuccess)
        print('TEST verification success     (meanshape): ', meanshape_verification_success)
        print('TEST verification insuccess   (meanshape): ', meanshape_verification_insuccess)


def main():

        scale = 1000
        
        paths = os.listdir(hand_base + path_in)

        n_people, _ = countPeoplePhoto(hand_base + path_in)

        # save scores
        # shape_normalization, g_scores, d_scores, o_scores = saveScores(n_people, NUM_IMGS, hand_base + path_in, path_out, dist_path, hand_base, pickle_base + scores_path)
        
        # load scores
        shape_normalization = np.load(pickle_base + scores_path + 'tot_shape.npy')
        g_scores = np.load(pickle_base + scores_path + 'tot_geom.npy')
        d_scores = np.load(pickle_base + scores_path + 'tot_distance.npy')
        o_scores = np.load(pickle_base + scores_path + 'tot_orientation.npy')
         
        scores = [
                ( g_scores, ('g', 'geom')       ),
                ( d_scores, ('d', 'distance')   ),
                ( o_scores, ('o', 'orientation'))  
        ]

        I = getFeatureVectors(shape_normalization, g_scores, d_scores, o_scores, NUM_IMGS)


        # TO-DO
        # now we have I composed as slack: I = [prims, centroids, mean_shapes]
        # where prims = [geom, dist, orien] features of first element
        # where centroids = [geom, dist, orien] features of centroid elements
        # where mean_shapes = [geom, dist, orien] features of mean_shapes
        # and where geom, dist, orien are list of features
        


        saveMatrix(scores, measures, pickle_base, norms_path, row_path, NUM_IMGS)

        scores_reduct_all = [
                ( 'g', 'geom'       ),
                ( 'd', 'distance'   ),
                ( 'o', 'orientation'),
                ( 'f', 'fusion'     )
        ]

        fusionScores(measures, NUM_IMGS, pickle_base, norms_path)

        saveParams(scores_reduct_all, measures, NUM_IMGS, pickle_base, params_path, norms_path, scale)

        saveFigures(scores_reduct_all, measures, pickle_base,  params_path, path_figs, thresholds, scale)

        test(measures, path_test, hand_path, pickle_path, norms_path, pickle_base + row_path , NUM_IMGS)
        
        

if __name__== "__main__":
  main()


