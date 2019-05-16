from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from Utils import *
from geometricalFeaturesExtraction import *
from ComputeScores import *
import matplotlib.pyplot as plt
import copy
import json
from operator import itemgetter

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


def saveScores(w, h, path_in, hand_base, scores_path):

        paths = os.listdir(path_in)

        paths.sort()

        print(paths)

        # matrices with scores for all people and all imgs ( features_scores[person][img] )

        geom_scores = [[0 for x in range(w)] for y in range(h)]
        distance_scores = [[0 for x in range(w)] for y in range(h)]
        orientation_scores = [[0 for x in range(w)] for y in range(h)]
        
        # print('\n\n n pers:', h, 'n images: ', w)

        for i, name_img in enumerate(paths):
                _ = None
                
                # name_img = '022_4.JPG'
                print('\n \n ---> ' ,name_img)
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                
                print(person_idx, img_idx)
                
                # read image in a grey scale way
                img_bgr = cv2.imread(path_in + name_img)

                # apply the preprocessing to the grey scale image
                # hand_mask, contour, center_of_mass, _  = getHand( img_grey )
                hand_mask, contour, center_of_mass, ellipse  = getHand( img_bgr )
                # print("getHand: ")

                # save image in output path    
                # cv2.imwrite(hand_base + path_out + name_img, hand_mask)
                # cv2.imwrite(hand_base + path_ell + name_img, ellipse)
                
                """ img_points_hand = cv2.imread('./hands/rotates/001_1.JPG')
                img_points_hand = draw(_, contour, (0, 255, 0), [], [255,0,0], _)
                img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [0,0,255], _)
                img_points_hand = draw(img_points_hand, [], _, [[[321, 38]]], [0,0,255], _)
                cv2.imwrite(hand_base + path_ell + name_img, img_points_hand) """

                # returns ordinated points starting from little finger(0)
                finger_points, valley_points, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                # rotate based on middle finger point to center of mass axes
                hand_mask_rotated, finger_points, valley_points, contour, center_of_mass = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                        # save image in output path    
                # cv2.imwrite(hand_base + path_rot + name_img, hand_mask_rotated)

                
                        # draw contour and finger points to image
                # img_points_hand = draw(_, contour, (0, 255, 0), finger_points, [255,0,0], _)
                r_point_normal, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                # img_points_hand = draw(_, contour, (0, 255, 0), finger_points, [255,0,0], _)
                # img_points_hand = draw(img_points_hand, [], _, valley_points, [0,0,255], _)
                # img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [0,0,255], _)
                # cv2.line(img_points_hand, (int(center_of_mass[0]), 0),(int(center_of_mass[0]), img_points_hand.shape[0]), [0,0,255])
                # img_points_hand = draw(img_points_hand, [], _, [r_point_normal], [255,0,0], [0,0,255])
                # img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [0,0,255], _)
                # cv2.line(img_points_hand, (int(center_of_mass[0]), 0),(int(center_of_mass[0]), img_points_hand.shape[0]), [0,0,255])
                # img_points_hand = draw(img_points_hand, [], _, [r_point_normal], [255,0,0], [0,0,255])
                # cv2.imwrite(hand_base + path_ell + name_img, img_points_hand)

                # img_points_hand = draw(_, contour, (0, 255, 0), [], [255,0,0], _)
                
                # print('update contour')
                r_point, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # draw center of mass to image
                # img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [0,0,255], _)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                        # draw medium to image
                        # print('medium')
                # img_points_hand = draw(img_points_hand, [], _, medium_points, [255,255,255], _)
                        # draw valley to image
                        # print('valley')
                # img_points_hand = draw(img_points_hand, [], _, r_based_contour[valley_indexes], [0,0,255], _)
                        # draw complementary valley to image
                        # print('complementary valley')
                # img_points_hand = draw(img_points_hand, [], _, r_based_contour[comp_valley_indexes], [0,255,255], _)
                # for f, m in zip(r_based_contour[r_based_fingers_indexes], medium_points):
                #         # (int(f[0][0]), int(f[0][1])), tuple(int(m[0][0]), int(m[0][1])
                #         cv2.line(img_points_hand, (int(f[0][0]), int(f[0][1])), (int(m[0][0]), int(m[0][1])), [255,255,255])
                
                # cv2.line(img_points_hand, (int(medium_points[0][0][0]), int(medium_points[0][0][1]) ) , (int(medium_points[3][0][0]), int(medium_points[3][0][1]) ) , [255,255,255])
                # cv2.line(img_points_hand, (int(medium_points[3][0][0]), int(medium_points[3][0][1]) ) , (int(medium_points[4][0][0]), int(medium_points[4][0][1]) ) , [255,255,255])


                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                # print("n geom features: ",len(geom_features))

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                        # draw new contour to image
                # img_points_hand = draw(img_points_hand, updated_contour, (255, 255, 255), [], _, _)

                # cv2.imwrite(hand_base + path_pts + name_img, img_points_hand)

                # to extract shape features updated contours are used
                # print(r_point)

                distance_features, orientation_features, dm_u, om_u  = extractShapeFeatures(updated_contour, 0)
                # print("n dist, orient features: ",len(distance_features), len(orientation_features))

                plt.plot(range(len(updated_contour)), dm_u, 'b--', label="distance map")
                        # print(r_based_fingers_indexes, dm)
                # plt.scatter(r_based_fingers_indexes, [ dm_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
                # plt.scatter(valley_indexes, [ dm_u[idx] for idx in valley_indexes], c='g', label='valley points')
                # plt.scatter(comp_valley_indexes, [ dm_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
                plt.savefig(hand_base + dist_path + new_name_img + '_dmap_update.png')
                plt.close()

                plt.plot(range(len(updated_contour)), om_u, 'b--', label="distance map")
                # print(r_based_fingers_indexes, dm)
                # plt.scatter(r_based_fingers_indexes, [ om_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
                # plt.scatter(valley_indexes, [ om_u[idx] for idx in valley_indexes], c='g', label='valley points')
                # plt.scatter(comp_valley_indexes, [ om_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
                plt.savefig(hand_base + dist_path + new_name_img + '_omap_update.png')
                plt.close()

                geom_scores[i % h][int(i / h)] = geom_features
                distance_scores[i % h][int(i / h)] = distance_features
                orientation_scores[i % h][int(i / h)] = orientation_features

        geom_scores = np.array(geom_scores)
        distance_scores = np.array(distance_scores)
        orientation_scores = np.array(orientation_scores)

        np.save( scores_path + 'geom', geom_scores)
        np.save( scores_path + 'distance', distance_scores)
        np.save( scores_path + 'orientation', orientation_scores)


def saveMatrix(scores, measures, pickle_base, norms_path, row_path):

        for cod, (measure, mea) in measures:

                for score, (l, file_name) in scores:
                
                        row_scores, matrix_distances, centroids_indexes = allScores(score, cod, measure)

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


def test(measures, path_test, hand_path, pickle_path , norms_path, row_path):

        print(' ----            TEST            ---- ')
        
        tests = os.listdir(path_test + hand_path + path_in)
        tests.sort()

        dataset = os.listdir(hand_base + path_in)
        dataset = [ image.replace('.JPG', '').split("_")[0] for image in dataset] 
        
        dataset = set(dataset)
        dataset = list(dataset)
        dataset.sort()

        n_people = len(tests)

        saveScores(n_people, 1, path_test + hand_path + path_in, path_test + hand_path, path_test + pickle_path + scores_path)

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

                        _, g_big_matrix, _ = allScores(np.array([np.append(g_centroid, [g], axis=0)]), cod, measure)
                        g_norm = matrixNormalizationMiniMaxi(g_big_matrix, g_mn, g_mx)
                        _, d_big_matrix, _ = allScores(np.array([np.append(d_centroid, [d], axis=0)]), cod, measure)
                        d_norm = matrixNormalizationMiniMaxi(d_big_matrix, d_mn, d_mx)
                        _, o_big_matrix, _ = allScores(np.array([np.append(o_centroid, [o], axis=0)]), cod, measure)
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

                        _, g_big_matrix, _ = allScores(np.array([np.append(g_meanshape, [g], axis=0)]), cod, measure)
                        g_norm = matrixNormalizationMiniMaxi(g_big_matrix, g_mn, g_mx)
                        _, d_big_matrix, _ = allScores(np.array([np.append(d_meanshape, [d], axis=0)]), cod, measure)
                        d_norm = matrixNormalizationMiniMaxi(d_big_matrix, d_mn, d_mx)
                        _, o_big_matrix, _ = allScores(np.array([np.append(o_meanshape, [o], axis=0)]), cod, measure)
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

        saveScores(n_people, 5, hand_base + path_in, hand_base, pickle_base + scores_path)

        g_scores = np.load(pickle_base + scores_path + 'geom.npy')
        d_scores = np.load(pickle_base + scores_path + 'distance.npy')
        o_scores = np.load(pickle_base + scores_path + 'orientation.npy')

        num_imgs = g_scores.shape[0]
        
        scores = [
                ( g_scores, ('g', 'geom')       ),
                ( d_scores, ('d', 'distance')   ),
                ( o_scores, ('o', 'orientation'))  
        ]

        saveMatrix(scores, measures, pickle_base, norms_path, row_path)

        scores_reduct_all = [
                ( 'g', 'geom'       ),
                ( 'd', 'distance'   ),
                ( 'o', 'orientation'),
                ( 'f', 'fusion'     )
        ]

        fusionScores(measures, num_imgs, pickle_base, norms_path)

        saveParams(scores_reduct_all, measures, num_imgs, pickle_base, params_path, norms_path, scale)

        saveFigures(scores_reduct_all, measures, pickle_base,  params_path, path_figs, thresholds, scale)

        test(measures, path_test, hand_path, pickle_path, norms_path, pickle_base + row_path )
        
        

if __name__== "__main__":
  main()


