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

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                        # draw new contour to image
                # img_points_hand = draw(img_points_hand, updated_contour, (255, 255, 255), [], _, _)

                # cv2.imwrite(hand_base + path_pts + name_img, img_points_hand)

                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                # print("n geom features: ",len(geom_features))
                
                # to extract shape features updated contours are used
                # print(r_point)

                distance_features, orientation_features, dm_u, om_u  = extractShapeFeatures(updated_contour, 0)
                # print("n dist, orient features: ",len(distance_features), len(orientation_features))

                # plt.plot(range(len(updated_contour)), dm_u, 'b--', label="distance map")
                        # print(r_based_fingers_indexes, dm)
                # plt.scatter(r_based_fingers_indexes, [ dm_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
                # plt.scatter(valley_indexes, [ dm_u[idx] for idx in valley_indexes], c='g', label='valley points')
                # plt.scatter(comp_valley_indexes, [ dm_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
                # plt.savefig(hand_base + dist_path + new_name_img + '_dmap_update.png')
                # plt.close()

                # plt.plot(range(len(updated_contour)), om_u, 'b--', label="distance map")
                        # print(r_based_fingers_indexes, dm)
                # plt.scatter(r_based_fingers_indexes, [ om_u[idx] for idx in r_based_fingers_indexes], c='r', label='finger points')
                # plt.scatter(valley_indexes, [ om_u[idx] for idx in valley_indexes], c='g', label='valley points')
                # plt.scatter(comp_valley_indexes, [ om_u[idx] for idx in comp_valley_indexes] , c='y', label='valley points')
                # plt.savefig(hand_base + dist_path + new_name_img + '_omap_update.png')
                # plt.close()

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


def saveParams(scores, measures, num_imgs, pickle_base, params_path, norms_path):

        for cod, (measure, mea) in measures:

                # matrixes = []

                for l, file_name in scores:
                        # 
                        centroids_indexes, matrix_distances_norm = np.load( pickle_base + norms_path + file_name + '_' + mea + '.npy' )
                        
                        performance_params = threshPerformanceParams(matrix_distances_norm, num_imgs, centroids_indexes)

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


def saveFigures(scores_reduct_all, measures, pickle_base, params_path, path_figs, thresholds):

        h = 100
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

        x = np.arange(0, 1.0, 0.01)

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

        for i, name_img in enumerate(tests):

                print('\n \n ---> ' ,name_img)
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                print(person_idx, img_idx)

                for cod, (measure, mea) in measures:
                        
                        (r_g, ci_g)     = np.load( row_path + 'geom_' + mea + '.npy' )                # g_centroids_indexes
                        (g_mn, g_mx)    = np.load(pickle_base + 'mini_maxi/' + 'geom_' + mea + '.npy')

                        (r_d, ci_d)     = np.load( row_path + 'distance_' + mea + '.npy' )            # d_centroids_indexes
                        (d_mn, d_mx)    = np.load(pickle_base + 'mini_maxi/' + 'distance_' + mea + '.npy')

                        (r_o, ci_o)     = np.load( row_path + 'orientation_' + mea + '.npy' )         # o_centroids_indexes
                        (o_mn, o_mx)    = np.load(pickle_base + 'mini_maxi/' + 'orientation_' + mea + '.npy')

                        (z_big_centroids_indexes, _) = np.load(pickle_base + norms_path + 'fusion_' + mea + '.npy')
                        (f_mn, f_mx)    = np.load(pickle_base + 'mini_maxi/' + 'fusion_' + mea + '.npy')

                        g, d, o = GG[0][i], DD[0][i], OO[0][i]

                        g_centroid = ci_g[np.ix_(r_g)]
                        d_centroid = ci_d[np.ix_(r_d)]
                        o_centroid = ci_o[np.ix_(r_o)]

                        _, g_big_matrix, g_big_centroids_indexes = allScores(np.array([np.append(g_centroid, [g], axis=0)]), cod, measure)
                        g_norm = matrixNormalizationMiniMaxi(g_big_matrix, g_mn, g_mx)
                        _, d_big_matrix, d_big_centroids_indexes = allScores(np.array([np.append(d_centroid, [d], axis=0)]), cod, measure)
                        d_norm = matrixNormalizationMiniMaxi(d_big_matrix, d_mn, d_mx)
                        _, o_big_matrix, o_big_centroids_indexes = allScores(np.array([np.append(o_centroid, [o], axis=0)]), cod, measure)
                        o_norm = matrixNormalizationMiniMaxi(o_big_matrix, o_mn, o_mx)
                                        
                        g_dist_norm = g_norm[-1]
                        d_dist_norm = d_norm[-1]
                        o_dist_norm = o_norm[-1]

                        pd = np.multiply(d_dist_norm, o_dist_norm)
                        f_dist = np.minimum(pd, g_dist_norm)
                        f_dist_norm = matrixNormalizationMiniMaxi(f_dist, f_mn, f_mx)

                        g_maybe = [ x for x, y in enumerate( g_dist_norm[:-1] ) if y < EERs_g[mea][0]]
                        d_maybe = [ x for x, y in enumerate( d_dist_norm[:-1] ) if y < EERs_d[mea][0]]
                        o_maybe = [ x for x, y in enumerate( o_dist_norm[:-1] ) if y < EERs_o[mea][0]]
                        f_maybe = [ x for x, y in enumerate( f_dist_norm[:-1] ) if y < EERs_f[mea][0]]

                        print('geom is similar images n: ', g_maybe, ' for measure ', mea)
                        print('dMap is similar images n: ', d_maybe, ' for measure ', mea)
                        print('oMap is similar images n: ', o_maybe, ' for measure ', mea)
                        print('fusi is similar images n: ', f_maybe, ' for measure ', mea)


def main():
        
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

        saveParams(scores_reduct_all, measures, num_imgs, pickle_base, params_path, norms_path)

        saveFigures(scores_reduct_all, measures, pickle_base,  params_path, path_figs, thresholds)

        test(measures, path_test, hand_path, pickle_path, norms_path, pickle_base + row_path )
        
        

if __name__== "__main__":
  main()


