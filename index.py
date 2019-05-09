from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from Utils import *
from geometricalFeaturesExtraction import *
from ComputeScores import *
import matplotlib.pyplot as plt
import copy
import csv

path_in = './dataset_images/'

path_out = './hands/masks/'
path_rot = './hands/rotates/'
path_pts = './hands/points/'
path_ell = './hands/ellipses/'

thresholds = './pickles/'

scores_path = './pickles/scores/'
params_path = './pickles/params/'
norms_path = './pickles/norms/'
row_path = './pickles/rows/'

path_figs = './figures/'
path_test = './tests/'

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


def saveScores(w, h, path_in, scores_path):

        paths = os.listdir(path_in)

        # matrices with scores for all people and all imgs ( features_scores[person][img] )

        geom_scores = [[0 for x in range(w)] for y in range(h)]
        distance_scores = [[0 for x in range(w)] for y in range(h)]
        orientation_scores = [[0 for x in range(w)] for y in range(h)]
        
        print('n pers:', h, 'n images: ', w)

        for name_img in paths:
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
                print("getHand: ")

                # save image in output path    
                cv2.imwrite(path_out + name_img, hand_mask)
                cv2.imwrite(path_ell + name_img, ellipse)

                # returns orinated points starting from little finger(0)
                finger_points, valley_points, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                hand_mask_rotated, finger_points, valley_points, contour, center_of_mass = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                # save image in output path    
                cv2.imwrite(path_rot + name_img, hand_mask_rotated)

                _ = None

                # draw contour and finger points to image
                img_points_hand = draw(_, contour, (0, 255, 0), finger_points, [255,0,0], _)

                r_point, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                print('update contour')
                r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # draw center of mass to image
                img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [0,0,255], _)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                # draw medium to image
                print('medium')
                img_points_hand = draw(img_points_hand, [], _, medium_points, [255,255,255], _)
                # draw valley to image
                print('valley')
                img_points_hand = draw(img_points_hand, [], _, r_based_contour[valley_indexes], [0,0,255], _)
                # draw complementary valley to image
                print('complementary valley')
                img_points_hand = draw(img_points_hand, [], _, r_based_contour[comp_valley_indexes], [0,255,255], _)

                updated_contour = fingerRegistration(r_based_contour, center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                # draw new contour to image
                img_points_hand = draw(img_points_hand, updated_contour, (255, 255, 255), [], _, _)

                cv2.imwrite(path_pts + name_img, img_points_hand)

                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                print("n geom features: ",len(geom_features))
                
                # to extract shape features updated contours are used
                distance_features, orientation_features = extractShapeFeatures(updated_contour, r_point)
                print("n dist, orient features: ",len(distance_features), len(orientation_features))

                geom_scores[int(img_idx)-1][int(person_idx)-1] = geom_features
                distance_scores[int(img_idx)-1][int(person_idx)-1] = distance_features
                orientation_scores[int(img_idx)-1][int(person_idx)-1] = orientation_features

        geom_scores = np.array(geom_scores)
        distance_scores = np.array(distance_scores)
        orientation_scores = np.array(orientation_scores)

        np.save(scores_path + 'geom', geom_scores)
        np.save(scores_path + 'distance', distance_scores)
        np.save(scores_path + 'orientation', orientation_scores)


def saveMatrix(scores, measures, norms_path, row_path):

        for cod, (measure, mea) in measures:

                for score, (l, file_name) in scores:
                
                        row_scores, matrix_distances, centroids_indexes = allScores(score, cod, measure)

                        matrix_distances_norm = matrixNormalization(matrix_distances)

                        np.save(norms_path + file_name + '_' + mea, (centroids_indexes, matrix_distances_norm))
                        np.save(row_path + file_name + '_' + mea, (centroids_indexes, row_scores))


def saveParams(scores, measures, num_imgs, params_path, norms_path):

        for cod, (measure, mea) in measures:

                # matrixes = []

                for l, file_name in scores:
                        # 
                        centroids_indexes, matrix_distances_norm = np.load( norms_path + file_name + '_' + mea + '.npy' )
                        
                        performance_params = threshPerformanceParams(matrix_distances_norm, num_imgs, centroids_indexes)

                        # matrixes.append(matrix_distances_norm)

                        np.save(params_path + file_name + '_' + mea, performance_params)
                        

                # fusion = fusionScores( matrixes, cod, measure )

                # np.save(norms_path + file_name + '_' + mea, fusion)
                

        """ 
        
        x = np.arange(0, 1.0, 0.01)
        y_FAR = performance_params[:,0]
        y_FRR = performance_params[:,1]
        y_TAR = performance_params[:,2]
        y_DI  = performance_params[:,3]

        idx_EER = np.argmin(np.abs(np.subtract(y_FAR, y_FRR)))
        EER = (x[idx_EER], y_FAR[idx_EER]) 
        
        plt.plot(x, y_FAR, 'r', label='FAR')
        plt.plot(x, y_FRR, 'b', label='FRR')
        plt.scatter(EER[0], EER[1], c='g' , label='EER')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('FIG 12 (a)')
        plt.legend()
        plt.show()


        plt.plot(y_FAR, y_TAR, 'r', label=l)
        plt.xlabel('FAR')
        plt.ylabel('TAR')
        plt.title('ROC'+' '+measure)
        plt.legend()
        plt.show()
        
        """
        

def fusionScores(measures, num_imgs, norms_path):

        for cod, (measure, mea) in measures:
                print('done')
                (_, g) = np.load( norms_path + 'geom_' + mea + '.npy' )               # g_centroids_indexes
                (_, d) = np.load( norms_path + 'distance_' + mea + '.npy' )            # d_centroids_indexes
                (_, o) = np.load( norms_path + 'orientation_' + mea + '.npy' )         # o_centroids_indexes

                pd = np.multiply(d, o)
                sm = np.add(d, o)
                G = np.minimum(pd, g)
                I = np.maximum(sm, g)

                Z = copy.deepcopy(I)

                for s in range(0, Z.shape[0], num_imgs):
                        Z[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))] = G[np.ix_(range(s, s+num_imgs), range(s, s+num_imgs))]

                Z = matrixNormalization(Z)

                # print('normalized matrix: ', Z)

                centroids_indexes = allIndexes( num_imgs, Z )

                np.save(norms_path + 'fusion_' + mea, (centroids_indexes, Z))


def saveFigures(scores_reduct_all, measures, params_path, path_figs, thresholds):

        h = 100
        w = len(scores_reduct_all)

        perf_matrix = np.matrix( np.zeros(shape=( h * w, w * w )))

        # print(perf_matrix.shape)

        # fig_12_a = []
        
        for i, (cod, (measure, mea)) in enumerate(measures):

                for j, (l, file_name )in enumerate(scores_reduct_all):
                        
                        perf_matrix[np.ix_(range( i * h, (i+1) * h), range( j * w, (j+1) * w))] = np.load(params_path + file_name + '_' + mea +'.npy')

                # fig_12_a.append((measure, perf_matrix[i, -1]))

        
        saveThreshFigure(h, w, [ measure for _, (measure, _) in measures], perf_matrix[np.ix_(range( h * w), range(- w, 0))], path_figs, thresholds)
        
        for j, (_, title)  in enumerate(scores_reduct_all):
                saveROCScoreFigure(h, w, perf_matrix[np.ix_(range( h * w), range( j * w, (j+1) * w))], title, measures)

        for i, (cod, (measure, mea))  in enumerate(measures):
                saveROCMeasureFigure(h, w, perf_matrix[np.ix_(range( i * h, (i+1) * h), range( w * w ))], measure, scores_reduct_all)

        
def saveThreshFigure(h, w, measures, performance_params_list, path_figs, thresholds):

        x = np.arange(0, 1.0, 0.01)

        # print(len(performance_params_list))
        EERs = []

        for i, measure in enumerate(measures):
                performance_params = performance_params_list[np.ix_(range( i * h, h * (i+1)), range( w ))]

                # print(len(performance_params))
        
                y_FAR = performance_params[:,0]
                # print(len(y_FAR))
                y_FRR = performance_params[:,1]

                idx_EER = np.argmin(np.abs(np.subtract(y_FAR, y_FRR)))
                # print(idx_EER)
                EER = (x[idx_EER], y_FAR[idx_EER])

                EERs.append(EER) 
                
                plt.plot(x, y_FAR, 'r', label='FAR')
                plt.plot(x, y_FRR, 'b', label='FRR')
                plt.scatter([EER[0]], [EER[1]], c='g' , label='EER')
                plt.xlabel('Threshold')
                plt.ylabel('Probability')
                plt.title('Finding best Threshold - '+measure)
                plt.legend()
                plt.savefig(path_figs + 'thresh_fusion_' + measure + '.png')
                plt.close()

        with open(thresholds + 'thresholds.txt', 'w') as f:
                for EER, measure in zip(EERs, measures):
                        f.write(measure + ': ' + str(EER) + '\n')



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


# def test(measures, path_test, norms_path, row_path):
        
#         tests = os.listdir(path_test)

#         # saveScores(n_people, n_imgs, path_test, path_test+'scores/')

#         for test in tests:

#                 for cod, (measure, mea) in measures:
                
#                         (r_g, ci_g) = np.load( row_path + 'geom_' + mea + '.npy' )               # g_centroids_indexes
#                         (r_d, ci_d) = np.load( row_path + 'distance_' + mea + '.npy' )            # d_centroids_indexes
#                         (r_o, ci_o) = np.load( row_path + 'orientation_' + mea + '.npy' )         # o_centroids_indexes
                        
#                         # print(r_g)
#                         # print(len(ci_g))
#                         g_centroid = ci_g[np.ix_(r_g)]
#                         d_centroid = ci_d[np.ix_(r_d)]
#                         o_centroid = ci_o[np.ix_(r_o)]

                        

                        




def main():
        
        paths = os.listdir(path_in)
        
        n_imgs = 5
        n_people = int( len([name for name in paths])/5 )
        print('persone: ',n_people)

        saveScores(n_people, n_imgs, path_in, scores_path)

        g_scores = np.load(scores_path + 'geom.npy')
        d_scores = np.load(scores_path + 'distance.npy')
        o_scores = np.load(scores_path + 'orientation.npy')

        num_imgs = g_scores.shape[0]
        
        scores = [
                ( g_scores, ('g', 'geom')       ),
                ( d_scores, ('d', 'distance')   ),
                ( o_scores, ('o', 'orientation'))  
        ]

        saveMatrix(scores, measures, norms_path, row_path)

        scores_reduct_all = [
                ( 'g', 'geom'       ),
                ( 'd', 'distance'   ),
                ( 'o', 'orientation'),
                ( 'f', 'fusion'     )
        ]

        fusionScores(measures, num_imgs, norms_path)

        saveParams(scores_reduct_all, measures, num_imgs, params_path, norms_path)

        saveFigures(scores_reduct_all, measures, params_path, path_figs, thresholds)

        # test(measures, path_test, norms_path, row_path )
        
        

if __name__== "__main__":
  main()


