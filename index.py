from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from Utils import *
from geometricalFeaturesExtraction import *
from ComputeScores import *
from tempfile import TemporaryFile

outfile = TemporaryFile()

path_in = './dataset_images/'
path_out = './a_masks/'
path_rot = './a_rotates/'
path_pts = './a_points/'

paths = os.listdir(path_in)

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
            ( 0    ,   'l1'        ),
            ( 0  ,   'euclidean'    ),
            ( 0   ,   'cosine' ),
            ( 1  ,   'chi-square'      )	
]




def main():

        genuine_scores = []
        imposter_scores = []

        n_imgs = 5
        n_people = int( len([name for name in paths])/5 )
        print('persone: ',n_people)

        w, h = n_imgs, n_people

        # saveFile(w, h)

        geom_scores = np.load('./pickles/geom.npy')
        distance_scores = np.load('./pickles/distance.npy')
        orientation_scores = np.load('./pickles/orientation.npy')

        for cod, measure in measures:
                genuine_scores_list, imposter_scores_list, centroids_indexes = calculateScores(geom_scores, cod, measure)

        # geom_Euclid_dist_new, geom_L1_dist_new, y_geom_genuine_predicted = calculateScores(n_people, n_imgs, geom_scores, distance_scores, orientation_scores)

        # print(len(geom_Euclid_dist_new), geom_Euclid_dist_new)

        # print(len(geom_L1_dist_new), geom_L1_dist_new)


        # for person in range(n_people):
        #         print('person : ', person)
        #         unos = [ 1 for i in range(len(y_geom_genuine_predicted[person]))]
        #         TP, FP, TN, FN = performanceMeasure(unos, y_geom_genuine_predicted[person])
        #         print(TP, FP, TN, FN)
        print(genuine_scores_list, imposter_scores_list, centroids_indexes)
        

if __name__== "__main__":
  main()


def saveFile(w, h):
        # Matrix = [[0 for x in range(w)] for y in range(h)] 

        # matrices with scores for all people and all imgs ( features_scores[person][img] )

        geom_scores = [[0 for x in range(w)] for y in range(h)]
        distance_scores = [[0 for x in range(w)] for y in range(h)]
        orientation_scores = [[0 for x in range(w)] for y in range(h)]
        
        print('n pers:', len(geom_scores), 'n images: ', len(geom_scores[0]))
        curr_img = '001_1.JPG'
        curr_person, curr_img = curr_img.replace('.JPG', '').split("_")[:]

        for name_img in paths:

                print('---> ',name_img)
                new_name_img = name_img.replace('.JPG', '')
                person_idx, img_idx = new_name_img.split("_")[:]
                print(person_idx, img_idx)
                
                # read image in a grey scale way
                img_grey = cv2.imread(path_in + name_img, cv2.IMREAD_GRAYSCALE)

                # apply the preprocessing to the grey scale image
                # hand_mask, contour, center_of_mass , ellipse_mask = getHand(img_grey)
                hand_mask, contour, center_of_mass, _  = getHand( img_grey )

                # save image in output path    
                cv2.imwrite(path_out + name_img, hand_mask)

                # returns orinated points starting from little finger(0)
                finger_points, valley_points, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                hand_mask_rotated, finger_points, valley_points, contour = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                # save image in output path    
                cv2.imwrite(path_rot + name_img, hand_mask_rotated)

                _ = None

                # draw contour and finger points to image
                img_points_hand = draw(_, contour, (0, 255, 0), finger_points, [255,0,0], _)

                r_point, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                print('update contour')
                r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # draw center of mass to image
                img_points_hand = draw(img_points_hand, [], _, [[center_of_mass]], [255,0,0], _)

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

                geom_scores[int(person_idx)-1][int(img_idx)-1] = geom_features
                distance_scores[int(person_idx)-1][int(img_idx)-1] = distance_features
                orientation_scores[int(person_idx)-1][int(img_idx)-1] = orientation_features


        geom_scores = np.array(geom_scores)
        distance_scores = np.array(distance_scores)
        orientation_scores = np.array(orientation_scores)

        np.save('./pickles/geom', geom_scores)
        np.save('./pickles/distance', distance_scores)
        np.save('./pickles/orientation', orientation_scores)