from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from Utils import *
from geometricalFeaturesExtraction import *

path_in = './dataset_images/'
path_out = './a_masks/'
path_rot = './a_rotates/'
path_pts = './a_points/'
path_ell = './a_ellipses/'

# path_clo_out = './closed/'

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


def main():

        for name_img in paths:
                # name_img = '022_4.JPG'
                print('\n \n ---> ' ,name_img)
                
                t0 = time.time()
                # read image in a grey scale way
                img_bgr = cv2.imread(path_in + name_img)
                t1 = time.time()
                print("imread: ", t1-t0)

                # apply the preprocessing to the grey scale image
                # hand_mask, contour, center_of_mass, _  = getHand( img_grey )
                hand_mask, contour, center_of_mass, ellipse  = getHand( img_bgr )
                print("getHand: ")

                # save image in output path    
                cv2.imwrite(path_out + name_img, hand_mask)
                # cv2.imwrite(path_ell + name_img, ellipse)

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



if __name__== "__main__":
  main()