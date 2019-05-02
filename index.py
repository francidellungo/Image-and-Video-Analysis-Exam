from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from Utils import *

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


def main():

        for name_img in paths:
                # name_img = '035_3.JPG'
                print('\n \n ---> ' ,name_img)

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



if __name__== "__main__":
  main()