from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *
from extractionShapeFeatures import *
from geometricalFeaturesExtraction import *

path_in = './dataset_images/'
path_out = './a_masks/'
path_ell = './a_ellipses/'
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
                # name_img = '031_5.JPG'
                print('\n \n ---> ' ,name_img)

                # read image in a grey scale way
                img_grey = cv2.imread(path_in + name_img, cv2.IMREAD_GRAYSCALE)

                # apply the preprocessing to the grey scale image
                # hand_mask, contour, _ , ellipse_mask = getHand(cv2.flip( img_grey, 0 ))
                hand_mask, contour, center_of_mass , ellipse_mask = getHand( img_grey )

                # save image in output path    
                cv2.imwrite(path_out + name_img, hand_mask)

                # save image with mask background and axes and ellipse
                cv2.imwrite(path_ell + name_img, ellipse_mask)

                # returns orinated points starting from little finger(0)
                img_points_hand, finger_points, valley_points, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                # save image in output path    
                cv2.imwrite(path_pts + name_img, img_points_hand)

                r_point, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                # draw line from middle finger point to center of mass point
                cv2.line(img_points_hand, tuple([ int(x) for x in center_of_mass]), tuple([int(x) for x in finger_points[2][0]]), [0,255,0], 2)
                # draw reference point and cener of mass point
                        # cv2.circle(img_points_hand, tuple(r_point[0]), 5, [0,0,255], -1)
                cv2.circle(img_points_hand, tuple([ int(x) for x in center_of_mass]), 8, [255,0,0], -1)

                # save image in output path    
                cv2.imwrite(path_pts + name_img, img_points_hand)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(contour, valley_indexes, fingers_indexes, r_index)

                medium_points = [ (int(x[0][0]), int(x[0][1])) for x in medium_points ]
                valley_points = [ (int(x[0][0]), int(x[0][1])) for x in contour[valley_indexes] ]
                comp_valley_points = [ (int(x[0][0]), int(x[0][1])) for x in contour[comp_valley_indexes] ]
                
                font = cv2.FONT_HERSHEY_SIMPLEX

                for i, (m_ptr, v_ptr, c_v_ptr) in enumerate(zip(medium_points,valley_points,comp_valley_points)):
                        cv2.circle(img_points_hand, m_ptr, 8, [255,255,255], -1)
                        # cv2.putText(img_points_hand, str(i), m_ptr, font, 2,(255,255,255),2,cv2.LINE_AA)

                        cv2.circle(img_points_hand, c_v_ptr, 8, [0,255,255], -1)
                        # cv2.putText(img_points_hand, str(i), c_v_ptr, font, 2,(0,255,255),2,cv2.LINE_AA)
                        
                        cv2.circle(img_points_hand, v_ptr, 8, [0,0,255], -1)
                        # cv2.putText(img_points_hand, str(i), v_ptr, font, 2,(0,0,255),2,cv2.LINE_AA)

                cv2.imwrite(path_pts + name_img, img_points_hand)

                _, geom_features = extractGeometricalFeatures(finger_points, medium_points)


if __name__== "__main__":
  main()