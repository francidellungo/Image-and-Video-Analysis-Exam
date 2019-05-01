from fingerCoordinates import *
from preprocessing import *
from fingerFeaturePoints import *

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

                cv2.line(img_points_hand, tuple([ int(x) for x in center_of_mass]), tuple([int(x) for x in finger_points[2][0]]), [0,255,0], 2)
                cv2.circle(img_points_hand, tuple(r_point[0]), 5, [0,0,255], -1)

                # save image in output path    
                cv2.imwrite(path_pts + name_img, img_points_hand)


if __name__== "__main__":
  main()