from fingerCoordinates import *
from preprocessing import *

path_in = './images/'
path_out = './a_masks/'
path_ell = './a_ellipses/'

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
        # read image in a grey scale way
        img_grey = cv2.imread(path_in + name_img, cv2.IMREAD_GRAYSCALE)

        # apply the preprocessing to the grey scale image
        hand_mask, contour, _ , ellipse_mask = getHand(img_grey)

        # save image in output path    
        cv2.imwrite(path_out + name_img, hand_mask)

        # save image with mask background and axes and ellipse
        cv2.imwrite(path_ell + name_img, ellipse_mask)

        getFingerCoordinates(contour, hand_mask)
        

if __name__== "__main__":
  main()