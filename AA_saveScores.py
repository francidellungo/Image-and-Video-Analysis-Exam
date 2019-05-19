import os
import cv2
from preprocessing import getHand
from fingerCoordinates import getFingerCoordinates
from Utils import rotateHand, draw
from extractionShapeFeatures import getAngle
from fingerFeaturePoints import getReferencePoint, updateContour, calculateMediumPoints
from geometricalFeaturesExtraction import extractGeometricalFeatures
from extractionShapeFeatures import fingerRegistration, extractShapeFeatures
import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

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

def saveScores(w, h, path_in, path_out, dist_path, hand_base, scores_path):

        paths = os.listdir(path_in)
        paths.sort()

        geom_scores = []
        distance_scores = []
        orientation_scores = []
        shape_normalization = []
        
        for i, name_img in enumerate(paths):                
                print('\n \n ---> ' ,name_img)
                                
                # read image in a grey scale way
                img_bgr = cv2.imread(path_in + name_img)

                # apply the preprocessing to the grey scale image
                hand_mask, contour, center_of_mass, _  = getHand( img_bgr )

                # returns ordinated points starting from little finger(0)
                finger_points, _, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                # rotate based on middle finger point to center of mass axis
                _, _, _, contour, center_of_mass = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                print(center_of_mass)

                _, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                _, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)


                # return: contour, medium_points, center of mass has to be the same
                r_based_contour, medium_points, center_of_mass = shapeNormalization(r_based_contour, center_of_mass, r_based_fingers_indexes, medium_points)
                shape_normalization.append(((r_based_contour, center_of_mass), (medium_points, r_based_fingers_indexes, comp_valley_indexes, valley_indexes)))


                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                distance_features, orientation_features, _, _  = extractShapeFeatures(updated_contour, 0)

                geom_scores.append(geom_features)
                distance_scores.append(distance_features)
                orientation_scores.append(orientation_features)
        
        np.save( scores_path + 'tot_shape', shape_normalization)

        # if there are less than 50 coeff for some images, we take for all the uqual number of coeff that is the min
        d_coeff = np.min([ len(ele) for ele in distance_scores])
        distance_scores = [ d_score[:d_coeff] for d_score in distance_scores]

        o_coeff = np.min([ len(ele) for ele in orientation_scores])
        orientation_scores = [ o_score[:o_coeff] for o_score in orientation_scores]
        
        np.save( scores_path + 'tot_geom', geom_scores)
        # print(geom_scores)
        np.save( scores_path + 'tot_distance', distance_scores)
        # print(distance_scores)
        np.save( scores_path + 'tot_orientation', orientation_scores)
        # print(orientation_scores)

        return shape_normalization, geom_scores, distance_scores, orientation_scores


# def saveRepresentatives(scores, score_name, scores_path):
def shapeNormalization(r_based_contour, center_of_mass, r_based_fingers_indexes, medium_points):
        LENGTH = 100
        # print(center_of_mass)
        # print(r_based_contour[r_based_fingers_indexes[2]])
        distance = np.linalg.norm(center_of_mass - r_based_contour[r_based_fingers_indexes[2]]) 
        # print(distance)
        scale_factor = LENGTH/distance
        print(center_of_mass)
        distance_vector = center_of_mass - center_of_mass*scale_factor
        print(distance_vector)
        contour = np.array([ [[ int(x) for x in point[0]*scale_factor+distance_vector]] for point in r_based_contour])
        
        # img = draw(None, r_based_contour, [255, 0, 0], [], [255, 0, 0], None)
        # img = draw(img, contour, [0, 255, 0], [], [255, 0, 0], None)
        # cv2.imshow('ciao', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        medium_points =  [ [ [x*scale_factor+distance_vector[i] for i, x in enumerate(point[0])]] for point in medium_points]
        center_of_mass = [ x*scale_factor+distance_vector[i] for i, x in enumerate(center_of_mass)]
        # print(medium_points)
        
        return contour, medium_points, center_of_mass



def getFeatureVectors(shape_normalization, g_scores, d_scores, o_scores, NUM_IMGS):
        I = []
        prims = []

        # prims list contains lists of geometrical, distance and orientation scores
        prims.append([np.array(x).tolist() for x in g_scores[0::NUM_IMGS]])
        prims.append([np.array(x).tolist() for x in d_scores[0::NUM_IMGS]])
        prims.append([np.array(x).tolist() for x in o_scores[0::NUM_IMGS]])
        I.append(prims)


        mean_shapes, shapes, maxi_centroid = findMeanshape(shape_normalization, NUM_IMGS)

        centroids_indexes = calculateCentroidIndexes(mean_shapes, shapes, NUM_IMGS)

        centroids = []
        centroids.append([np.array(x).tolist() for x in g_scores[np.ix_(centroids_indexes)]])
        centroids.append([np.array(x).tolist() for x in d_scores[np.ix_(centroids_indexes)]])
        centroids.append([np.array(x).tolist() for x in o_scores[np.ix_(centroids_indexes)]])
        I.append(centroids)

        means = []
        g_means = []
        d_means = []
        o_means = []
        for shape in mean_shapes:

                mask = np.zeros((500, 500), np.uint8)
                cv2.fillPoly(mask, pts =[shape], color=(255))
                mask[mask == 255]=1
                properties = regionprops(mask, coordinates='xy')
                center_of_mass = properties[0].centroid[::-1]

                finger_points, _, fingers_indexes, valley_indexes = getFingerCoordinates(shape, mask)
                
                _, r_index = getReferencePoint(shape, fingers_indexes, center_of_mass)

                _, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(shape, valley_indexes, fingers_indexes, r_index)

                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                r_based_contour, medium_points, center_of_mass = shapeNormalization(r_based_contour, np.array(list(center_of_mass)), r_based_fingers_indexes, medium_points)

                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                distance_features, orientation_features, _, _  = extractShapeFeatures(updated_contour, 0)

                g_means.append(geom_features)
                d_means.append(distance_features.tolist())
                o_means.append(orientation_features.tolist())

        means.append(g_means)
        means.append(d_means)
        means.append(o_means)
        I.append(means)

        return I


def findMeanshape(shape_normalization, NUM_IMGS):

        # find shape with maximum center of mass (maxi)
        maxi = [0, 0]
        for (_, c), _ in shape_normalization:
                if maxi < c:
                        maxi = c
        
        # align shapes: move each contour component to be centered in one center of mass (the same for all the shapes)
        shapes = [ [ [[ int(x) for x in  (np.array(point[0])+np.array(maxi)-np.array(c)).tolist()] ] for point in cnt] for (cnt, c), _ in shape_normalization]

        # print(shapes[0])
        # img = np.zeros((500, 500, 3), np.uint8)
        # img = draw(img, [], None, [[maxi]], [0, 0, 255], None)
        # for i, cnt in enumerate(shapes):
        #         img = draw(img, np.array(cnt), list(colours[i%len(colours)][0]), [], [255, 0, 0], None)
        # cv2.imshow('ciao', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        step = int(255/NUM_IMGS)
        mean_shapes = []
        # print(len(shape_normalization)/NUM_IMGS)
        for i in range(int(len(shape_normalization)/NUM_IMGS)):
                allin = np.zeros((500, 500), np.uint8)
                person_shapes = shapes[i*NUM_IMGS:(i+1)*NUM_IMGS]
                for ele in person_shapes:
                        mask = np.zeros((500, 500), np.uint8)
                        cv2.fillPoly(mask, pts =[np.array(ele)], color=(step))
                        allin = allin + mask

                allin = cv2.medianBlur(allin,5)
                allin = cv2.medianBlur(allin,5)
                
                # cv2.imshow('allin', allin)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                allin[allin <= 255-2*step] = 0
                allin[allin > 255-2*step] = 1

                contours, _ = cv2.findContours(allin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # plt.imshow(allin, cmap='gray')
                # plt.show()
                # plt.close()

                cnt = contours[0]
                mean_shapes.append(cnt)

        return mean_shapes, shapes,  maxi


def calculateCentroidIndexes(mean_shape, shapes, NUM_IMGS):

        centroids_indexes = []
        for i in range(len(mean_shape)):
                person_shapes = shapes[i*NUM_IMGS:(i+1)*NUM_IMGS]
                step = int(255/2)
                areas = []
                for shape in person_shapes:
                        allin = np.zeros((500, 500), np.uint8)

                        mask = np.zeros((500, 500), np.uint8)
                        cv2.fillPoly(mask, pts =[np.array(shape)], color=(step))
                        allin = allin + mask

                        mask = np.zeros((500, 500), np.uint8)
                        cv2.fillPoly(mask, pts =[np.array(mean_shape[i])], color=(step))
                        allin = allin + mask

                        allin[allin <= step] = 0
                        allin[allin > step] = 1
                        contours, _ = cv2.findContours(allin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt = contours[0]

                        area = cv2.contourArea(cnt)
                        areas.append(area)
                
                idx = np.argmax(areas)
                centroids_indexes.append(idx)

        return np.array(centroids_indexes)