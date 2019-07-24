import os
import cv2
from preprocessing import getHand
from fingerCoordinates import getFingerCoordinates
from Utils import rotateHand, draw, line, get_numbers_with_idx, distance
from extractionShapeFeatures import getAngle
from fingerFeaturePoints import getReferencePoint, updateContour, calculateMediumPoints
from geometricalFeaturesExtraction import extractGeometricalFeatures
from extractionShapeFeatures import fingerRegistration, extractShapeFeatures
import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import math

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

epsilon = 0.3

def saveScores(LENGTH, w, h, path_in, path_out, dist_path, hand_base, scores_path):

        paths = os.listdir(path_in)
        paths.sort()

        geom_scores = []
        distance_scores = []
        orientation_scores = []
        shape_normalized = []
        
        for i, name_img in enumerate(paths):                
                print('\n \n ---> ' ,name_img)
                                
                # read image in a grey scale way
                img_bgr = cv2.imread(path_in + name_img)

                # apply the preprocessing to the grey scale image
                hand_mask, contour, center_of_mass, _  = getHand( img_bgr )

                cv2.imwrite(hand_base + path_out + name_img, hand_mask)

                # returns ordinated points starting from little finger(0)
                finger_points, _, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                # rotate based on middle finger point to center of mass axis
                _, _, _, contour, center_of_mass = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                _, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                _, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                img = draw(None, r_based_contour, None, r_based_contour[valley_indexes] , point_list_colors = [0,255,0])
                img = draw(img, [], None, r_based_contour[comp_valley_indexes], point_list_colors = [0,0,255])
                cv2.imshow('hand normalized', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                

                # modify contour so that wrist is cutted off
                # reference point will be the last complementary valley: on the right of the thumb
                new_contour = r_based_contour[comp_valley_indexes[4]:valley_indexes[0]+1]

                # find and add the points of the line connecting the two extreme points to the contour
                # (the line is the one that goes from the first valley point to the last complementary valley point)
                line_points = line(r_based_contour[valley_indexes[0]][0][0], r_based_contour[valley_indexes[0]][0][1], r_based_contour[comp_valley_indexes[4]][0][0], r_based_contour[comp_valley_indexes[4]][0][1])
                # update indexes of valley points, complementary valley points and finger points
                shift_idx = comp_valley_indexes[4]
               
                valley_indexes = [x - shift_idx for x in valley_indexes]
                comp_valley_indexes = [x - shift_idx for x in comp_valley_indexes]
                r_based_fingers_indexes = [x - shift_idx for x in r_based_fingers_indexes]

                print('compl v idx [4]: ', comp_valley_indexes[4])
                #remove first and last elements that are already considered in the contour
                line_points = line_points[1:-1]
                new_contour = np.append(new_contour, [ [[ line_point[0], line_point[1] ]] for line_point in line_points  ], axis = 0  )

                img = draw(None, new_contour, None, new_contour[valley_indexes] , point_list_colors = [0,255,0])
                img = draw(img, [], None, new_contour[comp_valley_indexes], point_list_colors = [0,0,255])
                # img = draw(img, [], None, medium_points)
                cv2.circle(img, (int(center_of_mass[0]), int(center_of_mass[1])), 4, (255,0,255), -1)
                cv2.imshow('hand normalized', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


                # return: contour, medium_points, center of mass has to be the same
                new_contour, medium_points, center_of_mass = shapeNormalization(LENGTH, new_contour, center_of_mass, r_based_fingers_indexes, medium_points)
                shape_normalized.append(((new_contour, center_of_mass), (medium_points, r_based_fingers_indexes, comp_valley_indexes, valley_indexes)))


                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(new_contour[r_based_fingers_indexes], medium_points)

                print('\n saveScores \n')
                updated_contour = fingerRegistration(copy.deepcopy(new_contour), center_of_mass, new_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                distance_features, orientation_features, _, _  = extractShapeFeatures(updated_contour, 0)

                geom_scores.append(geom_features)
                distance_scores.append(distance_features)
                orientation_scores.append(orientation_features)
        
        np.save( scores_path + 'tot_shape', shape_normalized)

        # if there are less than 50 coeff for some images, we take for all an equal number of coeff that is the min
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

        return shape_normalized, geom_scores, distance_scores, orientation_scores


# def saveRepresentatives(scores, score_name, scores_path):
def shapeNormalization(LENGTH, r_based_contour, center_of_mass, r_based_fingers_indexes, medium_points):
        
        # print(center_of_mass)
        # print(r_based_contour[r_based_fingers_indexes[2]])
        distance = np.linalg.norm(center_of_mass - r_based_contour[r_based_fingers_indexes[2]]) 
        # print(distance)
        scale_factor = LENGTH/distance
        # print(center_of_mass)
        distance_vector = center_of_mass - center_of_mass*scale_factor
        # print(distance_vector)
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



def getFeatureVectors(LENGTH, shape_normalization, g_scores, d_scores, o_scores, path, NUM_IMGS, pickle_base, features_matrix_path):
        I = []
        prims = []

        # prims list contains lists of geometrical, distance and orientation scores
        prims.append([np.array(x).tolist() for x in g_scores[0::NUM_IMGS]])
        prims.append([np.array(x).tolist() for x in d_scores[0::NUM_IMGS]])
        prims.append([np.array(x).tolist() for x in o_scores[0::NUM_IMGS]])
        I.append(prims)


        mean_shapes, shapes, maxi_centroid, size = findMeanshape(int(LENGTH/50), shape_normalization, NUM_IMGS)

        # TODO remove centroids
        centroids_indexes = calculateCentroidIndexes(mean_shapes, shapes, NUM_IMGS)

        centroids = []
        centroids.append([np.array(x).tolist() for x in [g_scores[i] for i in centroids_indexes]])
        centroids.append([np.array(x).tolist() for x in [d_scores[i] for i in centroids_indexes]])
        centroids.append([np.array(x).tolist() for x in [o_scores[i] for i in centroids_indexes]])
        I.append(centroids)
        # print('centroids ', centroids)

        means = []
        g_means = []
        d_means = []
        o_means = []
        for idx, shape in enumerate(mean_shapes):

                mask = np.zeros(size, np.uint8)
                cv2.fillPoly(mask, pts =[shape], color=(255))
                cv2.imwrite(path + str(idx)+'_meanshape.JPG', mask)
                mask[mask == 255]=1
                properties = regionprops(mask, coordinates='xy')
                center_of_mass = properties[0].centroid[::-1]

                finger_points, _, fingers_indexes, valley_indexes = getFingerCoordinates(shape, mask)
                
                _, r_index = getReferencePoint(shape, fingers_indexes, center_of_mass)

                _, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(shape, valley_indexes, fingers_indexes, r_index)

                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                r_based_contour, medium_points, center_of_mass = shapeNormalization(LENGTH, r_based_contour, np.array(list(center_of_mass)), r_based_fingers_indexes, medium_points)

                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)
                
                print('\n getFeatureVectors \n')

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                distance_features, orientation_features, _, _  = extractShapeFeatures(updated_contour, 0)
                g_means.append(geom_features)
                d_means.append(distance_features.tolist())
                o_means.append(orientation_features.tolist())

        d_coeff = np.min([ len(ele) for ele in d_means])
        d_means = [ d_score[:d_coeff] for d_score in d_means]

        o_coeff = np.min([ len(ele) for ele in o_means])
        o_means = [ o_score[:o_coeff] for o_score in o_means]

        means.append(g_means)
        means.append(d_means)
        means.append(o_means)
        # print('means ', means)
        I.append(means)
        
        # save imposter matrix of features
        np.save( pickle_base + features_matrix_path + 'Imposter.npy', I)

        return I


def findMeanshape(alpha, shape_normalization, NUM_IMGS):

        # find shape with maximum center of mass (maxi)
        max_centroid = [0, 0]
        for (_, c), _ in shape_normalization:
                if max_centroid < c:
                        max_centroid = c
        
        # align shapes: move each contour component to be centered in one center of mass (the same for all the shapes)
        shapes = [ [ [[ int(x) for x in  (np.array(point[0])+np.array(max_centroid)-np.array(c)).tolist()] ] for point in cnt] for (cnt, c), _ in shape_normalization]
        rects = [ cv2.boundingRect(np.array(cnt)) for cnt in shapes]

        # print(shapes[0])

        # img = np.zeros((1000, 1000, 3), np.uint8)
        # img = draw(img, [], None, [[maxi]], [0, 0, 255], None)
        # for i, (cnt, rect) in enumerate(zip(shapes, rects)):
        #         x,y,w,h = rect
        #         img = draw(img, np.array(cnt), list(colours[i%len(colours)][0]), [], [255, 0, 0], None)
        #         cv2.rectangle(img,(x,y),(x+w,y+h),colours[i%len(colours)][0],2)

        # cv2.imshow('ciao', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mini, maxi = (min(x for x,y,w,h in rects)-alpha, min(y for x,y,w,h in rects)-alpha) , (max(x+w for x,y,w,h in rects)+alpha, max(y+h for x,y,w,h in rects)+alpha)
        # img = draw(img, [], None, [[mini]], [0, 0, 255], None)
        # img = draw(img, [], None, [[maxi]], [0, 0, 255], None)
        # cv2.imshow('ciao', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        shapes = [ [ [[ int(x) for x in  (np.array(point[0])-np.array(c)+np.array(max_centroid)-np.array(mini)).tolist()] ] for point in cnt] for (cnt, c), _ in shape_normalization]
        rects = [ cv2.boundingRect(np.array(cnt)) for cnt in shapes]
        centroid = (np.array(max_centroid)-np.array(mini)).tolist()
        # print(shapes[0])
        size = tuple((np.array(maxi)-np.array(mini)).tolist()[::-1])

        # img = np.zeros((size[0], size[1], 3), np.uint8)
        # img = draw(img, [], None, [[centroid]], [0, 0, 255], None)
        # for i, (cnt, rect) in enumerate(zip(shapes, rects)):
        #         x,y,w,h = rect
        #         img = draw(img, np.array(cnt), list(colours[i%len(colours)][0]), [], [255, 0, 0], None)
        #         cv2.rectangle(img,(x,y),(x+w,y+h),colours[i%len(colours)][0],2)

        # cv2.imshow('ciao', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        #TODO things to change for meanshape

        # for each 5 imgs of each person do:
        # define points that will be landmarks
        # find correspondant points and do the mean of the position
        # joint the points with lines (maybe better with an interpolating function?)

        mean_shapes = []
        NUM_POINTS = 100

        for person_idx in  range(int(len(shape_normalization)/NUM_IMGS)):
                person_shapes = shapes[person_idx*NUM_IMGS:(person_idx+1)*NUM_IMGS]
                landmarks = [[] for i in range(NUM_IMGS)]
                print('           PERSONA: ', person_idx)
                for shape_idx, elem in enumerate(person_shapes):
                        _, (_, fingers_indexes, comp_valley_indexes, valley_indexes) = shape_normalization[person_idx*NUM_IMGS+shape_idx]
                        # i = 4,3,2,1,0 finger and valleys idx
                        print('                      MANO:   ', shape_idx)
                        for i in range(4,-1,-1):
                                # print('num elements btw the finger point and the valley: ', i,' ', fingers_indexes[i] - comp_valley_indexes[i], ' ', int((fingers_indexes[i]-comp_valley_indexes[i])/NUM_POINTS) )
                                # points_list = []

                                # generate landmarks points for every hand
                                landmarks_idx = get_numbers_with_idx(comp_valley_indexes[i], fingers_indexes[i], NUM_POINTS)
                                landmarks_points = [elem[idx] for idx in landmarks_idx]
                                landmarks[shape_idx].extend(landmarks_points)
                                
                                landmarks_idx = get_numbers_with_idx(fingers_indexes[i], valley_indexes[i],  NUM_POINTS)
                                landmarks_points = [elem[idx] for idx in landmarks_idx]
                                landmarks[shape_idx].extend(landmarks_points)

                                # print('indice: ', i, 'len landmarks:  ', len(landmarks[shape_idx]))
                

                # calculate means of landmarks points for every person 
                # convert landmarks list into nparray, then calculate mean x,y for the N images of each  person
                # then return mean_landmarks as the list of the mean coordinates of all the landmarks.
                mean_landmarks = []
                landmarks = np.asarray(landmarks)
                for i in range(len(landmarks[0])):
                        # print('landmarks[:,',i,'] : ', landmarks[:,i])
                        value = landmarks[:,i]
                        # print(np.mean(value, axis = 0))
                        mean_landmarks.append( np.round(np.mean(value, axis = 0)).astype(int).tolist() )
                print('len mean landmarks: ', len(mean_landmarks), len(mean_landmarks[0]), len(mean_landmarks[0][0]))

                # generate points btw the mean landmarks if they are not consecutive
                mean_shape_cnt = []
                mean_shape_cnt.append(mean_landmarks[0])

                for i in range(len(mean_landmarks)-1):
                        if distance(mean_landmarks[i][0], mean_landmarks[i+1][0]) > math.sqrt(2) + epsilon:
                                a = line(mean_landmarks[i][0][0], mean_landmarks[i][0][1], mean_landmarks[i+1][0][0], mean_landmarks[i+1][0][1])
                                a = [[list(j)] for j in a]
                                mean_shape_cnt.extend( a[1:] )
                        else:
                                mean_shape_cnt.append(mean_landmarks[i+1])

                wrist_line = line(mean_landmarks[-1][0][0], mean_landmarks[-1][0][1], mean_landmarks[0][0][0], mean_landmarks[0][0][1])
                wrist_line = [[list(j)] for j in wrist_line]
                mean_shape_cnt.extend( wrist_line[1:-1] )

                mean_shape_cnt = np.array(mean_shape_cnt)
                mean_shapes.append(mean_shape_cnt)

                person_shape = np.array(person_shapes[0])
                img = draw(None, person_shape , cnt_color = [255,0,0], point_lists = [] )
                person_shape = np.array(person_shapes[1])
                img = draw(img, person_shape , cnt_color = [255,0,0], point_lists = [] )
                person_shape = np.array(person_shapes[2])
                img = draw(img, person_shape , cnt_color = [255,0,0], point_lists = [] )
                person_shape = np.array(person_shapes[3])
                img = draw(img, person_shape , cnt_color = [255,0,0], point_lists = [] )
                person_shape = np.array(person_shapes[4])
                img = draw(img, person_shape , cnt_color = [255,0,0], point_lists = [] )

                img = draw(img, mean_shape_cnt, None, [] )
                cv2.imshow('hand mean shape', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        
        # old mean shape 
        
        # step = int(255/NUM_IMGS)
        # mean_shapes = []
        # # print(len(shape_normalization)/NUM_IMGS)
        # for i in range(int(len(shape_normalization)/NUM_IMGS)):
        #         allin = np.zeros(size, np.uint8)
        #         person_shapes = shapes[i*NUM_IMGS:(i+1)*NUM_IMGS]
        #         for ele in person_shapes:
        #                 mask = np.zeros(size, np.uint8)
        #                 cv2.fillPoly(mask, pts =[np.array(ele)], color=(step))
        #                 allin = allin + mask

        #         allin = cv2.medianBlur(allin,5)
        #         allin = cv2.medianBlur(allin,5)
                
        #         # cv2.imshow('allin', allin)
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()
                
        #         allin[allin <= 255-2*step] = 0
        #         allin[allin > 255-2*step] = 1

        #         contours, _ = cv2.findContours(allin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #         # plt.imshow(allin, cmap='gray')
        #         # plt.show()
        #         # plt.close()

        #         cnt = contours[0]
        #         mean_shapes.append(cnt)


        return mean_shapes, shapes, centroid, size


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

        return centroids_indexes