import os
import cv2
from preprocessing import getHand
from fingerCoordinates import getFingerCoordinates
from Utils import rotateHand
from extractionShapeFeatures import getAngle
from fingerFeaturePoints import getReferencePoint, updateContour, calculateMediumPoints
from geometricalFeaturesExtraction import extractGeometricalFeatures
from extractionShapeFeatures import fingerRegistration, extractShapeFeatures
import copy
import numpy as np
import matplotlib.pyplot as plt

def saveScores(w, h, path_in, path_out, dist_path, hand_base, scores_path):

        paths = os.listdir(path_in)

        paths.sort()

        print(paths)

        # matrices with scores for all people and all imgs ( features_scores[person][img] )

        geom_scores = []
        distance_scores = []
        orientation_scores = []
        
        # print('\n\n n pers:', h, 'n images: ', w)

        for i, name_img in enumerate(paths):
                _ = None
                
                # name_img = '022_4.JPG'
                print('\n \n ---> ' ,name_img)
                                
                # read image in a grey scale way
                img_bgr = cv2.imread(path_in + name_img)

                # apply the preprocessing to the grey scale image
                hand_mask, contour, center_of_mass, _  = getHand( img_bgr )

                # returns ordinated points starting from little finger(0)
                finger_points, _, fingers_indexes, valley_indexes = getFingerCoordinates(contour, hand_mask)

                # rotate based on middle finger point to center of mass axes
                _, _, _, contour, center_of_mass = rotateHand(hand_mask.shape, contour, getAngle(finger_points[2],[list(center_of_mass)]), center_of_mass, fingers_indexes, valley_indexes)

                _, r_index = getReferencePoint(contour, fingers_indexes, center_of_mass)

                _, r_based_contour, r_based_valley_indexes, r_based_fingers_indexes = updateContour(contour, valley_indexes, fingers_indexes, r_index)

                # valley_indexes has 5 points updating after complementary search
                medium_points, valley_indexes, comp_valley_indexes = calculateMediumPoints(r_based_contour, r_based_valley_indexes, r_based_fingers_indexes)

                # to extract geometrical features we used non rotated fingers 
                _, geom_features = extractGeometricalFeatures(r_based_contour[r_based_fingers_indexes], medium_points)

                updated_contour = fingerRegistration(copy.deepcopy(r_based_contour), center_of_mass, r_based_contour[r_based_fingers_indexes], medium_points, comp_valley_indexes, valley_indexes)

                distance_features, orientation_features, _, _  = extractShapeFeatures(updated_contour, 0)

                geom_scores.append(geom_features)
                distance_scores.append(distance_features)
                orientation_scores.append(orientation_features)

        d_coeff = np.min([ len(ele) for ele in distance_scores])
        print(d_coeff)
        distance_scores = [ d_score[:d_coeff] for d_score in distance_scores]

        o_coeff = np.min([ len(ele) for ele in orientation_scores])
        print(o_coeff)
        orientation_scores = [ o_score[:o_coeff] for o_score in orientation_scores]
        
        np.save( scores_path + 'tot_geom', geom_scores)
        np.save( scores_path + 'tot_distance', distance_scores)
        np.save( scores_path + 'tot_orientation', orientation_scores)

        return geom_scores, distance_scores, orientation_scores


# def saveRepresentatives(scores, score_name, scores_path):
