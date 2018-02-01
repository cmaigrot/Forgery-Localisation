#!/usr/bin/python
import argparse, json, sys, os
import random, cv2, glob, time

import numpy as np
from scipy import stats
from scipy import misc
from scipy import ndimage
from threading import Thread


# # # # # # # # # # # # # # # # # # #
#
# MAIN FUNCTION
#
# # # # # # # # # # # # # # # # # # #

# entries :
# i1 : a numpy image corresponding to the forged image
# i2 : a numpy image corresponding to the candidate image
# output_directory : path to the directory where the results will be saved
def compare_images(i1, i2, output_directory="./output"):
    print "Start of the analyze..."
    print "Creating repertories ..."
    createRepertories(output_directory)
    queryName = os.path.basename(i1)

    # We resize the image with a maximum of 900px for the width and the height
    print "Resizing images ..."
    img_1 = resize_image(cv2.imread(i1, 0), 900)
    img_2 = resize_image(cv2.imread(i2, 0), 900)
    img_1_color = cv2.cvtColor(resize_image(cv2.imread(i1), 900), cv2.COLOR_BGR2RGB)

    print "Computing SURF descriptors ..."
    dkp_1, ddes_1 = get_descriptors_SURF(img_1)
    dkp_2, ddes_2 = get_descriptors_SURF(img_2)

    print "Computing matching between descriptors from image 1 to image 2 ..."
    matches, all_matches, goodDMatch = matching(ddes_1, ddes_2, 'Lowe')

    print "Computing homography from image 1 to image 2 ..."
    H, mask, matchesMask = homography(matches, dkp_1, dkp_2)

    print "Analyzing the necessity to crop the query ..."
    img_1, img_1_color, dkp_temp, ddes__temp, matches, all_matches, goodDMatch, matchesMask, H, dst, border_crop = \
        cropping(img_1, img_2, img_1_color, H, dkp_1, dkp_2, ddes_1, ddes_2, output_directory, queryName)

    print "Computing the outliers ..."
    outliers = detect_outliers(H, all_matches, dkp_1, dkp_2, img_2.shape[0], img_2.shape[1],\
                        img_1_color, output_directory, queryName, 'dense_surf')


    if len(outliers) > 0:
        img_1_reload = resize_image(cv2.imread(i1, 0), 900)

        print "Binarization ..."
        dilation(outliers,img_1_reload.shape[0],img_1_reload.shape[1],output_directory, queryName, border_crop)
        density(outliers,img_1_reload.shape[0],img_1_reload.shape[1],output_directory, queryName, border_crop)
        createIllustration(i1, output_directory)

# # # # # # # # # # # # # # # # # # #
#
# FUNCTIONS
#
# # # # # # # # # # # # # # # # # # #

# entries :
# img : the image where desciptors will be extracted
# stride : interval for the dense extraction pf the descriptors
#
# outputs :
# dkp :  list of the descriptors' coordinates
# ddes :  list of the descriptors' values
def get_descriptors_SURF(img, stride=10):
    surf = cv2.xfeatures2d.SURF_create()

    dkp = [cv2.KeyPoint(x, y, scale) for y in range(0, img.shape[0], stride)
             for x in range(0, img.shape[1], stride)
             for scale in [16, 24, 32, 40]]
    dkp, ddes = surf.compute(img, dkp)
    ddes = np.float32(ddes)
    return dkp, ddes

# entries :
# ddes_1 : descriptors of the first image
# ddes_2 : descriptors of the second image
# mode : mode of selection for the inliers
#
# outputs :
# good_threshold :  list of inliers (each inliers have the 2NN)
# all_matches :  list of matches
# goodDMatch :  list of inliers (only the most  NN)
def matching(ddes_1, ddes_2, mode='rec-1nn'):
    bfr = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(ddes_1, ddes_2, k=2)
    if mode == 'rec-1nn':
        goodDMatch = []
        return bfr.match(ddes_1, ddes_2), all_matches
    elif mode == 'Lowe' or mode == 'radius':
        # Calculation of min distance between keypoints descriptors
        distances = []
        for m in all_matches:
            distances.append(m[0].distance)  # m[0] only needed if k=2
        minDist = min(distances)
        if mode == 'Lowe':
            good_Lowe = []
            goodDMatch = []
            for m, n in all_matches:  # parcours des k=2 Dmatch pour chaque descripteurs (= 2 ppv)
                ## ratio test as per Lowe's paper
                if m.distance < 0.75 * n.distance:
                    good_Lowe.append([m, n])
            for m, n in all_matches:  # parcours des k=2 Dmatch pour chaque descripteurs (= 2 ppv)
                ## ratio test as per Lowe's paper
                if m.distance < 0.75 * n.distance:
                    goodDMatch.append(m)
            return good_Lowe, all_matches, goodDMatch
        elif mode == 'radius':
            good_threshold = []
            goodDMatch = []
            for m, n in all_matches:
                ##  radiusMatch: test vs min distance
                ## "good" matches are those whose distance is less than 2*min_dist, or a small arbitary value ( 0.02 ) in the event that min_dist is very small or equals to 0
                if m.distance <= max(2 * minDist, 0.02):
                    good_threshold.append([m, n])
            for m, n in all_matches:
                if m.distance <= max(2 * minDist, 0.02):
                    goodDMatch.append(m)
            return good_threshold, all_matches, goodDMatch

# entries :
# matches : list of the matches
# kp1 : list of keypoints of the first image
# kp2 : list of keypoints of the second image
#
# outputs :
# TO DO
def homography(matches, kp1, kp2):
    # compute homography on relevant matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m, n in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m, n in matches])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return H, mask, matchesMask

# entries :
# img : the image to resize (if necessary)
# size_max : the maximun size for width and height
#
# output :
# the image 'img' with a width and a height of 'size_max'
def cropping(img_1, img_2, img_1_color, H, dkp_1, dkp_2, ddes_1, ddes_2, output_directory, queryName):
    h_1, w_1 = img_1.shape[:2]
    h_2, w_2 = img_2.shape[:2]

    pts = np.float32([[0, 0], [0, h_1-1], [w_1-1, h_1-1], [w_1-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    border_crop = [0, h_1-1, 0, w_1-1]
    resized = False

    t = 10 # minimun threshold (in pixels) to launch the process of cropping
    for pt in dst: # for each projection coordinates
        for x, y in pt:
            if not resized and (abs(x) > t or abs(y) > t or abs(x - w_1) > t or abs(y - h_1) > t):
                matches, all_matches, goodDMatch = matching(ddes_2, ddes_1, 'Lowe')

                if len(goodDMatch) < 10:
                    print "Less than 10 matches."
                    return img_1, img_1_color, border_crop

                H, mask, matchesMask = homography(matches, dkp_2, dkp_1)

                pts = np.float32([[0, 0], [0, h_2 - 1], [w_2 - 1, h_2 - 1], [w_2 - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)

                pt1 = dst[0][0]
                pt2 = dst[1][0]
                pt3 = dst[2][0]
                pt4 = dst[3][0]

                yMin = max(0, min(int(pt1[0]), int(pt2[0])))
                yMax = min(w_1-1, max(int(pt3[0]), int(pt4[0])))
                xMin = max(0, min(int(pt1[1]), int(pt4[1])))
                xMax = min(h_1-1, max(int(pt2[1]), int(pt3[1])))

                img_1 = img_1[xMin:xMax, yMin:yMax]
                img_1_color = img_1_color[xMin:xMax, yMin:yMax, :]
                border_crop = [xMin, xMax, yMin, yMax]

                misc.imsave(os.path.join(output_directory, "forged", queryName), img_1_color)

                dkp_temp, ddes__temp = get_descriptors_SURF(img_1)

                matches, all_matches, goodDMatch = matching(ddes__temp, ddes_2, 'Lowe')
                H, mask, matchesMask = homography(matches, dkp_temp, dkp_2)
                print 'Cropping applied !'
                resized = True

    if not resized:
        img_1_color = cv2.cvtColor(resize_image(cv2.imread(img_1), 900), cv2.COLOR_BGR2RGB)


    img_visu = cv2.polylines(img_2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_visu = cv2.drawMatches(img_1, dkp_1, img_visu, dkp_2, goodDMatch, None, **draw_params)
    misc.imsave(os.path.join(output_directory, "homography", queryName), img_visu)

    return img_1, img_1_color, dkp_temp, ddes__temp, matches, all_matches, goodDMatch, matchesMask, H, dst, border_crop

def detect_outliers(H, all_matches, dkp_1, dkp_2, h_2, w_2, img_1_color,\
             output_directory, queryName, mode='dense_surf'):
    outliers_matches = []
    # try:
    # compute outliers on all matches
    src_pts = np.float32([dkp_1[m[0].queryIdx].pt for m in all_matches])  # m[0] only needed if k=2
    dst_pts = np.float32([dkp_2[m[0].trainIdx].pt for m in all_matches])
    inliers_matches = []
    outliers_points = []
    inliers_points = []
    # threshold for outliers detection
    diag = np.linalg.norm(np.float32([0, 0]) - np.float32([h_2, w_2]), 2)
    if mode == 'dense_surf':
        scale_num = 4  # retain only one keypoint amongst the 4 scales
    elif mode == 'sparse_surf':
        scale_num = 1
    for p in range(len(src_pts) / scale_num):
        is_in = []
        src_pt = src_pts[p].reshape(1, 2)
        src_kp = src_pts[p]
        for i in range(scale_num * p, scale_num * p + scale_num):
            src_pt = src_pts[i].reshape(1, 2)
            dest_pt = dst_pts[i].reshape(1, 2)
            pt_proj = cv2.perspectiveTransform(np.array([src_pt]), H)
            dist = np.linalg.norm(dest_pt - pt_proj[0], 2)
            is_in.append(dist <= 0.1 * diag)  # is_in = True if src_pt is an inlier

        if any(is_in):  # if at least one scale amongst the 4 has a match, the point is considered as inlier
            inliers_matches.append(all_matches[p][0])
            inliers_points.append(src_pt)
            cv2.circle(img_1_color, (src_pt[0][0], src_pt[0][1]), 4, (0, 255, 0), -1)

        else:
            outliers_matches.append(all_matches[p][0])
            outliers_points.append(src_pt[0])
            if (len(outliers_points) > 1):
                cv2.circle(img_1_color, (src_pt[0][0], src_pt[0][1]), 4, (0, 0, 255), -1)
            else:
                cv2.circle(img_1_color, (src_pt[0][0], src_pt[0][1]), 4, (255, 0, 255), -1)
    misc.imsave(os.path.join(output_directory, "outliers", queryName), img_1_color)

    # Filtrage of the isolated outliers
    transf = []
    for t1 in [-10, 0, 10]:
        for t2 in [-10, 0, 10]:
            if t1 != 0 or t2 != 0:
                transf.append([t1, t2])
    outliers_copy = list(outliers_points)
    for pt in range(len(outliers_copy)):
        Neigh_cpt = 0
        for t_pt in transf:
            for pt2 in outliers_copy:
                if outliers_copy[len(outliers_copy) - pt - 1][0] + t_pt[0] == pt2[0] and \
                                        outliers_copy[len(outliers_copy) - pt - 1][1] + t_pt[1] == pt2[1]:
                    Neigh_cpt += 1
        if Neigh_cpt < 1 :
            del outliers_points[len(outliers_copy) - pt - 1]
    # END -- Filtrage of the isolated outliers

    for p in range(len(src_pts)):
        src_pt = src_pts[p].reshape(1, 2)
        cv2.circle(img_1_color, (src_pt[0][0], src_pt[0][1]), 4, (0, 255, 0), -1)
    for p in outliers_points:
        cv2.circle(img_1_color, (p[0], p[1]), 4, (255, 0, 0), -1)
    misc.imsave(os.path.join(output_directory, "outliers_cleared", queryName), img_1_color)

    return outliers_points

def dilation(outliers_list, h1, w1, output_directory, queryName, border_crop) :
    size = 10  # size=stride
    es_dilation = np.ones((size + 1, size + 1), dtype=np.uint8)
    outliers = np.zeros((h1, w1), dtype=np.uint8)
    for x, y in outliers_list:
        try:
            outliers[int(y)+border_crop[0], int(x)+border_crop[2]] = 1
        except:
            print "one outlier is out of range ..."
    dilate = ndimage.morphology.binary_dilation(outliers, structure=es_dilation).astype(np.int)
    misc.imsave(os.path.join(output_directory, "dilation", queryName), dilate)

    #### filtering
    es = np.ones((size + 3, size + 3), dtype=np.uint8)
    close = ndimage.binary_closing(dilate, structure=es).astype(np.int)
    open = ndimage.binary_opening(close, structure=es).astype(np.int)
    oc = ndimage.binary_closing(open, structure=es).astype(np.int)

    mask_filled = ndimage.morphology.binary_fill_holes(oc)
    misc.imsave(os.path.join(output_directory, "morpho", queryName), mask_filled)

def density(outliers_list, h1, w1, output_directory, queryName, border_crop) :
    #(outliers_points, h1, w1, img_fname_base, dirname, border_crop, name):
    outliers_points_t = np.array([[y+border_crop[0], x+border_crop[2]] for x, y in outliers_list]).T
    outliers_map = np.zeros(h1 * w1).reshape([h1, w1])
    for pt in outliers_list:
        outliers_map[int(pt[1])+border_crop[0], int(pt[0])+border_crop[2]] = 255
    misc.imsave(os.path.join(output_directory, "pts", queryName), outliers_map)

    if len(outliers_points_t) > 1:
        values = np.vstack([outliers_points_t[0], outliers_points_t[1]])
        kde = stats.gaussian_kde(values)
        X, Y = np.mgrid[0:h1:1, 0:w1:1]
        positions = np.vstack([X.ravel(), Y.ravel()])
        estim = np.reshape(kde(positions).T, X.shape)
        misc.imsave(os.path.join(output_directory, "density", queryName), estim)

        # Calculation of max density value
        max_density = np.amax(estim)
        for percent in range(21):
            bin_map = (estim > (0.05*percent) * max_density).astype(np.int_)
            bin_map[0,0] = 0
            bin_map[0,1] = 1
            misc.imsave(os.path.join(output_directory, str(percent*5), queryName), bin_map)

# entries :
# img : the image to resize (if necessary)
# size_max : the maximun size for width and height
#
# output :
# the image 'img' with a width and a height of 'size_max'
def resize_image(img, size_max):
    h, w = img.shape[:2]
    if w > size_max or h > size_max:
        coef = 1.0 / (float(max(w, h)) / float(size_max))
        res = cv2.resize(img, (int(coef * float(w)), int(coef * float(h))), interpolation=cv2.INTER_LINEAR)
        return res
    else:
        return img

def createRepertories(output_path):
    dirs = ["homography", "forged", "density", "morpho", "visualization/redAndBlue/morpho",
            "points_outliers", "outliers", "pts", "outliers", "dilation", "outliers_cleared"]

    for i in range(21):
        dirs.append(str(i * 5))
        dirs.append("visualization/redAndBlue/" + str(i * 5))

    for dir in dirs:
        if not os.path.exists(os.path.join(output_path, dir)):
            os.makedirs(os.path.join(output_path, dir))



def createIllustration(forged_path, output_directory):
    forged = misc.imread(forged_path)
    queryName = os.path.basename(forged_path)

    list_of_dirs = ["morpho"]
    for i in range(21):
        list_of_dirs.append(str(i*5))

    for threshold in list_of_dirs:
        mask = misc.imread(os.path.join(output_directory, threshold, queryName))
        mask = misc.imresize(mask,forged.shape)
        test = misc.imread(forged_path)


        for rownum in range(len(forged)):
            for colnum in range(len(forged[rownum])):
                baseP = np.sum(mask[rownum][colnum])
                if (float(baseP)) / 3 > 20:
                    test[rownum][colnum][0] = 255
                else:
                    test[rownum][colnum][0] = 0

        misc.imsave(os.path.join(output_directory, "visualization", "redAndBlue", threshold, queryName), test)



