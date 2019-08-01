"""
TO DO LIST
    ( O ) [1] "Normalization" of CCV_N, CCV_D, CCV_A, + Coh / InCoh vals
    ( X ) [2] Calculate Euclidean "Distance" between two images per each factors --> 일단 16.jpg하고 1.jpg하고 비교!!
    ( X ) ---- decide weights for each factors --
    ( X ) [3] add up each factor's distance*weights => and calculate total similarity indicator value
    ( X ) ---- decide threshold values (in deciding whether those two are identical or not)
    ( X ) [4] make a function that returns boolean val // whether those two are identical targets
"""

import argparse
import numpy as np
import cv2
import json
from . ccv_improved import *

THRESHOLD = 0.0003
PATH = 'detected/'
# PATH = 'detected/two_people/'
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--input1", required=True)
ap.add_argument("-i2", "--input2", required=True)
ap.add_argument("-n", "--threshold", type=int)
args = vars(ap.parse_args())

# temporary weight values --> must modify these with some reasonable values later on
# have to discuss this issue with SkJ opp
alpha_weight = 0.2
beta_weight = 0.2
region_num_weight = 0.2
distance_weight = 0.2
angle_weight = 0.2
# 일단 1:1:1:1:1로 했음

class N_img_factors: # N for "normalized"
    def __init__(self, n_alpha, n_beta, n_num, n_R, n_theta):
        self.coh = n_alpha
        self.incoh = n_beta
        self.region_num = n_num
        self.distance = n_R
        self.angle = n_theta

    def calc_total_distance(self, img_factors_2): # using Euclidean distance per each factor
        global alpha_weight
        global beta_weight
        global region_num_weight
        global distance_weight
        global angle_weight
        # how can I use loops to simplify the process below...?

        coh_dist = img_factors_2.coh - self.coh
        coh_dist = [np.sqrt(s**2) for s in coh_dist]

        incoh_dist = img_factors_2.incoh - self.incoh
        incoh_dist = [np.sqrt(s ** 2) for s in incoh_dist]

        region_num_dist = img_factors_2.region_num - self.region_num
        region_num_dist = [np.sqrt(s ** 2) for s in region_num_dist]

        distance_dist = img_factors_2.distance - self.distance
        distance_dist = [np.sqrt(s ** 2) for s in distance_dist]

        angle_dist = img_factors_2.angle - self.angle
        angle_dist = [np.sqrt(s ** 2) for s in angle_dist]

        result = coh_dist*alpha_weight + incoh_dist*beta_weight + region_num_dist*region_num_weight + distance_dist*distance_weight + angle_dist*angle_weight
        return result


#####################################################################################################################
"""
add some functions later on
"""
#####################################################################################################################


if __name__ == '__main__':
    img1 = cv2.imread(PATH + args["input1"])
    img2 = cv2.imread(PATH + args["input2"])
    n = args["threshold"]
    # work on the first img first
    alpha, beta, num, R, theta = ccv(img1, tau=0, n=n)
    N_alpha, N_beta, N_num, N_R, N_theta = normalize_factors(img1, alpha, beta, num, R, theta)
    img1_info = N_img_factors(N_alpha, N_beta, N_num, N_R, N_theta) # first instantiation successful
    # work on the second img next
    alpha, beta, num, R, theta = ccv(img2, tau=0, n=n)
    N_alpha, N_beta, N_num, N_R, N_theta = normalize_factors(img2, alpha, beta, num, R, theta)
    img2_info = N_img_factors(N_alpha, N_beta, N_num, N_R, N_theta)  # second instantiation successful
    # compare
    similarity_indicator_val = img1_info.calc_total_distance(img2_info)
    print(similarity_indicator_val)