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
import json
from os import listdir
from os.path import isfile, join
import cv2
from ccv_improved import * # ccv, normalize_factors

mypath = 'detected/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))


THRESHOLD = 0.0003
PATH = 'detected/'
# PATH = 'detected/two_people/'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
# ap.add_argument("-i2", "--input2", required=True)
ap.add_argument("-n", "--threshold", type=int)
args = vars(ap.parse_args())

"""
weight vals in percentage (sum==1)

alpha_weight = (float)(1/120)*(1/2)
beta_weight = (float)(1/120)*(1/2)
region_num_weight = (float)(1/120)*29
distance_weight = (float)(1/120)*46
angle_weight = (float)(1/120)*44
"""

alpha_weight = 50
beta_weight = 50
region_num_weight = 2900
distance_weight = 4600
angle_weight = 4400


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

        for i in range(0, len(self.region_num)):
            region_num_dist = [ s - self.region_num[i] for s in img_factors_2.region_num]
        # region_num_dist = img_factors_2.region_num - self.region_num
        region_num_dist = [np.sqrt(s ** 2) for s in region_num_dist]

        distance_dist = img_factors_2.distance - self.distance
        distance_dist = [np.sqrt(s ** 2) for s in distance_dist]

        for i in range(0, len(self.angle)):
            angle_dist = [ s - self.angle[i] for s in img_factors_2.angle]
        # angle_dist = img_factors_2.angle - self.angle
        angle_dist = [np.sqrt(s ** 2) for s in angle_dist]

        total_alpha = 0
        total_beta = 0
        total_region_num = 0
        total_dist = 0
        total_angle = 0
        for s in coh_dist : total_alpha += s
        for s in incoh_dist : total_beta += s
        for s in region_num_dist : total_region_num += s
        for s in distance_dist : total_dist += s
        for s in angle_dist : total_angle += s

        result = total_alpha*alpha_weight + total_beta*beta_weight + total_region_num*region_num_weight + total_dist*distance_weight + total_angle*angle_weight
        return result


#####################################################################################################################
"""
add some functions later on
"""
#####################################################################################################################


if __name__ == '__main__':
    img1 = images[0]
    img2 = images[16]
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


