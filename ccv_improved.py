"""
http://www.ccis2k.org/iajit/PDF/vol.7,no.3/891.pdf
Improving the Effectiveness of the Color Coherence Vector
The modified Color Coherence Vector based on
    1. the number of the color coherence regions (CCV_N)
    2. the distance of the color coherence regions (CCV_D)
    3. the angle of the color coherence regions (CCV_A)
"""

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
import matplotlib.pyplot as plt
import json

THRESHOLD = 0.0003
PATH = 'detected/'
# PATH = 'detected/two_people/'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-n", "--threshold", type=int)
args = vars(ap.parse_args())

def QuantizeColor(img, n=64):
    div = 256 // n
    rgb = cv2.split(img)
    q = []
    for ch in rgb:
        vf = np.vectorize(lambda x, div: int(x // div) * div)
        quantized = vf(ch, div)
        q.append(quantized.astype(np.uint8))
    d_img = cv2.merge(q)
    return d_img


def ccv(src, tau=0, n=64):
    img = src.copy()
    row, col, channels = img.shape
    print("width:{}, height:{}".format(col, row))
    center_x = int(col / 2)
    center_y = int(row / 2)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = QuantizeColor(img, n)
    bgr = cv2.split(img)
    if tau == 0:
        tau = row * col * THRESHOLD
    alpha = np.zeros(n)
    beta = np.zeros(n)
    R = []      # CCV_D
    theta = []  # CCV_A
    num = []    # CCV_N

    # labeling
    for i, ch in enumerate(bgr):
        retval, th = cv2.threshold(ch, 127, 255, 0)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image=th,
                                                                            labels=None,
                                                                            stats=cv2.CC_STAT_AREA,
                                                                            centroids=None,
                                                                            connectivity=8)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nbgr:",i)
        # print("retval:\n{}\nlabels:\n{}\nstats:\n{}\ncentroids:\n{}".format(retval, labels, stats, centroids))
        num.append(retval)      # CCV_N

        _R = 0
        _theta = 0
        for i,(x,y) in enumerate(centroids):
            if i is 0: # background
                continue
            x -= center_x
            y -= center_y
            dist = np.sqrt(x**2 + y**2)     # CCV_D
            angle = np.arctan2(y, x) * 180 / np.pi      # CCV_A
            _R += dist
            _theta += angle
        R.append(_R)
        theta.append(_theta)

        # generate ccv
        areas = [[v[4], label_idx] for label_idx, v in enumerate(stats)]
        coord = [[v[0], v[1]] for label_idx, v in enumerate(stats)]
        # Counting
        for a, c in zip(areas, coord):
            area_size = a[0]
            x, y = c[0], c[1]
            if (x < ch.shape[1]) and (y < ch.shape[0]):
                bin_idx = int(ch[y, x] // (256 // n))
                if area_size >= tau:
                    alpha[bin_idx] = alpha[bin_idx] + area_size
                else:
                    beta[bin_idx] = beta[bin_idx] + area_size

    # for i in range(retval):
    #     _x = int(centroids[i][0])
    #     _y = int(centroids[i][1])
    #     cv2.circle(img, (_x, _y), 5, (255,0,0),1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return alpha, beta, num, R, theta


def ccv_plot(img, alpha, beta, n=64):
    X = [x for x in range(n * 2)]
    Y = alpha.tolist() + beta.tolist()
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 1, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.bar(X, Y, align='center')
    plt.yscale('log')
    plt.xticks(X, (['alpha']*n)+(['beta']*n))
    plt.savefig('output/two_people/{}_plot.png'.format(args["input"]))
    plt.show()

def normalize_factors(src, alpha, beta, num, R, theta): # output would be noramlized ver of all these inputs
    img = src.copy()
    row, col, channels = img.shape
    img_size = row*col
    center_x = int(col / 2)
    center_y = int(row / 2)

    # 1. Normalize coh/incoh (alpha / beta)
    norm_alpha = alpha/img_size
    norm_beta = beta / img_size

    # 2. Normalize num (of coh regions)
    norm_num = [float(s)*THRESHOLD for s in num]
    #norm_num = float(num)*THRESHOLD

    # 3. Normalize R (distance sum)
    x_diff = int(col-center_x)-center_x
    y_diff = int(row-center_y)-center_y
    max_dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
    norm_R = R / (max(num) * max_dist)

    # 4. Normalize theta (angle sum)
    norm_theta = [s / (max(num) * 359) for s in theta]
    # norm_theta = theta / (max(num) * 359)

    return norm_alpha, norm_beta, norm_num, norm_R, norm_theta

if __name__ == '__main__':
    img = cv2.imread(PATH + args["input"])
    n = args["threshold"]
    alpha, beta, num, R, theta = ccv(img, tau=0, n=n)
    a_l = alpha.tolist()
    b_l = beta.tolist()
    CCV = list(zip(a_l, b_l))
    print("CCV:", CCV, len(CCV))
    print("num:{}\nR:{}\ntheta:{}".format(num, R, theta))
    # CCV = alpha.tolist() + beta.tolist()
    # ccv_plot(img, alpha, beta, n)

    # normalized factor vals
    N_alpha, N_beta, N_num, N_R, N_theta = normalize_factors(img, alpha, beta, num, R, theta)
    print("N_alpha:{}\n N_beta:{}\n N_num:{}\n N_R:{}\n N_theta:{}\n ".format(N_alpha, N_beta, N_num, N_R, N_theta))

    with open('output/two_people/{}_ccv.csv'.format(args["input"]), 'w') as f:
        f.write(str(CCV))
