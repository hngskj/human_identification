"""
http://www.ccis2k.org/iajit/PDF/vol.7,no.3/891.pdf
Improving the Effectiveness of the Color Coherence Vector
The modified Color Coherence Vector based on
    1. the number of the color coherence regions (CCV_N)
    2. the distance of the color coherence regions (CCV_D)
    3. the angle of the color coherence regions (CCV_A)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from ccv import config


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
        tau = row * col * config.THRESHOLD_tau
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
    plt.savefig(config.OUTPUT_PATH + '{}_plot.png'.format(config.IMAGE_NAME))
    plt.show()


if __name__ == '__main__':
    img = cv2.imread(config.IMAGE_PATH + config.IMAGE_NAME)
    # n = args["threshold"]
    alpha, beta, num, R, theta = ccv(img, tau=0, n=config.THRESHOLD_qauntizer)
    a_l = alpha.tolist()
    b_l = beta.tolist()
    CCV = list(zip(a_l, b_l))
    print("CCV:", CCV, len(CCV))
    print("num:{}\nR:{}\ntheta:{}".format(num, R, theta))
    # CCV = alpha.tolist() + beta.tolist()
    # ccv_plot(img, alpha, beta, n)

    # with open('output/two_people/{}_ccv.csv'.format(args["input"]), 'w') as f:
    #     f.write(str(CCV))
