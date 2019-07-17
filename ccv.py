# -*- coding:utf-8 -*-
# https://github.com/tamanobi/ccv/blob/master/ccv.py

# USAGE
# python ccv.py --input images/goose.jpg --threshold 5

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-n", "--threshold", type=int, default=64)
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


"""
Proccess of Computing CCV(color coherence vector)
1. Blur
2. Quantizing color
3. Thresholding
4. Labeling
5. Counting
"""

def ccv(src, tau=0, n=64):
    img = src.copy()
    row, col, channels = img.shape
    # blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # quantize color
    img = QuantizeColor(img, n)
    bgr = cv2.split(img)
    # bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    if tau == 0:
        tau = row * col * 0.1
    alpha = np.zeros(n)
    beta = np.zeros(n)
    # labeling
    for i, ch in enumerate(bgr):
        ret, th = cv2.threshold(ch, 127, 255, 0)
        ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None, connectivity=8)
        # generate ccv
        areas = [[v[4], label_idx] for label_idx, v in enumerate(stat)]
        coord = [[v[0], v[1]] for label_idx, v in enumerate(stat)]
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
    return alpha, beta


def ccv_plot(img, alpha, beta, n=64):
    import matplotlib.pyplot as plt
    X = [x for x in range(n * 2)]
    Y = alpha.tolist() + beta.tolist()
    with open('output/ccv.csv', 'w') as f:
        f.write(str(Y))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.subplot(2, 1, 2)
    plt.bar(X, Y, align='center')
    # plt.yscale('log')
    plt.xticks(X, (['alpha']*n)+(['beta']*n))
    plt.savefig('output/plot.png')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread(args["input"])
    n = args["threshold"]
    alpha, beta = ccv(img, tau=0, n=n)
    CCV = alpha.tolist() + beta.tolist()
    assert (sum(CCV) == img.size)
    assert (n == len(alpha) and n == len(beta))
    ccv_plot(img, alpha, beta, n)
