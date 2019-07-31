"""
http://www.ccis2k.org/iajit/PDF/vol.7,no.3/891.pdf
Improving the Effectiveness of the Color Coherence Vector
The modified Color Coherence Vector based on
    1. the number of the color coherence regions (CCV_N)
    2. the distance of the color coherence regions (CCV_D)
    3. the angle of the color coherence regions (CCV_A)
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


PATH = 'detected/two_people/'
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
    # blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # quantize color
    img = QuantizeColor(img, n)
    bgr = cv2.split(img)
    # bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if tau == 0:
        tau = row * col * 0.1
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = [0,0]
    # labeling
    for i, ch in enumerate(bgr):
        ret, th = cv2.threshold(ch, 127, 255, 0)
        ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None, connectivity=8)
        # generate ccv
        areas = [[v[4], label_idx] for label_idx, v in enumerate(stat)]
        coord = [[v[0], v[1]] for label_idx, v in enumerate(stat)]
        # print(areas, coord)
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
    return alpha, beta, gamma


def ccv_plot(img, alpha, beta, gamma, n=64):
    X = [x for x in range(n * 2)]
    Y = alpha.tolist() + beta.tolist()
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 1, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.bar(X, Y, align='center')
    plt.yscale('log')
    plt.xticks(X, (['alpha']*n)+(['beta']*n))
    plt.savefig('output/two_people/{}_plot.png'.format(args["input"]))
    plt.show()


if __name__ == '__main__':
    img = cv2.imread(PATH + args["input"])
    n = args["threshold"]
    alpha, beta, gamma = ccv(img, tau=0, n=n)
    a_l = alpha.tolist()
    b_l = beta.tolist()
    CCV = list(zip(a_l, b_l))
    print(CCV)
    # CCV = alpha.tolist() + beta.tolist()
    ccv_plot(img, alpha, beta, gamma, n)

    with open('output/two_people/{}_ccv.csv'.format(args["input"]), 'w') as f:
        f.write(str(CCV))
