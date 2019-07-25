# skeleton code from https://github.com/kohjingyu/color-coherence-vectors/blob/master/ccv.ipynb

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
import glob
import random


def is_adjacent(x1, y1, x2, y2):
    ''' Returns true if (x1, y1) is adjacent to (x2, y2), and false otherwise '''
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    return not (x_diff == 1 and y_diff == 1) and (x_diff <= 1 and y_diff <= 1)
    # 대각선은 adjacent하지 않다고 봄


def find_max_cliques(arr, n):
    ''' Returns a 2*n dimensional vector
    v_i, v_{i+1} describes the number of coherent and incoherent pixels respectively a given color
    '''
    tau = int(arr.shape[0] * arr.shape[1] * 0.01)  # Classify as coherent is area is >= 1%
    ccv = [0 for i in range(n ** 3 * 2)]
    unique = np.unique(arr) #  입력된 배열에서 중복되지 않는 고유한 요소들의 배열 리턴
    for u in unique:
        x, y = np.where(arr == u) # 좌표값
        groups = []
        coherent = 0
        incoherent = 0

        for i in range(len(x)):
            found_group = False
            for group in groups:
                if found_group:
                    break

                for coord in group:
                    xj, yj = coord
                    if is_adjacent(x[i], y[i], xj, yj):
                        found_group = True
                        group[(x[i], y[i])] = 1
                        break
            if not found_group:
                groups.append({(x[i], y[i]): 1})

        for group in groups:
            num_pixels = len(group)
            if num_pixels >= tau:
                coherent += num_pixels
            else:
                incoherent += num_pixels

        assert (coherent + incoherent == len(x))

        index = int(u)
        ccv[index * 2] = coherent
        ccv[index * 2 + 1] = incoherent

    return ccv
    # returns 2*n 행렬 (1행엔 coherent 정보/ 2행엔 incoherent 정보)




def get_ccv(img, n): # n은 한 채널의 컬러히스토그램에서의 bin의 개수 (color를 몇개로 나눌꺼냐)
    # Blur pixel slightly using avg pooling with 3x3 kernel
    blur_img = cv2.blur(img, (3, 3))
    blur_flat = blur_img.reshape(32 * 32, 3)

    # Discretize colors
    hist, edges = np.histogramdd(blur_flat, bins=n)
    # hist = n-d array
    # edges = A list of bin edges for each dimension

    graph = np.zeros((img.shape[0], img.shape[1]))
    result = np.zeros(blur_img.shape)

    total = 0
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                rgb_val = [edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]]
                previous_edge = [edges[0][i], edges[1][j], edges[2][k]]
                coords = ((blur_img <= rgb_val) & (blur_img >= previous_edge)).all(axis=2)
                result[coords] = rgb_val
                graph[coords] = i + j * n + k * n ** 2

    result = result.astype(int)
    return find_max_cliques(graph, n)
    # returns 2*n 행렬 (1행엔 coherent 정보/ 2행엔 incoherent 정보)
    # CCV for n # of color bins

n = 2  # indicating 2^3 discretized colors
feature_size = n ** 3 * 2  # Number of discretized colors * 2 for coherent and incoherent


def extract_features(image):
    return get_ccv(image, n)  # image.flatten()


def shuffle_data(data, labels):
    p = np.random.permutation(len(data))
    return data[p], labels[p]


def load_data(dataset="train", classes=["airplane", "automobile", "bird", "cat"]):
    random.seed(1337)

    data = []
    labels = []

    for i, c in enumerate(classes):
        for file in glob.glob("data/{}/{}/*.jpg".format(dataset, c)):
            one_hot_label = np.zeros(len(classes))
            one_hot_label[i] = 1
            labels.append(one_hot_label)

            img = np.array(Image.open(file))
            features = extract_features(img)
            data.append(features)

    data, labels = np.array(data), np.array(labels)

    if dataset == "train":
        data, labels = shuffle_data(data, labels)

    return data, labels