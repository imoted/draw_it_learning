#!/usr/bin/env python3

import numpy as np
import os
import pickle
import cv2
import math

files = os.listdir("./data")
files.sort()
# print(files)
x = []
x_load = []
y = []
y_load = []

# img_cv2 = cv2.imread("./qd_emo/0.png")
# print(type(img_cv2))
# print(img_cv2)
# cv2.imshow("color",img_cv2)
# k = cv2.waitKey(0)

def load_data():
    count = 0
    for file in files:
        file = "./data/" + file

        x = np.load(file)
        # x = x.astype('float32') / 255.       
        x = x[0:10000, :]
        x_load.append(x)
        y = [count for _ in range(10000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


features, labels = load_data()
# print(type(features))
print(np.array(features))
features = np.array(features)
for i in range(3):
    print(features.shape[i]) # 15 (ジャンル) / 10000 (ジャンル分けされた画像枚数)) / 784 (１画像の配列)

# for i in range(len(features)):
# sqrt_len = int(math.sqrt(len(features[0][1])))
# img_reshape = features[0][3].reshape( sqrt_len,sqrt_len)

# cv2.imshow("color",img_reshape)
# k = cv2.waitKey(0)

features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

print(features[0][0])
# features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
# labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])