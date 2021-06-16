import os
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

images = glob('./Test_R2_HYPEROPT_SMOTE/shap_T_ac_plot/*.png')
room_list = [img.split('\\')[-1].split('.')[0] for img in images]
image_list = np.array([cv2.imread(img) for img in images])
image_list = image_list.reshape(len(image_list), -1)
print(room_list)
print(images)

kmeans = MiniBatchKMeans(4)
kmeans.fit(image_list)
print(kmeans.labels_)
cate_a, cate_b, cate_c, cate_d = [], [], [], []

os.mkdir('catea/')
os.mkdir('cateb/')
os.mkdir('catec/')
os.mkdir('cated/')
for ind, i in enumerate(kmeans.labels_):
    if i == 0:
        cate_a.append(room_list[ind])
        copyfile(images[ind], 'catea/{}.png'.format(room_list[ind]))
    if i == 1:
        cate_b.append(room_list[ind])
        copyfile(images[ind], 'cateb/{}.png'.format(room_list[ind]))
    if i == 2:
        cate_c.append(room_list[ind])
        copyfile(images[ind], 'catec/{}.png'.format(room_list[ind]))
    if i == 3:
        cate_d.append(room_list[ind])
        copyfile(images[ind], 'cated/{}.png'.format(room_list[ind]))
category = {1: cate_a, 2: cate_b, 3: cate_c, 4: cate_d}
np.save('./category.npy', category)
