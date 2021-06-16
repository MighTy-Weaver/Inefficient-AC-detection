import os
from datetime import datetime
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

categories = 3
available_cates = ["KMeans", "MiniKMeans"]
cate_name = "KMeans"

folder_name = '{}_cate{}_{}/'.format(datetime.now().strftime("%Y%m%d%H%M%S"), categories, cate_name)
cate_symbol = ['a', 'b', 'c', 'd']
cates = [[] for _ in range(categories)]

images = glob('./Test_R2_HYPEROPT_SMOTE/shap_T_ac_plot/*.png')
room_list = [img.split('\\')[-1].split('.')[0] for img in images]
image_list = np.array([cv2.imread(img) for img in images])
image_list = image_list.reshape(len(image_list), -1)

if cate_name == "KMeans":
    cluster = KMeans(categories)
elif cate_name == "MiniKMeans":
    cluster = MiniBatchKMeans(categories)
else:
    cluster = KMeans(categories)
cluster.fit(image_list)
print(cluster.labels_)

os.mkdir('{}/'.format(folder_name))
for i in range(categories):
    os.mkdir('{}/{}'.format(folder_name, cate_symbol[i]))
for ind, i in enumerate(cluster.labels_):
    cates[i].append(room_list[ind])
    copyfile(images[ind], '{}/{}/{}.png'.format(folder_name, cate_symbol[i], room_list[ind]))

np.save('./{}/category.npy'.format(folder_name), cates)
