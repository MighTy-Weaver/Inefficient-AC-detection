import os
import pickle
from datetime import datetime
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

categories = 4
available_cates = ["KMeans", "MiniKMeans"]
cate_name = "MiniKMeans"
cate_base = "plot"  # "Value" or "plot"
model_folder_name = 'Test_R2_HYPEROPT_SMOTE'


def original_dataloader(room: int):
    """
    This function is the observation dataloader
    """
    data = pd.read_csv('summer_data_compiled.csv', index_col=0).drop(['Time', 'Hour', 'Date'], axis=1)
    data = data[data.AC > 0]
    room_data = data[data.Location == room]
    y = room_data['AC']
    X = room_data.drop(['AC', 'Location'], axis=1)
    return X, y


folder_name = '{}_cate{}_{}_22_33/'.format(datetime.now().strftime("%Y%m%d%H%M%S"), categories, cate_name)
os.mkdir('{}/'.format(folder_name))
cate_symbol = ['a', 'b', 'c', 'd']
cates = [[] for _ in range(categories)]

if cate_name == "KMeans":
    cluster = KMeans(categories)
elif cate_name == "MiniKMeans":
    cluster = MiniBatchKMeans(categories)
else:
    cluster = KMeans(categories)

if cate_base == 'plot':
    images = glob('./Test_R2_HYPEROPT_SMOTE_clustering/shap_T_ac_plot/*.png')
    room_list = [img.split('\\')[-1].split('.')[0] for img in images]
    image_list = np.array([cv2.imread(img) for img in images])
    print(image_list.shape)
    image_list = image_list.reshape(len(image_list), -1)
    cluster.fit(image_list)
else:
    rooms = [mdl.split('\\')[-1].split('.')[0] for mdl in glob('./Test_R2_HYPEROPT_SMOTE/models/*.pickle.bat')]
    rooms_shap = []
    for room in tqdm(rooms, desc="Loading room's shapley value:"):
        model = pickle.load(open('./{}/models/{}.pickle.bat'.format(model_folder_name, room), 'rb'))
        explainer = shap.TreeExplainer(model)
        X, y = original_dataloader(int(room))
        temperatures = list(X['Temperature'])
        shap_values = [i[5] for i in explainer.shap_values(X)]
        # rooms_shap.append(np.array(shap_values))
        rooms_shap.append(np.array([(temperatures[i], shap_values[i]) for i in range(len(temperatures))]))
    rooms_shap = np.array(rooms_shap)
    np.save('./{}/shap.npy'.format(folder_name), rooms_shap)
    cluster.fit(rooms_shap)

print(cluster.labels_)

for i in range(categories):
    os.mkdir('{}/{}'.format(folder_name, cate_symbol[i]))
for ind, i in enumerate(cluster.labels_):
    cates[i].append(room_list[ind])
    copyfile(images[ind], '{}/{}/{}.png'.format(folder_name, cate_symbol[i], room_list[ind]))

np.save('./{}/category.npy'.format(folder_name), cates)
