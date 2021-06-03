import os
from shutil import copy

import pandas as pd
from tqdm import trange

folder_name = "Test_R2_Best_Among_SMOTE_SMOGN"
if not os.path.exists('./{}/'.format(folder_name)):
    os.mkdir('./{}/'.format(folder_name))
    os.mkdir('./{}/models/'.format(folder_name))
    os.mkdir('./{}/trntst_models/'.format(folder_name))
SMOTE = pd.read_csv('./Test_R2_HYPEROPT_SMOTE/error.csv', index_col=None)
SMOGN = pd.read_csv('./Test_R2_HYPEROPT_SMOGN/error.csv', index_col=None)
SMOTE_prediction = pd.read_csv('./Test_R2_HYPEROPT_SMOTE/prediction.csv', index_col=None)
SMOGN_prediction = pd.read_csv('./Test_R2_HYPEROPT_SMOGN/prediction.csv', index_col=None)
error = pd.DataFrame(columns=list(SMOTE))
prediction = pd.DataFrame(columns=list(SMOTE_prediction))

for i in trange(len(SMOTE)):
    if SMOTE.loc[i, 'test-R2-mean'] < SMOGN.loc[i, 'test-R2-mean']:
        error = error.append(SMOGN.loc[i], ignore_index=True)
        prediction = prediction.append(SMOGN_prediction.loc[i], ignore_index=True)
        copy('./Test_R2_HYPEROPT_SMOGN/models/{}.pickle.bat'.format(int(SMOGN.loc[i, 'room'])),
             './{}/models/'.format(folder_name))
        copy('./Test_R2_HYPEROPT_SMOGN/trntst_models/{}.pickle.bat'.format(int(SMOGN.loc[i, 'room'])),
             './{}/trntst_models/'.format(folder_name))
    else:
        error = error.append(SMOTE.loc[i], ignore_index=True)
        prediction = prediction.append(SMOTE_prediction.loc[i], ignore_index=True)
        copy('./Test_R2_HYPEROPT_SMOTE/models/{}.pickle.bat'.format(int(SMOTE.loc[i, 'room'])),
             './{}/models/'.format(folder_name))
        copy('./Test_R2_HYPEROPT_SMOTE/trntst_models/{}.pickle.bat'.format(int(SMOTE.loc[i, 'room'])),
             './{}/trntst_models/'.format(folder_name))
error.to_csv('./{}/error.csv'.format(folder_name), index=False)
prediction.to_csv('./{}/prediction.csv'.format(folder_name), index=False)
