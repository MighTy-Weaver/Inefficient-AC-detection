import json
import os
import pickle
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import DMatrix, cv

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("Summer_data_prev_AC_updated.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

if not os.path.exists('./shap_importance_plot'):
    os.mkdir("./shap_importance_plot")
if not os.path.exists('./shap_TH_ac_plot'):
    os.mkdir('./shap_TH_ac_plot')
if not os.path.exists('./distribution_plot2'):
    os.mkdir('./distribution_plot2')
if not os.path.exists('./models2/'):
    os.mkdir('./models2')


def ten_percent_accuracy(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    total = len(predt)
    truth_value = dtrain.get_label()
    correct = 0
    for i in range(total):
        if predt[i] * 0.9 < truth_value[i] < predt[i] * 1.1:
            correct += 1
    return "10%Accuracy", float(correct / total)


def overall_ten_percent_accuracy(pred: np.ndarray, real: np.ndarray):
    total = len(pred)
    correct = 0
    real90 = real * 0.9
    real110 = real * 1.1
    for i in range(len(pred)):
        if real90[i] < pred[i] < real110[i]:
            correct += 1
    return float(correct / total), real90, real110


error_csv = pd.DataFrame(
    columns=['room', 'train-10%Accuracy-mean', 'train-10%Accuracy-std', 'train-rmse-mean', 'train-rmse-std',
             'test-10%Accuracy-mean', 'test-10%Accuracy-std', 'test-rmse-mean', 'test-rmse-std'])
prediction_csv = pd.DataFrame(columns=['room', 'real', 'predict'])

# ranging through all the rooms
for room in tqdm(data['Location'].unique()):
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue
    room_data = data[data.Location == room]
    room_data['SMOTE_split'] = (room_data['AC'] > 0.7).astype('int')
    X = room_data.drop(['SMOTE_split'], axis=1)
    y = room_data['SMOTE_split']
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, smote_split = model_smote.fit_resample(X, y)
    room_data_smote = pd.concat([room_data_smote, smote_split], axis=1)
    y = room_data_smote['AC']
    X = room_data_smote.drop(['AC', 'Location', 'SMOTE_split'], axis=1)
    param_dict = {'objective': 'reg:squarederror', 'max_depth': 10, 'reg_alpha': 7, 'min_child_weight': 0.1,
                  'n_estimators': 300, 'colsample_bytree': 0.8, 'learning_rate': 0.05}
    xgb_model = None
    for train, test in KFold(n_splits=10, shuffle=True, random_state=621).split(X, y):
        input_matrix = DMatrix(data=X.loc[train], label=y.loc[train])
        test_matrix = DMatrix(data=X.loc[test], label=y.loc[test])
        watchlist = [(test_matrix, 'eval'), (input_matrix, 'train')]
        if xgb_model is None:
            xgb_model = xgb.train(params=param_dict, dtrain=input_matrix, num_boost_round=300, evals=watchlist,
                                  feval=ten_percent_accuracy, maximize=True, xgb_model=None, verbose_eval=False)
        else:
            xgb_model = xgb.train(params=param_dict, dtrain=input_matrix, num_boost_round=300, evals=watchlist,
                                  feval=ten_percent_accuracy, maximize=True, xgb_model=xgb_model, verbose_eval=False)
    data_matrix = DMatrix(data=X, label=y)
    xgb_cv_result = cv(dtrain=data_matrix, params=param_dict, as_pandas=True, num_boost_round=300, nfold=10,
                       shuffle=True, seed=621, feval=ten_percent_accuracy, maximize=True)
    xgb_cv_result['room'] = room
    pickle.dump(xgb_model, open("./models2/" + str(room) + ".pickle.bat", "wb"))
    error_csv.loc[len(error_csv)] = xgb_cv_result.loc[len(xgb_cv_result) - 1]
    prediction = xgb_model.predict(data_matrix)
    real = np.array(y)
    prediction_csv.loc[len(prediction_csv)] = {'room': room, 'real': json.dumps(real.tolist()),
                                               'predict': json.dumps(prediction.tolist())}

error_csv.to_csv('./error2.csv', index=False)
prediction_csv.to_csv('./prediction2.csv', index=False)
