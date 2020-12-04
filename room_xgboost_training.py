# This is the code to train the xgboost model with cross-validation for each unique room in the dataset.
# Models are dumped into ./models and results are dumped into two csv files in the current work directory.

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

# Ignore all the warnings and set pandas to display every column and row everytime we print a dataframe
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

# Make some system path to store the results.
if not os.path.exists('./shap_TH_ac_plot'):
    os.mkdir('./shap_TH_ac_plot')
if not os.path.exists('./distribution_plot'):
    os.mkdir('./distribution_plot')
if not os.path.exists('./models/'):
    os.mkdir('./models')


# Define our own evaluation function for training and cross-validation, here we define a correct prediction
# interval as 10% error interval.
def ten_percent_accuracy(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    total = len(predt)
    truth_value = dtrain.get_label()
    correct = 0
    for i in range(total):
        if predt[i] * 0.9 < truth_value[i] < predt[i] * 1.1:
            correct += 1
    return "10%Accuracy", float(correct / total)


# Create two dataframes to store the result during the training and after the training.
error_csv = pd.DataFrame(
    columns=['room', 'train-10%Accuracy-mean', 'train-10%Accuracy-std', 'train-rmse-mean', 'train-rmse-std',
             'test-10%Accuracy-mean', 'test-10%Accuracy-std', 'test-rmse-mean', 'test-rmse-std'])
prediction_csv = pd.DataFrame(columns=['room', 'real', 'predict'])

# ranging through all the rooms and do the training and cross-validation for each room.
for room in tqdm(data['Location'].unique()):
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue

    # We extract the data of particular room and run the SMOTE algorithm on it.
    room_data = data[data.Location == room]

    # Label all the AC data by 0.7, all AC above 0.7 will be marked as 1, otherwise 0. Split into X and y
    room_data['SMOTE_split'] = (room_data['AC'] > 0.7).astype('int')
    X = room_data.drop(['SMOTE_split'], axis=1)
    y = room_data['SMOTE_split']

    # Run the SMOTE algorithm and retrieve the result.
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, smote_split = model_smote.fit_resample(X, y)

    # concat the result from SMOTE and split the result into X and y for training.
    room_data_smote = pd.concat([room_data_smote, smote_split], axis=1)
    y = room_data_smote['AC']
    X = room_data_smote.drop(['AC', 'Location', 'SMOTE_split'], axis=1)

    # setup our training parameters and a model variable as model checkpoint
    param_dict = {'objective': 'reg:squarederror', 'max_depth': 10, 'reg_alpha': 7, 'min_child_weight': 0.1,
                  'n_estimators': 300, 'colsample_bytree': 0.8, 'learning_rate': 0.05}
    xgb_model = None

    # We use KFold in scikit-learn as our cross-validation mechanism. 10 folds is used for each room.
    for train, test in KFold(n_splits=10, shuffle=True, random_state=621).split(X, y):
        # Build the input and output matrix for training in each fold and setup the watchlist for training validation.
        input_matrix = DMatrix(data=X.loc[train], label=y.loc[train])
        test_matrix = DMatrix(data=X.loc[test], label=y.loc[test])
        watchlist = [(test_matrix, 'eval'), (input_matrix, 'train')]

        # If there is already a checkpoint model, we'll keep training the checkpoint,
        # Otherwise, it will train from the beginning. It will ultimately bring us the cross-validated xgboost model.
        if xgb_model is None:
            xgb_model = xgb.train(params=param_dict, dtrain=input_matrix, num_boost_round=300, evals=watchlist,
                                  feval=ten_percent_accuracy, maximize=True, xgb_model=None, verbose_eval=False)
        else:
            xgb_model = xgb.train(params=param_dict, dtrain=input_matrix, num_boost_round=300, evals=watchlist,
                                  feval=ten_percent_accuracy, maximize=True, xgb_model=xgb_model, verbose_eval=False)

    # Build another full data matrix for the built-in cross validation function to work.
    data_matrix = DMatrix(data=X, label=y)
    # Use the built-in cv function to do the cross validation, still with ten folds, this will return us the results.
    xgb_cv_result = cv(dtrain=data_matrix, params=param_dict, as_pandas=True, num_boost_round=300, nfold=10,
                       shuffle=True, seed=621, feval=ten_percent_accuracy, maximize=True)

    # Dump our model into the folder and results of cross validation into the dataframe.
    xgb_cv_result['room'] = room
    pickle.dump(xgb_model, open("./models/" + str(room) + ".pickle.bat", "wb"))
    error_csv.loc[len(error_csv)] = xgb_cv_result.loc[len(xgb_cv_result) - 1]

    # Do the prediction on the full original dataset, and save both ground truth and prediction value into the dataframe
    prediction = xgb_model.predict(data_matrix)
    real = np.array(y)
    prediction_csv.loc[len(prediction_csv)] = {'room': room, 'real': json.dumps(real.tolist()),
                                               'predict': json.dumps(prediction.tolist())}

# dump the two dataframes into csv files.
error_csv.to_csv('./error.csv', index=False)
prediction_csv.to_csv('./prediction.csv', index=False)
