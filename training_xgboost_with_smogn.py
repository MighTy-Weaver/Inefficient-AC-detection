# This is the code to train the xgboost model with cross-validation for each unique room in the dataset.
# Models are dumped into ./models and results are dumped into two csv files in the current work directory.

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from xgboost import DMatrix, cv

# Set up an argument parser to decide the metric function
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=['1-R2', 'RMSE', '10acc', 'MSE'], type=str, required=False, default='1-R2',
                    help="The objective metric you want to use to train the XGBoost model")
parser.add_argument("--log", choices=[0, 1, 100], type=int, required=False, default=0,
                    help="Whether to print out the training progress")
args = parser.parse_args()

# Ignore all the warnings and set pandas to display every column and row everytime we print a dataframe
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("summer_data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

# Create some directory to store the models and future analysis figures.
log_folder_name = "Test_{}_{}".format(args.metric, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir('./{}'.format(log_folder_name))
os.mkdir('./{}/models/'.format(log_folder_name))


# Define our evaluation functions
def ten_percent_accuracy(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    total = len(predt)
    truth_value = dtrain.get_label()
    correct = sum([1 for i in range(total) if predt[i] * 0.9 < truth_value[i] < predt[i] * 1.1])
    return "10%Accuracy", float(correct / total)


def RMSE(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    truth_value = dtrain.get_label()
    squared_error = sum([np.power(truth_value[i] - predt[i], 2) for i in range(len(predt))])
    return "RMSE", float(np.sqrt(squared_error / len(predt)))


def MSE(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    truth_value = dtrain.get_label()
    return "MSE", float(sum([np.power(truth_value[i] - predt[i], 2) for i in range(len(predt))]) / len(predt))


def R2(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    truth_value = dtrain.get_label()
    var = np.var(truth_value)
    _, mse = MSE(predt, dtrain)
    R2 = 1 - float(mse / var)
    # Note that R2<=1, so in order to make it the loss, we have to make 1-R2 as small as possible.
    return "1-R2", 1 - R2


objective_dict = {'RMSE': RMSE, '1-R2': R2, 'MSE': MSE, '10acc': ten_percent_accuracy}

print("Start Training The Models")
# Create two dataframes to store the result during the training and after the training.
error_csv = pd.DataFrame(
    columns=['room', 'train-{}-mean'.format(args.metric), 'train-{}-std'.format(args.metric), 'train-rmse-mean',
             'train-rmse-std', 'test-{}-mean'.format(args.metric), 'test-{}-std'.format(args.metric), 'test-rmse-mean',
             'test-rmse-std'])
prediction_csv = pd.DataFrame(columns=['room', 'real', 'predict'])

# ranging through all the rooms and do the training and cross-validation for each room.
for room in tqdm(data['Location'].unique()):
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue

    # We extract the data of particular room and run the SMOTE algorithm on it.
    room_data = data[data.Location == room]

    if len(room_data) < 500:
        continue

    room_data = pd.read_csv('./SMOGN_processed/{}.csv'.format(room), index_col=0)
    y = room_data['AC']
    X = room_data.drop(['AC'], axis=1)

    # Apply the sample weight strategy to the AC value
    class_sample = pd.cut(y, bins=10)
    weight = compute_sample_weight(class_weight="balanced", y=class_sample)

    # setup our training parameters and a model variable as model checkpoint
    param_dict = {'objective': 'reg:squarederror', 'max_depth': 10, 'reg_alpha': 7, 'min_child_weight': 1,
                  'colsample_bytree': 0.8, 'learning_rate': 0.005}

    X = X.to_numpy()

    # Build another full data matrix for the built-in cross validation function to work.
    data_matrix = DMatrix(data=X, label=y, weight=weight)
    # Use the built-in cv function to do the cross validation, still with ten folds, this will return us the results.
    xgb_cv_result = cv(dtrain=data_matrix, params=param_dict, as_pandas=True, num_boost_round=300, nfold=3,
                       shuffle=True, seed=621, feval=objective_dict[args.metric], maximize=True)

    watchlist = [(data_matrix, 'eval'), (data_matrix, 'train')]
    xgb_model = xgb.train(params=param_dict, dtrain=data_matrix, num_boost_round=300, evals=watchlist,
                          verbose_eval=args.log, feval=objective_dict[args.metric])
    pickle.dump(xgb_model, open('./{}/models/{}.pickle.bat'.format(log_folder_name, room), 'wb'))

    xgb_cv_result['room'] = room
    error_csv.loc[len(error_csv)] = xgb_cv_result.loc[len(xgb_cv_result) - 1]

    # Do the prediction on the full original dataset, and save both ground truth and prediction value into the dataframe
    prediction = xgb_model.predict(data_matrix)
    real = np.array(y)
    prediction_csv.loc[len(prediction_csv)] = {'room': room, 'real': json.dumps(real.tolist()),
                                               'predict': json.dumps(prediction.tolist())}

print("Training finished!")

# dump the two dataframes into csv files.
error_csv.to_csv('./{}/error.csv'.format(log_folder_name), index=False)
prediction_csv.to_csv('./{}/prediction.csv'.format(log_folder_name), index=False)
