# This is the code to train the xgboost model with cross-validation for each unique room in the dataset.
# Models are dumped into ./models and results are dumped into two csv files in the current work directory.

import argparse
import json
import math
import os
import pickle
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numpy.random import RandomState
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import DMatrix, cv

# Set up an argument parser to decide the metric function
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=['R2', 'RMSE', '10acc'], type=str, required=False, default='R2',
                    help="The evaluation metric you want to use to train the XGBoost model")
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
# log_folder_name = "Test_{}_{}".format(args.metric, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
log_folder_name = "Test_R2_HYPEROPT_v11"
previous_parameter_folder = "Test_R2_HYPEROPT_v10"

if not os.path.exists('./{}/'.format(log_folder_name)):
    os.mkdir('./{}'.format(log_folder_name))
    os.mkdir('./{}/models/'.format(log_folder_name))
    os.mkdir('./{}/trntst_models/'.format(log_folder_name))


# Define our evaluation functions
def ten_percent_accuracy(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    total = len(predt)
    truth_value = dtrain.get_label()
    correct = sum([1 for i in range(total) if predt[i] * 0.9 < truth_value[i] < predt[i] * 1.1])
    return "10%Accuracy", float(correct / total)


def RMSE(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    truth_value = dtrain.get_label()
    root_squard_error = math.sqrt(mean_squared_error(truth_value, predt))
    return "RMSE", root_squard_error


def R2(predt: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    truth_value = dtrain.get_label()
    r2_value = r2_score(truth_value, predt)
    return "R2", r2_value


def fobjective(space):
    param_dict_tunning = {'max_depth': int(space['max_depth']),
                          'learning_rate': space['learning_rate'],
                          'colsample_bytree': space['colsample_bytree'],
                          'min_child_weight': int(space['min_child_weight']),
                          'reg_alpha': int(space['reg_alpha']),
                          'reg_lambda': space['reg_lambda'],
                          'subsample': space['subsample'],
                          'min_split_loss': space['min_split_loss'],
                          'objective': 'reg:squarederror'}

    xgb_cv_result = xgb.cv(dtrain=data_matrix, params=param_dict_tunning, nfold=5,
                           early_stopping_rounds=30, as_pandas=True, num_boost_round=200,
                           seed=seed, metrics='rmse', maximize=False, shuffle=True)

    return {"loss": (xgb_cv_result["test-rmse-mean"]).tail(1).iloc[0], "status": STATUS_OK}


eval_dict = {'RMSE': RMSE, 'R2': R2, '10acc': ten_percent_accuracy}

print("Start Training The Models")
# Create two dataframes to store the result during the training and after the training.
error_csv = pd.DataFrame(
    columns=['room', 'train-{}-mean'.format(args.metric), 'train-{}-std'.format(args.metric), 'train-rmse-mean',
             'train-rmse-std', 'test-{}-mean'.format(args.metric), 'test-{}-std'.format(args.metric), 'test-rmse-mean',
             'test-rmse-std'])
prediction_csv = pd.DataFrame(columns=['room', 'observation', 'prediction'])

room_list = data['Location'].unique()

# ranging through all the rooms and do the training and cross-validation for each room.
for room in tqdm(room_list):
    seed = 2030 + room
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 826 or room == 917 or room == 1001:
        continue

    # We extract the data of particular room and run the SMOTE algorithm on it.
    room_data = data[data.Location == room].drop(['Location'], axis=1).reset_index(drop=True)

    y = room_data['AC'].fillna(method='pad')
    X = room_data.drop(['AC'], axis=1).fillna(method='pad')

    X = X.to_numpy()

    # Build another full data matrix for the built-in cross validation function to work.
    data_matrix = DMatrix(data=X, label=y)

    # Cross_validation with hyper-parameter tuning
    space = {'max_depth': hp.quniform("max_depth", 3, 10, 1),
             'learning_rate': hp.uniform("learning_rate", 0.1, 3),
             'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
             'min_child_weight': hp.quniform("min_child_weight", 1, 20, 1),
             'reg_alpha': hp.quniform("reg_alpha", 0, 100, 1),
             'reg_lambda': hp.uniform("reg_lambda", 0, 2),
             'subsample': hp.uniform("subsample", 0.5, 1),
             'min_split_loss': hp.uniform("min_split_loss", 0, 9)}

    if os.path.exists('./{}/models/{}_parameter.npy'.format(previous_parameter_folder, room)):
        best_param_dict = np.load('./{}/models/{}_parameter.npy'.format(previous_parameter_folder, room),
                                  allow_pickle=True).item()
        np.save('./{}/models/{}_parameter.npy'.format(log_folder_name, room), best_param_dict)
    else:
        trials = Trials()
        best_hyperparams = fmin(fn=fobjective, space=space, algo=tpe.suggest, max_evals=400, trials=trials,
                                rstate=RandomState(seed))

        # setup our training parameters and a model variable as model checkpoint
        best_param_dict = {'objective': 'reg:squarederror', 'max_depth': int(best_hyperparams['max_depth']),
                           'reg_alpha': best_hyperparams['reg_alpha'], 'reg_lambda': best_hyperparams['reg_lambda'],
                           'min_child_weight': best_hyperparams['min_child_weight'],
                           'colsample_bytree': best_hyperparams['colsample_bytree'],
                           'learning_rate': best_hyperparams['learning_rate'],
                           'subsample': best_hyperparams['subsample'],
                           'min_split_loss': best_hyperparams['min_split_loss']}
        np.save('./{}/models/{}_parameter.npy'.format(log_folder_name, room), best_param_dict)

    # Use the built-in cv function to do the cross validation, still with ten folds, this will return us the results.
    xgb_cv_result = cv(dtrain=data_matrix, params=best_param_dict, nfold=5,
                       early_stopping_rounds=30, as_pandas=True, num_boost_round=200,
                       seed=seed, shuffle=True, feval=eval_dict[args.metric], maximize=True)

    xgb_cv_result['room'] = room
    error_csv.loc[len(error_csv)] = xgb_cv_result.loc[len(xgb_cv_result) - 1]

    # Use one training_testing for ploting, and save both ground truth and prediction value into the dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    d_train = DMatrix(X_train, label=y_train)
    d_test = DMatrix(X_test, label=y_test)

    watchlist = [(d_test, 'eval'), (d_train, 'train')]

    xgb_model_train_test = xgb.train(params=best_param_dict, dtrain=d_train, num_boost_round=200, evals=watchlist,
                                     verbose_eval=args.log, xgb_model=None, feval=eval_dict[args.metric], maximize=True)

    prediction = np.array(xgb_model_train_test.predict(d_test)).tolist()
    real = np.array(y_test).tolist()

    prediction_csv.loc[len(prediction_csv)] = {'room': room, 'observation': json.dumps(real),
                                               'prediction': json.dumps(prediction)}

    # Dump the error dataframes into csv files.
    error_csv.to_csv('./{}/error.csv'.format(log_folder_name), index=False)
    prediction_csv.to_csv('./{}/prediction.csv'.format(log_folder_name), index=False)

    # Develop a model using the whole orignial dataset, and save the model
    xgb_model_full = xgb.train(params=best_param_dict, dtrain=data_matrix, num_boost_round=200, evals=watchlist,
                               verbose_eval=args.log, xgb_model=None, feval=eval_dict[args.metric], maximize=True)

    pickle.dump(xgb_model_train_test, open('./{}/trntst_models/{}.pickle.bat'.format(log_folder_name, room), 'wb'))
    pickle.dump(xgb_model_full, open('./{}/models/{}.pickle.bat'.format(log_folder_name, room), 'wb'))

print("Training finished!")
