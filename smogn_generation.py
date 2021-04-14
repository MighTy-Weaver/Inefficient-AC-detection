# This is the code to train the xgboost model with cross-validation for each unique room in the dataset.
# Models are dumped into ./models and results are dumped into two csv files in the current work directory.

import os
import warnings

import pandas as pd
import smogn
from tqdm import tqdm

# Ignore all the warnings and set pandas to display every column and row everytime we print a dataframe
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("summer_data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

# Create some directory to store the models and future analysis figures.
log_folder_name = "SMOGN_processed"
if not os.path.exists('./{}/'.format(log_folder_name)):
    os.mkdir('./{}'.format(log_folder_name))

# ranging through all the rooms and do the training and cross-validation for each room.
for room in tqdm(data['Location'].unique()):
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue

    # We extract the data of particular room and run the SMOTE algorithm on it.
    room_data = data[data.Location == room].drop(['Location'], axis=1).reset_index(drop=True).fillna(method='pad')

    if len(room_data) < 500 or room <= 812:
        continue

    room_data_smogn = smogn.smoter(data=room_data, y='AC', rel_coef=0.1)
    room_data_smogn.to_csv('./{}/{}.csv'.format(log_folder_name, room))
    y = room_data_smogn['AC']
    X = room_data_smogn.drop(['AC'], axis=1)
