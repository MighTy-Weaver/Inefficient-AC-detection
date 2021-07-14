# This is the code for demonstrating the effect of SMOTE algorithm.
# By plotting the distribution histogram of data, we can show that SMOTE is
# capable of helping imbalanced data to distribute more evenly, in order to avoid over-fitting.

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

folder_name = 'Test_R2_HYPEROPT_SMOTE'

# Make the folder to store the result
if not os.path.exists('./{}/SMOTE_room/'.format(folder_name)):
    os.mkdir('./{}/SMOTE_room/'.format(folder_name))

# Ignore the warnings.
warnings.filterwarnings('ignore')

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("summer_data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

# Set some general settings of all the matplotlib.
plt.rcParams.update({'font.size': 15})
plt.rc('font', family='Times New Roman')

# Set up two empty list to record the AC before and after for each room
AC_before = []
AC_after = []

# ranging through all the rooms
for room in tqdm(data['Location'].unique()):
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue
    data_room = data[data.Location == room]

    # plot the histogram of the AC value before the SMOTE algorithm. It will be saved to the current work directory
    plt.hist(data_room['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("AC value")
    plt.ylabel("Occurrence")
    plt.title("AC Value Distribution for Room {} Before SMOTE".format(room))
    plt.savefig('./{}/SMOTE_room/room{}before.png'.format(folder_name, room))
    plt.clf()

    AC_before.extend(list(data_room['AC']))

    # Label all the AC data by 0.7, all AC above 0.7 will be marked as 1, otherwise 0. Split into X and y
    data_room['SMOTE_split'] = (data_room['AC'] > 0.7).astype('int')
    X = data_room.drop(['SMOTE_split'], axis=1)
    y = data_room['SMOTE_split']

    # Run the SMOTE algorithm and retrieve the result.
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, smote_split = model_smote.fit_resample(X, y)

    # Plot the AC distribution histogram after the SMOTE algorithm.
    plt.hist(room_data_smote['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("AC value")
    plt.ylabel("Occurrence")
    plt.title("AC Value Distribution for Room {} After SMOTE".format(room))
    plt.savefig('./{}/SMOTE_room/room{}after.png'.format(folder_name, room))
    plt.clf()

    AC_after.extend(room_data_smote['AC'])

# With the AC values in each room before and after the SMOTE, we can plot a distribution histogram for all rooms.
plt.hist(AC_before, bins=100, facecolor="blue", edgecolor="black", align='mid')
plt.xlabel("Hourly AC Electricity Consumption/kWh", fontsize=18)
plt.ylabel("Number of Samples", fontsize=18)
plt.ylim(ymax=12500, ymin=0)
plt.savefig('./{}/SMOTE_Before.png'.format(folder_name), bbox_inches='tight')
plt.clf()

plt.hist(AC_after, bins=100, facecolor="blue", edgecolor="black", align='mid')
plt.xlabel("Hourly AC Electricity Consumption/kWh", fontsize=18)
plt.ylabel("Number of Samples", fontsize=18)
plt.ylim(ymax=12500, ymin=0)
plt.savefig('./{}/SMOTE_After.png'.format(folder_name), bbox_inches='tight')
plt.clf()
