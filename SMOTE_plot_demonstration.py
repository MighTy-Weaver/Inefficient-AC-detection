# This is the code for demonstrating the effect of SMOTE algorithm.
# By plotting the distribution histogram of data, we can show that SMOTE is
# capable of helping imbalanced data to distribute more evenly, in order to avoid over-fitting.

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE

# Ignore the warnings.
warnings.filterwarnings('ignore')

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)

# Set some general settings of all the matplotlib.
plt.rcParams.update({'font.size': 15})
plt.rc('font', family='Times New Roman')

# plot the histogram of the AC value before the SMOTE algorithm. It will be saved to the current work directory
plt.hist(data['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("AC value")
plt.ylabel("Occurrence")
plt.title("AC Value Distribution Before SMOTE Algorithm")
plt.savefig('./SMOTE_before.png')
plt.clf()

# Label all the AC data by 0.7, all AC above 0.7 will be marked as 1, otherwise 0. Split into X and y
data['SMOTE_split'] = (data['AC'] > 0.7).astype('int')
X = data.drop(['SMOTE_split'], axis=1)
y = data['SMOTE_split']

# Run the SMOTE algorithm and retrieve the result.
model_smote = SMOTE(random_state=621, k_neighbors=3)
room_data_smote, smote_split = model_smote.fit_resample(X, y)

# Plot the AC distribution histogram after the SMOTE algorithm.
plt.hist(room_data_smote['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("AC value")
plt.ylabel("Occurrence")
plt.title("AC Value Distribution After SMOTE Algorithm")
plt.savefig('./SMOTE_After.png')
plt.clf()
