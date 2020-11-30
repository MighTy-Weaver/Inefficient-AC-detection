import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams.update({'font.size': 15})
plt.rc('font', family='Times New Roman')
# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
data = pd.read_csv("data_compiled.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1).reset_index(drop=True)
plt.hist(data['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("AC value")
plt.ylabel("Occurrence")
plt.title("AC Value Distribution Before SMOTE Algorithm")
plt.savefig('./SMOTE_before.png')
plt.clf()
data['SMOTE_split'] = (data['AC'] > 0.7).astype('int')
X = data.drop(['SMOTE_split'], axis=1)
y = data['SMOTE_split']
model_smote = SMOTE(random_state=621, k_neighbors=3)
room_data_smote, smote_split = model_smote.fit_resample(X, y)
plt.hist(room_data_smote['AC'], bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("AC value")
plt.ylabel("Occurrence")
plt.title("AC Value Distribution After SMOTE Algorithm")
plt.savefig('./SMOTE_After.png')
plt.clf()
