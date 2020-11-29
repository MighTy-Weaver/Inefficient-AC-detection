import pandas as pd

# Load the data with a positive AC electricity consumption value, and drop the time data as we don't need them
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedKFold

data = pd.read_csv("Summer_data_prev_AC_updated.csv", index_col=0)
data = data[data.AC > 0].drop(['Time', 'Date', 'Hour'], axis=1)

# Generate the room list containing all the rooms
room_list = data['Location'].unique()

# ranging through all the rooms
for room in room_list:
    # Four rooms have low quality data and we delete them manually
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue
    room_data = data[data.Location == room]
    room_data['SMOTE_split'] = (room_data['AC'] > 0.7).astype('int')
    X = room_data.drop(['SMOTE_split'], axis=1)
    y = room_data['SMOTE_split']
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, _ = model_smote.fit_resample(X, y)
    y = room_data_smote['AC']
    X = room_data_smote.drop(['AC'], axis=1)
    rkf = RepeatedKFold(n_splits=6, n_repeats=2, random_state=621)
    for train, test in rkf.split(X, y):
        print(y[test])
