import json
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings('ignore')
room_list = pd.read_csv('./Summer_data_prev_AC_updated.csv')['Location'].unique()
plt.rc('font', family='Times New Roman')
plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams["savefig.bbox"] = "tight"


def original_dataloader(room: int):
    data = pd.read_csv('./Summer_data_prev_AC_updated.csv', index_col=0).drop(['Time', 'Hour', 'Date'], axis=1)
    room_data = data[data.Location == room]
    room_data['SMOTE_split'] = (room_data['AC'] > 0.7).astype('int')
    X = room_data.drop(['SMOTE_split'], axis=1)
    y = room_data['SMOTE_split']
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, smote_split = model_smote.fit_resample(X, y)
    room_data_smote = pd.concat([room_data_smote, smote_split], axis=1)
    y = room_data_smote['AC']
    X = room_data_smote.drop(['AC', 'Location', 'SMOTE_split'], axis=1)
    return X, y


def prediction_dataloader():
    data = pd.read_csv('./prediction.csv', index_col=None)
    room_list = []
    original_list = []
    predict_list = []
    for i in range(len(data)):
        room_list.append(data.loc[i, 'room'])
        original_list.append(json.loads(data.loc[i, 'real']))
        predict_list.append(json.loads(data.loc[i, 'predict']))
    return room_list, original_list, predict_list


def plot_shap(room: int):
    model = pickle.load(open('./models2/{}.pickle.bat'.format(room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)
    shap.dependence_plot("Temperature", shap_values, X, interaction_index="Humidity", save=True,
                         path="./shap_TH_ac_plot/{}.png".format(room), show=False)


def plot_distribution(room: int, real: list, predict: list):

    sns.set_style(style="whitegrid")
    sns.set(font_scale=5)
    plt.rc('font', family='Times New Roman')
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.rcParams["savefig.bbox"] = "tight"
    color = abs(real - prediction) / real
    sns.regplot(x=real, y=prediction, scatter=True, y_jitter=0.45, x_jitter=0.1, marker="x",
                scatter_kws={"s": 80, "color": "black"}, line_kws={"color": "red"})
    # plt.scatter(x=real, y=prediction, c=color, marker='x')
    real_range = np.linspace(min(real), max(real))
    sns.lineplot(x=real_range, y=real_range, color='blue', dashes=True)
    sns.lineplot(x=real_range, y=0.9 * real_range, color='blue', dashes=True)
    sns.lineplot(x=real_range, y=1.1 * real_range, color='blue', dashes=True)
    plt.title("Prediction Validation Graph of Room {}".format(room))
    plt.ylabel("Prediction")
    plt.xlabel("Original AC\nAccuracy: {}%".format(round(100 * accuracy, 2)))
    plt.savefig("./distribution_plot2/room{}.png".format(room))
    plt.clf()


for room in tqdm(room_list):
    if room == 309 or room == 312 or room == 917 or room == 1001:
        continue
    plot_shap(room)
