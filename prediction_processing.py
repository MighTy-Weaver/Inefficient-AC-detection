import json
import pickle
import warnings
from math import sqrt
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
room_list = pd.read_csv('data_compiled.csv')['Location'].unique()
plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"


def original_dataloader(room: int):
    data = pd.read_csv('data_compiled.csv', index_col=0).drop(['Time', 'Hour', 'Date'], axis=1)
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


def prediction_dataloader(room: int):
    data = pd.read_csv('./prediction.csv', index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = json.loads(data.loc[0, 'real'])
    predict = json.loads(data.loc[0, 'predict'])
    correct = 0
    for i in range(len(real)):
        if 0.9 * real[i] <= predict[i] <= 1.1 * real[i]:
            correct += 1
    return float(correct / len(real)), real, predict


def view_shap_importance(room: int):
    model = pickle.load(open('./models2/{}.pickle.bat'.format(room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)


def calculate_RMSE(room: int):
    data = pd.read_csv('./prediction.csv', index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = json.loads(data.loc[0, 'real'])
    predict = json.loads(data.loc[0, 'predict'])
    square_sum = sum([(real[i] - predict[i]) ** 2 for i in range(len(real))])
    correct = 0
    for i in range(len(real)):
        if 0.9 * real[i] <= predict[i] <= 1.1 * real[i]:
            correct += 1
    return float(correct / len(real)), sqrt(square_sum / len(real))


def plot_shap_interact(room: int):
    model = pickle.load(open('./models2/{}.pickle.bat'.format(room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.dependence_plot("Temperature", shap_values, X, interaction_index="Humidity", save=True,
                         path="./shap_TH_ac_plot/{}.png".format(room), show=False)


def plot_distribution(room: int):
    accuracy, real, prediction = prediction_dataloader(room)
    plt.rc('font', family='Times New Roman')
    sns.regplot(x=real, y=prediction, scatter=True, y_jitter=0.45, x_jitter=0.2, marker=".",
                scatter_kws={"s": 15, "color": "black"},
                line_kws={"color": "red", "label": "Real-Prediction Regression Line"}, label="(Real, Prediction)")
    real_range = np.linspace(min(real), max(real))
    sns.lineplot(x=real_range, y=real_range, color='blue', dashes=True, style=True, label="Identity Line")
    sns.lineplot(x=real_range, y=0.9 * real_range, color='blue', dashes=True, style=True)
    sns.lineplot(x=real_range, y=1.1 * real_range, color='blue', dashes=[real_range], style=True)
    plt.title("Prediction Validation Graph of Room {}".format(room))
    plt.ylabel("Prediction")
    plt.legend()
    plt.xlabel("Original AC\nAccuracy: {}%".format(round(100 * accuracy, 2)))
    plt.savefig("./distribution_plot/room{}.png".format(room))
    plt.clf()


def plot_error_distribution():
    accuracy_list, rmse_list = [], []
    for room_f in room_list:
        if room_f == 309 or room_f == 312 or room_f == 917 or room_f == 1001:
            continue
        acc, rmse = calculate_RMSE(room_f)
        accuracy_list.append(acc * 100)
        rmse_list.append(rmse)
    plt.rcParams.update({'font.size': 15})
    plt.hist(accuracy_list, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Accuracy(%)\nMean Accuracy: {}%".format(round(mean(accuracy_list), 2)))
    plt.ylabel("Occurrence")
    plt.title("The Accuracy Distribution Histogram")
    plt.savefig('./AccDis.png')
    plt.clf()
    plt.hist(rmse_list, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Root Mean Square Error\nMean RMSE: {}".format(round(mean(rmse_list), 4)))
    plt.ylabel("Occurrence")
    plt.title("The RMSE Distribution Histogram")
    plt.savefig('./RMSEDis.png')
    plt.clf()


if __name__ == "__main__":
    # for room in tqdm(room_list):
    #     if room == 309 or room == 312 or room == 917 or room == 1001:
    #         continue
    #
    #     plot_shap_interact(room)
    #     plot_distribution(room)
    plot_error_distribution()
