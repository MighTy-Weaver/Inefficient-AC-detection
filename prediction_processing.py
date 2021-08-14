# This is the code for result visualization and model analysis.
# Please run this code only after you've got the models trained and dumped.
# Kindly notice that all the plots are default to save directly, it will not show out during the process.

import json
import math
import os
import pickle
import warnings
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import shap
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from xgboost import DMatrix

folder_name = 'Test_R2_HYPEROPT_SMOTE_clustering'

# Ignore the warnings, let pandas print the full message and do some overall settings for matplotlib.
warnings.filterwarnings('ignore')
room_list = pd.read_csv('./{}/prediction.csv'.format(folder_name))['room'].unique()
plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"

trntst_r2_list, trntst_rmse_list = [], []


# Define our own original data loader
def original_dataloader(room: int):
    """
    This function is the observation dataloader
    """
    data = pd.read_csv('summer_data_compiled.csv', index_col=0).drop(['Time', 'Hour', 'Date'], axis=1)
    data = data[data.AC > 0]
    room_data = data[data.Location == room]
    y = room_data['AC']
    X = room_data.drop(['AC', 'Location'], axis=1)
    return X, y


# Define our prediction dataloader.
def prediction_dataloader(room: int):
    """This function is a prediction dataloader, it will return the real value, prediction value, accuracy and RMSE of
    the predictions along with the accuracy. It take the room number as the input parameter."""
    data = pd.read_csv('./{}/prediction.csv'.format(folder_name), index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = np.array(json.loads(data.loc[0, 'observation']))
    predict = np.array(json.loads(data.loc[0, 'prediction']))
    return real, predict


# This is the function to view the shapley additive explanation's importance plot.
def view_shap_importance(room: int):
    """This function will plot the importance of the models by shapely additive explanations. It takes the room
    number as the input parameter."""
    model = pickle.load(open('./{}/models/{}.pickle.bat'.format(folder_name, room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)


# This function plots the interacted shapley value for each room's temperature and humidity.
def plot_shap_interact(room: int):
    plt.rcParams.update({'font.size': 20})
    model = pickle.load(open('./{}/models/{}.pickle.bat'.format(folder_name, room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    if room in [903, 332]:
        shap.dependence_plot("Temperature", shap_values, X, interaction_index="Wifi_count", save=True,
                             path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                             title="Shapley Value for Temperature & WIFI count of Room {}".format('1'),
                             xlimit=[22, 33], ylimit=[-0.25, 0.2], fontsize=21)
    elif room in [328, 630]:
        shap.dependence_plot("Temperature", shap_values, X, interaction_index="Wifi_count", save=True,
                             path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                             title="Shapley Value for Temperature & WIFI count of Room {}".format('2'),
                             xlimit=[22, 33], ylimit=[-0.25, 0.2], fontsize=21)
    elif room in [910, 821]:
        shap.dependence_plot("Temperature", shap_values, X, interaction_index="Wifi_count", save=True,
                             path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                             title="Shapley Value for Temperature & WIFI count of Room {}".format('3'),
                             xlimit=[22, 33], ylimit=[-0.25, 0.2], fontsize=21)
    elif room in [1007, 1011]:
        shap.dependence_plot("Temperature", shap_values, X, interaction_index="Wifi_count", save=True,
                             path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                             title="Shapley Value for Temperature & WIFI count of Room {}".format('4'),
                             xlimit=[22, 33], ylimit=[-0.25, 0.2], fontsize=21)
    else:
        shap.dependence_plot("Temperature", shap_values, X, interaction_index="Wifi_count", save=True,
                             path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                             title="Shapley Value for Temperature & WIFI count of Room {}".format(room),
                             xlimit=[22, 33], ylimit=[-0.25, 0.2], fontsize=21)
    shap.dependence_plot("Temperature", shap_values, X, interaction_index=None, save=True,
                         path="./{}/shap_T_ac_plot/{}.png".format(folder_name, room), show=False,
                         title="Shapley Value for Temperature of Room {}".format(room), xlimit=[22, 33],
                         ylimit=[-0.25, 0.2], fontsize=1, axis_hide=True)
    plt.clf()


# This is the function to plot the distribution of the predictions according to the ground truth.
def plot_distribution(room: int):
    plt.rcParams.update({'font.size': 13})

    input, real = original_dataloader(room)
    matrix = DMatrix(input, label=real)
    model = pickle.load(open('./{}/trntst_models/{}.pickle.bat'.format(folder_name, room), 'rb'))
    predict = list(model.predict(matrix))
    full_r2 = round(r2_score(real, predict), 4)
    full_rmse = round(math.sqrt(mean_squared_error(real, predict)), 4)

    error = pd.read_csv('./{}/error.csv'.format(folder_name), index_col=None)
    room_error = error[error.room == room].reset_index(drop=True)
    plt.rc('font', family='Times New Roman')
    # Define our own color bar for the plot.
    newcolors = cm.get_cmap('viridis', 256)(np.linspace(0, 1, 256))
    newcolors[:128, :] = newcolors[128:, :]
    newcolors[128:, :] = np.flipud(newcolors[128:, :])
    newcmp = ListedColormap(newcolors)
    # Use the scatter plot to plot the distribution with our own color bar.
    plt.scatter(real, predict, c=real - predict, marker='o', label="(Observation, Prediction)", s=10, cmap=newcmp,
                vmin=-0.8, vmax=0.8)
    # real_range = np.linspace(min(real), max(real))

    if room == 916:
        real_range = np.linspace(0, 2)
    else:
        real_range = np.linspace(0, 1.25)

    # real_range = np.linspace(0, 2)

    # Plot the identity line of y=x
    plt.plot(real_range, real_range, color='m', linestyle="-.", linewidth=1, label="Identity Line (y=x)")
    if room == 328:
        plt.title("Prediction Validation Graph of Room {}".format('A'), fontsize=20)
    elif room == 621:
        plt.title("Prediction Validation Graph of Room {}".format('B'), fontsize=20)
    elif room == 819:
        plt.title("Prediction Validation Graph of Room {}".format('C'), fontsize=20)
    else:
        plt.title("Prediction Validation Graph of Room {}".format(room), fontsize=20)
    plt.ylabel("Prediction", fontsize=17)
    plt.legend(frameon=False, fontsize=15)
    cb = plt.colorbar()
    cb.set_label("Error (Observation - Prediction)", size=17)
    plt.xlabel(
        "Observation\n#data: {}  R2-score: {}  RMSE: {}".format(
            len(real), round(room_error.loc[0, 'test-R2-mean'], 4), round(room_error.loc[0, 'test-rmse-mean'], 4)),
        fontsize=18)
    if room == 916:
        plt.xlim([0, 2])
        plt.ylim([0, 2])
    else:
        plt.xlim([0, 1.25])
        plt.ylim([0, 1.25])
    plt.savefig("./{}/distribution_plot/room{}.png".format(folder_name, room), bbox_inches='tight')
    # plt.show()
    plt.clf()
    trntst_r2_list.append(full_r2)
    trntst_rmse_list.append(full_rmse)


# This is the function to plot the error and root mean square error distribution of all the rooms.
def plot_error_distribution(bin=20):
    r2_list, rmse_list = [], []
    stat_log = pd.read_csv('./{}/error.csv'.format(folder_name), index_col=None)
    # Looping through all the room and collect the statistics we need.
    for room_f in room_list:
        if room_f == 309 or room_f == 312 or room_f == 826 or room_f == 917 or room_f == 1001:
            continue
        if stat_log[stat_log.room == room_f].reset_index(drop=True).loc[0, 'test-R2-mean'] > 0:
            r2_list.append(stat_log[stat_log.room == room_f].reset_index(drop=True).loc[0, 'test-R2-mean'])
            rmse_list.append(stat_log[stat_log.room == room_f].reset_index(drop=True).loc[0, 'test-rmse-mean'])
        else:
            continue
    r2_mean, r2_std = np.mean(r2_list), np.std(r2_list)
    rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
    plt.rcParams.update({'font.size': 15})

    # Use the historgram in matplotlib to plot the accuracy distribution histogram.
    fig, ax = plt.subplots()
    n, r2_bins, patches = ax.hist(r2_list, bins=bin, density=True, facecolor="blue", rwidth=0.8, alpha=0.7)
    y = ((1 / (np.sqrt(2 * np.pi) * r2_std)) *
         np.exp(-0.5 * (1 / r2_std * (r2_bins - r2_mean)) ** 2))
    ax.plot(r2_bins, y, '--')
    plt.xlabel("R2 Score\nMean R2 Score: {}".format(round(mean(r2_list), 4)), fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title("The R2 Score Distribution Histogram (Cross-Validation)", fontsize=20)
    plt.savefig('./{}/R2Dis_CV_bin{}.png'.format(folder_name, bin), bbox_inches='tight')
    plt.clf()

    # Use the same to plot the root mean square error distribution histogram.
    fig, ax = plt.subplots()
    _, rmse_bins, _ = ax.hist(rmse_list, bins=bin, density=True, facecolor="blue", rwidth=0.8, alpha=0.7)
    y = ((1 / (np.sqrt(2 * np.pi) * rmse_std)) *
         np.exp(-0.5 * (1 / rmse_std * (rmse_bins - rmse_mean)) ** 2))
    ax.plot(rmse_bins, y, '--')
    plt.xlabel("Root Mean Square Error\nMean RMSE: {}".format(round(mean(rmse_list), 4)), fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title("The RMSE Distribution Histogram (Cross-Validation)", fontsize=20)
    plt.savefig('./{}/RMSEDis_CV_bin{}.png'.format(folder_name, bin), bbox_inches='tight')
    plt.clf()

    r2_mean, r2_std = np.mean(trntst_r2_list), np.std(trntst_r2_list)
    rmse_mean, rmse_std = np.mean(trntst_rmse_list), np.std(trntst_rmse_list)
    plt.rcParams.update({'font.size': 15})

    if trntst_r2_list:
        # Use the historgram in matplotlib to plot the accuracy distribution histogram.
        fig, ax = plt.subplots()
        n, r2_bins, patches = ax.hist(trntst_r2_list, bins=bin, density=True, facecolor="blue", rwidth=0.8, alpha=0.7)
        y = ((1 / (np.sqrt(2 * np.pi) * r2_std)) *
             np.exp(-0.5 * (1 / r2_std * (r2_bins - r2_mean)) ** 2))
        ax.plot(r2_bins, y, '--')
        plt.xlabel("R2 Score\nMean R2 Score: {}".format(round(mean(trntst_r2_list), 4)))
        plt.ylabel("Frequency")
        plt.title("The R2 Score Distribution Histogram (Trained on 80% data)")
        plt.savefig('./{}/R2Dis_trntst_bin{}.png'.format(folder_name, bin), bbox_inches='tight')
        plt.clf()

    if trntst_rmse_list:
        # Use the same to plot the root mean square error distribution histogram.
        fig, ax = plt.subplots()
        _, rmse_bins, _ = ax.hist(trntst_rmse_list, bins=bin, density=True, facecolor="blue", rwidth=0.8, alpha=0.7)
        y = ((1 / (np.sqrt(2 * np.pi) * rmse_std)) *
             np.exp(-0.5 * (1 / rmse_std * (rmse_bins - rmse_mean)) ** 2))
        ax.plot(rmse_bins, y, '--')
        plt.xlabel("Root Mean Square Error\nMean RMSE: {}".format(round(mean(trntst_rmse_list), 4)))
        plt.ylabel("Frequency")
        plt.title("The RMSE Distribution Histogram (Trained on 80% data)")
        plt.savefig('./{}/RMSEDis_trntst_bin{}.png'.format(folder_name, bin), bbox_inches='tight')
        plt.clf()


def plot_room_number_data_and_R2():
    stat_log = pd.read_csv('./{}/error.csv'.format(folder_name), index_col=None)
    predic = pd.read_csv('./{}/prediction.csv'.format(folder_name), index_col=None)
    r2_list, rmse_list, data_number = [], [], []
    for room_f in room_list:
        if room_f == 309 or room_f == 312 or room_f == 826 or room_f == 917 or room_f == 1001:
            continue
        r2_list.append(stat_log[stat_log.room == room_f].reset_index(drop=True).loc[0, 'test-R2-mean'])
        rmse_list.append(stat_log[stat_log.room == room_f].reset_index(drop=True).loc[0, 'test-rmse-mean'])
        data_number.append(len(json.loads(predic[predic.room == room_f].reset_index(drop=True).loc[0, 'observation'])))
    seaborn.regplot(x=data_number, y=r2_list, scatter=True, label="R2 Score", marker="x", color="red",
                    scatter_kws={"s": 15})
    seaborn.regplot(x=data_number, y=rmse_list, label="RMSE", marker="o", scatter_kws={"s": 15})
    plt.legend()
    plt.savefig('./{}/data_number.png'.format(folder_name), bbox_inches='tight')


# The main function
if __name__ == "__main__":
    if not os.path.exists('./{}/shap_TH_ac_plot/'.format(folder_name)):
        os.mkdir('./{}/shap_TH_ac_plot/'.format(folder_name))
    if not os.path.exists('./{}/shap_T_ac_plot/'.format(folder_name)):
        os.mkdir('./{}/shap_T_ac_plot/'.format(folder_name))
    if not os.path.exists('./{}/distribution_plot/'.format(folder_name)):
        os.mkdir('./{}/distribution_plot/'.format(folder_name))
    # if not os.path.exists('./{}/SMOTE_room/'.format(folder_name)):
    #     os.mkdir('./{}/SMOTE_room/'.format(folder_name))
    for room in tqdm(room_list):
        # Delete the rooms with low quality data manually.
        if room == 309 or room == 312 or room == 826 or room == 917 or room == 1001:
            continue
        # view_shap_importance(room)  # This function will pop up a demo window for each room.
        plot_shap_interact(room)
        # plot_distribution(room)
    # plot_error_distribution(23)
    # plot_room_number_data_and_R2()
