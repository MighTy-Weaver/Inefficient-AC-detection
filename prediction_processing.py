# This is the code for result visualization and model analysis.
# Please run this code only after you've got the models trained and dumped.
# Kindly notice that all the plots are default to save directly, it will not show out during the process.

import json
import os
import pickle
import warnings
from math import sqrt
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score
from tqdm import tqdm

# Ignore the warnings, let pandas print the full message and do some overall settings for matplotlib.
warnings.filterwarnings('ignore')
room_list = pd.read_csv('summer_data_compiled.csv')['Location'].unique()
plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"

folder_name = 'Original_Version_Models'


# Define our own original data loader
def original_dataloader(room: int):
    """This is the function to load the original data and run the SMOTE algorithm on them.
    It will return the X and y split from the original data after SMOTE. It takes the room number as the input
    parameter."""
    data = pd.read_csv('summer_data_compiled.csv', index_col=0).drop(['Time', 'Hour', 'Date'], axis=1)
    room_data = data[data.Location == room]
    room_data['SMOTE_split'] = (room_data['AC'] > 0.7).astype('int')
    X = room_data.drop(['SMOTE_split'], axis=1)
    y = room_data['SMOTE_split']
    model_smote = SMOTE(random_state=621, k_neighbors=3)
    room_data_smote, smote_split = model_smote.fit_resample(X, y)
    room_data_smote = pd.concat([room_data_smote, smote_split], axis=1)
    y = room_data_smote['AC']
    X = room_data_smote.drop(['AC', 'Location'], axis=1)
    return X, y


# Define our prediction dataloader.
def prediction_dataloader(room: int):
    """This function is a prediction dataloader, it will return the real value, prediction value, accuracy and RMSE of
    the predictions along with the accuracy. It take the room number as the input parameter."""
    data = pd.read_csv('./{}/prediction.csv'.format(folder_name), index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = np.array(json.loads(data.loc[0, 'real']))
    predict = np.array(json.loads(data.loc[0, 'predict']))
    r2 = r2_score(real, predict)
    rmse = sqrt(sum([(real[i] - predict[i]) ** 2 for i in range(len(real))]) / len(real))
    return real, predict, r2, rmse


# This is the function to view the shapley additive explanation's importance plot.
def view_shap_importance(room: int):
    """This function will plot the importance of the models by shapely additive explanations. It takes the room
    number as the input parameter."""
    model = pickle.load(open('./{}/models/{}.pickle.bat'.format(folder_name, room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)


# This is the function to calculate the RMSE value of a room, based on the model's predictions.
def calculate_R2_MSE_RMSE(room: int):
    data = pd.read_csv('./{}/prediction.csv'.format(folder_name), index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = json.loads(data.loc[0, 'real'])
    predict = json.loads(data.loc[0, 'predict'])
    square_sum = sum([(real[i] - predict[i]) ** 2 for i in range(len(real))])
    R2 = r2_score(real, predict)
    return R2, square_sum / len(real), sqrt(square_sum / len(real))


# This function plots the interacted shapley value for each room's temperature and humidity.
def plot_shap_interact(room: int):
    plt.rcParams.update({'font.size': 20})
    model = pickle.load(open('./{}/models/{}.pickle.bat'.format(folder_name, room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.dependence_plot("Temperature", shap_values, X, interaction_index="Humidity", save=True,
                         path="./{}/shap_TH_ac_plot/{}.png".format(folder_name, room), show=False,
                         title="Shapley Value for Temperature & Humidity of Room {}".format(room))
    plt.clf()


# This is the function to plot the distribution of the predictions according to the ground truth.
def plot_distribution(room: int):
    plt.rcParams.update({'font.size': 13})
    real, predict, r2, rmse = prediction_dataloader(room)
    plt.rc('font', family='Times New Roman')
    # Define our own color bar for the plot.
    newcolors = cm.get_cmap('viridis', 256)(np.linspace(0, 1, 256))
    newcolors[:128, :] = newcolors[128:, :]
    newcolors[128:, :] = np.flipud(newcolors[128:, :])
    newcmp = ListedColormap(newcolors)
    # Use the scatter plot to plot the distribution with our own color bar.
    plt.scatter(real, predict, c=real - predict, marker='o', label="(Real, Prediction)", s=10, cmap=newcmp, vmin=-0.8,
                vmax=0.8)
    real_range = np.linspace(min(real), max(real))
    # Plot the identity line of y=x
    plt.plot(real_range, real_range, color='m', linestyle="-.", linewidth=1, label="Identity Line (y=x)")
    plt.title("Prediction Validation Graph of Room {}".format(room))
    plt.ylabel("Prediction")
    plt.legend(frameon=False)
    plt.colorbar(label="Error (Real Value - Prediction)")
    plt.xlabel(
        "Original AC\nR2 score: {}\nRoot Mean Square Error: {}".format(round(r2, 4), round(rmse, 4)))
    plt.savefig("./{}/distribution_plot/room{}.png".format(folder_name, room), bbox_inches='tight')
    # plt.show()
    plt.clf()


# This is the function to plot the error and root mean square error distribution of all the rooms.
def plot_error_distribution():
    r2_list, mse_list, rmse_list = [], [], []
    # Looping through all the room and collect the statistics we need.
    for room_f in room_list:
        if room_f == 309 or room_f == 312 or room_f == 917 or room_f == 1001:
            continue
        r2, mse, rmse = calculate_R2_MSE_RMSE(room_f)
        r2_list.append(r2)
        rmse_list.append(rmse)
        mse_list.append(mse)
    plt.rcParams.update({'font.size': 15})
    # Use the historgram in matplotlib to plot the accuracy distribution histogram.
    plt.hist(r2_list, bins=50, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("R2 Score\nMean R2 Score: {}".format(round(mean(r2_list), 2)))
    plt.ylabel("Occurrence")
    plt.title("The R2 Score Distribution Histogram")
    plt.savefig('./{}/R2Dis.png'.format(folder_name), bbox_inches='tight')
    plt.clf()
    # Use the same to plot the root mean square error distribution histogram.
    plt.hist(rmse_list, bins=50, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Root Mean Square Error\nMean RMSE: {}".format(round(mean(rmse_list), 4)))
    plt.ylabel("Occurrence")
    plt.title("The RMSE Distribution Histogram")
    plt.savefig('./{}/RMSEDis.png'.format(folder_name), bbox_inches='tight')
    plt.clf()
    plt.hist(mse_list, bins=50, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Mean Square Error\nMean MSE: {}".format(round(mean(mse_list), 4)))
    plt.ylabel("Occurrence")
    plt.title("The MSE Distribution Histogram")
    plt.savefig('./{}/MSEDis.png'.format(folder_name), bbox_inches='tight')
    plt.clf()


# The main function
if __name__ == "__main__":
    if not os.path.exists('./{}/shap_TH_ac_plot/'.format(folder_name)):
        os.mkdir('./{}/shap_TH_ac_plot/'.format(folder_name))
    if not os.path.exists('./{}/distribution_plot/'.format(folder_name)):
        os.mkdir('./{}/distribution_plot/'.format(folder_name))
    if not os.path.exists('./{}/SMOTE_room/'.format(folder_name)):
        os.mkdir('./{}/SMOTE_room/'.format(folder_name))
    for room in tqdm(room_list):
        # Delete the rooms with low quality data manually.
        if room == 309 or room == 312 or room == 917 or room == 1001:
            continue
        # view_shap_importance(room)  # This function will pop up a figure window for each room.
        plot_shap_interact(room)
        plot_distribution(room)
    plot_error_distribution()
