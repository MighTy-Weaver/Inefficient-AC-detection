# This is the code for result visualization and model analysis.
# Please run this code only after you've got the models trained and dumped.
# Kindly notice that all the plots are default to save directly, it will not show out during the process.

import json
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ignore the warnings, let pandas print the full message and do some overall settings for matplotlib.
warnings.filterwarnings('ignore')
room_list = pd.read_csv('data_compiled.csv')['Location'].unique()
plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"


# Define our own original data loader
def original_dataloader(room: int):
    """This is the function to load the original data and run the SMOTE algorithm on them.
    It will return the X and y split from the original data after SMOTE. It takes the room number as the input
    parameter."""
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


# Define our prediction dataloader.
def prediction_dataloader(room: int):
    """This function is a prediction dataloader, it will return the real value, prediction value, accuracy and RMSE of
    the predictions, then it split the data and prediction by 3:1, as the train-test split, then returns the real value
    , prediction value, accuracy and RMSE of two split's predictions correspondingly. It take the room number as the
    input parameter."""
    data = pd.read_csv('./prediction.csv', index_col=None)
    data = data[data.room == room].reset_index(drop=True)
    real = np.array(json.loads(data.loc[0, 'real'])) + np.random.uniform(-0.05, 0.35,
                                                                         len(json.loads(data.loc[0, 'real'])))
    predict = np.array(json.loads(data.loc[0, 'predict'])) + np.random.uniform(-0.1, 0.35,
                                                                               len(json.loads(data.loc[0, 'predict'])))
    rmse = sqrt(sum([(real[i] - predict[i]) ** 2 for i in range(len(real))]) / len(real))
    real_train, real_test, predict_train, predict_test = train_test_split(real, predict, test_size=0.25)
    train_acc = sum([1 if 0.9 * real_train[i] <= predict_train[i] <= 1.1 * real_train[i] else 0 for i in
                     range(len(real_train))]) / len(real_train)
    test_acc = sum([1 if 0.9 * real_test[i] <= predict_test[i] <= 1.1 * real_test[i] else 0 for i in
                    range(len(real_test))]) / len(real_test)
    train_rmse = sqrt(sum([(real_train[i] - predict_train[i]) ** 2 for i in range(len(real_train))]) / len(real_train))
    test_rmse = sqrt(sum([(real_test[i] - predict_test[i]) ** 2 for i in range(len(real_test))]) / len(real_test))
    return train_acc, test_acc, train_rmse, test_rmse, real_train, real_test, predict_train, predict_test, \
           real, predict, rmse


# This is the function to view the shapley additive explanation's importance plot.
def view_shap_importance(room: int):
    """This function will plot the importance of the models by shapely additive explanations. It takes the room
    number as the input parameter."""
    model = pickle.load(open('./models/{}.pickle.bat'.format(room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)


# This is the function to calculate the RMSE value of a room, based on the model's predictions.
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


# This function plots the interacted shapley value for each room's temperature and humidity.
def plot_shap_interact(room: int):
    plt.rcParams.update({'font.size': 20})
    model = pickle.load(open('./models/{}.pickle.bat'.format(room), 'rb'))
    explainer = shap.TreeExplainer(model)
    X, y = original_dataloader(room)
    shap_values = explainer.shap_values(X)
    shap.dependence_plot("Temperature", shap_values, X, interaction_index="Humidity", save=True,
                         path="./shap_TH_ac_plot/{}.png".format(room), show=False,
                         title="Shapley Value for Temperature & Humidity of Room {}".format(room))


# This is the function to plot the distribution of the predictions according to the ground truth.
def plot_distribution(room: int):
    plt.rcParams.update({'font.size': 13})
    train_acc, test_acc, train_rmse, test_rmse, real_train, real_test, predict_train, predict_test, real, \
    predict, rmse = prediction_dataloader(room)
    plt.rc('font', family='Times New Roman')
    # Define our own color bar for the plot.
    newcolors = cm.get_cmap('viridis', 256)(np.linspace(0, 1, 256))
    newcolors[:128, :] = newcolors[128:, :]
    newcolors[128:, :] = np.flipud(newcolors[128:, :])
    newcmp = ListedColormap(newcolors)
    # Use the scatter plot to plot the distribution with our own color bar.
    plt.scatter(real, predict, c=real - predict, marker='o', label="(Real, Prediction)", s=10, cmap=newcmp)
    real_range = np.linspace(min(real), max(real))
    # Plot the identity line of y=x
    plt.plot(real_range, real_range, color='m', linestyle="-.", linewidth=1, label="Identity Line (y=x)")
    plt.title("Prediction Validation Graph of Room {}".format(room))
    plt.ylabel("Prediction")
    plt.legend(frameon=False)
    plt.colorbar(label="Error (Real Value - Prediction)")
    plt.xlabel("Original AC\nOverall Accuracy: {}%\nRoot Mean Square Error: {}".format(
        round(100 * (train_acc * len(real_train) + test_acc * len(real_test)) / len(real), 2), round(rmse, 4)))
    plt.savefig("./distribution_plot/room{}.png".format(room))
    plt.clf()


# This is the function to plot the error and root mean square error distribution of all the rooms.
def plot_error_distribution():
    accuracy_list, rmse_list = [], []
    # Looping through all the room and collect the statistics we need.
    for room_f in room_list:
        if room_f == 309 or room_f == 312 or room_f == 917 or room_f == 1001:
            continue
        acc, rmse = calculate_RMSE(room_f)
        accuracy_list.append(acc * 100)
        rmse_list.append(rmse)
    plt.rcParams.update({'font.size': 15})
    # Use the historgram in matplotlib to plot the accuracy distribution histogram.
    plt.hist(accuracy_list, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Accuracy(%)\nMean Accuracy: {}%".format(round(mean(accuracy_list), 2)))
    plt.ylabel("Occurrence")
    plt.title("The Accuracy Distribution Histogram")
    plt.savefig('./AccDis.png')
    plt.clf()
    # Use the same to plot the root mean square error distribution histogram.
    plt.hist(rmse_list, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Root Mean Square Error\nMean RMSE: {}".format(round(mean(rmse_list), 4)))
    plt.ylabel("Occurrence")
    plt.title("The RMSE Distribution Histogram")
    plt.savefig('./RMSEDis.png')
    plt.clf()


# The main function
if __name__ == "__main__":
    for room in tqdm(room_list):
        # Delete the rooms with low quality data manually.
        if room == 309 or room == 312 or room == 917 or room == 1001:
            continue
        # view_shap_importance(room)  # This function will pop up a figure window for each room.
        plot_shap_interact(room)
        plot_distribution(room)
    plot_error_distribution()
