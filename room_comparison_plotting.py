import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

folder_name = "20210616130636_cate2_KMeans"
mode = "CLU"  # "CSV" or "CLU"
categories = ["High", "Low"]

# Ignore all the warnings and Set some general settings of all the matplotlib.
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 22})
plt.rc('font', family='Times New Roman')

colors = ['brown', 'darkblue', 'skyblue']

# Read the original data
data = pd.read_csv('./summer_data_compiled.csv', index_col=0)
print(len(data))
data = data[(data.AC > 0) & (data.Prev_1hr)]
print(len(data))

if mode == "CLU":
    lists = np.load('{}/category.npy'.format(folder_name), allow_pickle=True)
    num_cates = len(lists)

    dataframes = [[data[data.Location == int(i)] for i in lis] for lis in lists]
    dataframe_list = [pd.concat(dataframes[i]) for i in range(num_cates)]
    mean_list = [round(np.array(df['AC']).mean(), 4) for df in dataframe_list]
    print(mean_list)

    for i in range(num_cates):
        sns.distplot(dataframe_list[i]['AC'], bins=sorted(dataframe_list[i]['AC'].unique()),
                     label="{} efficiency, mean={}kWh, #data={}".format(categories[i], mean_list[i],
                                                                        len(dataframe_list[i])), color=colors[i],
                     hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    plt.legend()
    plt.xlabel("Hourly AC Electricity Consumption/kWh")
    plt.ylabel("Kernel Density")
    plt.show()

elif mode == "CSV":
    room_split = pd.read_csv('./room_classification.csv', index_col=None)

    # Read the List of all rooms and classify them into three efficient list by human observation
    room_list = data['Location'].unique()
    efficient_list = room_split[room_split['Room AC Status- Lu-20210521'] == 'High efficiency']['Room Number']
    moderate_list = room_split[room_split['Room AC Status- Lu-20210521'] == 'Moderate efficiency']['Room Number']
    inefficient_list = room_split[room_split['Room AC Status- Lu-20210521'] == 'Low efficiency']['Room Number']

    # create three dataframe list to store the dataframes
    high_list = [data[data.Location == i] for i in efficient_list]
    mid_list = [data[data.Location == i] for i in moderate_list]
    low_list = [data[data.Location == i] for i in inefficient_list]

    # Concat three lists into three dataframes
    high_df = pd.concat(high_list)
    mid_df = pd.concat(mid_list)
    low_df = pd.concat(low_list)

    # Calculate the mean of the data
    high_mean, mid_mean, low_mean = round(np.array(high_df['AC']).mean(), 4), round(np.array(mid_df['AC']).mean(),
                                                                                    4), round(
        np.array(low_df['AC']).mean(), 4)
    print(high_mean, mid_mean, low_mean)

    # Plot the distribution plot.
    sns.distplot(high_df['AC'], bins=sorted(high_df['AC'].unique()),
                 label="High efficiency, mean={}kWh, #data{}".format(high_mean, len(high_df)), color="brown",
                 hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    sns.distplot(mid_df['AC'], bins=sorted(mid_df['AC'].unique()),
                 label="Moderate efficiency, mean={}kWh, #data{}".format(mid_mean, len(mid_df)), color="darkblue",
                 hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    sns.distplot(low_df['AC'], bins=sorted(low_df['AC'].unique()),
                 label="Low efficiency, mean={}kWh, #data{}".format(low_mean, len(low_df)), color="skyblue",
                 hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})

    plt.legend()
    plt.xlabel("Hourly AC Electricity Consumption/kWh")
    plt.ylabel("Kernel Density")
    plt.show()
