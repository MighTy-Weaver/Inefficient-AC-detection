import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

folder_name = "20210713191456_cate3_KMeans_22_33"
mode = "CLU"  # "CSV" or "CLU"
categories = ['Normal', 'Low', 'Moderate']

# Ignore all the warnings and Set some general settings of all the matplotlib.
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 22})
plt.rc('font', family='Times New Roman')

colors = ['brown', 'darkblue', 'skyblue']

# Read the original data
data = pd.read_csv('./summer_data_compiled.csv', index_col=0)
data = data[(data.AC > 0)]
print(len(data))
data = data[(data.AC > 0) & (data.Prev_1hr)]
print(len(data))
for room in [328, 1007, 821, 332]:
    print(room, len(data[data.Location == room]))

replaced_2016 = [714, 503, 1012, 235, 520, 735, 220, 335, 619, 817, 807, 202, 424, 801, 211, 402, 201, 326, 306, 429,
                 414, 715, 311, 330]
replaced_2017 = [432, 802, 227, 231, 733, 210, 315, 427, 430, 612, 613, 626, 630, 704, 914, 123, 307, 903]
replaced_2018 = [219, 516, 417, 605, 816, 703, 803, 818, 915, 122, 207, 310, 320, 824, 518, 530, 913]
replaced_2019 = [822, 730, 608]
replaced_2020 = [808, 819, 403, 716, 303, 334, 832, 401, 622]
replaced_2021 = [604, 702, 735, 217, 517, 710]

if mode == "CLU":
    lists = np.load('{}/category.npy'.format(folder_name), allow_pickle=True)
    num_cates = len(lists)

    # --------------------------------------------------------
    lists[0].extend(lists[2])
    num_cates = 2
    # -------------------------------------------------------------------------

    dataframes = [[data[data.Location == int(i)] for i in lis] for lis in lists]
    dataframe_list = [pd.concat(dataframes[i]) for i in range(num_cates)]
    mean_list = [round(np.array(df['AC']).mean(), 4) for df in dataframe_list]
    print(mean_list)

    for i in range(num_cates):
        sns.distplot(dataframe_list[i]['AC'], bins=sorted(dataframe_list[i]['AC'].unique()),
                     label="{} efficiency, mean={}kWh, #data={}".format(categories[i], mean_list[i],
                                                                        len(dataframe_list[i])), color=colors[i],
                     hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
    plt.legend(frameon=False)
    plt.xlabel("Hourly AC Electricity Consumption/kWh")
    plt.ylabel("Kernel Density")
    plt.show()

    for i in range(num_cates):
        print(len(lists[i]))
        print(
            "In [{}] category:\n2016 Replaced: {}({}%)\t2017 Replaced: {}({}%)\t2018 Replaced: {}({}%)\t2019 Replaced: {}({}%)\t2020 Replaced: {}({}%)\t2021 Replaced: {}({}%)".format(
                categories[i], len([j for j in lists[i] if int(j) in replaced_2016]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2016]) / len(lists[i]), 4),
                len([j for j in lists[i] if int(j) in replaced_2017]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2017]) / len(lists[i]), 4),
                len([j for j in lists[i] if int(j) in replaced_2018]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2018]) / len(lists[i]), 4),
                len([j for j in lists[i] if int(j) in replaced_2019]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2019]) / len(lists[i]), 4),
                len([j for j in lists[i] if int(j) in replaced_2020]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2020]) / len(lists[i]), 4),
                len([j for j in lists[i] if int(j) in replaced_2021]),
                100 * round(len([j for j in lists[i] if int(j) in replaced_2021]) / len(lists[i]), 4)))
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

    plt.legend(fontsize=22)
    plt.xlabel("Hourly AC Electricity Consumption/kWh", fontsize=22)
    plt.ylabel("Kernel Density", fontsize=22)
    plt.show()
