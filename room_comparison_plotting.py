import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ignore all the warnings
warnings.filterwarnings('ignore')

# Read the original data
data = pd.read_csv('./summer_data_compiled.csv')
data = data[data.AC > 0]

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

# Set some general settings of all the matplotlib.
plt.rcParams.update({'font.size': 22})
plt.rc('font', family='Times New Roman')

# Calculate the mean of the data
high_mean, mid_mean, low_mean = round(np.array(high_df['AC']).mean(), 4), round(np.array(mid_df['AC']).mean(),
                                                                                4), round(np.array(low_df['AC']).mean(),
                                                                                          4)
# Filter out the data less than 1.5
high_df = high_df[high_df.AC < 1.5]
mid_df = mid_df[mid_df.AC < 1.5]
low_df = low_df[low_df.AC < 1.5]

# Plot the distribution plot.
sns.distplot(high_df['AC'], bins=sorted(high_df['AC'].unique()),
             label="High efficiency, mean={}kWh".format(high_mean), color="brown",
             hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
sns.distplot(mid_df['AC'], bins=sorted(mid_df['AC'].unique()),
             label="Moderate efficiency, mean={}kWh".format(mid_mean), color="darkblue",
             hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})
sns.distplot(low_df['AC'], bins=sorted(low_df['AC'].unique()),
             label="Low efficiency, mean={}kWh".format(low_mean), color="skyblue",
             hist_kws={"edgecolor": "black"}, kde_kws={"linewidth": "3"})

plt.legend()
plt.xlabel("Hourly AC Electricity Consumption/kWh")
plt.ylabel("Kernel Density")
plt.show()
# plt.savefig('./demo/Room_dis.png', bbox_inches='tight')
