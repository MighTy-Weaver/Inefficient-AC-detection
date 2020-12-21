# This is the code for compiling all the data.
# It's just demonstrating how the data is generated, no raw data will be released.

import glob
import os

import pandas as pd
from tqdm import trange

pd.set_option('display.max_columns', None)
energy_files = os.listdir('./Energy raw data set 20200101-20201211')
climate_files = os.listdir('./Climate raw data set 20200101-20201211')
energy = []
for i in trange(len(energy_files)):
    energy.append(pd.read_csv('./Energy raw data set 20200101-20201211/{}'.format(energy_files[i]), encoding='utf16',
                              header=0, sep='\t'))
energy_total = pd.concat(energy)
energy_total = energy_total.drop(
    ['Location Path', 'Location Description', 'Total (kWh)', 'Light (kWh)', 'Socket (kWh)', 'Water Heater (kWh)',
     'Mixed Usage (kWh)', 'Time (day of week)'], axis=1)
energy_total = energy_total.rename(
    columns={'Time (date)': 'Date', 'Time (hour)': 'Hour', 'Location': 'Location', 'AC (kWh)': 'AC'})

energy_total['Time'] = pd.to_datetime(
    energy_total['Date'].apply(str) + energy_total['Hour'].apply(str), format='%Y-%m-%d%H:%M')
energy_total['Time'] = energy_total['Time'].apply(str)
irr = glob.glob('./Climate raw data set 20200101-20201211/Irr*')
pre = glob.glob('./Climate raw data set 20200101-20201211/Pre*')
Rel = glob.glob('./Climate raw data set 20200101-20201211/Rel*')
Tem = glob.glob('./Climate raw data set 20200101-20201211/Tem*')
I = pd.concat([pd.read_csv(i, index_col=None) for i in irr]).drop(
    ['Source', 'Height', 'Status', 'Method ID', 'Details'], axis=1)
P = pd.concat([pd.read_csv(i, index_col=None) for i in pre]).drop(
    ['Source', 'Height', 'Status', 'Method ID', 'Details', 'Hour'], axis=1)
H = pd.concat([pd.read_csv(i, index_col=None) for i in Rel]).drop(
    ['Source', 'Height', 'Status', 'Method ID', 'Details'], axis=1)
T = pd.concat([pd.read_csv(i, index_col=None) for i in Tem]).drop(['Source', 'Height', 'Status', 'Method ID'], axis=1)
for i in [I, P, H, T]:
    i['Time'] = pd.to_datetime(i['Time'].apply(str), format='%Y/%m/%d %H:%M:%S')
    i['Time'] = i['Time'].apply(str)

data = pd.merge(pd.merge(pd.merge(pd.merge(energy_total, I, on='Time'), P, on='Time'), H, on='Time'), T, on='Time')
data = data.rename(
    columns={'w/m2': 'Irradiance', 'mm': 'Precipitation', '%': 'Humidity', 'Degree Celsius': 'Temperature'})
data = data.fillna(method='pad', axis=0)

print("Checking Merged Data:")
print(data.head(10))
print("Number of data: {}".format(len(data)))
print(data.isnull().any())

drop_list = ['10/F Public', '10/F Rooms', '3/F Public', '3/F Rooms', '2/F', '4/F', '5/F', '6/F Public', '6/F Rooms',
             '7/F', '8/F Public', '8/F Rooms', '9/F Public', '9/F Rooms', 'ST302', 'ST602', 'ST802', 'ST902',
             'Warden Flat']
for i in drop_list:
    data = data[data['Location'] != i]
data.to_csv('2020_data_initial.csv', index=False)

data = data.sort_values(by=['Location', 'Time'])
data.insert(1, 'Prev_2hrs', '')
data.insert(1, 'Prev_1hr', '')
data.insert(1, 'Prev_1hr_AC', '')
data.insert(2, 'Prev_3hr_AC', '')
data.insert(3, 'Prev_5hr_AC', '')
print("Start Generating Previous Data")
for index in trange(len(data)):
    if (index == 0) or (index == 1):
        data.loc[index, 'Prev_2hrs'] = False
        data.loc[index, 'Prev_1hr'] = False
    else:
        data.loc[index, 'Prev_2hrs'] = data.at[index - 1, 'AC'] > 0 and data.at[index - 2, 'AC'] > 0
        data.loc[index, 'Prev_1hr'] = data.at[index - 1, 'AC'] > 0
    if index <= 4:
        data.loc[index, 'Prev_1hr_AC'] = 0
        data.loc[index, 'Prev_3hr_AC'] = 0
        data.loc[index, 'Prev_5hr_AC'] = 0
    else:
        data.loc[index, 'Prev_1hr_AC'] = data.at[index - 1, 'AC']
        data.loc[index, 'Prev_3hr_AC'] = data.at[index - 1, 'AC'] + data.at[index - 2, 'AC'] + data.at[index - 3, 'AC']
        data.loc[index, 'Prev_5hr_AC'] = data.at[index - 1, 'AC'] + data.at[index - 2, 'AC'] + data.at[
            index - 3, 'AC'] + data.at[index - 4, 'AC'] + data.at[index - 5, 'AC']
data.to_csv('2020_data_compiled.csv', index=True)
print("Checking Generated Previous Data:")
print(data.head(10))
print("Number of data: {}".format(len(data)))
print(data.isnull().any())
