import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if not os.path.exists('./data_description/'):
    os.mkdir('./data_description/')
data = pd.read_csv('./summer_data_compiled.csv', index_col=0).drop(['Date', 'Hour', 'Time'], axis=1)
data = data[data.AC > 0]

plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"

xlabel_dict = {'Prev_1hr_AC': 'Previous 1 hour AC consumption (kWh)',
               'Prev_3hr_AC': 'Previous 3 hours AC consumption (kWh)',
               'Prev_5hr_AC': 'Previous 5 hours AC consumption (kWh)',
               'Prev_1hr': 'AC is turned on in previous 1 hour (True/False)',
               'Prev_2hrs': 'AC is turned on in previous 2 hours (True/False)',
               'AC': 'AC electricity consumption (kWh)', 'Temperature': 'Hourly outdoor temperature (°C)',
               'Humidity': 'Hourly outdoor relative humidity (%)', 'Irradiance': 'Hourly outdoor irradiance (W/m^2)',
               'Precipitation': 'Hourly precipitation (mm)',
               'Wifi_count': 'Average hourly Wi-Fi connection count'}

print("Total Number of Data : {}".format(len(data)))
types = list(data)

for i in types:
    t_data = list(data[i])
    if i == 'Location':
        continue
    print(i, len(t_data), data[i].dtypes, np.min(t_data), np.max(t_data), np.mean(t_data), np.std(t_data), sep='\t')
    sns.distplot(data[i], hist=True, kde=False,
                 bins=40, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.xlabel(xlabel_dict[i])
    plt.ylabel('Frequency')
    plt.title('Distribution histogram for {}'.format(i))
    plt.savefig('./data_description/{}.png'.format(i), bbox_inches='tight')
    plt.clf()

data_no_location = data.drop(['Location'], axis=1)
corr = data_no_location.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2).set_properties(**{'font-family': 'Times New Roman'})
html_text = corr.style.background_gradient(cmap='coolwarm').set_precision(2).render()

# save = open('./matrix.html', 'w')
# save.write(html_text)
# save.close()
