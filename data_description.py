import numpy as np
import pandas as pd

data = pd.read_csv('./summer_data_compiled.csv', index_col=0).drop(['Date', 'Hour', 'Time'], axis=1)
data = data[data.AC > 0]

print("Total Number of Data : {}".format(len(data)))
types = list(data)
for i in types:
    t_data = list(data[i])
    if i == 'Location':
        continue
    print(i, len(t_data), data[i].dtypes, np.min(t_data), np.max(t_data), np.mean(t_data), np.std(t_data), sep='\t')

data_no_location = data.drop(['Location'], axis=1)
corr = data_no_location.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
