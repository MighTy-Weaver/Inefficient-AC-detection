import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')
plt.rcParams["savefig.bbox"] = "tight"

# a = np.array([6, 4, 2, 0, 2, 0])  # 35
# b = np.array([4, 5, 7, 3, 4, 1])  # 48
a = np.array([7, 1, 3, 2, 1, 2, 0])
b = np.array([51, 6, 5, 4, 0, 3, 0])
c = np.array([33, 3, 1, 3, 2, 1, 1])
size = len(a)
x = np.arange(size)
total_width, n = 0.6, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='High Efficiency', color='blue')
plt.bar(x + width, b, width=width, label='Moderate Efficiency', color='lightblue')
plt.bar(x + 2 * width, c, width=width, label='Low Efficiency', color='darkblue')
plt.xlabel('Replacement year', fontsize=18)
plt.ylabel('Number of replaced ACs', fontsize=18)
plt.title('Number of ACs replaced by year', fontsize=20)
plt.xticks(np.arange(size), ['2015', '2016', '2017', '2018', '2019', '2020', '2021'], fontsize=15)
plt.yticks(fontsize=15)
plt.legend(frameon=False, fontsize=16)
plt.savefig('./replacement_barplot_with2015_3cate.png', bbox_inches='tight')
