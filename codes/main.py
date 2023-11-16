"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
data_heart = pd.read_csv('./heart.csv')

print('Dataset')
print(data_heart.head(5))
print()

print('Information')
data_heart.info()
print()

print('Description')
print(data_heart.describe())
print()

# print(pp.ProfileReport(data_heart))

# Covariance Matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data_heart.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()
