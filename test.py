import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

#test data between 470 and 64540 values

data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';')
del data['DateTime']
data["LiftWorkingPosition"] = data["LiftWorkingPosition"].astype(int)

data['CurrentMA'] = data['Current'].rolling(window=5).mean()
data['CurrentMA10'] = data['Current'].rolling(window=10).mean()


ilocs_min = argrelextrema(data.price.values, np.less_equal, order=3)[0]
ilocs_max = argrelextrema(data.price.values, np.greater_equal, order=3)[0]

data.price.plot(figsize=(20,8), alpha=.3)
# filter prices that are peaks and plot them differently to be visable on the plot
data.iloc[ilocs_max].price.plot(style='.', lw=10, color='red', marker="v");
data.iloc[ilocs_min].price.plot(style='.', lw=10, color='green', marker="^");

data.iloc[0:1000].plot(y = ["Current",  "CurrentMA", "CurrentMA10"], use_index=True)
plt.show()

corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
corr.head

