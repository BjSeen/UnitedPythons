import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#test data between 470 and 64540 values

data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';')
del data['DateTime']
data["LiftWorkingPosition"] = data["LiftWorkingPosition"].astype(int)

data.iloc[0:10000].plot(y = ["Current", "YAxisCurrent"], use_index=True)
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

