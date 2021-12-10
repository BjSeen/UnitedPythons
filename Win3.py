import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

#test data between 470 and 64540 values


data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';')
del data['DateTime']


#ARIMA
trainData = data['Current']
model = ARIMA(trainData, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())
output = model_fit.forecast()

corr = data.corr(method='kendall')
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