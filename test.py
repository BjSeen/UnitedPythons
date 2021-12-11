import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

#test data between 470 and 64540 values


## ONLY USE ONE OF THESE BELOW DEPENDING ON HOW LARGE UR SET HAS TO BE
#Importing Training Data 
data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';')

#Import Testing data 
data = pd.read_csv("Data/TestingData/testing3.csv", header=0, sep=';')

#removing Datetime and coding LiftWorkingPosition to 0 and 1
del data['DateTime']
data["LiftWorkingPosition"] = data["LiftWorkingPosition"].astype(int)

#Moving Averages for Current
data['CurrentMA'] = data['Current'].rolling(window=5).mean()
data['CurrentMA10'] = data['Current'].rolling(window=10).mean()

#Plot some variables
# data.iloc[0:1000].plot(y = ["Current",  "CurrentMA", "CurrentMA10"], use_index=True)

#Detect Extremes
ilocs_min = argrelextrema(data.Current.values, np.less_equal, order=20)[0]
ilocs_max = argrelextrema(data.Current.values, np.greater_equal, order=20)[0]

data.Current.plot(figsize=(20,8), alpha=.3)
# data.CurrentMA.plot(figsize=(20,8), alpha=.3)
# filter prices that are peaks and plot them differently to be visable on the plot
data.iloc[ilocs_max].Current.plot(style='.', lw=10, color='red', marker="v")
data.iloc[ilocs_min].Current.plot(style='.', lw=10, color='green', marker="^")
plt.show()

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

#Create List of Minimums (x-variable)
min_sorted = consecutive(ilocs_min)
min_list = []
for element in min_sorted:
    min_list.append(int(np.median(element)))

#Create List of Maximums (x-variable)
max_sorted = consecutive(ilocs_max)
max_list = []
for element in max_sorted:
    max_list.append(int(np.median(element)))



#Create List of Minimus (y-variable)
min_list_y = []
for element in min_list:
    min_list_y.append(data.iloc[element].Current)


#Create List of Maximums (y-variable)
max_list_y = []
for element in max_list:
    max_list_y.append(data.iloc[element].Current)

print(len(max_list), len(max_list_y))
print(len(min_list), len(min_list_y))

#Create min Dataframe
dfmin = pd.DataFrame(data = {'min_x': min_list, 'min_y': min_list_y})
#Create max Dataframe
dfmax = pd.DataFrame(data = {'max_x': max_list, 'max_y': max_list_y})

#Plot point cloud of min and max
ax = dfmin.plot.scatter(x="min_x", y="min_y", c="DarkBlue")
dfmax.plot.scatter(x="max_x", y="max_y", c="green", ax=ax)
plt.show()


#Correlation between variables
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


