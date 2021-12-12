from ctypes import WinDLL
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 

WINDOW_SIZE=250

data_og = pd.read_csv("Data/TestingData/testing2.csv", header=0, sep=';', usecols=["Current"])
data = data_og.to_numpy()
# a= 37000
# b= 370601 + WINDOW_SIZE
ground_truth = data[-850:]


model = tf.keras.models.load_model('saved_models/BigBigdata_ExcelUnknown_Overnight')
model = tf.keras.models.load_model('saved_models/Bigdata_Excel20_1241')

predicted_values = np.array([])
temp_Window = ground_truth[0:WINDOW_SIZE]

for i in range(600):
  #predict the next value
  temp_label = model.predict(np.expand_dims(temp_Window, axis=0))
  #add value to the Window and predicted_values
  predicted_values = np.append(predicted_values, temp_label)
  temp_Window = np.append(temp_Window, temp_label)
  #remove first value of Window
  temp_Window = np.delete(temp_Window,0).reshape(WINDOW_SIZE,1)


Root_Algo = tf.keras.metrics.RootMeanSquaredError()
Root_Algo.update_state(ground_truth[WINDOW_SIZE:], predicted_values)
error = Root_Algo.result().numpy()
print(error)



plt.figure(figsize=(15, 6)) 
plt.plot(list(range(WINDOW_SIZE)), ground_truth[0:WINDOW_SIZE],label = 'Given Data') 
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), predicted_values, label= 'Predicted Values')
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), ground_truth[WINDOW_SIZE:], label= 'Ground Truth')
plt.legend()
plt.show()