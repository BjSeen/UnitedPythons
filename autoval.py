from ctypes import WinDLL
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import os

directory = 'Data/TestingData'
modelname = 'Big_Data_Win300'
model = tf.keras.models.load_model('saved_models/' + modelname)

for filename in os.listdir(directory):
    if filename != 'testing9.csv': 
        data_og = pd.read_csv(os.path.join(directory, filename), header=0, sep=';', usecols=["Current"])

        WINDOW_SIZE=300

        data = data_og.to_numpy()
        ground_truth = data[-(600+WINDOW_SIZE):]

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
        with open('rmses.txt', 'a') as f:
            f.write(modelname + ', ' + filename + ': ' + str(error) + '\n')