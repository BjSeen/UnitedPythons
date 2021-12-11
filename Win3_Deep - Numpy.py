from ctypes import WinDLL
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

WINDOW_SIZE = 300

data_og = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';', usecols=["Current"])
data_og = pd.read_csv("Data/TestingData/testing1.csv", header=0, sep=';', usecols=["Current"])
#Stück ohne Null von 380.000 bis 520.000
data_og = data_og.to_numpy()
offset = 1000
a= 8000 +offset
b= 8600 + WINDOW_SIZE + offset
ground_truth = data_og[a:b]
data = data_og[400000:500000]


ground_truth.shape

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


windows, labels  = split_sequence(data, WINDOW_SIZE)

train_windows = windows[:int(windows.shape[0] * 0.8)]
train_labels = labels[:int(windows.shape[0] * 0.8)]

test_windows = windows[int(windows.shape[0] * 0.8):]
test_labels = labels[int(windows.shape[0] * 0.8):]


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,strides=1, padding="causal",activation="relu", input_shape=[WINDOW_SIZE,1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])

model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(train_windows, train_labels, epochs=5, validation_data=[test_windows, test_labels])


predicted_values = np.array([])
#Get the last Window oof the time Serias and reshape
temp_Window = ground_truth[0:WINDOW_SIZE]


for i in range(600):
  #predict the next value
  temp_label = model.predict(np.expand_dims(temp_Window, axis=0))
  #add value to the Window and predicted_values
  predicted_values = np.append(predicted_values, temp_label)
  temp_Window = np.append(temp_Window, temp_label)
  #remove first value of Window
  temp_Window = np.delete(temp_Window,0).reshape(WINDOW_SIZE,1)



plt.figure(figsize=(15, 6)) 
plt.plot(list(range(WINDOW_SIZE)), ground_truth[0:WINDOW_SIZE],label = 'Given Data') 
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), predicted_values, label= 'Predicted Values')
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), ground_truth[WINDOW_SIZE:], label= 'Ground Truth')
plt.legend()
plt.show()



Root_Algo = tf.keras.metrics.RootMeanSquaredError()
Root_Algo.update_state(ground_truth[WINDOW_SIZE:], predicted_values)
error = Root_Algo.result().numpy()
print(error)


# plt.figure(figsize=(15, 6)) 
# plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data') 
# plt.plot(list(range(split_index,len(data))), rnn_forecast, label = 'Predictions') 
# plt.legend()
# plt.show()


