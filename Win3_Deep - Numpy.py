import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# setting the window size
WINDOW_SIZE = 250

# reading the csv
data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';', usecols=["Current"])

# creating numpy arrays for training
data = data.to_numpy()
a= 500001
b= 500601 + WINDOW_SIZE
ground_truth = data[a:b]
data0 = data[400_000:500_000]
data1 = data[2_120_000:2_200_000]
data2 = data[1_065_000:1_085_000]
data3 = data[3_260_000:3_310_000]

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

# split training data into windows
windows0, labels0 = split_sequence(data0, WINDOW_SIZE)
windows1, labels1 = split_sequence(data1, WINDOW_SIZE)
windows2, labels2 = split_sequence(data2, WINDOW_SIZE)
windows3, labels3 = split_sequence(data3, WINDOW_SIZE)

# concatenate the windows from different sections of the training data
windows = np.concatenate((windows0, windows1, windows2, windows3))
labels = np.concatenate((labels0, labels1, labels2, labels3))

#splitting the trainings and test data and labels
train_windows = windows[:int(windows.shape[0] * 0.8)]
train_labels = labels[:int(windows.shape[0] * 0.8)]

test_windows = windows[int(windows.shape[0] * 0.8):]
test_labels = labels[int(windows.shape[0] * 0.8):]

# building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,strides=1, padding="causal",activation="relu", input_shape=[WINDOW_SIZE,1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])

# setting callback and optimizer function
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# compiling and fitting the model
model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(train_windows, train_labels, epochs=5, validation_data=[test_windows, test_labels], callbacks=[callback])

predicted_values = np.array([])
#Get the last Window of the time Series and reshape
temp_Window = ground_truth[0:WINDOW_SIZE]

for i in range(600):
  #predict the next value
  temp_label = model.predict(np.expand_dims(temp_Window, axis=0))
  #add value to the Window and predicted_values
  predicted_values = np.append(predicted_values, temp_label)
  temp_Window = np.append(temp_Window, temp_label)
  #remove first value of Window
  temp_Window = np.delete(temp_Window,0).reshape(WINDOW_SIZE,1)

#Plot figure with given data, predicted and actual values
plt.figure(figsize=(15, 6)) 
plt.plot(list(range(WINDOW_SIZE)), ground_truth[0:WINDOW_SIZE],label = 'Given Data') 
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), predicted_values, label= 'Predicted Values', color="red")
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), ground_truth[WINDOW_SIZE:], label= 'Ground Truth', color="yellow")
plt.legend()
plt.show()

#Evaluate with RMSE
rmse = tf.keras.metrics.RootMeanSquaredError()
rmse.update_state(ground_truth[WINDOW_SIZE:], predicted_values)
error = rmse.result().numpy()
print(error)