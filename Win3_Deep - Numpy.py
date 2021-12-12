from ctypes import WinDLL
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

WINDOW_SIZE = 470

data_og = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';', usecols=["Current"])

test_data_og = pd.read_csv("Data/TestingData/testing1.csv", header=0, sep=';', usecols=["Current"])
#Stück ohne Null von 380.000 bis 520.000
data_og = data_og.to_numpy()
test_data_og = test_data_og.to_numpy()
offset = 0
a= 500001 +offset
b= 500601 + WINDOW_SIZE + offset
ground_truth = data_og[a:b]
data0 = data_og[400_000:420_000]
data1 = data_og[2_120_000:2_140_000]
data2 = data_og[1_065_000:1_085_000]
data3 = data_og[3_260_000:3_280_000]


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


windows0, labels0 = split_sequence(data0, WINDOW_SIZE)
windows1, labels1 = split_sequence(data1, WINDOW_SIZE)
windows2, labels2 = split_sequence(data2, WINDOW_SIZE)
windows3, labels3 = split_sequence(data3, WINDOW_SIZE)
windows4, labels4 = split_sequence(data4, WINDOW_SIZE)
windows5, labels5 = split_sequence(data5, WINDOW_SIZE)
windows6, labels6 = split_sequence(data6, WINDOW_SIZE)
windows7, labels7 = split_sequence(data7, WINDOW_SIZE)

windows = np.concatenate((windows0, windows1, windows2, windows3, windows4, windows5, windows6, windows7))
labels = np.concatenate((labels0, labels1, labels2, labels3, labels4, labels5, labels6, labels7))

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


from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(train_windows, train_labels, epochs=50, validation_data=[test_windows, test_labels], callbacks=[callback])


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

model.save('saved_models/Big_Data')
# new_model = tf.keras.models.load_model('saved_models/Bigdata_Excel20_1241')


plt.figure(figsize=(15, 6)) 
plt.plot(list(range(WINDOW_SIZE)), ground_truth[0:WINDOW_SIZE],label = 'Given Data') 
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), predicted_values, label= 'Predicted Values', color="red")
plt.plot(list(range(WINDOW_SIZE +1,601+ WINDOW_SIZE)), ground_truth[WINDOW_SIZE:], label= 'Ground Truth', color="yellow")
plt.legend()
plt.show()


Root_Algo = tf.keras.metrics.RootMeanSquaredError()
Root_Algo.update_state(ground_truth[WINDOW_SIZE:], predicted_values)
error = Root_Algo.result().numpy()
print(error)


#SmallBigData
# data0 = data_og[400_000:420_000]
# data1 = data_og[2_120_000:2_140_000]
# data2 = data_og[1_065_000:1_085_000]
# data3 = data_og[3_260_000:3_280_000]

#BigBigData
# data0 = data_og[400_000:500_000]
# data1 = data_og[2_120_000:2_200_000]
# data2 = data_og[1_065_000:1_085_000]
# data3 = data_og[3_260_000:3_310_000]
# data4 = data_og[100_000:170_000]
# data5 = data_og[940_000:1_000_000]
# data6 = data_og[1_600_000:1_660_000]
# data7 = data_og[2_690_000:2_750_000]