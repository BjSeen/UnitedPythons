from ctypes import WinDLL
from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 


data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';', usecols=["Current"])
#Stück ohne Null von 380.000 bis 520.000
data = data.to_numpy()
data = data[400000:500000]

WINDOW_SIZE = 5

def ts_data_generator(data, window_size, batch_size, shuffle_buffer):
  '''
  Utility function for time series data generation in batches
  '''
  ts_data = tf.data.Dataset.from_tensor_slices(data)
  ts_data = ts_data.window(window_size + 1, shift=1, drop_remainder=True)
  ts_data = ts_data.flat_map(lambda window: window.batch(window_size + 1))
  ts_data = ts_data.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  ts_data = ts_data.batch(batch_size).prefetch(1)
  return ts_data

train_data = data[:int(0.8*(len(data)))]
test_data = data[int(0.8*(len(data))):]

tensor_train_data = tf.expand_dims(train_data, axis=-1)
tensor_test_data = tf.expand_dims(test_data, axis=-1)

tensor_train_dataset = ts_data_generator(tensor_train_data, WINDOW_SIZE, 32, 100)
tensor_test_dataset = ts_data_generator(tensor_test_data, WINDOW_SIZE, 32, 100)



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,strides=1, padding="causal",activation="relu", input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)])

model.summary()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(tensor_train_dataset, epochs=5, validation_data=tensor_test_dataset)

def model_forecast(model, data, window_size):
    ds = tf.data.Dataset.from_tensor_slices(data)
    print(ds)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# ds = tf.data.Dataset.from_tensor_slices(data[..., np.newaxis])
# ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
# ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
# ds = ds.batch(32).prefetch(1)

# len(list(ds.as_numpy_iterator()))
# 3125*32
# forecast = model.predict(ds)

c = model.predict(np.array([b,b]))
c[0:-1,-1,0]



# for window in ds:
#     print(window)


d = np.array([[[i + 2*j + 8*k for i in range(3)] for j in range(3)] for k in range(3)])
d
d[1,...,1]
import tensorflow_datasets as tfds
ds_numpy = tfds.as_numpy(ds)
ds

for ex in ds_numpy:
  # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
  print(ex)



a = [[[ 5.1]],[[ 3.4]],[[ 3.4]],[[ 3.4]],[[ 3.4]]]
b = np.array(a)





y=numpy.array([numpy.array(xi) for xi in x])

rnn_forecast = model_forecast(model, data[..., np.newaxis], WINDOW_SIZE) 
split_index = int(0.8 * data.shape[0])
rnn_forecast = rnn_forecast[split_index - WINDOW_SIZE:-1, -1, 0]

test_data = test_data.reshape(20000)
test_data.shape
rnn_forecast.shape

data[..., np.newaxis].shape

Root_Algo = tf.keras.metrics.RootMeanSquaredError()
Root_Algo.update_state(test_data, rnn_forecast)
error = Root_Algo.result().numpy()
print(error)


plt.figure(figsize=(15, 6)) 
plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data') 
plt.plot(list(range(split_index,len(data))), rnn_forecast, label = 'Predictions') 
plt.legend()
plt.show()