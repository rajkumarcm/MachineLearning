import numpy as np
import pandas as pd
from DNN import *

epochs = 100
batch_size = 50
lr = 1e-2

df = pd.read_csv('data/insurance.csv')
df.drop('region', 'columns', inplace=True)
df.sex.replace({'female': 1, 'male': 0}, inplace=True)
df.smoker.replace({'yes': 0, 'no': 1}, inplace=True)

df -= df.mean(axis=0)
df /= df.std(axis=0)

X_train = df.iloc[:-100, :-1].to_numpy()
y_train = df.iloc[:-100, -1].to_numpy()
y_train = np.reshape(y_train, [-1, 1])

X_val = df.iloc[-100:, :-1].to_numpy()
y_val = df.iloc[-100:, -1].to_numpy()
y_val = np.reshape(y_val, [-1, 1])

# X = np.random.random([100, 5])
# y = X @ np.random.random([5, 1])
#
# X_train = X[:50]
# y_train = y[:50]
#
# X_val = X[50:]
# y_val = y[50:]

l2_config = [Dense(X_train.shape[1], 5, LReLU, reg='l1'),
             Dense(5, 1, LReLU, reg='l1')]
l2_model = Model(l2_config, MSE)
l2_hist = l2_model.fit(X_train, y_train, X_val, y_val, epochs=epochs,
                       batch_size=batch_size, lr=lr, reg='l1',
                       verbose=True)

# print("\nTF training:\n")
# # TF model
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
#
# model = Sequential([Dense(5, activation='relu', kernel_regularizer='l1'),
#                     Dense(1, activation='relu', kernel_regularizer='l1')])
#
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
# model_hist = model.fit(X_train, y_train, batch_size=10, validation_data=(X_val, y_val), epochs=epochs)
#
# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(range(epochs), l2_hist['vl_loss'], '-r', label='My model')
# plt.plot(range(epochs), model_hist.history['val_loss'], '-b', label='TF model')
# plt.legend()
# plt.show()
# print(f'Ground truth: {y_val[10]}')
# print(f'TF model: {model.predict(X_val[10].reshape([1, -1]))}')
# print(f'My model: {l2_model(X_val[10])}')
# print('debug...')