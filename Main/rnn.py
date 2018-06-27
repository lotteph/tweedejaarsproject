# Example of LSTM to learn a sequence
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Activation, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from os import listdir
scaler = MinMaxScaler(feature_range=(0, 1))

def rnn(x_train, x_test, y_train, y_test, offset, set_size):
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    shaper = 18

    model = Sequential()
    model.add(LSTM(20, input_shape=(1,shaper), return_sequences=True))
    model.add(LSTM(12, input_shape=(1,shaper), return_sequences=True))
    model.add(LSTM(6))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(x_train, y_train, batch_size=32, epochs=1)
    predictions = model.predict(x_test)

    save_model(model)

    p = predictions.ravel()

    error = np.sqrt(np.sum(np.square(y_test-p)))/len(y_test)*offset

    p = scipy.ndimage.gaussian_filter(p,5)
    y_test = scipy.ndimage.gaussian_filter(y_test,5)
    plot(p, y_test)

    return error

def load_rnn(x_test, y_test, offset, set_size):
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # load json and create model
    json_file = open('model_8.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_8.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    score = loaded_model.predict(x_test)
    p = score.ravel()

    p = p * offset
    y_test = y_test * offset
    p = scipy.ndimage.gaussian_filter(p,5)
    y_test = scipy.ndimage.gaussian_filter(y_test,5)
    plot(p, y_test)

    return np.sqrt(np.sum(np.square(y_test-p)))/len(y_test)*offset

def plot(p, y_test):
    plt.plot(p,label='predicted output',color="red")
    plt.plot(y_test,label='real output',color="blue")
    plt.legend()
    plt.title("Neural network")
    plt.show()

def save_model(model):
    model_json = model.to_json()
    with open("model_8.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_8.h5")
    print("Saved model to disk")
