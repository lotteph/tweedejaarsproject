# Example of LSTM to learn a sequence
from pandas import DataFrame
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
scaler = MinMaxScaler(feature_range=(0, 1))

def rnn(x_train, x_test, y_train, y_test, offset, epochs, batch_size, name):
    ''' Initializes neural network and trains the network using x_train
        makes predictions about x_test and returns these predictions after 
        denormalizing it.
    '''
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

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    predictions = model.predict(x_test)
    save_model(model, name)
    # Makes sure the predictions have the same shape as the real output.
    p = predictions.ravel()
    return p*offset

def load_rnn(x_test, y_test, offset,name):
    ''' Loads a neural network and makes predictions about x_test.
        It returns these predictions after denormalizing it
    '''
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")

    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    score = loaded_model.predict(x_test)
    # Makes sure the predictions have the same shape as the real output.
    p = score.ravel()
    p = p * offset
    return p

def save_model(model,name):
    ''' Saves a neural network and it weights in two files
    '''
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")
