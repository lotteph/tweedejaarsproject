import pandas as pd
import numpy as np
import scipy
import scipy.ndimage
import csv
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import functions
import create_pandas
import neural_network_ruth as nnr
import rnn
from os import listdir
scaler = MinMaxScaler(feature_range=(0, 1))

# makes the data into pandas ready for use
def create_train_test():
    results, W, SP, set_size = create_pandas.create_weather_pandas("Train")
    x_train = W
    y_train = results

    results, W, SP, set_size = create_pandas.create_weather_pandas("Test")
    x_test =  W
    y_test = results

    offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]
    mean = np.mean(y_train)+np.zeros(len(y_test))
    return  x_train, y_train, x_test, y_test, mean, offset, set_size

# Actual performance of the functions
def regression():
    x_train, y_train, x_test, y_test, mean, offset,set_size = create_train_test()

    print("base: ",np.sqrt(sum(np.square(mean-y_test)*offset)/len(y_test)))
    print("ridge: ",functions.ridge_regression([-5], x_train, x_test, y_train, y_test, offset))
    print("bayes: ",functions.Bayes_regression([0.00980137, -0.00372394, -0.00682109, -0.04635455], x_train, x_test, y_train, y_test, offset))
    print("decision tree:", functions.decision_tree(x_train, x_test, y_train, y_test, offset))
    print("KNN: ",functions.k_nearest([1], x_train, x_test, y_train, y_test, offset))

# The running of the neural network
def train_nn():
    x_train, y_train, x_test, y_test, mean, offset, set_size = create_train_test()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    print("recurrent neural network: ", rnn.rnn(x_train, x_test, y_train, y_test, offset, set_size))

def test_nn():
    x_train, y_train, x_test, y_test, mean, offset, set_size = create_train_test()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    print("recurrent neural network: ", rnn.load_rnn(x_test, y_test, offset, set_size))

regression()
train_nn()
test_nn()
