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
scaler = MinMaxScaler(feature_range=(0, 1))

data = dict()
data["7325"] = range(2013,2019)
data["7559"] = range(2013,2019)
data["2134"] = range(2009,2018)
data["2201"] = range(2013,2019)
data["6591"] = range(2015,2019)
data["3481"] = range(2015,2019)
data["5384"] = range(2013,2016)
data["5552"] = range(2013,2017)
data["3994"] = range(2012,2018)

#choose which postal codes are used
postal_codes = ["7325","7559","2201","6591","3481","5384","5552","3994","2134"]

# makes the data into pandas ready for use
results, W, SP, set_size = create_pandas.create_weather_pandas(data, postal_codes)

train_size = len(results)-365*3

#offset = SP["Number_of_panels"].values[train_size:]*SP["Max_power"].values[train_size:]
offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]

# Actual performance of the functions
def regression(W, offset):
    x_train = W[:train_size,:]
    x_test =  W[train_size:,:]
    y_train = results[:train_size]
    y_test = results[train_size:]
    mean = np.mean(y_train)+np.zeros(len(y_test))

    print("base: ",np.sqrt(sum(np.square(mean-y_test)*offset)/len(y_test)))
    print("ridge: ",functions.ridge_regression([-5], x_train, x_test, y_train, y_test, offset))
    print("bayes: ",functions.Bayes_regression([0.00980137, -0.00372394, -0.00682109, -0.04635455], x_train, x_test, y_train, y_test, offset))
    print("decision tree:", functions.decision_tree(x_train, x_test, y_train, y_test, offset))
    print("KNN: ",functions.k_nearest([1], x_train, x_test, y_train, y_test, offset))

# The running of the neural network
def train_nn(W, offset, set_size):
    scaled = scaler.fit_transform(W)

    x_train = scaled[:train_size,:]
    x_test = scaled[train_size:,:]
    y_train = results[:train_size]
    y_test = results[train_size:]
    mean = np.mean(y_train)+np.zeros(len(y_test))

#    print("neural network: ", nnr.neural_net(x_train, x_test, y_train, y_test, offset))
    print("recurrent neural network: ", rnn.rnn(x_train, x_test, y_train, y_test, offset, set_size))

def test_nn(W, offset, set_size):
    scaled = scaler.fit_transform(W)

    x_train = scaled[:train_size,:]
    x_test = scaled[train_size:,:]
    y_train = results[:train_size]
    y_test = results[train_size:]
    mean = np.mean(y_train)+np.zeros(len(y_test))
    print("recurrent neural network: ", rnn.load_rnn(x_train, x_test, y_train, y_test, offset, set_size))

train_nn(W, offset, set_size)
test_nn(W, offset, set_size)


# # init first prompt, choose which training method to create model
# test_code = 0
# train_code = 0
#
# #DIT HOEFT NIET MEER, IVM FUNCTIES IN TERMINAL AANROEPEN
# answer_prompt = input("Would you like to test data? [Y/N]").upper()
# if answer_prompt == "Y":
#     functions.run_test()
#     # DO NOT USE ELSE, BUT LET IT FLOW THROUGH
# else:
#     while (train_code == 0):
#         answer = input("Use regression or neural network? [R/N]").upper()
#         if answer == "R":
#             regression(W, offset)
#             train_code = 1
#         elif answer == "N":
#             nn(W, offset)
#             train_code = 1
#         else:
#             train_code = 0
#             print("Choose either R or N")
