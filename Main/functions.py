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
import os, tempfile
from Weather_test import _main_

# Of all CSV files get postalcode and match years of data
# input: postal codes and data years, of PV panels in us

def ridge_regression(par, x_train, x_test, y_train, y_test, offset):
    alpha = par[0]
    regr = linear_model.Ridge(alpha,solver="svd")
    regr.fit(x_train,y_train)
    print(x_train.shape)
    print(x_test.shape)
    ridge_pred = regr.predict(x_test)
    plot(ridge_pred, y_test)
    error = np.square(ridge_pred-y_test)*offset
    return np.sqrt(sum(error)/len(y_test))

def Bayes_regression(par, x_train, x_test, y_train, y_test, offset):
    alpha_1 = par[0]
    alpha_2 = par[1]
    lambda_1 = par[2]
    lambda_2 = par[3]
    Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
    Bay.fit(x_train,y_train)
    Bay_pred = Bay.predict(x_test)
    plot(Bay_pred, y_test)
    return np.sqrt(sum(np.square(Bay_pred-y_test)*offset)/len(y_test))

def decision_tree(x_train, x_test, y_train, y_test, offset):
    dec = DecisionTreeRegressor()
    dec.fit(x_train, y_train)
    pred = dec.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(pred,5)
    plot(pred, y_test)
    return np.sqrt(sum(np.square(pred-y_test)*offset)/len(y_test))

def k_nearest(par, x_train, x_test, y_train, y_test, offset):
    neighbors = par[0]
    neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
    neigh.fit(x_train, y_train)
    neigh_pred = neigh.predict(x_test)
    plot(neigh_pred, y_test)
    return np.sqrt(sum(np.square(neigh_pred-y_test)*offset)/len(y_test))

def kn_opt(iterations):
    best = 999999999999999999
    par = 1
    for i in range(1, iterations):
        temp = k_nearest([i])
        if temp < best:
            best = temp
            par = [i]
    return(best,par)

def plot(prediction, y_test):
    pre = scipy.ndimage.gaussian_filter(prediction,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("KNN predicted vs real output of 2017")
    plt.show()

# def run_test():
#     csv = input("which file would you like to test on?")
#     if os.path.exists(csv):
#         postalcode = input("What is the postal code?")
#         start_date = input("what year does the data start?")
#         end_date = input("what year does the data end?")
#         data[postalcode] = (start_date, end_date)
#         # function to create corresponding weather files
#         for year in range(int(start_date), int(end_date)+1):
#             # still to cut to years
#             match.match_to_input(csv)
#
#     else:
#         print("File not found. Please add to main directory.")
