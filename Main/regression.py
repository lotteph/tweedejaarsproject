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

# Of all CSV files get postalcode and match years of data
# input: postal codes and data years, of PV panels in us

def ridge_regression(par, x_train, x_test, y_train, y_test, offset):
    alpha = par[0]
    regr = linear_model.Ridge(alpha,solver="svd")
    regr.fit(x_train,y_train)
    print(x_train.shape)
    print(x_test.shape)
    ridge_pred = regr.predict(x_test)
    return ridge_pred

def Bayes_regression(par, x_train, x_test, y_train, y_test, offset):
    alpha_1 = par[0]
    alpha_2 = par[1]
    lambda_1 = par[2]
    lambda_2 = par[3]
    Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
    Bay.fit(x_train,y_train)
    Bay_pred = Bay.predict(x_test)
    return Bay_pred

def decision_tree(x_train, x_test, y_train, y_test, offset):
    dec = DecisionTreeRegressor()
    dec.fit(x_train, y_train)
    pred = dec.predict(x_test)
    return pred

def k_nearest(par, x_train, x_test, y_train, y_test, offset):
    neighbors = par[0]
    neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
    neigh.fit(x_train, y_train)
    neigh_pred = neigh.predict(x_test)
    return neigh_pred

def multiple_years(years,N,W,results):
    bas = np.zeros(len(years)*12)
    rid = np.zeros(len(years)*12)
    bay = np.zeros(len(years)*12)
    dec = np.zeros(len(years)*12)
    knn = np.zeros(len(years)*12)
    for i in range(0,N):
        #W, results = shuffle(W, results)
        for j in range(0,len(years)*12):
            new_W = W[:365*1+12*(j+1)]
            new_results = results[:365*1+12*(j+1)]
            train_size = len(new_results)-365*1
            x_train = new_W[:train_size,:]
            x_test =  new_W[train_size:,:]
            y_train = new_results[:train_size]
            y_test = new_results[train_size:]
            offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]
            bas[j] += np.sqrt(np.sum(np.square(np.mean(y_train)-y_test)))/len(y_test)*offset
            rid[j] += ridge_regression(x_train,y_train,x_test,y_test,offset,[-5])
            bay[j] += Bayes_regression(x_train,y_train,x_test,y_test,offset,[-3.63600029e-04,  2.33234414e-03,  5.52569969e-02, -4.99181236e-01])
            dec[j] += decision_tree(x_train,y_train,x_test,y_test,offset)
            knn[j] += k_nearest(x_train,y_train,x_test,y_test,offset,5)
    plt.ylim(ymax=0.01)
    plt.plot(bas/365,label='base')
    plt.plot(rid/365,label='ridge')
    plt.plot(bay/365,label='bayes')
    plt.plot(dec/365,label='decision tree')
    plt.legend()
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
