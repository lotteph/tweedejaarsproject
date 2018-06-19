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

def make_csv(solar, weather):
    new = weather
    del new["Unnamed: 0"]
    new["number_of_panels"] = solar["Number_of_panels"]
    new["max_power"] = solar["Max_power"]
    new["system_size"] = solar["System_size"]
    new["number_of_inverters"] = solar["Number_of_inverters"]
    new["inverter_size"] = solar["Inverter_size"]
    new["tilt"] = solar["Tilt"]
    return np.array(new)

years = ["2013","2014","2015","2016","2017","2018"]
#years = ["2017"]
postal_code = ["7325","7559","2201"]

SP = False
W = False
for code in range(0,len(postal_code)):
    for year in range(0,len(years)):
        W2 = pd.read_csv("../data/"+postal_code[code]+ "_" + years[year] + "_W.csv")
        if type(W) != type(False):
            W = pd.DataFrame.append(W,W2)
        else:
            W = W2
        SP2 = pd.read_csv("../data/"+postal_code[code]+ "_" + years[year] + "_S.csv")
        if type(SP) != type(False):
            SP = pd.DataFrame.append(SP,SP2)
        else:
            SP = SP2
results = np.array(SP["Generated"])

W = make_csv(SP,W)

train_size = len(results)-365

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

def ridge_regression(par):
    alpha = par[0]
    regr = linear_model.Ridge(alpha,solver="svd")
    regr.fit(x_train,y_train)
    ridge_pred = regr.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(ridge_pred,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("ridge predicted vs real output of 2017")
    plt.show()
    error = np.square(ridge_pred-y_test)
    return sum(error)/len(y_test)

def lasso_regression(par):
    alpha = par[0]
    ep = par[1]
    Lasso = linear_model.LassoLars (alpha,eps = ep)
    Lasso.fit(x_train,y_train)
    lasso_pred = Lasso.predict(x_test)
    plt.plot(np.square(lasso_pred-y_test),label='mean training output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("lasso predicted vs real output of 2017")
    plt.show()
    return sum(np.square(lasso_pred-y_test))/len(y_test)

def Bayes_regression(par):
    alpha_1 = par[0]
    alpha_2 = par[1]
    lambda_1 = par[2]
    lambda_2 = par[3]
    Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
    Bay.fit(x_train,y_train)
    Bay_pred = Bay.predict(x_test)
    # pre = scipy.ndimage.gaussian_filter(Bay_pred,5)
    # plt.plot(pre,label='predicted output',color="red")
    # real = scipy.ndimage.gaussian_filter(y_test,5)
    # plt.plot(real,label='real output',color="blue")
    # plt.legend()
    # plt.xlabel("time (days)")
    # plt.ylabel("solar panel output (kWh)")
    # plt.title("bayes predicted vs real output of 2017")
    # plt.show()
    return sum(np.square(Bay_pred-y_test))/len(y_test)

def decision_tree():
    dec = DecisionTreeRegressor()
    dec.fit(x_train, y_train)
    pred = dec.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(pred,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("decision tree predicted vs real output of 2017")
    plt.show()
    return sum(np.square(pred-y_test))/len(y_test)

def k_nearest(par):
    neighbors = par[0]
    neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
    neigh.fit(x_train, y_train)
    neigh_pred = neigh.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(neigh_pred,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("KNN predicted vs real output of 2017")
    plt.show()
    return sum(np.square(neigh_pred-y_test))/len(y_test)

def kn_opt(iterations):
    best = 999999999999999999
    par = 1
    for i in range(1, iterations):
        temp = k_nearest([i])
        if temp < best:
            best = temp
            par = [i]
    return(best,par)

print("base: ",sum(np.square(np.mean(y_train)-y_test))/len(y_test))
print("ridge: ",ridge_regression([-5]))
res = scipy.optimize.minimize(Bayes_regression,[0.000006,0.000006,0.000006,0.000006],method="L-BFGS-B")
print(res.x)
print("new bayes: ",Bayes_regression(res.x))
#print("decision tree:", decision_tree())
#print("KNN: ",k_nearest([1]))
