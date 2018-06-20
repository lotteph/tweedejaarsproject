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

data = dict()
data["7325"] = range(2013,2019)
data["7559"] = range(2013,2019)
data["2134"] = range(2013,2019)
data["2201"] = range(2013,2019)
data["6591"] = range(2015,2019)
data["3481"] = range(2015,2019)
postal_codes = ["2134","7325","2201","6591","7559","3481"]
SP = False
W = False
print(data.keys())
for code in postal_codes:
    for year in data[code]:
        W2 = pd.read_csv("../data/"+str(code)+ "_" + str(year) + "_W.csv")
        if type(W) != type(False):
            W = pd.DataFrame.append(W,W2)
        else:
            W = W2
        SP2 = pd.read_csv("../data/"+ str(code) + "_" + str(year) + "_S.csv")
        if type(SP) != type(False):
            SP = pd.DataFrame.append(SP,SP2)
        else:
            SP = SP2
results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])
W["time"] = 0
W = make_csv(SP,W)

train_size = len(results)-365*3

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

# NORMALIZE DATA:
# scaler = MinMaxScaler(feature_range=(0, 1))
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# y_train = y_train/np.linalg.norm(y_train)
# y_test = y_test/np.linalg.norm(y_test)

offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]

def ridge_regression(par):
    alpha = par[0]
    regr = linear_model.Ridge(alpha,solver="svd")
    regr.fit(x_train,y_train)
    ridge_pred = regr.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(ridge_pred,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    mean = np.mean(y_train)+np.zeros(len(real))
    plt.plot(mean,label='mean output',color="green")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("ridge predicted vs real output of 2017")
    plt.show()
    error = np.square(ridge_pred-y_test)
    return np.sqrt(sum(error)/len(y_test))*offset

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
    return np.sqrt(sum(np.square(lasso_pred-y_test))/len(y_test))*offset

def Bayes_regression(par):
    alpha_1 = par[0]
    alpha_2 = par[1]
    lambda_1 = par[2]
    lambda_2 = par[3]
    Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
    Bay.fit(x_train,y_train)
    Bay_pred = Bay.predict(x_test)
    pre = scipy.ndimage.gaussian_filter(Bay_pred,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("bayes predicted vs real output of 2017")
    plt.show()
    return np.sqrt(sum(np.square(Bay_pred-y_test))/len(y_test))*offset

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
    return np.sqrt(sum(np.square(pred-y_test))/len(y_test))*offset

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
    return np.sqrt(sum(np.square(neigh_pred-y_test))/len(y_test))*offset

def kn_opt(iterations):
    best = 999999999999999999
    par = 1
    for i in range(1, iterations):
        temp = k_nearest([i])
        if temp < best:
            best = temp
            par = [i]
    return(best,par)

mean = np.mean(y_train)+np.zeros(len(y_test))
print("base: ",np.sqrt(sum(np.square(mean-y_test))/len(y_test))*offset)
print("ridge: ",ridge_regression([-5]))
print("bayes: ",Bayes_regression([0.00980137, -0.00372394, -0.00682109, -0.04635455]))
print("decision tree:", decision_tree())
print("KNN: ",k_nearest([1]))
