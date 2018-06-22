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
from scipy.optimize import minimize

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

years = ["2013","2014","2015","2016","2017"]
postal_code = "7559"
#years = ["2015","2017","2016"]

W = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    W["time"] = 0
    SP2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)

results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])

W["time"] = 0
# W["cloudCover"] = 0
# W["sunsetTime"] = 0
# W["sunriseTime"] = 0
# W["temperatureLow"] = 0
# W["temperatureHigh"] = 0
# W["temperatureMax"] = 0
# W["temperatureMin"] = 0
# W["visibility"] = 0
# W["longitude"] = 0
# W["latitude"] = 0
# W["month"] = 0

W = W.values

train_size = len(results)-365*1

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

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
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("ridge predicted vs real output of 2017")
    plt.show()
    error = np.square(ridge_pred-y_test)*offset
    return np.sqrt(sum(error)/len(y_test))

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
    return np.sqrt(sum(np.square((Bay_pred-y_test)*offset))/len(y_test))


def decision_tree():
    dec = DecisionTreeRegressor(min_samples_split=0.1, presort=True)
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
    return np.sqrt(sum(np.square(neigh_pred-y_test))/len(y_test))*offset

def kn_opt(iterations):
    best = 999999999999999999
    par = 1
    for i in range(1, iterations):
        temp = k_nearest([i])
        if temp < best:
            best = temp
            par = [i]
    return(best, par)

print("base: ",np.sqrt(sum(np.square(np.mean(y_train)-y_test))/len(y_test))*offset)
# print("ridge: ",ridge_regression([-5]))
print("new bayes: ",Bayes_regression([-3.63600029e-04,  2.33234414e-03,  5.52569969e-02, -4.99181236e-01]))
# print("decision tree:", decision_tree())
# print("KNN: ",kn_opt(5)[0])
