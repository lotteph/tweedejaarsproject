import pandas as pd
import numpy as np
import scipy
import csv
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
# import matplotlib.pyplot as plt

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
W = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    SP2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)
results = np.array(SP["Generated"])

#W = pd.read_csv("1078Weather" + years[0] + ".csv")
#SP = pd.read_csv("2016_1078_Solarpanel.csv")
#results = np.array(SP["Generated"])
#for year in range(1,len(years)):
#    W2 = pd.read_csv("1078Weather" + years[year] + ".csv")
#    W = pd.DataFrame.append(W,W2)
#    SP2 = pd.read_csv("2016_1078_Solarpanel.csv")
#    SP = pd.DataFrame.append(SP,SP2)
#results = np.array(SP["Generated"])

#W = make_csv(pd.DataFrame(SP), pd.DataFrame(W))
W = (W.values)
#W, results = shuffle(W,results, random_state=0)
train_size = int(len(results)*(5/6))

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]
def ridge_regression(par):
    alpha = par[0]
    regr = linear_model.Ridge(alpha,solver="svd")
    regr.fit(x_train,y_train)
    joblib.dump(regr, 'ridge.pkl')
    ridge_pred = regr.predict(x_test)
    # plt.plot(ridge_pred)
    # plt.plot(y_test)
    # plt.show()
    return sum(np.square(ridge_pred-y_test))/len(y_test)

def lasso_regression(par):
    alpha = par[0]
    ep = par[1]
    Lasso = linear_model.LassoLars (alpha,eps = ep)
    Lasso.fit(x_train,y_train)
    joblib.dump(Lasso, 'lasso.pkl')
    lasso_pred = Lasso.predict(x_test)
    return sum(np.square(lasso_pred-y_test))/len(y_test)

def Bayes_regression():
    Bay = linear_model.BayesianRidge()
    Bay.fit(x_train,y_train)
    joblib.dump(Bay, 'bayes.pkl')
    Bay_pred = Bay.predict(x_test)
    return sum(np.square(Bay_pred-y_test))/len(y_test)

def decision_tree(par):
    dec = DecisionTreeRegressor(max_depth=par[0])
    dec.fit(x_train, y_train)
    joblib.dump(dec, 'dec_tree.pkl')
    pred = dec.predict(x_test)
    return sum(np.square(pred-y_test))/len(y_test)

def k_nearest(par):
    neighbors = par[0]
    neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
    neigh.fit(x_train, y_train)
    neigh_pred = neigh.predict(x_test)
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
# plt.plot(np.mean(y_train))
res = scipy.optimize.minimize(ridge_regression,[0.5])
print("ridge: ",ridge_regression(res.x))
res = scipy.optimize.minimize(lasso_regression,[1,1])
print("lasso: ",lasso_regression([1,1]))
res = scipy.optimize.minimize(decision_tree,[10])
print("decision tree:", decision_tree(res.x))
print("KNN: ",kn_opt(5)[0])

#log = linear_model.LogisticRegression()
#log.fit(x_train.astype('int'),y_train.astype('int'))
#log_pred = log.predict(x_test.astype('int'))
#print(sum(np.square(log_pred-y_test))/len(results))
