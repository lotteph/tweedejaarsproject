import pandas as pd
import numpy as np
import scipy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor


W = np.transpose(pd.read_csv("1078Weather2016.csv").values)
W = np.transpose(W)[:,1:10]
SP = pd.read_csv("2016_1078_Solarpanel.csv")
results = np.array(SP["Generated"])

W, results = shuffle(W,results, random_state=0)

train_size = int(len(results)*(2/3))

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

def ridge_regression(alpha):
    regr = linear_model.Ridge (alpha)
    regr.fit(x_train,y_train)
    ridge_pred = regr.predict(x_test)
    return sum(np.square(ridge_pred-y_test))/len(results)

def lasso_regression(par):
    alpha = par[0]
    ep = par[1]
    Lasso = linear_model.LassoLars (alpha,eps = ep)
    Lasso.fit(x_train,y_train)
    lasso_pred = Lasso.predict(x_test)
    return sum(np.square(lasso_pred-y_test))/len(results)

def Bayes_regression():
    Bay = linear_model.BayesianRidge()
    Bay.fit(x_train,y_train)
    Bay_pred = Bay.predict(x_test)
    return sum(np.square(Bay_pred-y_test))/len(results)

def decision_tree(max_depth):
    dec = DecisionTreeRegressor(max_depth=2)
    dec.fit(x_train, y_train)
    pred = dec.predict(x_test)
    return sum(np.square(pred-y_test))/len(results)


print(decision_tree(1))
#res = scipy.optimize.minimize(ridge_regression,[1])
#print(ridge_regression(res.x))
#print(res.x)
#res = scipy.optimize.minimize(lasso_regression,[1,4])
#print(lasso_regression(res.x))
#print(res.x)
#print(sum(np.square(np.mean(y_train))-y_test)/len(results))
