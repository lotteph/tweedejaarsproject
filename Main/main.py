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
import create_pandas
import neural_network_ruth as nnr
import rnn
from os import listdir
import multilayer_perceptron as mlp

class Predictor(object):
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_train, self.x_train, _ = create_pandas.create_weather_pandas("Train")
        self.y_test, self.x_test, self.offset = create_pandas.create_weather_pandas("Test")
        self.prediction = False
        print("loaded train and test data")

# def regression():
#     # Actual performance of the functions.
#     x_train, y_train, x_test, y_test, mean, offset,set_size = create_train_test()
#     print("base: ",np.sqrt(sum(np.square(mean-y_test)*offset)/len(y_test)))
#     print("ridge: ",functions.ridge_regression([-5], x_train, x_test, y_train, y_test, offset))
#     print("bayes: ",functions.Bayes_regression([0.00980137, -0.00372394, -0.00682109, -0.04635455], x_train, x_test, y_train, y_test, offset))
#     print("decision tree:", functions.decision_tree(x_train, x_test, y_train, y_test, offset))
#     print("KNN: ",functions.k_nearest([1], x_train, x_test, y_train, y_test, offset))

    def train_nn(self, epochs=200, batch_size=32,name="model",smoothing_factor = 1):
        # The running of the neural network
        x_train = self.scaler.fit_transform(self.x_train)
        x_test = self.scaler.fit_transform(self.x_test)
        nn_prediction = rnn.rnn(x_train, x_test, self.y_train, self.y_test, self.offset, epochs, batch_size, name)
        self.prediction = nn_prediction
        nn_prediction = scipy.ndimage.gaussian_filter(nn_prediction,smoothing_factor)
        plt.plot(nn_prediction,label='nn output',color="green")

    def test_nn(self,name="model",smoothing_factor = 1):
        x_test = self.scaler.fit_transform(self.x_test)
        nn_prediction = rnn.load_rnn(x_test, self.y_test, self.offset, name)
        self.prediction = nn_prediction
        nn_prediction = scipy.ndimage.gaussian_filter(nn_prediction,smoothing_factor)
        plt.plot(nn_prediction,label='nn output',color="green")

    def train_mlp(self,smoothing_factor=1):
        pred = mlp.run_session(self.x_train, self.x_test, self.y_train, self.y_test, self.offset)[0]*self.offset
        self.prediction = pred
        pred = scipy.ndimage.gaussian_filter(pred,smoothing_factor)
        plt.plot(pred,label='mlp output',color="pink")

    def ridge_regression(self,par=[-5],smoothing_factor = 1):
        alpha = par[0]
        regr = linear_model.Ridge(alpha,solver="svd")
        regr.fit(self.x_train,self.y_train)
        ridge_pred = regr.predict(self.x_test)*self.offset
        self.prediction = ridge_pred
        ridge_pred = scipy.ndimage.gaussian_filter(ridge_pred,smoothing_factor)
        plt.plot(ridge_pred,label='Ridge output',color="pink")

    def Bayes_regression(self,par=[-3.63600029e-04,  2.33234414e-03,  5.52569969e-02, -4.99181236e-01],smoothing_factor = 1):
        alpha_1 = par[0]
        alpha_2 = par[1]
        lambda_1 = par[2]
        lambda_2 = par[3]
        Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
        Bay.fit(self.x_train,self.y_train)
        Bay_pred = Bay.predict(self.x_test)*self.offset
        self.prediction = Bay_pred
        Bay_pred = scipy.ndimage.gaussian_filter(Bay_pred,smoothing_factor)
        plt.plot(Bay_pred,label='Bayes output',color="blue")

    def decision_tree(self,smoothing_factor = 1):
        dec = DecisionTreeRegressor()
        dec.fit(self.x_train, self.y_train)
        pred = dec.predict(self.x_test)*self.offset
        self.prediction = pred
        pred = scipy.ndimage.gaussian_filter(pred,smoothing_factor)
        plt.plot(pred,label='Decision tree output',color="red")


    def k_nearest(self, par=[5],smoothing_factor=1):
        neighbors = par[0]
        neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
        neigh.fit(self.x_train, self.y_train)
        neigh_pred = neigh.predict(self.x_test)*self.offset
        self.prediction = neigh_pred
        neigh_pred = scipy.ndimage.gaussian_filter(neigh_pred,smoothing_factor)
        plt.plot(neigh_pred,label='KNN output',color="green")


    def plot_real(self,smoothing_factor = 1):
        real = scipy.ndimage.gaussian_filter(self.y_test,smoothing_factor)*self.offset
        plt.plot(real,label='real output',color="black",linewidth=2.0)

    def show_plot(self):
        plt.legend()
        plt.xlabel("time (days)")
        plt.ylabel("solar panel output (kWh)")
        plt.show()

    def calculate_error(self):
        return np.sqrt(sum(np.square(self.prediction-self.y_test))/len(self.y_test))

def plot_all():
    predictor = Predictor()
    #predictor.train_nn(100, 32,"model_11",5)
    #predictor.test_nn("model_11", smoothing_factor = 5)
    #predictor.train_mlp()
    predictor.ridge_regression(smoothing_factor = 5)
    print("ridge",predictor.calculate_error())
    predictor.Bayes_regression(smoothing_factor = 5)
    print("Bayes",predictor.calculate_error())
    predictor.decision_tree(smoothing_factor = 5)
    print("Decision tree",predictor.calculate_error())
    predictor.k_nearest(smoothing_factor = 5)
    print("KNN",predictor.calculate_error())
    predictor.plot_real(smoothing_factor = 5)
    predictor.show_plot()
#regression()
#train_nn()
#test_nn()
