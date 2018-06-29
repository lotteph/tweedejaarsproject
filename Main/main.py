import pandas as pd
import numpy as np
import scipy
import scipy.ndimage
import csv
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import create_pandas
import rnn
import multilayer_perceptron as mlp

class Predictor(object):
    #Can load train and test data. And can run different algorithms to predict solar panel output.

    def __init__(self):
        #Loads data and sets variables.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_train, self.x_train, _ = create_pandas.create_weather_pandas("Train")
        self.y_test, self.x_test, self.offset = create_pandas.create_weather_pandas("Test")
        self.prediction = False
        print("loaded train and test data")

    def train_nn(self, epochs=5000, batch_size=32,name="model",smoothing_factor = 1):
        #Trains and tests a RNN.
        x_train = self.scaler.fit_transform(self.x_train)
        x_test = self.scaler.fit_transform(self.x_test)
        nn_prediction = rnn.rnn(x_train, x_test, self.y_train, self.y_test, self.offset, epochs, batch_size, name)
        self.prediction = nn_prediction
        nn_prediction = scipy.ndimage.gaussian_filter(nn_prediction,smoothing_factor)
        plt.plot(nn_prediction,label='nn output',color="green")

    def test_nn(self,name="model",smoothing_factor = 1):
        #Loads a existing network to test on.
        x_test = self.scaler.fit_transform(self.x_test)
        nn_prediction = rnn.load_rnn(x_test, self.y_test, self.offset, name)
        self.prediction = nn_prediction
        nn_prediction = scipy.ndimage.gaussian_filter(nn_prediction,smoothing_factor)
        plt.plot(nn_prediction,label='nn output',color="green")

    def train_mlp(self,smoothing_factor=1):
        pred = mlp.run_mlp()
        self.prediction = pred
        pred = scipy.ndimage.gaussian_filter(pred,smoothing_factor)
        plt.plot(pred,label='mlp output',color="pink")

    def ridge_regression(self,par=[-5],smoothing_factor = 1):
        #Trains and tests using ridge regression.
        alpha = par[0]
        x_train = self.scaler.fit_transform(self.x_train)
        x_test = self.scaler.fit_transform(self.x_test)
        regr = linear_model.Ridge(alpha,solver="svd")
        regr.fit(x_train,self.y_train)
        ridge_pred = regr.predict(x_test)*self.offset
        self.prediction = ridge_pred
        ridge_pred = scipy.ndimage.gaussian_filter(ridge_pred,smoothing_factor)
        plt.plot(ridge_pred,label='Ridge output',color="red")

    def Bayes_regression(self,par=[-3.63600029e-04,  2.33234414e-03,  5.52569969e-02, -4.99181236e-01],smoothing_factor = 1):
        #Trains and tests using bayes regression.
        alpha_1 = par[0]
        alpha_2 = par[1]
        lambda_1 = par[2]
        lambda_2 = par[3]
        x_train = self.scaler.fit_transform(self.x_train)
        x_test = self.scaler.fit_transform(self.x_test)
        Bay = linear_model.BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1,lambda_2=lambda_2)
        Bay.fit(x_train,self.y_train)
        Bay_pred = Bay.predict(x_test)*self.offset
        self.prediction = Bay_pred
        Bay_pred = scipy.ndimage.gaussian_filter(Bay_pred,smoothing_factor)
        plt.plot(Bay_pred,label='Bayes output',color="green")

    def decision_tree(self,smoothing_factor = 1):
        #Trains and tests using decision tree.
        dec = DecisionTreeRegressor()
        x_train = self.scaler.fit_transform(self.x_train)
        x_test = self.scaler.fit_transform(self.x_test)
        dec.fit(x_train, self.y_train)
        pred = dec.predict(x_test)*self.offset
        self.prediction = pred
        pred = scipy.ndimage.gaussian_filter(pred,smoothing_factor)
        plt.plot(pred,label='Decision tree output',color="red")


    def k_nearest(self, par=[5],smoothing_factor=1):
        #Trains and tests using KNN.
        neighbors = par[0]
        neigh = KNeighborsRegressor(n_neighbors=int(neighbors))
        neigh.fit(self.x_train, self.y_train)
        neigh_pred = neigh.predict(self.x_test)*self.offset
        self.prediction = neigh_pred
        neigh_pred = scipy.ndimage.gaussian_filter(neigh_pred,smoothing_factor)
        plt.plot(neigh_pred,label='KNN output',color="green")


    def plot_real(self,smoothing_factor = 1):
        #Plots the real output of the solar panel
        real = scipy.ndimage.gaussian_filter(self.y_test,smoothing_factor)*self.offset
        plt.plot(real,label='real output',color="black",linewidth=2.0)

    def show_plot(self):
        #Shows the queued plots with labels and legend.
        plt.legend()
        plt.xlabel("time (days)")
        plt.ylabel("solar panel output (kWh)")
        plt.show()

    def calculate_error(self):
        #Calculates the error of the last predicted model.
        return np.sqrt(np.sum(np.square(self.prediction-(self.y_test*self.offset)))/len(self.y_test))
        #return sum(np.square(self.prediction-self.y_test*self.offset))/len(self.y_test)

def plot_all():
    #Plots all the real and predicted outputs.
    predictor = Predictor()
    predictor.test_nn("model_real_specific_2", smoothing_factor = 10)
    print("NN",predictor.calculate_error())
    predictor.train_mlp(smoothing_factor=10)
    print("MLP",predictor.calculate_error())
    predictor.ridge_regression(smoothing_factor = 10)
    print("Ridge",predictor.calculate_error())
    predictor.Bayes_regression(smoothing_factor = 10)
    print("Bayes",predictor.calculate_error())
    predictor.decision_tree(smoothing_factor = 10)
    print("Decision tree",predictor.calculate_error())
    predictor.k_nearest(smoothing_factor = 10)
    print("KNN",predictor.calculate_error())
    predictor.plot_real(smoothing_factor = 10)
    predictor.show_plot()

def error_all():
    #Tests all models and prints the errors of all.
    predictor = Predictor()
    predictor.test_nn("model_real", smoothing_factor = 1)
    print("NN",predictor.calculate_error())
    predictor.ridge_regression(smoothing_factor = 1)
    print("Ridge",predictor.calculate_error())
    predictor.Bayes_regression(smoothing_factor = 1)
    print("Bayes",predictor.calculate_error())
    predictor.decision_tree(smoothing_factor = 1)
    print("Decision tree",predictor.calculate_error())
    predictor.k_nearest(smoothing_factor = 1)
    print("KNN",predictor.calculate_error())
