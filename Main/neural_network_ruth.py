import pandas as pd
import numpy as np
import scipy
import scipy.ndimage
import csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def neural_net(x_train, x_test, y_train, y_test, offset):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

    mlp = MLPRegressor(hidden_layer_sizes=(50,),  activation='tanh', solver='adam',    alpha=0.0001,batch_size=32,
                   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000000, shuffle=True,
                   random_state=None, tol=0.00001, verbose=False, warm_start=False, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-08)

    mlp.fit(x_train,y_train)
    predictions = mlp.predict(x_test)

    mean = np.mean(y_train)+np.zeros(len(y_test))

    pre = scipy.ndimage.gaussian_filter(predictions,5)
    plt.plot(pre,label='predicted output',color="red")
    real = scipy.ndimage.gaussian_filter(y_test,5)
    plt.plot(real,label='real output',color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("NN tree predicted vs real output of 2017")
    plt.show()

    return np.sqrt(sum(np.square(predictions-y_test))/len(y_test))
