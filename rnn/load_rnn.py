from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Activation, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
scaler = MinMaxScaler(feature_range=(0, 1))



data = dict()
data["7325"] = range(2013,2019)
data["7559"] = range(2013,2019)
data["2134"] = range(2009,2018)
data["2201"] = range(2013,2019)
data["6591"] = range(2015,2019)
data["3481"] = range(2015,2019)
data["5384"] = range(2013,2016)
data["5552"] = range(2013,2017)
data["3994"] = range(2012,2018)
postal_codes = ["2134"]

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# load json and create model
json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_1.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mean_squared_error', optimizer='adam')
score = loaded_model.predict(x_test)
p = score.ravel()

# error = np.square((y_test-p)*offset)
print("error:")
# print(np.sum(error)/len(y_test))

# print((np.sum(np.square(p-y_test))/len(y_test))*offset)
print(np.sqrt(np.sum(np.square(y_test-p)))/len(y_test)*offset)
p = p * offset
y_test = y_test * offset
p = scipy.ndimage.gaussian_filter(p,5)
y_test = scipy.ndimage.gaussian_filter(y_test,5)
plt.plot(p,label='predicted output',color="red")
plt.plot(y_test,label='real output',color="blue")
plt.legend()
plt.title("Neural network")
plt.show()
