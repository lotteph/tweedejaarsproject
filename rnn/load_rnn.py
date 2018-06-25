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

def make_panda(solar, weather):
    new = weather
    del new["Unnamed: 0"]
    new["number_of_panels"] = solar["Number_of_panels"]
    new["max_power"] = solar["Max_power"]
    new["system_size"] = solar["System_size"]
    new["number_of_inverters"] = solar["Number_of_inverters"]
    new["inverter_size"] = solar["Inverter_size"]
    new["tilt"] = solar["Tilt"]
    return new

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

SP = False
W = False
print(data.keys())
for code in postal_codes:
    for year in data[code]:
        W2 = pd.read_csv("data/" + str(code) + "_" + str(year) + "_W.csv")
        if type(W) != type(False):
            W = pd.DataFrame.append(W,W2)
        else:
            W = W2
        SP2 = pd.read_csv("data/" + str(code) + "_" + str(year) + "_S.csv")
        if type(SP) != type(False):
            SP = pd.DataFrame.append(SP,SP2)
        else:
            SP = SP2

results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])
W["time"] = 0
# W = make_panda(SP,W)
del W["Unnamed: 0"]
scaled = scaler.fit_transform(W)

train_size = len(results)-365*1

x_train = scaled[:train_size,:]
x_test =  scaled[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]

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
