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
postal_codes = ["3481","7325","2134","2201","6591","7559"]
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

offset = SP["Number_of_panels"].values[train_size:]*SP["Max_power"].values[train_size:]

scaled = scaler.fit_transform(W)


x_train = scaled[:train_size,:]
x_test = scaled[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

mlp = MLPRegressor(hidden_layer_sizes=(50,),  activation='tanh', solver='adam',    alpha=0.0001,batch_size=32,
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000000, shuffle=True,
               random_state=None, tol=0.00001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

print(x_train.shape)
mlp.fit(x_train,y_train)
predictions = mlp.predict(x_test)

mean = np.mean(y_train)+np.zeros(len(y_test))
print("base: ",np.sqrt(sum(np.square(mean-y_test)*offset)/len(y_test)))
print(np.sqrt(sum(np.square(predictions-y_test))/len(y_test)))

pre = scipy.ndimage.gaussian_filter(predictions,5)
plt.plot(pre,label='predicted output',color="red")
real = scipy.ndimage.gaussian_filter(y_test,5)
plt.plot(real,label='real output',color="blue")
plt.legend()
plt.xlabel("time (days)")
plt.ylabel("solar panel output (kWh)")
plt.title("NN tree predicted vs real output of 2017")
plt.show()
