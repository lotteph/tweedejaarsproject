# Example of LSTM to learn a sequence
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Activation, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
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
postal_codes = ["7325","7559","2201","6591","3481","5384","5552","3994","2134"]

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
W = make_panda(SP,W)
# del W["Unnamed: 0"]
scaled = scaler.fit_transform(W)

train_size = len(results)-365*9

x_train = scaled[:train_size,:]
x_test =  scaled[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


model = Sequential()
# model.add(LSTM(20, input_shape=(1,12), return_sequences=True))
# model.add(LSTM(6, input_shape=(1,12), return_sequences=True))
model.add(LSTM(20, input_shape=(1,18), return_sequences=True))
model.add(LSTM(12, input_shape=(1,18), return_sequences=True))
# model.add(Flatten())
model.add(LSTM(6))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(optimizer='adam', loss='mean_squared_error')


# 1 l:
# history = model.fit(x_train, y_train, batch_size=7, epochs=100) .1243 7559
# history = model.fit(x_train, y_train, batch_size=7, epochs=500) .1079 7559
# history = model.fit(x_train, y_train, batch_size=1, epochs=100) .114 7559
# history = model.fit(x_train, y_train, batch_size=7, epochs=2000) .1049 7559
# history = model.fit(x_train, y_train, batch_size=10, epochs=2000) .1019 7559
# history = model.fit(x_train, y_train, batch_size=128, epochs=20000) .1007 7559
# history = model.fit(x_train, y_train, batch_size=1, epochs=20000) ...
# 2 l:
 # history = model.fit(x_train, y_train, batch_size=128, epochs=20000) .0784 7559
 # 3 l:
# history = model.fit(x_train, y_train, batch_size=128, epochs=10000) .0514 7559
# history = model.fit(x_train, y_train, batch_size=32, epochs=10000) .0241 7559
history = model.fit(x_train, y_train, batch_size=32, epochs=1000)

# loss, accuracy = model.evaluate(X, y)
predictions = model.predict(x_test)

model_json = model.to_json()
with open("model_8.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_8.h5")
print("Saved model to disk")

print(history.history.keys())

p = predictions.ravel()


# error = (np.sum(np.square(p-y_test))/len(y_test))*offset
error = np.sqrt(np.sum(np.square(y_test-p)))/len(y_test)*offset
# error = np.square((p-y_test)*offset)
print("error:")
print(error)


p = scipy.ndimage.gaussian_filter(p,5)
y_test = scipy.ndimage.gaussian_filter(y_test,5)
plt.plot(p,label='predicted output',color="red")
plt.plot(y_test,label='real output',color="blue")
plt.legend()
plt.title("Neural network")
plt.show()

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
