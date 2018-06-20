import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import matplotlib.pyplot as plt

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

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 13, 5, 1
epochs = 300
seq_length = 20
lr = 0.1

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

W = pd.read_csv(postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv(postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    W["time"] = 0
    SP2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)
results = np.array(SP["Generated"])

W = W.values
train_size = len(results)-365

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

print(x_train.shape)

data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = Variable(torch.Tensor(np.transpose(x_train)).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(np.transpose(y_train)).type(dtype), requires_grad=False)

w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
init.normal(w1, 0.0, 0.4)
w1 =  Variable(w1, requires_grad=True)
w2 = torch.FloatTensor(hidden_size, output_size).type(dtype)
init.normal(w2, 0.0, 0.3)
w2 = Variable(w2, requires_grad=True)

def forward(input, context_state, w1, w2):
  xh = torch.cat((input, context_state), 1)
  context_state = torch.tanh(xh.mm(w1))
  out = context_state.mm(w2)
  return  (out, context_state)

for i in range(epochs):
  total_loss = 0
  context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
  for j in range(x.size(0)):
    input = x[j:(j+1)]
    target = y[j:(j+1)]
    (pred, context_state) = forward(input, context_state, w1, w2)
    loss = (pred - target).pow(2).sum()/2
    total_loss += loss
    loss.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    context_state = Variable(context_state.data)
  if i % 10 == 0:
     print("Epoch: {} loss {}".format(i, total_loss.data[0]))

context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
predictions = []

for i in range(x.size(0)):
  input = x[i:i+1]
  (pred, context_state) = forward(input, context_state, w1, w2)
  context_state = context_state
  predictions.append(pred.data.numpy().ravel()[0])
plt.plot(predictions)
plt.show()
