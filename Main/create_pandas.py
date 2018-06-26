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

#make this into pandas
def make_pandas(solar, weather):
    new = weather
    del new["Unnamed: 0"]
    new["number_of_panels"] = solar["Number_of_panels"]
    new["max_power"] = solar["Max_power"]
    new["system_size"] = solar["System_size"]
    new["number_of_inverters"] = solar["Number_of_inverters"]
    new["inverter_size"] = solar["Inverter_size"]
    new["tilt"] = solar["Tilt"]
    return np.array(new)


def create_weather_pandas(data, postal_codes):
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
    nr_codes = len(postal_codes)
    if (nr_codes > 1):
        W = make_pandas(SP,W)
        set_size = 1
    else:
        set_size = 0
    return results, W, SP, set_size
