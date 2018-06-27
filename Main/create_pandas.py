import pandas as pd
import numpy as np
import scipy
import scipy.ndimage
import csv
import os
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from os import listdir
import re

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

def create_weather_pandas(input_set):
    SP = False
    W = False

    data, postal_codes = create_postal_codes(input_set)

    for code in postal_codes:
        for year in data[code]:
            W2 = pd.read_csv("../"+input_set+"/"+str(code)+ "_" + str(year) + "_W.csv")
            if type(W) != type(False):
                W = pd.DataFrame.append(W,W2)
            else:
                W = W2
            SP2 = pd.read_csv("../"+input_set+"/"+ str(code) + "_" + str(year) + "_S.csv")
            if type(SP) != type(False):
                SP = pd.DataFrame.append(SP,SP2)
            else:
                SP = SP2
    results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])
    W["time"] = 0
    W = make_pandas(SP,W)
    set_size = set_array_size(postal_codes)
    return results, W, SP, set_size

def create_postal_codes(input_set):
    postal_code = []
    d = dict()
    files = os.listdir("../"+input_set+"/")
    for name in files:
        codes = re.search("(\d{4})\_\d{4}", name)
        if codes != None:
            codes = codes.group(1)
            years = re.search("(\d{4})\_[A-Z]", name).group(1)
            if codes not in postal_code:
                postal_code.append(codes)
                d[codes] = [years]
            elif years not in d[codes]:
                d[codes].append(years)
    return d, postal_code

def set_array_size(postal_codes):
    nr_codes = len(postal_codes)
    if (nr_codes > 1):
        set_size = 1
    else:
        set_size = 0
    return set_size
