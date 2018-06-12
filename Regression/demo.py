import scipy
import numpy as np
from sklearn.externals import joblib
import sklearn
import requests
import csv
import datetime
from time import mktime
import json

regr = joblib.load('ridge.pkl')
SECRET_KEY = "4026e043d76e5963c2c57ab535353197"
FACTORS = ["time","cloudCover","sunriseTime","sunsetTime","temperatureHigh","temperatureLow","temperatureMin","temperatureMax","visibility"]

def read_file():
#Reads the postal code and coordinates file
    with open("4pp.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]
    return(data)

def get_LL(postalCode,data):
#Gets the coordinates from a postal code
    for i in data:
        if i[1] == postalCode:
            return [i[-2],i[-3]]

def get_features(longitude,latitude,date):
#Gets a json_object with the weather of a given date on the given coordinate
    unix_date = mktime(date.timetuple())
    URL = "https://api.darksky.net/forecast/" + SECRET_KEY + "/" + str(latitude) + "," + str(longitude) + "," + str(int(unix_date)) +"?exclude=currently,minutely,hourly,alerts"
    r = requests.get(URL)
    json_str = r.content.decode("utf8").replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

def get_Weather(longitude,latitude,year,month,day):
#Gets the weather of a calendar year
    date = datetime.date(year,month,day)
    features = get_features(longitude,latitude,date)["daily"]["data"][0]
    toAdd = []
    for key in FACTORS:
        if key in features.keys():
            toAdd.append(features[key])
        else:
            toAdd.append(0)
    return toAdd

LL = get_LL(str(1078),read_file())
Weather = get_Weather(LL[0],LL[1],2018,11,27)
print(Weather)
pred = regr.predict(np.reshape(Weather,(1,-1)))
print(pred)
