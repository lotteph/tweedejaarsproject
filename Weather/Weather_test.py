import requests
import csv
import json
import datetime
from time import mktime
import calendar
import numpy as np
import pandas as pd

#Reads the postal code and coordinates file
def read_file():
    with open("4pp.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]
    return(data)

#Gets the coordinates from a postal code
def get_LL(postalCode,data) :
    for i in data:
        if i[1] == postalCode:
            return [i[-2],i[-3]]

secret_key = "4026e043d76e5963c2c57ab535353197"

LL = get_LL("1078",read_file())
longitude = LL[0]
latitude = LL[1]

def get_features(longitude,latitude,date):
    unix_date = mktime(date.timetuple())
    URL = "https://api.darksky.net/forecast/" + secret_key + "/" + str(latitude) + "," + str(longitude) + "," + str(int(unix_date)) +"?exclude=currently,minutely,hourly,alerts"
    r = requests.get(URL)
    json_str = r.content.decode('utf8').replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

factors = ['time','cloudCover','sunriseTime','sunsetTime','temperatureHigh','temperatureLow','temperatureMin','temperatureMax','visibility']

#Gets the weather of a calendar year
def get_year(longitude,latitude,year):
    if calendar.isleap(year):
        size = (364,len(factors))
    else:
        size = (365,len(factors))
    data = np.zeros(size)
    nday = 0
    for month in range(1,13):
        for day in range(1,calendar.monthrange(year,month)[1]+1):
        #for day in range(1,2):
            date = datetime.date(year,month,day)
            features = get_features(longitude,latitude,date)['daily']['data'][0]
            toAdd = []
            for key in factors:
                if key in features.keys():
                    toAdd.append(features[key])
                else:
                    toAdd.append(0)
            data[nday] = toAdd
            print(nday)
            nday += 1
    return data


def make_database():
    result = get_year(longitude,latitude,2017)
    res = pd.DataFrame(result,columns=factors)
    res.to_csv('Weather.csv')

def normalize(colomn):
    return colomn - np.mean(colomn)

res = pd.read_csv('Weather.csv')
for column in res:
    res[column] = normalize(res[column])
res.to_csv('Weather.csv')
