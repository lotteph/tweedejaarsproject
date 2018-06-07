import requests
import csv
import json
import datetime
from time import mktime
import calendar
import numpy as np
import pandas as pd


SECRET_KEY = "4026e043d76e5963c2c57ab535353197"
FACTORS = ["time","cloudCover","sunriseTime","sunsetTime","temperatureHigh","temperatureLow","temperatureMin","temperatureMax","visibility"]


def read_file():
#Reads the postal code and coordinates file
    with open("4pp.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]
    return(data)

def get_LL(postalCode,data) :
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

def get_data(longitude,latitude,year):
#Gets the weather of a calendar year
    if calendar.isleap(year):
        size = (364,len(FACTORS))
    else:
        size = (365,len(FACTORS))
    data = np.zeros(size)
    nday = 0
    for month in range(1,13):
        print('month: ', month)
        '''
        Comment the line below and uncomment the line
        below that to reduce the amount of API requests
        '''
        for day in range(1,calendar.monthrange(year,month)[1]+1):
        #for day in range(1,2):
            date = datetime.date(year,month,day)
            features = get_features(longitude,latitude,date)["daily"]["data"][0]
            toAdd = []
            for key in FACTORS:
                if key in features.keys():
                    toAdd.append(features[key])
                else:
                    toAdd.append(0)
            data[nday] = toAdd
            nday += 1
    return data

def make_database(file_name,year):
    #Makes a weather database and saves it
    result = get_data(longitude,latitude,year)
    res = pd.DataFrame(result,columns=FACTORS)
    res.to_csv(file_name)

def normalize(colomn):
    #normalizes a colomn
    return col - np.mean(col)

def normalize_data_base(db):
    #normalizes a database
    for col in db:
        db[column] = normalize(db[col])

def _main_(postal_code,year,file_name):
    [longitude, latitude] = get_LL("1078",read_file())
    db = []
    if type(year) == int:
        db.append(get_data(longitude,latitude,year))
    elif type(year) == list:
        for i in year:
            print('year:',i)
            db.append(get_data(longitude,latitude,i))
    db = normalize_data_base(db)
    res.to_csv(file_name)

_main_(2152,[2016,2017],"Weather2017.csv")
