import requests
import csv
import json
import datetime
from time import mktime
import calendar
import numpy as np
import pandas as pd

FACTORS = ["time","cloudCover","sunriseTime","sunsetTime","temperatureHigh","temperatureLow","temperatureMin","temperatureMax","visibility"]

#SECRET_KEY = "59e1462995d5ab68f2c1f29478f98081"
#SECRET_KEY = "4026e043d76e5963c2c57ab535353197"
#SECRET_KEY = "0cd188a858bdb7fda3eb65f83811fedc"
#SECRET_KEY = "da2aefea8bbb0c1a91c7059ad966008d"
SECRET_KEY = "d07fa9b030dfc7ae1a1897828f0e01de"

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
    json_str = r.content.decode('utf8').replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

def get_weather(longitude,latitude,date):
    #Gets the weather of a location on a day
    features = get_features(longitude,latitude,date)['daily']['data'][0]
    toAdd = []
    for key in FACTORS:
        if key in features.keys():
            toAdd.append(features[key])
        else:
            toAdd.append(0)
    return toAdd

def get_data(longitude,latitude,year):
    #Gets the weather of a calendar year
    if calendar.isleap(year):
        size = (367,len(FACTORS))
    else:
        size = (367,len(FACTORS))
    data = np.zeros(size)
    nday = 0
    for month in range(1,13):
        print('month: ', month)
        '''
        Comment the line below and uncomment the line
        below that to reduce the amount of API requests
        '''
        # for day in range(1,calendar.monthrange(year,month)[1]+1):
        for day in range(1,2):
            date = datetime.date(year,month,day)
            data[nday] = get_weather(longitude,latitude,date)
            nday += 1
    return data

def relative_times(db):
    #Makes sunset and sunrise time relative to daytime
    db["sunriseTime"] = db["sunriseTime"] - db["time"]
    db["sunsetTime"] = db["sunsetTime"] - db["time"]
    return db

def make_database(longitude,latitude,year):
    #Makes a weather database and saves it
    result = get_data(longitude,latitude,year)
    res = pd.DataFrame(result,columns=FACTORS)
    db = relative_times(res)
    db["longitude"] = longitude
    db["latitude"] = latitude
    return db

def _main_(postal_code,year):
    [longitude, latitude] = get_LL(str(postal_code),read_file())
    if type(year) == int:
        return make_database(longitude,latitude,year)
    elif type(year) == list:
        for i in range(0,len(year)):
            print('year:',year[i])
            return make_database(longitude,latitude,year[i])
