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
    json_str = r.content.decode('utf8').replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

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
        for day in range(1,calendar.monthrange(year,month)[1]+1):
        #for day in range(1,2):
            date = datetime.date(year,month,day)
            features = get_features(longitude,latitude,date)['daily']['data'][0]
            toAdd = []
            for key in FACTORS:
                if key in features.keys():
                    toAdd.append(features[key])
                else:
                    toAdd.append(0)
            data[nday] = toAdd
            nday += 1
    return data

def relative_times(db):
    #Makes sunset and sunrise time relative to daytime
    count = 0
    for col in db:
        if count == 2 or count == 3:
            db[col] = db[col] - db["time"]
        count += 1
    return db

def make_database(longitude,latitude,file_name,year):
    #Makes a weather database and saves it
    result = get_data(longitude,latitude,year)
    res = pd.DataFrame(result,columns=FACTORS)
    db = relative_times(res)
    db["longitude"]= longitude
    db["latitude"] = latitude
    db.to_csv(file_name)

def _main_(postal_code,year,file_name):
    [longitude, latitude] = get_LL("7325",read_file())
    if type(year) == int:
        db = make_database(longitude,latitude,str(year) + "_" + str(postal_code) + file_name + "_w.csv",year)
    elif type(year) == list:
        for i in range(0,len(year)):
            print('year:',year[i])
            db = make_database(longitude,latitude,str(year[i]) + "_" + str(postal_code) + file_name + "_w.csv",year[i])

_main_(7559,[2013,2014],"Weather")
