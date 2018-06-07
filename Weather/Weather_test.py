import requests
import csv
import json
import datetime
from time import mktime
import calendar
import numpy as np
import pandas as pd

SECRET_KEY = "4026e043d76e5963c2c57ab535353197"
FACTORS = ['time','cloudCover','sunriseTime','sunsetTime','temperatureHigh','temperatureLow','temperatureMin','temperatureMax','visibility']

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

<<<<<<< HEAD
def get_features(longitude,latitude,date):
#Gets a json_object with the weather of a given date on the given coordinate
    unix_date = mktime(date.timetuple())
    URL = "https://api.darksky.net/forecast/" + SECRET_KEY + "/" + str(latitude) + "," + str(longitude) + "," + str(int(unix_date)) +"?exclude=currently,minutely,hourly,alerts"
    r = requests.get(URL)
    json_str = r.content.decode('utf8').replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

def get_year(longitude,latitude,year):
#Gets the weather of a calendar year
    if calendar.isleap(year):
        size = (364,len(FACTORS))
    else:
        size = (365,len(FACTORS))
    data = np.zeros(size)
    nday = 0
    for month in range(1,13):
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
            print(nday)
            nday += 1
    return data


def make_database():
    result = get_year(longitude,latitude,2017)
    res = pd.DataFrame(result,columns=FACTORS)
    res.to_csv('Weather.csv')

def normalize(colomn):
    return colomn - np.mean(colomn)

=======
LL = get_LL("2152",read_file())
secret_key = ""
>>>>>>> 11c2aeb95556a34a181e038d3d26f167b1da9e96

[longitude, latitude] = get_LL("1078",read_file())

res = pd.read_csv('Weather.csv')
for column in res:
    res[column] = normalize(res[column])
res.to_csv('Weather.csv')
