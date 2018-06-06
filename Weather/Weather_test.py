import requests
import csv
import json
import datetime
from time import mktime
from calendar import monthrange
import numpy as np

def read_file():
    with open("4pp.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]
    return(data)

def get_LL(postalCode,data) :
    for i in data:
        if i[1] == postalCode:
            return [i[-2],i[-3]]

secret_key = "59e1462995d5ab68f2c1f29478f98081"

LL = get_LL("2152",read_file())
longitude = LL[0]
latitude = LL[1]

def get_features(longitude,latitude,date):
    unix_date = mktime(date.timetuple())
    URL = "https://api.darksky.net/forecast/" + secret_key + "/" + str(latitude) + "," + str(longitude) + "," + str(int(unix_date)) +"?exclude=currently,minutely,hourly,alerts"
    r = requests.get(URL)
    json_str = r.content.decode('utf8').replace("'", '"')
    json_object = json.loads(json_str)
    return json_object

def get_year(longitude,latitude,year):
    size = (365,29)
    data = np.zeros(size)
    nday = 0
    for month in range(1,2):
        #for day in monthrange(year,month):
        for day in range(1,2):
            nday += 1
            date = datetime.date(year,month,day)
            features = list(get_features(longitude,latitude,date)['daily']['data'][0].items())
            print(features)
            data[nday] = features
    return data

result = get_year(longitude,latitude,2010)
print(result)
