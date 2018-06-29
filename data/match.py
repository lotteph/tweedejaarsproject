import csv
import calendar
import numpy as np
import pandas as pd
from scrape import main
from Weather_test import _main_

def add_month(weather, year):
    ''' Makes an array with the amount of days
        per months to add to the weather data panda.
    '''
    months = []
    for i in range(1,13):
        days = calendar.monthrange(year, i)
        # days is a tuple with the amount as second argument.
        for j in range(days[1]):
            months.append(i)

    months = np.array(months)
    months = pd.DataFrame(months)
    weather["month"] = months
    return weather

def solar_csv(solar):
    ''' Cleans the solarpanel data from outliers and
        sorts it according to the date.
    '''
    solar = solar[int(solar["Generated"]) != 0.0]
    solar = solar.sort_values("Date")
    solar = solar.reset_index(drop=True)
    return solar

def weather_csv(solar, weather):
    ''' Creates a weather data panda with the same
        days as the solar panel panda and in the same
        order.
    '''
    date = solar["Date"]
    new = pd.DataFrame(weather[:0])
    for row in date:
        w = weather.loc[weather["time"] == row]
        new = new.append(w)
    new = new.reset_index(drop=True)
    return new

def match_data(postalcode, year, id, sid):
    ''' Creates two csv files one with weather and one with
        solar panel data. 
    '''
    solar_file = str(postalcode) + "_" + str(year) + "_S.csv"
    weather_file = str(postalcode) + "_" + str(year) + "_W.csv"

    solarpanel = main(1, year, 1, year+1, id, sid)
    weather = _main_(postalcode, year)

    weather = add_month(weather, year)
    solarpanel = solar_csv(solarpanel)
    weather = weather_csv(solarpanel, weather)

    weather.to_csv(weather_file)
    solarpanel.to_csv(solar_file)
    
# for every year in the list data arrays are made 
for y in [2013,2014]:
    match_data(7559, y,"10324", "8645")
