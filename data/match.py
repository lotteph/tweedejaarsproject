import csv
import calendar
import numpy as np
import pandas as pd
from scrape import main
from Weather_test import _main_

def add_month(weather, year):
    months = []

    for i in range(1,13):
        days = calendar.monthrange(year, i)
        for j in range(days[1]):
            months.append(i)

    months = np.array(months)
    months = pd.DataFrame(months)
    weather["month"] = months
    return weather

def solar_csv(solar):
    solar = solar[solar["Generated"] != 0.0]
    solar = solar.sort_values("Date")
    solar = solar.reset_index(drop=True)
    return solar

def weather_csv(solar, weather):
    date = solar["Date"]
    new = pd.DataFrame(weather[:0])
    for row in date:
        w = weather.loc[weather["time"] == row]
        new = new.append(w)
    new = new.reset_index(drop=True)
    return new

def match_data(postalcode, year, id, sid):
    solar_file = str(postalcode) + "_" + str(year) + "_S.csv"
    weather_file = str(postalcode) + "_" + str(year) + "_W.csv"

    solarpanel = main(1, year, 1, year+1, id, sid)
    weather = _main_(postalcode, year)

    weather = add_month(weather, year)
    solarpanel = solar_csv(solarpanel)
    weather = weather_csv(solarpanel, weather)

    weather.to_csv(solar_file)
    solarpanel.to_csv(weather_file)

match_data(7325, 2013,"13448", "13242")
