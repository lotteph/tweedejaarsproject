import csv
import calendar
import numpy as np
import pandas as pd

def percentages(file):
    data = pd.read_csv(file)
    base = data["all"]

    cloud_cover = data["cloudCover"]
    sunset_time = data["sunsetTime"]
    sunrise_time = data["sunriseTime"]
    temp_low = data["tempLow"]
    temp_high = data["tempHigh"]
    temp_max = data["tempMax"]
    temp_min = data["tempMin"]
    visibility = data["visibility"]
    month = data["month"]

    p_cloud_cover = ((base - cloud_cover) / base) * 100
    p_sunset_time = ((base - sunset_time) / base) * 100
    p_sunrise_time = ((base - sunrise_time) / base) * 100
    p_temp_low = ((base - temp_low) / base) * 100
    p_temp_high = ((base - temp_high) / base) * 100
    p_temp_max = ((base - temp_max) / base) * 100
    p_temp_min = ((base - temp_min) / base) * 100
    p_visibility = ((base - visibility) / base) * 100
    p_month = ((base - month) / base) * 100

    p_cloud_cover = pd.DataFrame(p_cloud_cover)
    p_sunset_time = pd.DataFrame(p_sunset_time)
    p_sunrise_time = pd.DataFrame(p_sunrise_time)
    p_temp_low = pd.DataFrame(p_temp_low)
    p_temp_high = pd.DataFrame(p_temp_high)
    p_temp_max = pd.DataFrame(p_temp_max)
    p_temp_min = pd.DataFrame(p_temp_min)
    p_visibility = pd.DataFrame(p_visibility)
    p_month = pd.DataFrame(p_month)

    data["p_cloud_cover"] = p_cloud_cover
    data["p_sunset_time"] = p_sunset_time
    data["p_sunrise_time"] = p_sunrise_time
    data["p_temp_low"] = p_temp_low
    data["p_temp_high"] = p_temp_high
    data["p_temp_max"] = p_temp_max
    data["p_temp_min"] = p_temp_min
    data["p_visibility"] = p_visibility
    data["p_month"] = p_month

    data.to_csv("percentages_parameters.csv")

percentages("specificModel_parameters.csv")
