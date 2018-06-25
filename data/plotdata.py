import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data(solar, weather):
    solar = pd.read_csv(solar)
    weather = pd.read_csv(weather)

    generated = solar["Generated"]
    cloudCover = weather["cloudCover"]
    tempLow = weather["temperatureLow"]
    tempHigh = weather["temperatureHigh"]
    visibility = weather["visibility"]
    tempMin = weather["temperatureMin"]
    tempMax = weather["temperatureMax"]

    generated = np.array(generated)[:50]
    cloudCover = np.array(cloudCover)[:50]
    tempLow = np.array(tempLow)[:50]
    tempHigh = np.array(tempHigh)[:50]
    visibility = np.array(visibility)[:50]
    tempMin = np.array(tempMin)[:50]
    tempMax = np.array(tempMax)[:50]


    print(np.correlate(cloudCover, visibility))

    plt.subplot(211)
    # plt.plot(tempLow, label="temperature low", color="green")
    # plt.plot(tempHigh, label="temperature high", color="orange")
    plt.plot(tempMin, label="temperature min", color="purple")
    plt.plot(tempMax, label="temperature max", color="pink")

    plt.legend()


    plt.subplot(212)
    plt.plot(generated, label="generated kWh", color="red")
    plt.plot(cloudCover, label="cloud cover", color="blue")
    plt.plot(visibility, label="visibility", color="yellow")

    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar and weather data")
    plt.title("data")
    plt.show()

plot_data("6591_2017_S.csv", "6591_2017_W.csv")
