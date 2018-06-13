import csv
import calendar
import numpy as np
import pandas as pd

def add_month(weather_file, year):
    months = []

    for i in range(1,13):
        days = calendar.monthrange(year, i)
        for j in range(days[1]):
            months.append(i)

    months = np.array(months)
    months = pd.DataFrame(months)
    weather = pd.read_csv(weather_file)
    del weather["Unnamed: 0"]
    weather["month"] = months
    weather.to_csv(weather_file)


def make_solarcsv(solar_file, postal_code, year):
    new_file_name = str(postal_code) + "_" + str(year) + "_S.csv"
    new_file = open(new_file_name, "w")

    with open(solar_file, "rU") as csvfile1:
        solardata = csv.reader(csvfile1)

        count = 0
        for row in solardata:
            if count == 0:
                firstvalue = True
                for value in row:
                    if firstvalue == True:
                        new_file.write(str(value))
                        firstvalue = False
                    else:
                        new_file.write("," + str(value))
                new_file.write("\n")

            if row[1] != "0000" and count != 0:
                firstvalue = True
                for value in row:
                    if firstvalue == True:
                        new_file.write(str(value))
                        firstvalue = False
                    else:
                        new_file.write("," + str(value))
                new_file.write("\n")

            count += 1

    csvfile1.close()
    new_file.close()
    return(new_file_name)


def make_weathercsv(solarcsv, weather_file, postal_code, year):
    with open(weather_file, "rU") as csvfile2, open(solarcsv, "rU") as csvfile1:
        weather = csv.reader(csvfile2)
        solar = csv.reader(csvfile1)
        new_file = open(str(postal_code) + "_" + str(year) + "_W.csv", "w")

        count = 0

        for row1 in solar:
            file = open(weather_file, "rU")
            weather = csv.reader(file, delimiter=",")

            for row2 in weather:
                if count == 0:
                    firstvalue = True
                    for value in row2:
                        if firstvalue == True:
                            new_file.write(str(value))
                            firstvalue = False
                        else:
                            new_file.write("," + str(value))
                    new_file.write("\n")

                if row1[0] == row2[1] and count != 0:

                    firstvalue = True
                    for value in row2:
                        if firstvalue == True:
                            new_file.write(str(value))
                            firstvalue = False
                        else:
                            new_file.write("," + str(value))
                    new_file.write("\n")

                    break;

                count += 1

        csvfile1.close()
        csvfile2.close()
        new_file.close()

if __name__ == "__main__":
    add_month("2018_7559_W.csv", 2018)
    solarcsv = make_solarcsv("2018_7559_S.csv", "7559", "2018")
    make_weathercsv(solarcsv, "2018_7559_W.csv", "7559", "2018")
