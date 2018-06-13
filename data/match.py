import csv
import pandas as pd

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
    solarcsv = make_solarcsv("test.csv", "test2", "test2")
    make_weathercsv(solarcsv, "2013_7325_W.csv", "test2", "test2")
