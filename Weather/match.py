import csv
import pandas as pd

def main(solar_file, weather_file):
    with open(solar_file, "rU") as csvfile1, open(weather_file, "rU") as csvfile2:
        solar = csv.reader(csvfile1)
        weather = csv.reader(csvfile2)

        new_file = open('weather_matched.csv', 'w')

        count = 0

        for row1 in solar:
            file = open(weather_file, "rU")
            weather = csv.reader(file, delimiter=",")

            for row2 in weather:
                if count == 0:
                    firstvalue = True
                    for value in row2:
                        print(value)
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
    main("../scrape/temp.csv", "1078Weather2016.csv")
