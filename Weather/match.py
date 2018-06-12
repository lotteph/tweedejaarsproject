import csv

def main(solar_file, weather_file):
    with open(solar_file, 'r') as csvfile1, open(weather_file, 'r') as csvfile2:
        solar_csv = csv.reader(csvfile1, ',')
        weather_csv = csv.reader(csvfile2, ',')

        new_file = open('weather_matched.csv', 'w')

        count = 0
        found = False
        for row1 in solar_csv:
            if count == 0:
                new_file.writer(row1)

            for row2 in weather_csv:
                if row1[0] == row2[0]:
                    new_file.writer(row2)
                    break;

            count += 1

        csvfile1.close()
        csvfile2.close()
        new_file.close()


_main_("../scrape/", "1078Weather2016.csv")
