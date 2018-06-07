from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import time
import calendar

# The month and year where the scraping must begin.
BEGIN_MONTH = 1 
BEGIN_YEAR = 2017

# The month and year where the scraping must end.
END_MONTH = 1
END_YEAR = 2018

# The output file.
FILE = "temp.csv"

# The first row in the output file.
COLUMNS = ("Date,Generated,Efficiency,Number_of_panels,Max_power,System_size,"
    + "Number_of_inverters,Inverter_size,Postcode,Install_date,Tilt\n")

def panel_info(begin_date, end_date):
    ''' Function to scrape the information from the solarpanel used 
        to generate energy.
    '''
    url = ("https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date
        + "&id=54314&sid=49389&t=m&v=3")
    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    panel_info = soup.find("a", attrs={"class": "system1"})

    # Open the url with the information about the solarpanel.
    url = panel_info.get("href")
    page = urlopen("https://pvoutput.org/" + url)
    soup = BeautifulSoup(page, "html.parser")
    info = soup.findAll("input", attrs={"class": "display"})

    # Add the needed information to the output list.
    output = []
    counter = 0
    for display in info:
        # Excludes some information from the solarpanel.
        if counter != 3 and counter != 4 and counter != 6 and counter != 10:
            value = display.get("value")
            output.append(value)
        counter += 1
    return output[:-1]

def add_data(data, panel, file):
    ''' Function to write data in a csv file. It adds both generated 
        energy and the panel data itself. 
    '''
    for row in data:
        cells = row.findAll("td")
        counter = 0

        for cell in cells:
            # Changes the date to year month day notation.
            if counter == 0:
                text = cell.text
                dt = datetime.strptime(text, "%d/%m/%y")
                if dt.month < 10 and dt.day < 10:
                    text = (str(dt.year) + "0" + str(dt.month) + "0" 
                    + str(dt.day))
                elif dt.day < 10:
                    text = str(dt.year) + str(dt.month) + "0" + str(dt.day)
                elif dt.month < 10:
                    text = str(dt.year) + "0" + str(dt.month) + str(dt.day)
                else:
                    text = str(dt.year) + str(dt.month) + str(dt.day)
                file.write(text + ",")

            # Strips the energy values from their unit of measurements.
            if 0 < counter < 3:
                text = cell.text
                text = text.replace("kWh/kW", "")
                text = text.replace("kWh", "")
                file.write(text + ",")

            # Strips the panel information from their unit measurements.
            if counter == 3:
                first = True
                for item in panel:
                    item = item.replace("W", "")
                    item = item.replace(" Degrees", "")
                    if first == True:
                        first = False
                        file.write(item)
                    else:
                        file.write("," + item)
                file.write("\n")
            counter += 1

def retrieve_data(begin_date, end_date, panel, file, second):
    ''' Scrapes energy data from pvoutput.org per month given the begin and end
        date. When the month has 31 days the data is located on two pages. These
        both gets scraped in this case. 
    '''
    url = ("https://pvoutput.org/list.jsp?df=" + begin_date + "&dt="
        + end_date + "&id=54314&sid=49389&t=m&gs=0&v=3")
    sec_url = ("https://pvoutput.org/list.jsp?p=1&id=54314&sid=49389&gs=0&df="
        + begin_date + "&dt=" + end_date +"&v=3&o=date&d=desc")

    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    data = soup.findAll("tr", attrs={"class": ["e2", "o2"]})
    add_data(data, panel, file)

    # If there is a second page scrape that site to. 
    if second == True:
        sec_page = urlopen(sec_url)
        sec_soup = BeautifulSoup(sec_page, "html.parser")
        data = sec_soup.findAll("tr", attrs={"class": ["e2", "o2"]})
        add_data(data, panel, file)

def main():
    ''' Runs all the functions to create a csv file with all the 
        important data from the solarpanels. 
    '''
    y = BEGIN_YEAR
    m = BEGIN_MONTH

    if m < 10:
        begin = str(y)+"0"+str(m)+"01"
        end_d = calendar.monthrange(y, m)
        end = str(y)+"0"+str(m)+str(end_d[1])
    else:
        begin = str(y)+str(m)+"01"
        end_d = calendar.monthrange(y, m)
        end = str(y)+str(m)+str(end_d[1])

    panel = panel_info(begin, end)
    csv_file = open(FILE,"a")
    csv_file.write(COLUMNS)

    # Retrieves data for each month 
    while y != END_YEAR or m != END_MONTH:
        end_d = calendar.monthrange(y, m)
        if m < 10:
            begin = str(y)+"0"+str(m)+"01"
            end = str(y)+"0"+str(m)+str(end_d[1])
        else:
            begin = str(y)+str(m)+"01"
            end = str(y)+str(m)+str(end_d[1])

        if end_d[1] > 30:
            time.sleep(1)
            retrieve_data(begin, end, panel, csv_file, True)
        else:
            time.sleep(1)
            retrieve_data(begin, end, panel, csv_file, False)

        if m != 12:
            m += 1
        else:
            y += 1
            m = 1
    csv_file.close()

if __name__ == "__main__":
    main()
