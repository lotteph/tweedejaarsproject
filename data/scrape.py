from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import time
import calendar
import pandas as pd

# The first row in the output file.
COLUMNS = ["Date","Generated","Efficiency","Number_of_panels","Max_power","System_size",
"Number_of_inverters","Inverter_size","Postcode","Install_date","Tilt"]

def panel_info(begin_date, end_date, id, sid):
    ''' Function to scrape the information from the solarpanel used
        to generate energy.
    '''
    url = ("https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date
        + "&id=" + id + "&sid=" + sid + "&t=m&gs=0&v=0")
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
            if counter == 9:
                value = display.get("value")
                dt = datetime.strptime(value, "%d/%m/%y")
                unix = datetime(dt.year, dt.month, dt.day, 0, 0)
                unix = time.mktime(unix.timetuple())
                output.append(str(unix))
            else:
                value = display.get("value")
                output.append(value)
        counter += 1
    return output[:-1]

def add_data(data, panel,df):
    ''' Function to write data in a panda. It adds both generated
        energy and the panel data itself.
    '''
    for row in data:
        cells = row.findAll("td")
        counter = 0
        info = []
        for cell in cells:
            # Changes the date to year month day notation.
            if counter == 0:
                text = cell.text
                dt = datetime.strptime(text, "%d/%m/%y")
                unix = datetime(dt.year, dt.month, dt.day, 0, 0)
                unix = time.mktime(unix.timetuple())
                info.append(unix)

            # Strips the energy values from their unit of measurements.
            if 0 < counter < 3:
                text = cell.text
                text = text.replace("kWh/kW", "")
                text = text.replace("kWh", "")
                info.append(text)

            # Strips the panel information from their unit measurements.
            if counter == 3:
                for item in panel:
                    item = item.replace("W", "")
                    item = item.replace(" Degrees", "")
                    info.append(item)
            counter += 1
        df.loc[len(df)] = info
    return df

def retrieve_data(begin_date, end_date, panel, df, second, id, sid):
    ''' Scrapes energy data from pvoutput.org per month given the begin and end
        date. When the month has 31 days the data is located on two pages. In
        that case both pages get scraped.
    '''
    url = ("https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date
        + "&id=" + id + "&sid=" + sid + "&t=m&gs=0&v=0")
    sec_url = ("https://pvoutput.org/list.jsp?p=1&id=" + id + "&sid=" + sid +
    "&gs=0&df=" + begin_date + "&dt=" + end_date +"&v=0&o=date&d=desc")

    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    data = soup.findAll("tr", attrs={"class": ["e2", "o2"]})
    df = add_data(data, panel, df)

    # If there is a second page scrape that page to.
    if second == True:
        sec_page = urlopen(sec_url)
        sec_soup = BeautifulSoup(sec_page, "html.parser")
        data = sec_soup.findAll("tr", attrs={"class": ["e2", "o2"]})
        df = add_data(data, panel, df)
    return df

def main(m, y, end_m, end_y, id, sid):
    ''' Runs all the functions to create a panda with all the
        important data from the solarpanels.
    '''
    if m < 10:
        begin = str(y) + "0" + str(m) + "01"
        end_d = calendar.monthrange(y, m)
        end = str(y) + "0" + str(m) + str(end_d[1])
    else:
        begin = str(y) + str(m) + "01"
        end_d = calendar.monthrange(y, m)
        end = str(y) + str(m) + str(end_d[1])

    panel = panel_info(begin, end, id, sid)
    df = pd.DataFrame(columns = COLUMNS)

    # Retrieves data for each month
    while y != end_y or m != end_m:
        end_d = calendar.monthrange(y, m)
        if m < 10:
            begin = str(y) + "0" + str(m) + "01"
            end = str(y) + "0" + str(m) + str(end_d[1])
        else:
            begin = str(y) + str(m) + "01"
            end = str(y) + str(m) + str(end_d[1])

        if end_d[1] > 30:
            time.sleep(.5)
            # Change to False when there are no second pages.
            df = retrieve_data(begin, end, panel, df, True, id, sid)
        else:
            time.sleep(.5)
            df = retrieve_data(begin, end, panel, df, False, id, sid)

        if m != 12:
            m += 1
        else:
            y += 1
            m = 1
    return df
