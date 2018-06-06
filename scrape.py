#import the library used to query a website
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import time
import calendar

BEGIN_MONTH = 1
BEGIN_YEAR = 2017

END_MONTH = 1
END_YEAR = 2018


FILE = "training.csv"

COLUMS = "Date,Generated,Efficiency,Number_of_panels,Max_power,System_size,Panel_brand,Orientation,Number_of_inverters,Inverter_brand,Inverter_size,Postcode,Install_date,Shading,Tilt\n"

def panel_info(begin_date, end_date):
	url = "https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date + "&id=54314&sid=49389&t=m&v=3"	
	page = urlopen(url)
	soup = BeautifulSoup(page, "html.parser")
	panel_info = soup.find('a', attrs={'class': 'system1'})
	url = panel_info.get('href')
	page = urlopen('https://pvoutput.org/' + url)
	soup = BeautifulSoup(page, "html.parser")
	info = soup.findAll('input', attrs={'class': 'display'})
	output = []
	for display in info:
		value = display.get('value')
		output.append(value)
	return output[:-1]

def add_data(data, panel, file):
	for row in data:
		cells = row.findAll('td')
		counter = 0
		for cell in cells:
			if counter < 3:
				text = cell.text
				file.write(text + ",")
			if counter == 3:
				first = True
				for item in panel:
					if first == True:
						first = False
						file.write(item)
					else:
						file.write(","+item)
				file.write("\n")
			counter += 1

def retrieve_data(begin_date, end_date, panel, file, second):
	url = "https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date + "&id=54314&sid=49389&t=m&gs=0&v=3"
	second_url = "https://pvoutput.org/list.jsp?p=1&id=54314&sid=49389&gs=0&df=" + begin_date + "&dt=" + end_date +"&v=3&o=date&d=desc"
	page = urlopen(url)
	soup = BeautifulSoup(page, "html.parser")
	data = soup.findAll('tr', attrs={'class': ['e2', 'o2']})
	add_data(data, panel, file)
	if second == True:
		second_page = urlopen(second_url)
		second_soup = BeautifulSoup(second_page, "html.parser")
		data = second_soup.findAll('tr', attrs={'class': ['e2', 'o2']})
		add_data(data, panel, file)

def main():
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
	csv_file = open(FILE,'a')
	csv_file.write(COLUMS)
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
