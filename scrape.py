#import the library used to query a website
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime

BEGIN_MONTH = 5
BEGIN_YEAR = 2017

END_MONTH = 5
END_YEAR = 2018

FILE = "training.csv"

def retrieve_data(begin_date, end_date):
	URL = "https://pvoutput.org/list.jsp?df=" + begin_date + "&dt=" + end_date + "&id=54314&sid=49389&t=m&v=3"	
	page = urlopen(URL)
	soup = BeautifulSoup(page, "html.parser")
	csv_file = open(FILE,'a')
	panel_info = soup.findAll('tr', attrs={'class': ['e2', 'o2']})
	for row in panel_info:
		cells = row.findAll('td')
		counter = 0
		for cell in cells:
			if counter < 3:
				text = cell.text
				csv_file.write(text + " ")
			if counter == 3:
				csv_file.write("\n")
			counter += 1
	csv_file.close()


def main():
	y = BEGIN_YEAR
	m = BEGIN_MONTH
	while y != END_YEAR or m != END_MONTH:
		if m < 10:
			begin = str(y)+"0"+str(m)+"01"
			end = str(y)+"0"+str(m)+"31"
		else:
			begin = str(y)+str(m)+"01"
			end = str(y)+str(m)+"31"
		retrieve_data(begin, end)
		if m != 12:
			m += 1
		else:
			y += 1
			m = 1


if __name__ == "__main__":
    main()
