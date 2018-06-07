import pandas as pd
import csv

with open('training.csv', 'rb') as csvfile:
	df = pd.read_csv(csvfile)
	saved_column = df.Date 
	output = []
	for item in saved_column:
		output.append(item)
	output.sort()

COLUMNS = "Date,Generated,Efficiency,Number_of_panels,Max_power,System_size,Panel_brand,Orientation,Number_of_inverters,Inverter_brand,Inverter_size,Postcode,Install_date,Shading,Tilt\n"

output_file = open('sorted_training.csv','a')
output_file.write(COLUMNS)
for item in output:
	file = open('training.csv', "rU")
	reader = csv.reader(file, delimiter=',')
	firstrow = True
	for row in reader:
		if firstrow:
			firstrow = False
		else:
			if int(row[0]) == item:
				firstvalue = True
				for value in row:
					if firstvalue:
						firstvalue = False
						output_file.write(str(value))
					else:
						output_file.write(","+str(value))
				output_file.write("\n")
output_file.close()




