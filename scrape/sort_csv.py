import pandas as pd
import csv

# The output file.
OUTPUT = "2017_1103_Solarpanel.csv"

# The input file
INPUT = "temp.csv"

# The first row in the output file.
COLUMNS = ("Date,Generated,Efficiency,Number_of_panels,Max_power,System_size,"
	+ "Number_of_inverters,Inverter_size,Postcode,Install_date,Tilt\n")

# Opens the input file to read the first column with the dates.
with open(INPUT, "rb") as csvfile:
	df = pd.read_csv(csvfile)
	saved_column = df.Date 
	output = []
	for item in saved_column:
		output.append(item)
	# Sorts the dates.
	output.sort()

output_file = open(OUTPUT,"a")
output_file.write(COLUMNS)

# Creates a new output file with sorted dates.
for item in output:
	file = open(INPUT, "rU")
	reader = csv.reader(file, delimiter=",")
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
						output_file.write("," + str(value))
				output_file.write("\n")
output_file.close()




