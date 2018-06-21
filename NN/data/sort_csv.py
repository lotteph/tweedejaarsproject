import pandas as pd
import numpy as np
import csv

# The output file.
OUTPUT = "1078_2016_S.csv"

# The input file
INPUT = "2016_1078_S.csv"

# The first row in the output file.
COLUMNS = ("Date,Generated,Efficiency,Number_of_panels,Max_power,System_size,"
	+ "Number_of_inverters,Inverter_size,Postcode,Install_date,Tilt\n")

# Opens the input file to read the first column with the dates.
with open(INPUT, "rb") as csvfile:
	df = pd.read_csv(csvfile)
	saved_column = df.Date
	output = []
	for item in saved_column:
		item = int(item)
		output.append(item)

	# output_normalized = []
	mean = np.mean(output)
	output_normalized = output - mean

	# Sorts the dates.
	output_normalized.sort()

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
			if float(row[0]) == float(item):
				firstvalue = True

				for value in row:
					if firstvalue:
						firstvalue = False
						output_file.write(str(value))
					else:
						output_file.write("," + str(value))
				output_file.write("\n")
output_file.close()
