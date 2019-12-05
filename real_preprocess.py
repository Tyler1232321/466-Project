# preprocess data

import csv
import random

# helper function to seperate data by class
def seperate_data( data ):
    n = len( data )
    seperated_data = {}
    for i in range(n):
        if data[i][0] not in seperated_data:
            seperated_data[data[i][0]] = []

        seperated_data[data[i][0]].append(data[i])
    return seperated_data

if __name__ == '__main__':

	data = []
	with open('ufc_data.csv', 'rt') as csv_file:
		reader = csv.reader( csv_file )

		next(reader)
		# get the data
		for row in reader:
			if row[0] == 'Red':
				row[0] = 1
			else:
				row[0] = 0
			if row[1] == 'TRUE':
				row[1] = 1
			else:
				row[1] = 0
			data.append(row)
	
	seperated_data = seperate_data( data )

	# now we have seperated red from blue, now we'll randomly pick 20% of 
	# each to be test data and 80% of each to be training data
	test_reds = random.sample( range( len( seperated_data[1] ) ), 483 )
	test_blues = random.sample( range( len( seperated_data[0] ) ), 265 )

	test_file = open("true_test_data.csv", mode='w')
	train_file = open("true_train_data.csv", mode='w')
	test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for n in range( len(seperated_data[1] ) ):
		if n in test_reds:
			test_writer.writerow(seperated_data[1][n])
		else:
			train_writer.writerow(seperated_data[1][n])

	for n in range( len( seperated_data[0] ) ):
		if n in test_blues:
			test_writer.writerow(seperated_data[0][n])
		else:
			train_writer.writerow(seperated_data[0][n])








