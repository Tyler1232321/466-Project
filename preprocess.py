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

		# get the data
		for row in reader:
			if row[0] == 'Red':
				row[0] = 1
			else:
				row[0] = 0
			data.append(row)

	# seperate into classes
	seperated_data = seperate_data(data)

	# n is the number of blue (or 0) samples, this number is important 
	# cause we are goiing to randomly resample n samples from the red
	# (or 1) samples, of which there are many more
	n = len( seperated_data[0] )

	# randomly resample n of the red points
	new_red_indices = random.sample( range( len( seperated_data[1] ) ), n )

	chosen_red = []
	for n in new_red_indices:
		chosen_red.append( seperated_data[1][n] )

	print(len(chosen_red))
	
	# now we have chosen red and all the blue, now we'll randomly pick 213 of 
	# each to be test data and 1000 of each to be training data
	test_reds = random.sample( range( len( chosen_red ) ), 213 )
	test_blues = random.sample( range( len( chosen_red ) ), 213 )

	test_file = open("test_data.csv", mode='w')
	train_file = open("train_data.csv", mode='w')
	test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for n in range( len(chosen_red ) ):
		if n in test_reds:
			test_writer.writerow(chosen_red[n])
		else:
			train_writer.writerow(chosen_red[n])
		if n in test_blues:
			test_writer.writerow(seperated_data[0][n])
		else:
			train_writer.writerow(seperated_data[0][n])








