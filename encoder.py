import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
import random
import pickle
from collections import Counter
import sys
lemmatizer = WordNetLemmatizer()

Desired_Data_Amount = 100000
Pickle_location = './DataForML'
Data_location = './TrainingData/'
if len(sys.argv) > 1:
	Data_location += sys.argv[1]

def dictionary():
	lexicon = []
	intents = []
	recipies = []
	categories = []
	calendar = []
	with open(Data_location, 'r', encoding="utf8") as train_file:
		lines = csv.reader(train_file)
		for line in lines:
			#Step 1 Process Query at location 1
			all_words = word_tokenize(line[0].lower())
			lexicon += list(all_words)
			#Step 2 Process Intents
			intents.append(line[1].lower())
			#Step 3 Process recipies
			recipies_per = line[2].lower().split(":")
			if len(recipies_per) > 1:
				for i in range(len(recipies_per)):
					#print(recipies_per[i])
					recipies.append(recipies_per[i])
			elif len(recipies_per) == 1:
				recipies += recipies_per
			#Step 4 categories
			cats_per = line[3].lower().split(":")
			if len(cats_per) > 1:
				for cats in cats_per:
					categories.append(cats)
			elif len(cats_per) == 1:
				recipies += cats_per
			#Step 5 calendar
			calendar.append(line[5].lower())
	#Step 6 lemmatize array
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	#Step 7 Count arrays
	word_counts = Counter(lexicon)
	intent_counts = Counter(intents)
	recipies_counts = Counter(recipies)
	categorie_counts = Counter(categories)
	calendar_counts = Counter(calendar)
	#Step 8 remove repeat
	word_array = []
	intents_array = []
	recipies_array = []
	categories_array = []
	calendar_array = []
	for word in word_counts:
		word_array.append(word)
	for intent in intent_counts:
		intents_array.append(intent)
	for rec in recipies_counts:
		recipies_array.append(rec)
	for category in categorie_counts:
		categories_array.append(category)
	for cal in calendar_counts:
		calendar_array.append(cal)
	#Step 7 return arrays
	return word_array, intents_array, recipies_array, categories_array, calendar_array

def gen_query_vec(line, lexicon):
	current_words = word_tokenize(line.lower())
	current_words = [lemmatizer.lemmatize(i) for i in current_words]

	len_row = len(current_words)
	length_of_row = 15
	if len_row < length_of_row:
		left_over = length_of_row - len_row

	feature_set = []
	for steps_in_words in range(length_of_row):
		features = np.zeros(len(lexicon))
		if steps_in_words < len(current_words):
			if current_words[steps_in_words].lower() in lexicon:
				index_value = lexicon.index(current_words[steps_in_words].lower())
				features[index_value] += 1
		feature_set.append(features)

	vec = np.reshape(feature_set, (len(feature_set) * len(feature_set[0])))
	return vec

def gen_intent_vec(line, intents):
	intents_amount = np.zeros(len(intents))
	intent_index_value = intents.index(line.lower())
	intents_amount[intent_index_value] += 1
	return intents_amount

def gen_M_vec(line, array):
	vec_amount = np.zeros(len(array))
	l = line.split(":")
	for rec in l:
		if rec != "" and rec.lower() in array:
			rep_index_value = array.index(rec.lower())
			vec_amount[rep_index_value] += 1
	return vec_amount

def gen_L_vec(line, word):
	line_split = line.split(" ")

	len_row = len(line_split)
	length_of_row = 15
	vec = np.zeros(length_of_row)
	if len_row < length_of_row:
		left_over = length_of_row - len_row

	for steps_in_words in range(length_of_row):
		if steps_in_words < len(line_split):
			if line_split[steps_in_words].lower() == word.lower():
				vec[steps_in_words] += 1
	return vec

def gen_L_vec_none():
	length_of_row = 15
	vec = np.zeros(length_of_row)
	return vec

def gen_vecs(lexicon, intents, recipes, categories, calendar, test_size=0.1):
	#Loop till have required lines stored
	Total_lines = Desired_Data_Amount
	file = 0
	while Total_lines > 0:
		#Add all lines into array
		lines_arrays = []
		with open(Data_location, 'r', encoding="utf8") as train_file:
			lines = csv.reader(train_file)
			for line in lines:
				lines_arrays.append(line)
		#Shuffle lines
		random.shuffle(lines_arrays)
		Total_lines -= len(lines_arrays)

		#Create Vecs for lines
		training_data = []
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])
		training_data.append([])

		for lines in lines_arrays:
			query = gen_query_vec(lines[0], lexicon)
			int = gen_intent_vec(lines[1], intents)
			rec = gen_M_vec(lines[2], recipes)
			cat = gen_M_vec(lines[3], categories)
			cal = gen_M_vec(lines[5], calendar)
			attr = gen_L_vec(lines[0], lines[4])
			plc = gen_L_vec(lines[0], lines[6])
			if len(lines) > 7:
				grp = gen_L_vec(lines[0], lines[7])
			else:
				grp = gen_L_vec_none()
			training_data[0].append(query)
			training_data[1].append(int)
			training_data[2].append(rec)
			training_data[3].append(cat)
			training_data[4].append(cal)
			training_data[5].append(attr)
			training_data[6].append(plc)
			training_data[7].append(grp)


		#Save training data batch in pickle
		training_data = np.array(training_data)
		testing_size = 50

		#Training data define
		train_query = list(training_data[0][:-testing_size])
		train_intent = list(training_data[1][:-testing_size])
		train_recipe = list(training_data[2][:-testing_size])
		train_categories = list(training_data[3][:-testing_size])
		train_calendar = list(training_data[4][:-testing_size])
		train_attributes = list(training_data[5][:-testing_size])
		train_places = list(training_data[6][:-testing_size])
		train_groups = list(training_data[7][:-testing_size])
		#Testing data define
		test_query = list(training_data[0][-testing_size:])
		test_intent = list(training_data[1][-testing_size:])
		test_recipe = list(training_data[2][-testing_size:])
		test_categories = list(training_data[3][-testing_size:])
		test_calendar = list(training_data[4][-testing_size:])
		test_attributes = list(training_data[5][-testing_size:])
		test_places = list(training_data[6][-testing_size:])
		test_groups = list(training_data[7][-testing_size:])

		with open( Pickle_location + '/Queries/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_query, test_query], f, protocol=2)
		with open( Pickle_location + '/Intent/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_intent, test_intent], f, protocol=2)
		with open( Pickle_location + '/Recipe/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_recipe, test_recipe], f, protocol=2)
		with open( Pickle_location + '/Category/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_categories, test_categories], f, protocol=2)
		with open( Pickle_location + '/Calendar/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_calendar, test_calendar], f, protocol=2)
		with open( Pickle_location + '/Attributes/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_attributes, test_attributes], f, protocol=2)
		with open( Pickle_location + '/Places/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_places, test_places], f, protocol=2)
		with open( Pickle_location + '/Grouping/' + str(file) + '.pickle', 'wb') as f:
			pickle.dump([train_groups, test_groups], f, protocol=2)
		file += 1



def main():
	#Generate dictionaries
	lexicon, intents, recipes, categories, calendar = dictionary()
	with open( './Dictionary/lexicon.pickle', 'wb') as f:
		pickle.dump([lexicon], f, protocol=2)
	with open( './Dictionary/intents.pickle', 'wb') as f:
		pickle.dump([intents], f, protocol=2)
	with open( './Dictionary/recipes.pickle', 'wb') as f:
		pickle.dump([recipes], f, protocol=2)
	with open( './Dictionary/categories.pickle', 'wb') as f:
		pickle.dump([categories], f, protocol=2)
	with open( './Dictionary/calendar.pickle', 'wb') as f:
		pickle.dump([calendar], f, protocol=2)

	#Generate vectors
	gen_vecs(lexicon, intents, recipes, categories, calendar)

if Data_location == './TrainingData/':
	print("Error, please parse argument: TrainingData file name")
else:
	print("Processing training files")
	main()
