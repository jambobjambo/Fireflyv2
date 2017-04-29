import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import os
import random
from collections import Counter
lemmatizer = WordNetLemmatizer()

def dictionary(training_data_location, word_index, name, columntype):
	lexicon = []
	characteristics = []
	with open(training_data_location, 'r', encoding="utf8") as train_file:
		lines = csv.reader(train_file)
		for line in lines:
			#Step 1 Process Query at location 1
			line_clean = line[word_index].split(":")
			for clean_line in line_clean:
				if(columntype == "Text"):
					all_words = word_tokenize(clean_line.lower())
				else:
					characteristics.append(clean_line.lower())

			if(columntype == "Text"):
				lexicon += list(all_words)
	#Step 6 lemmatize array
	dictionary_array = []
	if(columntype == "Text"):
		lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	#Step 7 Count arrays
		word_counts = Counter(lexicon)
	#Step 8 remove repeat
		for word in word_counts:
			dictionary_array.append(word)
	else:
		word_counts = Counter(characteristics)
		for word in word_counts:
			dictionary_array.append(word)
	if '' in dictionary_array:
		dictionary_array.pop(dictionary_array.index(''))
	#Step 7 return arrays
	with open('./Dictionary/' +  name + '.pickle', 'wb') as f:
		pickle.dump(dictionary_array, f, protocol=2)
	return './Dictionary/' +  name + '.pickle'

def gen_intent_vec(line, intents_local):
	intents = pickle.load(open(intents_local, 'rb'))
	intents_amount = np.zeros(len(intents))
	if line.lower() in intents:
		intent_index_value = intents.index(line.lower())
		intents_amount[intent_index_value] += 1
	return intents_amount

def gen_query_vec(line, lexicon_local, sentencelength):
	lexicon = pickle.load(open(lexicon_local, 'rb'))
	current_words = word_tokenize(line.lower())
	current_words = [lemmatizer.lemmatize(i) for i in current_words]
	len_row = len(current_words)
	length_of_row = sentencelength
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

def gen_L_vec(line, word, sentencelength):
	line_split = line.split(" ")
	len_row = len(line_split)
	length_of_row = sentencelength
	vec = np.zeros(length_of_row)
	if len_row < length_of_row:
		left_over = length_of_row - len_row
	if word != '':
		for steps_in_words in range(length_of_row):
			if steps_in_words < len(line_split):
				words = word.split(":")
				for word in words:
					if line_split[steps_in_words].lower() == word.lower():
						vec[steps_in_words] += 1
	return vec
