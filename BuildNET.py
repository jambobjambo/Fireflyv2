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
import Encoding
import logging

'''File_Location = "./TrainingData/training_data.csv"
Column_Label = ["Query","Intent","Recipe","Category","Attributes","Calendar","Places"]
Column_Type = ["Text","Class","Class","Class","Location","Class","Location"]
Input_Column = 0'''
class BuildNN():
	def Build(File_Location, Column_Label, Column_Type, Input_Column):
		logging.basicConfig(filename="Firefly.log")

		Default_Sentence_Length = 15
		Dictionary_Location = './Util/store.csv'
		Testing_Amount = 0.25
		Divercity = 5
		desired_lines = 100000

		input_type = []
		location_fortype = []
		with open(Dictionary_Location, 'r', encoding="utf8") as dict_local:
			lines = csv.reader(dict_local)
			for line in lines:
				input_type.append(line[0])
				location_fortype.append(line[1])

		with open('./Util/netstructure.txt') as f:
			lines = [ line.strip( ) for line in list(f) ]
			data_name = []
			data_structure = []
			data_position = []
			for line in lines:
				parts = line.split(":")
				data_name.append(parts[0])
				data_structure.append(parts[1])
				data_position.append(parts[2])

		for inputs_name_types in range(len(input_type)):
			if input_type[inputs_name_types] in data_name:
				print(input_type[inputs_name_types] + ' is in NetStructure')
			else:
				with open('./Util/netstructure.txt', "a") as netstructure:
					netstructure.write(input_type[inputs_name_types] + ':' + Column_Type[inputs_name_types] + ':' + 'NotTrained')
		dicts_store = []
		for charc in range(len(Column_Label)):
			inList = Column_Label[charc] in input_type
			if inList == True:
				dicts_store.append(location_fortype[input_type.index(Column_Label[charc])])
			elif Column_Type[charc] == "Text" or Column_Type[charc] == "Class":
				#Build dictionary
				stored_dict = Encoding.dictionary(File_Location, charc, Column_Label[charc], Column_Type[charc])
				add_to_store = Column_Label[charc] + ',' + stored_dict + '\n'
				with open(Dictionary_Location, "a") as myfile:
					myfile.write(add_to_store)
				dicts_store.append(stored_dict)
			else:
				dicts_store.append("None")

		trainingdata_Store = []
		vector_Store = []
		with open(File_Location, 'r', encoding="utf8") as trainingdata:
			lines = csv.reader(trainingdata)
			elements = len(Column_Label) #number of elements in column
			for element in range(elements):
				trainingdata_Store.append([])
				vector_Store.append([])
			lines = csv.reader(trainingdata)
			for line in lines:
				training_elements = []
				p = 0
				for parts in line:
					if p < elements:
						trainingdata_Store[p].append(parts)
					p += 1
		#Vectorise Training File
		increment_line = 0
		while increment_line < desired_lines:
			print("Line: ", str(increment_line), "Outof: ", str(desired_lines))
			for elements in range(len(trainingdata_Store)):
				folder_for_store = Column_Label[elements]
				if Column_Type[elements] == "Text":
					vector_gen_store = []
					for training_line in trainingdata_Store[Input_Column]:
						vector_gen_store.append(Encoding.gen_query_vec(training_line, dicts_store[elements], Default_Sentence_Length))
						#save to a pickle
					vector_Store[elements].append(list(vector_gen_store))
				elif Column_Type[elements] == "Class":
					vector_gen_store = []
					val = 0
					for training_line in trainingdata_Store[elements]:
						'''if dicts_store[elements] == "./Dictionary/Intent.pickle":
							print(trainingdata_Store[Input_Column][val])'''

						vector_gen_store.append(Encoding.gen_intent_vec(training_line, dicts_store[elements]))
						val += 1
					#save to a pickle
					vector_Store[elements].append(list(vector_gen_store))
				elif Column_Type[elements] == "Location":
					vector_gen_store = []
					line = 0
					for training_line in trainingdata_Store[Input_Column]:
						vector_gen_store.append(Encoding.gen_L_vec(training_line, trainingdata_Store[elements][line], Default_Sentence_Length))
						line += 1
					vector_Store[elements].append(list(vector_gen_store))
			increment_line += len(vector_Store[0][0])
		#Save elements
		for d in range(Divercity):
			random.seed(random.randint(0, 100000))
			testing_size = round(len(vector_Store[0][0])*Testing_Amount)
			for elements in range(len(trainingdata_Store)):
				if elements != Input_Column:
					random.shuffle(vector_Store[elements][0])
					random.shuffle(vector_Store[Input_Column][0])

					data_to_split = vector_Store[elements][0]
					input_data_to_split = vector_Store[Input_Column][0]

					training_data = list(data_to_split)
					testing_data = list(data_to_split[-testing_size:])

					intput_training_data = list(input_data_to_split)
					input_testing_data = list(input_data_to_split[-testing_size:])

					folder_for_store = Column_Label[Input_Column] + '$$$' + Column_Label[elements]
					folder_output_store = Column_Label[elements]
					folder_input_store = Column_Label[Input_Column]
					if not os.path.exists('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_output_store + '/'):
						os.makedirs('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_output_store + '/')
					if not os.path.exists('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_input_store + '/'):
						os.makedirs('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_input_store + '/')

					number_of_files_to_name = os.listdir('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_output_store + '/')

					with open('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_output_store + '/' + str(len(number_of_files_to_name)) + '.pickle', 'wb') as f:
						pickle.dump([training_data, testing_data], f, protocol=2)

					with open('./TrainingData/DataforTraining/' + folder_for_store + '/' + folder_input_store + '/' + str(len(number_of_files_to_name)) + '.pickle', 'wb') as f:
						pickle.dump([intput_training_data, input_testing_data], f, protocol=2)

		return  {"Complete": "Vectorising training data complete"}
