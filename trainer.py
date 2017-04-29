import tensorflow as tf
import numpy as np
import pickle
import random
import os
import Model
import logging

'''Training_Queue = [["Query", "Intent"],["Query", "Recipe"],["Query", "Calendar"],["Query", "Category"],["Query", "Places"],["Query", "Attributes"]]
Build = "New"'''
class Train():
	def train_net(Training_Queue, Build, Wait):
		logging.basicConfig(filename="Firefly.log")
		if Wait == "false":
			try:
				return ({"Complete":"Done", "Progress_Key": "11122233"})
			finally:
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

					for train_queue_point in Training_Queue:
						input_data = train_queue_point[0]
						output_data = train_queue_point[1]
						try:
							net_use = data_structure[data_name.index(output_data)]
							if net_use == "Class":
								print("Training ", str(output_data), "model")
								input_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + input_data + '/'
								output_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + output_data + '/'
								try:
									netin_train, netin_test = pickle.load(open(input_training_data + '0.pickle', 'rb'))
								except ValueError:
									logging.warning("No training data available for '" + input_data + "'. In Trainer.py")
								try:
									netout_train, netout_test = pickle.load(open(output_training_data + '0.pickle', 'rb'))
								except ValueError:
									logging.warning("No training data available for '" + output_data + "'. In Trainer.py")
								try:
									Model.classifier(output_data, input_training_data, output_training_data, len(netout_train[0]), len(netin_train[0]))
								except ValueError:
									logging.warning("Error when training '" + output_data + "' model. In Trainer.py")
							elif net_use == "Location":
								print("Training ", str(output_data), "model")
								input_training_data = './TrainingData/DataforTraining/' + input_data + '/'
								output_training_data = './TrainingData/DataforTraining/' + output_data + '/'
								try:
									netin_train, netin_test = pickle.load(open(input_training_data + '0.pickle', 'rb'))
								except ValueError:
									logging.warning("No training data available for '" + input_data + "'. In Trainer.py")
								try:
									netout_train, netout_test = pickle.load(open(output_training_data + '0.pickle', 'rb'))
								except ValueError:
									logging.warning("No training data available for '" + output_data + "'. In Trainer.py")
								try:
									Model.three_dem_trainer(output_data, input_training_data, output_training_data, len(netout_train[0]), len(netin_train[0]))
								except ValueError:
									logging.warning("Error when training '" + output_data + "' model. In Trainer.py")
							else:
								print("not classed")
						except ValueError:
							logging.warning("No training model clarrified for '" + output_data + "'. In Trainer.py")
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

			for train_queue_point in Training_Queue:
				input_data = train_queue_point[0]
				output_data = train_queue_point[1]
				try:
					net_use = data_structure[data_name.index(output_data)]

					if net_use == "Class":
						print("Training ", str(output_data), "model")
						input_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + input_data + '/'
						output_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + output_data + '/'
						try:
							netin_train, netin_test = pickle.load(open(input_training_data + '0.pickle', 'rb'))
						except ValueError:
							logging.warning("No training data available for '" + input_data + "'. In Trainer.py")
						try:
							netout_train, netout_test = pickle.load(open(output_training_data + '0.pickle', 'rb'))
						except ValueError:
							logging.warning("No training data available for '" + output_data + "'. In Trainer.py")
						try:
							Model.classifier(output_data, input_training_data, output_training_data, len(netout_train[0]), len(netin_train[0]))
						except ValueError:
							logging.warning("Error when training '" + output_data + "' model. In Trainer.py")
					elif net_use == "Location":
						print("Training ", str(output_data), "model")
						input_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + input_data + '/'
						output_training_data = './TrainingData/DataforTraining/' + input_data + '$$$' + output_data + '/' + output_data + '/'
						try:
							netin_train, netin_test = pickle.load(open(input_training_data + '0.pickle', 'rb'))
						except ValueError:
							logging.warning("No training data available for '" + input_data + "'. In Trainer.py")
						try:
							netout_train, netout_test = pickle.load(open(output_training_data + '0.pickle', 'rb'))
						except ValueError:
							logging.warning("No training data available for '" + output_data + "'. In Trainer.py")
						try:
							Model.three_dem_trainer(output_data, input_training_data, output_training_data, len(netout_train[0]), len(netin_train[0]))
						except ValueError:
							logging.warning("Error when training '" + output_data + "' model. In Trainer.py")
					else:
						print("not classed")
				except ValueError:
					logging.warning("No training model clarrified for '" + output_data + "'. In Trainer.py")

		if Wait == "true":
			return({"complete": "Net Trained"})
