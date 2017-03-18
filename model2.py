import tensorflow as tf
import numpy as np
import pickle
import csv
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

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

def gen_M_vec(line, array):
	vec_amount = np.zeros(len(array))
	l = line.split(":")
	for rec in l:
		if rec != "" and rec.lower() in array:
			rep_index_value = array.index(rec.lower())
			vec_amount[rep_index_value] += 1
	return vec_amount

class Model():
	def sample(query, req):
		tf.reset_default_graph()
		pickle_directory = './Data_Process/DataForML/'
		train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
		train_i, test_i = pickle.load(open(pickle_directory + '/Intent/' + '0.pickle', 'rb'))
		lexicon = pickle.load(open('./Data_Process/dictionary/lexicon.pickle', 'rb'))

		x = tf.placeholder('float')
		y = tf.placeholder('float')

		in_query = False
		output_array = []
		if(req == "intent"):
			output_array = pickle.load(open('./Data_Process/dictionary/intents.pickle', 'rb'))
			saver_locale = "./Checkpoints/Intent/model.ckpt"
		elif(req == "calendar"):
			output_array = pickle.load(open('./Data_Process/dictionary/calendar.pickle', 'rb'))
			saver_locale = "./Checkpoints/Calendar/model.ckpt"
		elif(req == "category"):
			output_array = pickle.load(open('./Data_Process/dictionary/categories.pickle', 'rb'))
			saver_locale = "./Checkpoints/Category/model.ckpt"
		elif(req == "recipe"):
			output_array = pickle.load(open('./Data_Process/dictionary/recipes.pickle', 'rb'))
			saver_locale = "./Checkpoints/Recipe/model.ckpt"
		elif(req == "grouping"):
			in_query = True
			output_array = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
			saver_locale = "./Checkpoints/Grouping/model.ckpt"
		elif(req == "attributes"):
			in_query = True
			output_array = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
			saver_locale = "./Checkpoints/Attributes/model.ckpt"
		elif(req == "places"):
			in_query = True
			output_array = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
			saver_locale = "./Checkpoints/Places/model.ckpt"

		output_array = output_array[0]
		n_classes = len(output_array)
		input_size = len(train_q[0])
		# input images
		# None -> batch size can be any size, 784 -> flattened mnist image
		x = tf.placeholder(tf.float32, shape=[None, input_size], name="x-input")
		# target 10 output classes
		y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")

		# model parameters will change during training so we use tf.Variable
		W = tf.Variable(tf.zeros([input_size, n_classes]))

		# bias
		b = tf.Variable(tf.zeros([n_classes]))

		# implement model
		# y is our prediction
		y = tf.nn.softmax(tf.matmul(x,W) + b)

		output_response = ""
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess, saver_locale)

			query_input = query
			query_input_vec = gen_query_vec(query_input, lexicon[0])
			input_array = np.array(query_input_vec)
			input_array = np.reshape(input_array, (-1, 1470))
			classification = y.eval({x: input_array})

			confidencerate = 0

			classification_array =[]
			for i in range(len(classification[0])):
				classification_array.append(classification[0][i])

			json_list = {}
			if in_query != True:
				for i in range(len(classification_array)):
					value = output_array[i]
					confidence = classification_array[i]
					#json_list += "{ 'value': " + value + ", 'confidence': " +  str(confidence) + "}"
					json_list[str(value)] = str(confidence)

			else:
				for i in range(len(classification_array)):
					value = i
					confidence = classification_array[i]
					#json_list += "{ 'value': " + str(value) + ", 'confidence': " +  str(confidence) + "}"
					json_list[str(value)] = str(confidence)

			return json_list
