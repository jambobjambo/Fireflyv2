import tensorflow as tf
import numpy as np
import pickle
import csv
import nltk
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

def neural_network_model(data, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer):
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

	return output


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

		n_nodes_hl1 = 1500
		n_nodes_hl2 = 1500
		n_nodes_hl3 = 1500

		hidden_1_layer = {'f_fum':n_nodes_hl1,
		                  'weight':tf.Variable(tf.random_normal([len(train_q[0]), n_nodes_hl1])),
		                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

		hidden_2_layer = {'f_fum':n_nodes_hl2,
		                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
		                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

		hidden_3_layer = {'f_fum':n_nodes_hl3,
		                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
		                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

		output_layer = {'f_fum':None,
		                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
		                'bias':tf.Variable(tf.random_normal([n_classes])),}

		# Add ops to save and restore all the variables.
		output_response = ""

		prediction = neural_network_model(x, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess, saver_locale)

			predy = tf.nn.softmax(prediction)
			query_input = query
			query_input_vec = gen_query_vec(query_input, lexicon[0])
			input_array = np.array(query_input_vec)
			input_array = np.reshape(input_array, (-1, 1470))
			classification = predy.eval({x: input_array})
			for i in range(len(classification[0])):
				if classification[0][i] == 1.:
					if in_query != True:
						output_response = output_array[i]
					else:
						output_response = i
		return output_response
