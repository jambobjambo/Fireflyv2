import tensorflow as tf
import numpy as np
import pickle
import os
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

pickle_directory = './Data_Process/DataForML/'

train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
train_i, test_i = pickle.load(open(pickle_directory + '/Intent/' + '0.pickle', 'rb'))
lexicon = pickle.load(open('./Data_Process/dictionary/lexicon.pickle', 'rb'))

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = len(train_i[0])
batch_size = 8000
hm_epochs = 0

x = tf.placeholder('float')
y = tf.placeholder('float')

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


# Nothing changes
def neural_network_model(data):
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

	return output

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

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver(tf.all_variables())
	saver_locale = "./Checkpoints/Intent/model.ckpt"
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess, saver_locale)

		for epoch in range(hm_epochs):
			epoch_loss = 0
			files = os.listdir(pickle_directory + '/Queries') # dir is your directory path
			file_num = 0

			test_queries = []
			test_intents = []

			for file in files:
				file_num += 1
				train_query, test_query = pickle.load(open(pickle_directory + 'Queries/' + file, 'rb'))
				train_intent, test_intent = pickle.load(open(pickle_directory + 'Intent/' + file, 'rb'))

				test_queries.append(test_query)
				test_intents.append(test_intent)
				i=0
				while i < len(train_query):
					start = i
					end = i+batch_size
					batch_x = np.array(train_query[start:end])
					batch_y = np.array(train_intent[start:end])

					_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
					                                              y: batch_y})
					epoch_loss += c
					i+=batch_size
				print("File: " + str(file_num) + " out of " + str(len(files)) + " trained")
			#print(test_queries[0][0])
			testing_x = np.reshape(test_queries, (len(test_queries) * len(test_queries[0]), len(test_queries[0][0])))
			testing_y = np.reshape(test_intents, (len(test_intents) * len(test_intents[0]), len(test_intents[0][0])))

			correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'Loss:',epoch_loss,"Accuracy:", accuracy.eval({x: testing_x, y: testing_y}))
			saver.save(sess, "./Checkpoints/Intent/model.ckpt")

			test_value = random.randint(0, len(testing_x))
			predy = tf.nn.softmax(prediction)
			input_array = np.array(testing_x[test_value])
			input_array = np.reshape(input_array, (-1, 1470))

			classification = predy.eval({x: input_array})
			print("Actual Intent: " + str(testing_y[test_value]))
			print("Prediction: " + str(classification[0]))

		'''query_input = input()
		predy = tf.nn.softmax(prediction)
		query_input_vec = gen_query_vec(query_input, lexicon[0])
		output_array = pickle.load(open('./Data_Process/dictionary/intents.pickle', 'rb'))
		input_array = np.array(query_input_vec)
		input_array = np.reshape(input_array, (-1, 1470))

		classification = predy.eval({x: input_array})
		print(classification[0])
		for i in range(len(classification[0])):
			if classification[0][i] == 1.:
				print(output_array[0][i])'''

train_neural_network(x)
