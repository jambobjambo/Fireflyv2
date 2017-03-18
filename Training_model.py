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

def train_firefly(location):
	train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
	train_i, test_i = pickle.load(open(pickle_directory + location + '/' + '0.pickle', 'rb'))
	# reset everything to rerun in jupyter
	tf.reset_default_graph()

	# config
	batch_size = 100
	learning_rate = 0.01
	training_epochs = 10

	n_classes = len(train_i[0])
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

	# specify cost function
	# this is our cost
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	# Accuracy
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# specify optimizer
	# optimizer is an "operation" which we can execute in a session
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

	with tf.Session() as sess:
		# variables need to be initialized before we can use them
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		for epoch in range(training_epochs):
			files = os.listdir(pickle_directory + '/Queries') # dir is your directory path
			file_num = 0

			test_queries = []
			test_intents = []

			for file in files:
				file_num += 1
				train_query, test_query = pickle.load(open(pickle_directory + 'Queries/' + file, 'rb'))
				train_intent, test_intent = pickle.load(open(pickle_directory + location + '/' + file, 'rb'))

				test_queries.append(test_query)
				test_intents.append(test_intent)
				i=0
				while i < len(train_query):
					start = i
					end = i+batch_size
					batch_x = np.array(train_query[start:end])
					batch_y = np.array(train_intent[start:end])

					sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
					i+=batch_size
				print("File: " + str(file_num) + " out of " + str(len(files)) + " trained")

			if epoch % 2 == 0:
				print("Epoch: ", epoch)

			testing_x = np.reshape(test_queries, (len(test_queries) * len(test_queries[0]), len(test_queries[0][0])))
			testing_y = np.reshape(test_intents, (len(test_intents) * len(test_intents[0]), len(test_intents[0][0])))

			print("Accuracy: ", accuracy.eval(feed_dict={x: testing_x, y_: testing_y}))
		saver.save(sess, "./Checkpoints/" + location + "/model.ckpt")

files = os.listdir(pickle_directory)
files.pop(6)

for file in files:
	train_firefly(file)

print("Training is complete")
