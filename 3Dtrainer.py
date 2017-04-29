import tensorflow as tf
import numpy as np
import pickle
import os
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pickle_directory = "./DataForML"
train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
train_i, test_i = pickle.load(open(pickle_directory + '/Grouping/' + '0.pickle', 'rb'))
train_r, test_r = pickle.load(open(pickle_directory + '/Recipe/' + '0.pickle', 'rb'))
lexicon = pickle.load(open('./Dictionary/lexicon.pickle', 'rb'))
# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 10
learning_rate = 0.01
training_epochs = 5

n_classes = len(train_i[0])
input_query_size = float(len(train_q[0]))
input_recipe = float(len(train_r[0]))

dictionary_length = len(lexicon)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
	#Input placeholder
	x = tf.placeholder(tf.float32, [None, input_query_size])
	#Define weight and bias
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	#reshape x
	x_image = tf.reshape(x, [-1, dictionary_length, 15, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	#Define weight and bias for second layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	#define weight and bias for final layer
	W_fc1 = weight_variable([28 * 28 * 32, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 28*28*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	#work out dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#define output layer weight and bias
	W_fc2 = weight_variable([1024, n_classes])
	b_fc2 = bias_variable([n_classes])
	#work out loss
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	#define output layer
	y_ = tf.placeholder(tf.float32, [None, n_classes])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		# variables need to be initialized before we can use them
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		for epoch in range(training_epochs):
			files = os.listdir(pickle_directory + '/Queries') # dir is your directory path
			file_num = 0

			test_queries = []
			test_intents = []
			test_recipes = []
			for file in files:
				file_num += 1
				train_query, test_query = pickle.load(open('./DataForML/Queries/' + file, 'rb'))
				train_intent, test_intent = pickle.load(open('./DataForML/Grouping/' + file, 'rb'))
				train_recipe, test_recipe = pickle.load(open('./DataForML/Recipe/' + file, 'rb'))

				test_queries.append(test_query)
				test_intents.append(test_intent)
				test_recipes.append(test_recipes)
				i=0
				while i < len(train_query):
					start = i
					end = i+batch_size
					batch_x = np.array(train_query[start:end], dtype=float)
					Zs = np.array(train_recipe[start:end])
					#batch_x = np.concatenate([Xs, Zs], axis=1)
					batch_y = np.array(train_intent[start:end], dtype=float)

					sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
					i+=batch_size
				print("File: " + str(file_num) + " out of " + str(len(files)) + " trained")

				testing_x = np.reshape(test_queries, (len(test_queries) * len(test_queries[0]), len(test_queries[0][0])))
				testing_y = np.reshape(test_intents, (len(test_intents) * len(test_intents[0]), len(test_intents[0][0])))
				print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: testing_x, y_: testing_y, keep_prob: 1.0}))
			if epoch % 2 == 0:
				print("Epoch: ", epoch)
		saver.save(sess, "./Checkpoints/Grouping/model.ckpt")
