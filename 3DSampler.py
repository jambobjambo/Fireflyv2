from multiprocessing import Process
import tensorflow as tf
import numpy as np
import pickle
import os
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#southampton tax for last year
#Create new user
#How was the revenue this month versus last month
#what gross margin did boston do on Saturday
#what gross margin did dundee do on Saturday -- not in the dataset
#Compare gross margin for boston versus london for last week

location = "Attributes"
queryin = "what gross margin did dundee do on Saturday"

pickle_directory = "./DataForML"
train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
train_i, test_i = pickle.load(open(pickle_directory + '/' + location + '/' + '0.pickle', 'rb'))
train_r, test_r = pickle.load(open(pickle_directory + '/Recipe/' + '0.pickle', 'rb'))
lexicon = pickle.load(open('./Dictionary/lexicon.pickle', 'rb'))
# reset everything to rerun in jupyter
tf.reset_default_graph()

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

def training():
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
	prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
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
		saver.restore(sess, "./Checkpoints/" + location + "/model.ckpt")
		#Reshape input
		vec_query = gen_query_vec(queryin, lexicon[0])
		input_array = np.array(vec_query, dtype=float)
		sample_array = np.reshape(input_array, (-1, 1470))
		classification = prediction.eval({x:sample_array, keep_prob: 1.0})
		#Find highest class
		highest = 50 + classification[0][0]
		highestindex = 0
		for i in range(len(classification[0])):
			val_check = 50 + classification[0][i]
			if val_check > highest:
				highest = val_check
				highestindex = i
		query_split = queryin.split(" ")
		#60% threshold
		if highest < 60:
			print("None found in query.")
		else:
			print("Value found: " + query_split[highestindex])

if __name__ == '__main__':
	process = Process(target=training, args=())
	process.start()
