import tensorflow as tf
import numpy as np
import pickle
import os
import random
import Encoding

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

def classifier_model(numberofclasses, input_size):
	learning_rate = 0.01
	tf.reset_default_graph()

	n_classes = numberofclasses
	input_size = input_size
	x = tf.placeholder(tf.float32, shape=[None, input_size], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")
	W = tf.Variable(tf.zeros([input_size, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	return accuracy, train_op, x, y_, y

def classifier(output_data, input_data_location, output_data_location, numberofclasses, input_size):
	accuracy, train_op, x, y_, y = classifier_model(numberofclasses, input_size)
	training_epochs = 30
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		files = os.listdir(input_data_location)
		test_input_store = []
		test_output_store = []
		for epoch in range(training_epochs):
			for file_stored in files:
				train_input, test_input = pickle.load(open(input_data_location + str(file_stored), 'rb'))
				train_output, test_output = pickle.load(open(output_data_location + str(file_stored), 'rb'))
				shuffle_array = np.array([train_input[0], train_output[0]])
				random.shuffle(shuffle_array)
				test_input_store.append(test_input)
				test_output_store.append(test_output)
				randomindex = random.randint(0, 3000)
				sess.run([train_op], feed_dict={x: train_input, y_: train_output})
			'''if epoch == 0:
				print(test_output_store[0])'''
			if epoch % 1 == 0:
				print("Epoch: ", epoch, "Accuracy: ", accuracy.eval(feed_dict={x: test_input_store[0], y_: test_output_store[0]}))
			if epoch % 5 == 0:
				test_val = random.randint(0, len(test_input_store[0]))
				sample_array = np.reshape(test_input_store[0][0], (-1, len(test_input_store[0][test_val])))
				print(y.eval(feed_dict={x: sample_array}))
				print(test_output_store[0][test_val])
		#save model
		folder_for_store = output_data
		if not os.path.exists('./TrainedModel/' + folder_for_store + '/'):
			os.makedirs('./TrainedModel/' + folder_for_store + '/')
		saver.save(sess, './TrainedModel/' + folder_for_store + '/model.ckpt')

def three_dem_trainer_model(numberofclasses, input_size):
	tf.reset_default_graph()

	learning_rate = 0.01
	n_classes = numberofclasses
	input_query_size = input_size

	x = tf.placeholder(tf.float32, [None, input_query_size])
	W_conv1 = weight_variable([98, 15, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1, 98, 15, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	W_fc1 = weight_variable([32*200, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 32*200])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	W_fc2 = weight_variable([1024, n_classes])
	b_fc2 = bias_variable([n_classes])

	prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	y_ = tf.placeholder(tf.float32, [None, n_classes])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return train_step, accuracy, x, y_, keep_prob, prediction

def three_dem_trainer(output_data, input_data_location, output_data_location, numberofclasses, input_size):
	train_step, accuracy, x, y_, keep_prob, prediction = three_dem_trainer_model(numberofclasses, input_size)

	training_epochs = 100
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		files = os.listdir(input_data_location)
		test_input_store = []
		test_output_store = []
		for epoch in range(training_epochs):
			batch_size = 10
			for file_stored in files:
				train_input, test_input = pickle.load(open(input_data_location + str(file_stored), 'rb'))
				train_output, test_output = pickle.load(open(output_data_location + str(file_stored), 'rb'))
				shuffle_array = np.array([train_input[0], train_output[0]])
				random.shuffle(shuffle_array)
				test_input_store.append(test_input)
				test_output_store.append(test_output)
				randomindex = random.randint(0, 3000)
				i=0
				while i < len(train_input):
					start = i
					end = i+batch_size
					batch_x = np.array(train_input[start:end], dtype=float)
					batch_y = np.array(train_output[start:end], dtype=float)
					sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
					shown = False
					i+=batch_size

			if epoch % 10 == 0:
				print("Epoch: ", epoch, "Accuracy: ", accuracy.eval(session=sess,feed_dict={x: test_input_store[0], y_: test_output_store[0], keep_prob: 1.0}))
		#save model
		folder_for_store = output_data
		if not os.path.exists('./TrainedModel/' + folder_for_store + '/'):
			os.makedirs('./TrainedModel/' + folder_for_store + '/')
		saver.save(sess, './TrainedModel/' + folder_for_store + '/model.ckpt')

def Sampling(Type, Input_Dictionary, Output_Dictionary, Model_local, Input, Sentence_Length):
	#Create code for sampling neural nets
	json_list = {}
	if(Type == "Class"):
		Classes = pickle.load(open(Output_Dictionary, 'rb'))

		#Vector input
		vec_query = Encoding.gen_query_vec(Input, Input_Dictionary, Sentence_Length)
		input_array = np.array(vec_query, dtype=float)
		sample_array = np.reshape(input_array, (-1, len(input_array)))
		accuracy, train_op, x, y_, y = classifier_model(len(Classes), len(input_array))
		with tf.Session() as sess:
			# variables need to be initialized before we can use them
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess, Model_local + "model.ckpt")

			classification = y.eval({x: sample_array})
			for i in range(len(classification[0])):
				json_list[str(Classes[i])] = str(round(classification[0][i],2))
	elif(Type == "Location"):
		vec_query = Encoding.gen_query_vec(Input, Input_Dictionary, Sentence_Length)
		input_array = np.array(vec_query, dtype=float)
		sample_array = np.reshape(input_array, (-1, len(input_array)))
		train_step, accuracy, x, y_, keep_prob, prediction = three_dem_trainer_model(Sentence_Length, len(input_array))

		with tf.Session() as sess:
			# variables need to be initialized before we can use them
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess, Model_local + "model.ckpt")
			classification = prediction.eval({x:sample_array, keep_prob: 1.0})

			for i in range(len(classification[0])):
				json_list[str(i)] = str(round((50 + classification[0][i]) * 0.01,2))

	return json_list
