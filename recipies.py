import tensorflow as tf
import numpy as np
import pickle
import os

pickle_directory = './Data_Process/DataForML/'

train_q, train_i, train_r, train_ct, train_cl, test_q, test_i, test_r, test_ct, test_cl = pickle.load(open(pickle_directory + '0.pickle', 'rb'))

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = len(train_r[0])
batch_size = 100
hm_epochs = 2000

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

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			files = os.listdir(pickle_directory) # dir is your directory path
			file_num = 0
			for file in files:
				file_num += 1
				train_query, train_intent, train_recipe, train_categories, train_calendar, test_query, test_intent, test_recipe, test_categories, test_calendar = pickle.load(open(pickle_directory + file, 'rb'))
				i=0
				while i < len(train_query):
					start = i
					end = i+batch_size
					batch_x = np.array(train_query[start:end])
					batch_y = np.array(train_recipe[start:end])

					_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
					                                              y: batch_y})
					epoch_loss += c
					i+=batch_size
				print("File: " + str(file_num) + " out of " + str(len(files)) + " trained")
			correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'Loss:',epoch_loss,"Accuracy:", accuracy.eval({x: test_q, y: test_r}))



train_neural_network(x)