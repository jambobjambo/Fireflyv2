import tensorflow as tf
import numpy as np
import pickle
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

pickle_directory = './Data_Process/DataForML/'

train_q, test_q = pickle.load(open(pickle_directory + '/Queries/' + '0.pickle', 'rb'))
train_i, test_i = pickle.load(open(pickle_directory + '/Intent/' + '0.pickle', 'rb'))
lexicon = pickle.load(open('./Data_Process/dictionary/lexicon.pickle', 'rb'))
intents = pickle.load(open('./Data_Process/dictionary/intents.pickle', 'rb'))


x = tf.placeholder('float')
y = tf.placeholder('float')

n_classes = 4

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

# Add ops to save and restore all the variables.
saver = tf.train.Saver({
	"l1-weight": hidden_1_layer['weight'],
	"l2-weight": hidden_2_layer['weight'],
	"l2-weight": hidden_3_layer['weight'],
	"out-weight": output_layer['weight'],
	"l1-bias": hidden_1_layer['bias'],
	"l2-bias": hidden_2_layer['bias'],
	"l2-bias": hidden_3_layer['bias'],
	"out-bias": output_layer['bias']
})
prediction = neural_network_model(x)
with tf.Session() as sess:
	saver.restore(sess, "./Checkpoints/Intent/model.ckpt")

	sess.run(tf.global_variables_initializer())

	predy = tf.nn.softmax(prediction)
	query_input = input()
	query_input_vec = gen_query_vec(query_input, lexicon[0])

	input_array = np.array(query_input_vec)
	input_array = np.reshape(input_array, (-1, 1470))

	classification = predy.eval({x: input_array})
	for i in range(len(classification[0])):
		if classification[0][i] == 1.:
			print(intents[0][i])
