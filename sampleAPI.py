import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from model2 import Model
import csv
app = Flask(__name__)


def get_most_confident_value(list):
	values_index = []
	values_confidence = []
	for values in list:
		values_confidence.append(float(list[values]))
		values_index.append(str(values))
	highest = 0
	highest_index = 0
	for val in range(len(values_confidence)):
		if highest < values_confidence[val]:
			highest = values_confidence[val]
			highest_index = val

	return(values_index[highest_index])


@app.route('/api', methods = ['POST'])
def api():
	if request.json['query']:
		query = request.json['query']
		intent = Model.sample(query, "intent")
		recipe = Model.sample(query, "recipe")
		calendar = Model.sample(query, "calendar")
		category = Model.sample(query, "category")
		attributes = Model.sample(query, "attributes")
		places = Model.sample(query, "places")
		grouping = Model.sample(query, "grouping")

		response = jsonify({'input_query': query,
				'intent': [intent],
				'recipe': [recipe],
				'calendar': [calendar],
				'category': [category],
				'attributes': [attributes],
				'places': [places],
				'grouping': [grouping]
				})

		log_intent = get_most_confident_value(intent)
		log_recipe = get_most_confident_value(recipe)
		log_calendar = get_most_confident_value(calendar)
		log_category = get_most_confident_value(category)

		saves = []
		saveline = "'" + query + "'" + ',' + "'" + log_intent + "'" + ',' + "'" + log_recipe + "'" + ',' "'" + log_calendar + "'" + ',' + "'" + log_category + "'" + '\n'
		saves.append(saveline)
		with open('save_log.csv') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				saves.append(row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + '\n')
			filesave = open('save_log.csv', 'w')
			for save in saves:
				filesave.write(str(save))

		return response
	else:
		return "Error, no input query provided"

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)
