import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from model import Model
import csv
app = Flask(__name__)

@app.route('/api', methods = ['POST'])
def api():
	print(request)
	if request.json['query']:
		query = request.json['query']
		intent, intent_accuracy = Model.sample(query, "intent")
		recipe, recipe_accuracy = Model.sample(query, "recipe")
		calendar, calendar_accuracy = Model.sample(query, "calendar")
		category, category_accuracy = Model.sample(query, "category")
		attributes, attributes_accuracy = Model.sample(query, "attributes")
		places, places_accuracy = Model.sample(query, "places")
		grouping, grouping_accuracy = Model.sample(query, "grouping")

		response = jsonify({'input_query': query,
				'intent': { 'value': intent, 'confidence': intent_accuracy },
				'recipe': { 'value': recipe, 'confidence': recipe_accuracy },
				'calendar': { 'value': calendar, 'confidence': calendar_accuracy },
				'category': { 'value': category, 'confidence': category_accuracy },
				'attributes': { 'value': attributes, 'confidence': attributes_accuracy },
				'places': { 'value': places, 'confidence': places_accuracy },
				'grouping': { 'value': grouping, 'confidence': grouping_accuracy }
				})
		saves = []
		saveline = "'" + query + "'" + ',' + "'" + intent + "'" + ',' + "'" + recipe + "'" + ',' "'" + calendar + "'" + ',' + "'" + category + "'" + ',' + "'" + attributes + "'" + ',' + "'" + places + "'" + ',' + "'" + grouping + "'" + '\n'
		saves.append(saveline)
		with open('save_log.csv') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				saves.append(row[0] + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',' + row[4] + ',' + row[5] + ',' + row[6] + ',' + row[7] + '\n')
			filesave = open('save_log.csv', 'w')
			for save in saves:
				filesave.write(str(save))

		return response
	else:
		return "Error, no input query provided"

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)
