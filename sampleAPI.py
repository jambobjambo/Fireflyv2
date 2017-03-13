import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from model import Model
app = Flask(__name__)

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
				'intent': intent,
				'recipe': recipe,
				'calendar': calendar,
				'category': category,
				'attributes': attributes,
				'places': places,
				'grouping': grouping
				})
		return response
	else:
		return "Error, no input query provided"

if __name__ == "__main__":
	app.run(host='0.0.0.0')
