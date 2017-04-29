import numpy as np
from flask import Flask, jsonify, request
from BuildNET import BuildNN
from Trainer import Train
from Sample import sample
import csv
import logging
app = Flask(__name__)

@app.route('/buildnet', methods = ['POST'])
def build_net():
	if request.json['File_Location'] and request.json['File_Location'] and request.json['Column_Type']:
		File_Location = request.json['File_Location']
		Column_Label = request.json['Column_Label']
		Column_Type = request.json['Column_Type']
		Input_Column = request.json['Input_Column']
		try:
			build_net_response = BuildNN.Build(File_Location, Column_Label, Column_Type, Input_Column)
			return jsonify(build_net_response)
		except ValueError:
			logging.warning("Error building training data")
			return jsonify({"error":"Error building training data"})
	else:
		return jsonify({"error":"Please ensure you have entered all inputs and that they are labelled correctly"})

@app.route('/train_net', methods = ['POST'])
def train_net():
	if request.json['Training_Queue'] and request.json['Build']:
		Training_Queue = request.json['Training_Queue']
		Build = request.json['Build']
		Wait = "true"
		if request.json['Wait']:
			Wait = request.json['Wait']
		return jsonify(Train.train_net(Training_Queue, Build, Wait))
	else:
		return jsonify({"error":"Please ensure you have entered all inputs and that they are labelled correctly"})

@app.route('/train_net/<model>', methods = ['POST'])
def train_net_model(model):
	if request.json['Build']:
		Training_Queue = [["Query",model]]
		Build = request.json['Build']
		Wait = "true"
		if request.json['Wait']:
			Wait = request.json['Wait']
		return jsonify(Train.train_net(Training_Queue, Build, Wait))
	else:
		return jsonify({"error":"Please ensure you have entered all inputs and that they are labelled correctly"})

@app.route('/sample', methods = ['POST'])
def sample_net():
	if request.json['Queue'] and request.json['Input']:
		Queue = request.json['Queue']
		Input = request.json['Input']
		return jsonify(sample.Sample_Model(Input, Queue))
	else:
		return jsonify({"error":"Please ensure you have entered all inputs and that they are labelled correctly"})

@app.route('/sample/<model>', methods = ['POST'])
def sample_net_model(model):
	if request.json['Input']:
		Queue = [["Query", model]]
		Input = request.json['Input']
		return jsonify(sample.Sample_Model(Input, Queue))
	else:
		return jsonify({"error":"Please ensure you have entered all inputs and that they are labelled correctly"})
'''
@app.route('/feedback', methods = ['POST'])

@app.route('/models', methods = ['GET'])'''

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000, debug=True)
