import Model
import logging
import os
'''Queue = [["Query", "Recipe"], ["Query", "Category"], ["Query", "Places"]]
Input =	"Show profit for Bristol last year"'''

class sample():
	def Sample_Model(Input, Queue):
		logging.basicConfig(filename="Firefly.log")
		Sentence_Length = 15
		with open('./Util/netstructure.txt') as f:
			lines = [ line.strip( ) for line in list(f) ]
			data_name = []
			data_structure = []
			data_position = []
			for line in lines:
				parts = line.split(":")
				data_name.append(parts[0])
				data_structure.append(parts[1])
				data_position.append(parts[2])
			json_response = {}
			for check in Queue:
				input_data = check[0]
				output_data = check[1]
				if output_data in data_name:
					net_use = data_structure[data_name.index(output_data)]

					Input_Dictionary = './Dictionary/' + input_data + '.pickle'
					Output_Dictionary = './Dictionary/' + output_data + '.pickle'
					Model_local = './TrainedModel/' + output_data + '/'
					try:
						response = Model.Sampling(net_use, Input_Dictionary, Output_Dictionary, Model_local, Input, Sentence_Length)
						if not os.path.exists('./Logging/' + output_data + '/'):
							os.makedirs('./Logging/' + output_data + '/')
						highestval = ""
						val_confidence = 0
						for res_values in response:
							if highestval == "":
								highestval = res_values
								val_confidence = response[res_values]
							if response[res_values] > val_confidence:
								highestval = res_values
								val_confidence = response[res_values]

						with open('./Logging/' + output_data + '/live.csv', "a") as mysamplefile:
							mysamplefile.write(Input + ',' + highestval + ',' + val_confidence + '\n')
						json_response[output_data] = response
					except ValueError:
						logging.warning("Error with sampling model: " + output_data)
				else:
					json_response[output_data] = "No model found, name may be incorrect"
			return json_response
