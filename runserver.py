#!flask/bin/python
from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource, reqparse

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pickle
import re
from prettytable import PrettyTable
import sys

global graph,model,tk

graph = tf.get_default_graph()

# Loading tokenizer for preprocessing
file = open("saved_models/tokenizer", "rb")
tk = pickle.load(file)
file.close()

# Loading model for classifying
file = open("saved_models/model", "rb")
model = pickle.load(file)
file.close()

app = Flask(__name__)
api = Api(app)

def preprocess(data):
	"""
	"""

	count = 0
	n_count = 0 

	surnames = pd.read_csv("data/indian_surname.csv").apply(lambda x: x.astype(str).str.lower().str.strip()).values

	it = 0

	t = PrettyTable(["original_name","cleaned_name"])

	final_df = []

	for value in  data:

		name = value.lower().encode("ascii", "ignore").decode("utf-8")
		name = re.sub(r'\b\w{1,2}\b', '', name)
		name = re.sub(r'[_]+', '', name).strip()
		name = re.sub(r'[^a-z]+', ' ', name)
		name_list = re.split(r'\s', name)

		# print(' '.join(name_list), end=" -> ")
		if len(name_list) > 1:
			for word in name_list:
				if not word or (word in surnames):
					name_list.pop(name_list.index(word))
			# print(name_list)
			if len(name_list) > 0 and name_list[0]:

				t.add_row([value, name_list[0]])
				count += 1

				final_df.append([value, name_list[0]])

			else:

				t.add_row([value, "unknown"])
				n_count += 1

				final_df.append([value, "unknown"])

		else:
			if len(name_list) > 0 and name_list[0]:

				t.add_row([value, name_list[0]])
				count += 1

				final_df.append([value, name_list[0]])

			else:

				t.add_row([value, "unknown"])
				n_count += 1

				final_df.append([value, "unknown"])

		it += 1

	print(final_df)
	return final_df

def nametogender(data):
	"""
	"""

	final_df = preprocess(data)

	final_df = pd.DataFrame.from_records(final_df, columns=["original_name","cleaned_name"])

		
	new = tk.texts_to_sequences(final_df["cleaned_name"].values)
	new = pad_sequences(new, maxlen=20, padding='post')
	new = np.array(new)

	with graph.as_default():
		predict = np.argmax(model.predict(new), axis=1)

	tb = []
	it = 0

	for i in predict:
		ls = final_df.values[it]
		if i == 0:
		    tb.append({'name':ls[0], 'gender':"female"})
		else:
		    tb.append({'name':ls[0], 'gender':"male"})
		it += 1

	return tb

# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('name', type=list, action="append")

@app.route('/')
def home():
	return render_template('home.html')

class Item(Resource):
	def get(self):
		# parser.add_argument('class', type=str)
		args = parser.parse_args()
		name_list = list(map(lambda x: ''.join(x), args['name']))

		print(name_list)

		data = nametogender(name_list)

		print(data)

		return jsonify({'classified_values': data})

api.add_resource(Item, '/api/v1.0/classify')

if __name__ == '__main__':

	if len(sys.argv) > 1:
		port = sys.argv[1]
	else:
		port = 5000
	app.run(host= '0.0.0.0', debug=True, port=port)