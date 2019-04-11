import pandas as pd
from prettytable import PrettyTable
import re
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

from numpy import argmax

from keras.utils import to_categorical


data = pd.read_csv("../raw_data/Indian-Male-Names.csv")
surnames = pd.read_csv("../raw_data/indian_surname.csv").apply(lambda x: x.astype(str).str.lower().str.strip()).values

data = data.dropna()

t = PrettyTable(["original_name","cleaned_name", "gender"])

count = 0
n_count = 0
it = 0

final_df = []

for value in  data.values:

	name = value[0].lower().encode("ascii", "ignore").decode("utf-8")
	name = re.sub(r'\b\w{1,2}\b', '', name)
	name = re.sub(r'[_]+', '', name).strip()
	name = re.sub(r'[^a-z]+', ' ', name)
	name = re.sub(r'\b\w{1,2}\b', '', name)
	name = re.sub(r'[^a-z_]+', ' ', name)
	name_list = re.split(r'\s', name)

	# print(' '.join(name_list), end=" -> ")
	if len(name_list) > 1:
		for word in name_list:
			if not word or (word in surnames):
				name_list.pop(name_list.index(word))
		# print(name_list)
		if len(name_list) > 0 and name_list[0]:

			t.add_row([value[0], name_list[0], value[1]])
			count += 1

			final_df.append([value[0], name_list[0], value[1]])

		else:

			t.add_row([value[0], "unknown", value[1]])
			n_count += 1

			final_df.append([value[0], "unknown", value[1]])

	else:
		if len(name_list) > 0 and name_list[0]:

			t.add_row([value[0], name_list[0], value[1]])
			count += 1

			final_df.append([value[0], name_list[0], value[1]])

		else:

			t.add_row([value[0], "unknown", value[1]])
			n_count += 1

			final_df.append([value[0], "unknown", value[1]])

	it += 1

final_df = pd.DataFrame.from_records(final_df, columns=["original_name","cleaned_name", "gender"]).drop_duplicates()

print(t)

print("---")

print(final_df.head())

print("---")

print("original:")
print(data.count())
print("final:")
print("Found: ",count," unknown: ",n_count)