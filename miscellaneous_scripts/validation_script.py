import pandas as pd
from prettytable import PrettyTable
import re
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

from numpy import argmax

from keras.utils import to_categorical


data = pd.read_csv("data/gender.csv")
surnames = pd.read_csv("data/indian_surname.csv").apply(lambda x: x.astype(str).str.lower().str.strip()).values

t = PrettyTable(["original_name","cleaned_name", "gender"])

count = 0
n_count = 0
it = 0

file = open("saved_models/tokenizer", "rb")
tk = pickle.load(file)

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

final_df = pd.DataFrame.from_records(final_df, columns=["original_name","cleaned_name", "gender"])

	
new = tk.texts_to_sequences(final_df["cleaned_name"].values)
new = pad_sequences(new, maxlen=20, padding='post')
new = np.array(new)

print(t)

print("---")

print(final_df.head())

print("---")

print("original:")
print(data.count())
print("final:")
print("Found: ",count," unknown: ",n_count)

file = open("saved_models/model", "rb")
model = pickle.load(file)
file.close()

predict = argmax(model.predict(new), axis=1)

test_class_list = [1 if x=='male' else 0 for x in final_df["gender"].values]
y_test = to_categorical(test_class_list)

scores = model.evaluate(new, y_test, verbose=0)

final_list = final_df.values

t = PrettyTable(["original_name","cleaned_name", "genderize", "predicted"])
tb = []

file = open("outputs/output.csv", "w+")
file.write("original_name,cleaned_name,genderize,predicted\n")

it = 0
for i in predict:
	ls = final_list[it]
	if i == 0:
	    t.add_row([ls[0], ls[1], ls[2], "female"])
	    tb.append([ls[0], ls[1], ls[2], "female"])
	    file.write("%s,%s,%s,%s\n" %(ls[0], ls[1], ls[2], "female"))
	else:
	    # print("|", ls[0], "|", ls[1], "|", ls[2], "| -> male")
	    t.add_row([ls[0], ls[1], ls[2], "male"])
	    tb.append(([ls[0], ls[1], ls[2], "male"]))
	    file.write("%s,%s,%s,%s\n" %(ls[0], ls[1], ls[2], "male"))
	it += 1

file.close()
print(t)

hit, miss = 0, 0

print(t)

print("Missed options: ")

for i in tb:

	if i[2] == i[3]:
		hit += 1
	else:
		miss += 1
		print("|%s|%s|%s|%s|" %(i[0], i[1], i[2], i[3]))

print("Number of hits: ", hit)
print("Number of miss: ", miss)

# print(scores)
print("Loss: {0:.2f}% Accuracy: {1:.2f}%".format(scores[0]*100, scores[1]*100))
# 