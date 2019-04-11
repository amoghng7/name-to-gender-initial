import pandas as pd
from prettytable import PrettyTable
import re

# file = open('cleaned_indian_name_dataset.csv','a+')

data = pd.read_csv('/home/amoghg/Downloads/Names2Gender/data/Indian-Male-Names.csv')

data = data.dropna()

for val in data.values:
	raw_name = val[0]
	gender = val[1]

	name_list = raw_name.split(' ')

	print(name_list, gender)


print("Original data shape:\n")
print(data.count())