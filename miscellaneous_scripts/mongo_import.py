from pymongo import MongoClient
import sys, pandas as pd, os
from pprint import pprint

key = ''

if not len(sys.argv) > 1:
	print("Filename not provided!\nCorrect format: python %s <filename>" %(sys.argv[0]))
	sys.exit()
if not os.path.isfile(sys.argv[1]) and str(sys.argv[1])[-4:] != '.csv' :
	print("The file %s does not exist or is not a valid csv file. Please provide a valid file." %(str(sys.argv[1])))
	sys.exit()

filename = str(sys.argv[1])

data = pd.read_csv(filename)

client = MongoClient('mongodb://localhost:27017/Name2Gender')
db = client.Name2Gender
count = 1

for var in data.values:
	name = db.dataset.find_one({'name':var[0].lower()})

	if name:
		print("[%d]\t%s already exists!" %(count, var[0]))
	else:
		print("[%d]\t%s not found, adding to the database..." %(var[0]))
		db_value = {
			'name':var[0],
			'1-gram':var[1],
			'2-gram':var[2],
			'3-gram':var[3],
			'last_letter_vowel': var[4],
			'vowel': var[5],
			'sonorant': var[6],
			'ratio_of_syllable': var[7],
			'gender': var[8],
			'region': 'India', # var[9]
		}
		result = db.dataset.insert_one(db_value)
		print('\tInserted with _id {0}\n'.format(result.inserted_id))
	count += 1