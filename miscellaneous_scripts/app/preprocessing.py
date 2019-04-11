import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

# To copy objects

from copy import deepcopy

# Pickle to save models

import pickle

# Utils

import os

root_dir = './models/preprocessing/'
# Creating a LabelBinarizer directory to store LabelBinarizer objects
lb_dir = root_dir+'LabelBinarizer/'
# Creating a LabelEncoder directory to store LabelEncoder objects
le_dir = root_dir+'LabelEncoder/'
# Creating a PCA directory to store PCA objects
pca_dir = root_dir+'PCA/'
# Creating a Standard Scaler directory to store Standard Scaler objects
sc_dir = root_dir+'StandardScaler/'

def sylco(word) :
		word = word.lower()

		# exception_add are words that need extra syllables
		# exception_del are words that need less syllables

		exception_add = ['serious','crucial']
		exception_del = ['fortunately','unfortunately']

		co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
		co_two = ['coapt','coed','coinci']

		pre_one = ['preach']

		syls = 0 #added syllable number
		disc = 0 #discarded syllable number

		#1) if letters < 3 : return 1
		if len(word) <= 3 :
			syls = 1
			return syls

		#2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
		# if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

		if word[-2:] == "es" or word[-2:] == "ed" :
			doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
			if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
				if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
					pass
				else :
					disc+=1

		#3) discard trailing "e", except where ending is "le"  

		le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

		if word[-1:] == "e" :
			if word[-2:] == "le" and word not in le_except :
				pass

			else :
				disc+=1

		#4) check if consecutive vowels exists, triplets or pairs, count them as one.

		doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
		tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
		disc+=doubleAndtripple + tripple

		#5) count remaining vowels in word.
		numVowels = len(re.findall(r'[eaoui]',word))

		#6) add one if starts with "mc"
		if word[:2] == "mc" :
			syls+=1

		#7) add one if ends with "y" but is not surrouned by vowel
		if word[-1:] == "y" and word[-2] not in "aeoui" :
			syls +=1

		#8) add one if "y" is surrounded by non-vowels and is not in the last word.

		for i,j in enumerate(word) :
			if j == "y" :
				if (i != 0) and (i != len(word)-1) :
					if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
						syls+=1

		#9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

		if word[:3] == "tri" and word[3] in "aeoui" :
			syls+=1

		if word[:2] == "bi" and word[2] in "aeoui" :
			syls+=1

		#10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

		if word[-3:] == "ian" : 
		#and (word[-4:] != "cian" or word[-4:] != "tian") :
			if word[-4:] == "cian" or word[-4:] == "tian" :
				pass
			else :
				syls+=1

		#11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

		if word[:2] == "co" and word[2] in 'eaoui' :

			if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
				syls+=1
			elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
				pass
			else :
				syls+=1

		#12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

		if word[:3] == "pre" and word[3] in 'eaoui' :
			if word[:6] in pre_one :
				pass
			else :
				syls+=1

		#13) check for "-n't" and cross match with dictionary to add syllable.

		negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

		if word[-3:] == "n't" :
			if word in negative :
				syls+=1
			else :
				pass   

		#14) Handling the exceptional words.

		if word in exception_del :
			disc+=1

		if word in exception_add :
			syls+=1     

		# calculate the output
		return numVowels - disc + syls

def calc_ratio_of_syllables(name):

	open_syllable = 0
	# print("First Letter: "+name[-1:].lower()+" Second Letter: "+name[0].lower())
	try:
		if name[-1:].lower() in 'aeiou':
			open_syllable += 1
		if name[0].lower() in 'aeiou':
			open_syllable += 1
		total_syllables = sylco(name)

		return open_syllable/total_syllables
	except Exception as e:
		return 0

def preprocess(X,y, model_type='binary'):
	""" Pre-Processing function """

	le = preprocessing.LabelEncoder()

	if not os.path.exists(lb_dir):
		os.makedirs(lb_dir)

	# Binary encoding if the model is of type binary
	if model_type == 'binary':

		lb = LabelBinarizer().fit(y.ix[:,0].values)
		y.ix[:,0] = lb.transform(y.ix[:,0].values)

		# Storing lbm objects using pickle for reusability
		outfile = open(lb_dir+y.ix[:,0].name, 'wb')
		pickle.dump(lb, outfile)
		outfile.close()
			

	dtypes = X.dtypes

	if not os.path.exists(le_dir):
		os.makedirs(le_dir)

	# Label encoding the categorical values
	for header, typ in dtypes.items():
		if typ == 'object':
			le.fit(X[header].astype(str))
			X[header] = le.transform(X[header].astype(str))

			# Storing different le objects using pickle for reusability
			outfile = open(le_dir+header, 'wb')
			pickle.dump(le, outfile)
			outfile.close()


	if not os.path.exists(pca_dir):
		os.makedirs(pca_dir)

	pca = PCA(n_components=4).fit(X)
	X = pca.transform(X)

	# Storing pca objects using pickle for reusability
	outfile = open(pca_dir+'pca', 'wb')
	pickle.dump(pca, outfile)
	outfile.close()

	X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)


	if not os.path.exists(sc_dir):
		os.makedirs(sc_dir)

	sc = StandardScaler().fit(X_train)
	X_train = sc.transform(X_train)
	X_test = sc.transform(X_test)

	# Storing StandardScaler objects using pickle for reusability
	outfile = open(sc_dir+'sc', 'wb')
	pickle.dump(sc, outfile)
	outfile.close()

	return X_train, X_test, y_train, y_test

def get_features(name, dtypes):
	"""
	Function to convert names to features so it can be predicted again
	"""

	vowels = ['a', 'e', 'i', 'o', 'u']
	sonorants = ['ha', 'mha', 'vha', 'lha', 'rha']
	
	if name[-3:].lower() in sonorants:
		sono = name[-3:]
	elif  name[-2:].lower()  in  sonorants:
		sono = name[-2:]
	else:
		sono = "na"
		
	ratio = calc_ratio_of_syllables(name)
	
	df = pd.DataFrame(columns=dtypes.index)
	
	if name[-1:] in vowels:
		df.loc[0] = [name[-1:], name[-2:], name[-3:], True, name[-1:], sono, ratio]
	else:
		df.loc[0] = [name[-1:], name[-2:], name[-3:], False, 'n', sono, ratio]
	
	encoder = preprocessing.LabelEncoder()
	for header, typ in dtypes.items():
		print(header)
		if typ == 'object':
			infile = open(le_dir+header, 'rb')
			encoder = pickle.load(infile)
			df[header] = encoder.transform(df[header].astype(str))

	infile = open(pca_dir+'pca', 'rb')
	pca = pickle.load(infile)
	df = pca.transform(df)
	infile.close()

	infile = open(sc_dir+'sc', 'rb')
	sc = pickle.load(infile)
	infile.close()

	return sc.transform(df)