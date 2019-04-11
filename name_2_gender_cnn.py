#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:51:21 2019

@author: amoghg
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.models import model_from_json

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.models import Model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

import numpy as np
	
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)    

# import mlflow.keras

data = pd.read_csv("/home/amoghg/Downloads/Names2Gender/data/indian_name_dataset.csv")

X = data['name'].values
y = data['gender'].values

# train_texts, test_texts, train_classes, test_classes  = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)

# train_texts = [s.lower() for s in train_texts]

# test_texts = [s.lower() for s in test_texts]

X = [s.lower() for s in X]

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(X)

X = tk.texts_to_sequences(X)

X = pad_sequences(X, maxlen=20, padding='post')

X = np.array(X)

# train_class_list = [1 if x=='M' else 0 for x in train_classes]

# test_class_list = [1 if x=='M' else 0 for x in test_classes]

# y_train = to_categorical(train_class_list)
# y_test = to_categorical(test_class_list)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

count = 1

# baseline model
def create_baseline():
	# create model
	global count
	model = Sequential()
	model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("count: %d" %(count))
	count += 1
	return model
	
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=7, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))