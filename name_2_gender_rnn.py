#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:28:41 2019

@author: amoghg
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.models import model_from_json

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import Bidirectional

import pylab as pl
import matplotlib.pyplot as plt

from keras.callbacks import Callback

from numpy import argmax

import pickle

plt.style.use('seaborn-whitegrid')

# Plots the loss vs epoch graph
def scatter_plot(loss,epochs,accuracies, title="Loss vs epoch", legend_1='loss', legend_2='accuracy', color_1='#76aad3', color_2='#5439a4'):
    
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(epochs, loss)
    plt.plot(epochs, accuracies)
    plt.show()          
    
    for i in range(len(loss)):
        c1 = pl.scatter(epochs[i], loss[i], c=color_1, marker='o')
        c2 = pl.scatter(epochs[i], accuracies[i], c=color_2, marker='o')
    
    pl.legend([c1, c2], [legend_1, legend_2])
    pl.title(title)
    pl.figure(figsize=(40,20))
    pl.show()

# Data preprocessing start
data = pd.read_csv("data/new_indian_name_dataset.csv")
data = data.dropna()

#print(data)

X = data['name'].values
y = data['gender'].values

train_texts, test_texts, train_classes, test_classes  = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)

train_texts = [s.lower() for s in train_texts]

test_texts = [s.lower() for s in test_texts]

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(train_texts)

# Saving the tokenizer function using pickle for later use
file = open("saved_models/tokenizer","wb")
pickle.dump(tk, file)
file.close()

train_sequences = tk.texts_to_sequences(train_texts)
test_texts = tk.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=20, padding='post')
test_data = pad_sequences(test_texts, maxlen=20, padding='post')

X_train = np.array(train_data)
X_test = np.array(test_data)

train_class_list = [1 if x=='M' else 0 for x in train_classes]
test_class_list = [1 if x=='M' else 0 for x in test_classes]

y_train = to_categorical(train_class_list)
y_test = to_categorical(test_class_list)

# Data preprocessing end

embedding_vecor_length = 29
embedding_layer = Embedding(7166, embedding_vecor_length, input_length=20)

model = Sequential()
model.add(embedding_layer)

# CNN Layer
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# First BLSTM layer
model.add(Bidirectional(LSTM(
    150,
    dropout=0.2,
    recurrent_dropout=0.2,
    return_sequences=True)))

# Second BLSTM layer
model.add(Bidirectional(LSTM(
    100,
    dropout=0.2,
    recurrent_dropout=0.2)))

# model.add(Dropout(0.2))
# First Dense layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# Final dense layer
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = []
accuracies = []
losses = []

# Callback function (Used to plot loss vs accuracy curve) 
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        losses.append(loss)
        accuracies.append(acc)
        epochs.append(epoch)
        print('\nTesting loss: {0:.2f}%, acc: {1:.2f}%\n'.format(loss*100, acc*100))

epoch = 45

print("\n---------- Training for %d epochs ----------\n" %(epoch))
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epoch,
    batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)

#scatter_plot(losses,epochs,accuracies)

print("Accuracy: %.2f%%" % (scores[1]*100))
# mlflow.keras.save_model(model, '/home/amoghg/Downloads/Names2Gender/data/')

# Saving model in pickle format
file = open("saved_models/model_%.2f_pkl" %(scores[1]*100),"wb")
pickle.dump(model, file)
file.close()

# Saving the model in JSON format
model_json = model.to_json()
with open("saved_models/model_with_accuracy_%.2f.json" %(scores[1]*100), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("saved_models/model_with_accuracy_%.2f.h5" %(scores[1]*100))
print("Saved model to disk")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import sys

# This is to check manual prediction
while True:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    name = input("Enter name to predict: ")
    
    if name == '0':
        sys.exit()
    
    new = tk.texts_to_sequences([name,])
    new = pad_sequences(new, maxlen=20, padding='post')
    
    new = np.array(new)

    predict = argmax(model.predict(new), axis=1)
    
    if predict[0] == 0:
        print("Female")
    else:
        print("Male")