#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:42:12 2019

@author: amoghg
"""

# For pre processing
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

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import Bidirectional
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import pickle

file = open("logs.txt", "a+")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def data():
    data = pd.read_csv("/home/amoghg/sparkstuff/python-examples/Scripts/new_indian_name_dataset.csv")

    data = data.dropna()
    
    X = data['name'].values
    y = data['gender'].values
    
    train_texts, test_texts, train_classes, test_classes  = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)
    
    train_texts = [s.lower() for s in train_texts]
    
    test_texts = [s.lower() for s in test_texts]
    
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(train_texts)
    
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
    
    return X_train, y_train, X_test, y_test

def create_model(X_train, y_train, X_test, y_test):
    """
    """
    
    # Embedding layer
    embedding_vecor_length = 29
    embedding_layer = Embedding(7166, embedding_vecor_length, input_length=20)
    
    # Using a sequential layer
    model = Sequential()
    model.add(embedding_layer)
    
    # CNN layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation={{choice(['relu', 'sigmoid'])}}))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    
    # First BLSTM layer
    model.add(Bidirectional(LSTM(
            {{choice([75, 100, 125, 150])}}, 
            dropout={{choice([0.1, 0.2, 0.3])}},
            recurrent_dropout={{choice([0.1, 0.2, 0.3])}},
            return_sequences=True)))
    
    #Second BLSTM layer
    model.add(Bidirectional(LSTM(
            {{choice([50, 75, 100])}},
            dropout={{choice([0.1, 0.2, 0.3])}},
            recurrent_dropout={{choice([0.1, 0.2, 0.3])}})))
    # model.add(Dropout(0.2))
    
    # First Dense layer
    model.add(Dense({{choice([64, 128, 256])}}, activation={{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout(0.2))
    
    # Final Dense layer
    model.add(Dense(2, activation='sigmoid'))
    model.compile(
            loss='binary_crossentropy',
            optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
            metrics=['accuracy'])
    
    history = model.fit(
            X_train, y_train, 
            validation_data=(X_test, y_test),
            epochs={{choice([25, 50, 75])}},
            batch_size={{choice([64, 128, 256])}})

    scores = model.evaluate(X_test, y_test, verbose=0)
    
    file.write("-----Accuracy: %.4f%%-----\n" % (scores[1]*100))
    file.write(model.summary())
    file.write("\n")
    
    pickle_file = open("model_with_accuracy_%.2f_pickle" %(scores[1]*100))
    pickle.dump(model, pickle_file)
    pickle_file.close()
    
    model_json = model.to_json()
    with open("model_with_accuracy_%.2f.json" %(scores[1]*100), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_with_accuracy_%.2f.h5" %(scores[1]*100))
    print("Saved model to disk")
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_history_accuracy_%.2f.png' %(scores[1]*100))
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_history_loss_%.2f.png' %(scores[1]*100))
    plt.show()
    
    validation_acc = np.amax(history.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
file.close()