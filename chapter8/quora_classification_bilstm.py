import numpy as np
import os
import pickle

import keras.backend as K
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import ModelCheckpoint
from keras.engine import Layer
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers import SpatialDropout1D, add, concatenate, SimpleRNN, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    return train_x, train_y, test_x, test_y, val_x, val_y


# convert text data to vector.
def data_preprocissing(train_x, test_x, val_x):
    CHARS_TO_REMOVE = r'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(list(train_x) + list(test_x) + list(val_x))  # Make dictionary

    # Text match to dictionary.
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    val_x = tokenizer.texts_to_sequences(val_x)

    temp_list = []
    total_list = list(train_x) + list(test_x) + list(val_x)

    for i in range(0, len(total_list)):
        temp_list.append(len(total_list[i]))

    max_len = max(temp_list)

    train_x = sequence.pad_sequences(train_x, maxlen=max_len)
    test_x = sequence.pad_sequences(test_x, maxlen=max_len)
    val_x = sequence.pad_sequences(val_x, maxlen=max_len)

    return train_x, test_x, val_x, tokenizer


def build_model_rnn(size):
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(10000, 300)(input_layer)
    embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

    bi_lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    dense_layer = Flatten()(bi_lstm_layer)

    result = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def evaluate(model, test_x, test_y):
    prediction = model.predict(test_x)
    y_pred = (prediction > 0.5)

    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(test_y, y_pred, target_names=["0", "1"]))


def main():
    train_dir = "quora_train_data.csv"
    test_dir = "quora_test_data.csv"

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)
    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)

    model = build_model_rnn(train_x.shape[1])
    model.fit(x=train_x, y=train_y, epochs=3, batch_size=128, validation_data=(val_x, val_y))

    evaluate(model, test_x, test_y)


if __name__ == '__main__':
    main()
