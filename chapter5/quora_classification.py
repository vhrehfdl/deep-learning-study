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
from keras.layers import SpatialDropout1D, add, concatenate
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


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding="utf-8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


# Pre-trained embedding match to my dataset.
def text_to_vector(word_index, path):

    # If you change your embedding.pickle file, you must make new embedding.pickle file.
    if os.path.isfile("embedding.pickle"):
        with open("embedding.pickle", 'rb') as rotten_file:
            embedding_matrix = pickle.load(rotten_file)

    else:
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                pass

        with open("embedding.pickle", 'wb') as handle:
            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix


# BI_LSTM
def build_model_lstm(size, embedding_matrix):
    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 512

    words = Input(shape=(size,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# TextCNN
def build_model_cnn(size, embedding_matrix):
    num_filters = 128
    filter_sizes = [3, 4, 5]

    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

    pooled_outputs = []

    for filter_size in filter_sizes:
        x = Conv1D(num_filters, filter_size, activation='relu')(embedding_layer)
        x = MaxPool1D(pool_size=2)(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs, axis=1)
    dense_layer = Flatten()(merged)

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


def create_callbacks(model_dir):
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5",
                                          monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    train_dir = "./data/quora_train_data.csv"
    test_dir = "./data/quora_test_data.csv"

    model_dir = "./model/"
    embedding_dir = "../embedding/glove.840B.300d.txt"

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)
    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)
    embedding_matrix = text_to_vector(tokenizer.word_index, embedding_dir)

    # model = build_model_lstm(train_x.shape[1], embedding_matrix)
    model = build_model_cnn(train_x.shape[1], embedding_matrix)

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=3, batch_size=128, validation_data=(val_x, val_y), callbacks=callbacks)

    evaluate(model, test_x, test_y)


if __name__ == '__main__':
    main()
