from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, add
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.models import Model


#1. load mnist data
def load_data(num_classes):
    (train_image, train_y), (test_image, test_y) = mnist.load_data()

    train_image_num, test_image_num = train_image.shape[0], test_image.shape[0]
    width, height = train_image.shape[1], test_image.shape[2]
    input_shape = width * height

    train_image = train_image.reshape(train_image_num, width * height)
    test_image = test_image.reshape(test_image_num, width * height)

    train_image = train_image.astype(np.float64)
    test_image = test_image.astype(np.float64)

    # convert class vectors to binary class matrices
    train_y = keras.utils.to_categorical(train_y, num_classes)
    test_y = keras.utils.to_categorical(test_y, num_classes)

    return train_image, train_y, test_image, test_y, input_shape


#2. make DL model
def build_model(input_shape, num_classes):
    input_layer = Input(shape=(input_shape,))

    fcnn_layer1 = Dense(256, activation='sigmoid')(input_layer)
    dropout1 = Dropout(0.5)(fcnn_layer1)

    fcnn_layer2 = Dense(128, activation='sigmoid')(dropout1)
    dropout2 = Dropout(0.5)(fcnn_layer2)

    fcnn_layer3 = Dense(64, activation='sigmoid')(dropout2)
    dropout3 = Dropout(0.5)(fcnn_layer3)

    output_layer = Dense(num_classes, activation='softmax')(dropout3)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()  # 모델의 요약을 보여준다.

    return model


def main():
    batch_size = 128  # 배치 사이즈 128로 설정.
    num_classes = 10  # 정답 라벨링 개수 ( 0 ~ 10 )
    epochs = 10  # Epoch 반복 회숫 3회로 설정

    train_image, train_y, test_image, test_y, input_shape = load_data(num_classes)
    model = build_model(input_shape, num_classes)
    model.fit(train_image, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_image, test_y))

    score = model.evaluate(test_image, test_y, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
