from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, add
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model


#1. load mnist data
def load_data(num_classes):
    img_rows, img_cols = 28, 28  # 이미지 가로 세로 배열.

    # MNIST 데이터 불러오기
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return  x_train, x_test, y_train, y_test, input_shape


#2. make DL model
def build_model(input_shape, num_classes):
    input_layer = Input(shape=(input_shape))

    conv_layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv2D(64, (3, 3), activation='relu')(conv_layer1)
    pooling_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer2)
    dropout_layer = Dropout(0.25)(pooling_layer)

    flatten_layer = Flatten()(dropout_layer)
    dense_laeyr = Dense(128, activation='relu')(flatten_layer)

    output_layer = Dense(num_classes, activation='softmax')(dense_laeyr)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()  # 모델의 요약을 보여준다.

    return model


def create_callbacks():
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

    return [checkpoint_callback]


def main():
    batch_size = 128  # 배치 사이즈 128로 설정.
    num_classes = 10  # 정답 라벨링 개수 ( 0 ~ 10 )
    epochs = 3  # Epoch 반복 회숫 3회로 설정
    callbacks = create_callbacks()

    x_train, x_test, y_train, y_test, input_shape = load_data(num_classes)
    model = build_model(input_shape, num_classes)
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)

    test_x = x_test[0:3]
    print(len(test_x))
    print(test_x)

    model.load_weights("./model-weights.02-0.988400.hdf5")
    prediction = model.predict(x_test)
    print(prediction[0])


if __name__ == '__main__':
    main()
