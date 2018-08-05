from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model


def jeremynet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], conv_layers=1) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    for _ in range(conv_layers):
        model.add(Conv2D(64, (3, 3), dilation_rate=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (1, 1), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    ##### Your code above (Lab 2)

    return model




