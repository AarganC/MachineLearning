from __future__ import absolute_import, division, print_function, unicode_literals
import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization, concatenate, multiply, add, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from datetime import datetime

import tensorflow as tf

if __name__ == "__main__":
    # Get param
    name_param = "test"
    print("name_param = " + name_param)
    name_modele = "lstm"
    print("name_modele = " + name_modele)
    batch_size = 4096
    print("batch_size = " + str(batch_size))
    epochs = 10
    print("epochs = " + str(epochs))
    activation = "relu"
    print("activation = " + activation)
    nb_layer = 3
    print("nb_layer = " + str(nb_layer))
    nb_filtre = 16
    print("nb_filtre = " + str(nb_filtre))
    final_activation = "adam"
    print("final_activation = " + final_activation)
    lera = 0.001
    print("lera = " + str(lera))

    # Loading data in ram
    train_input1 = np.load("../Data/train-input_1.npy", mmap_mode="r")
    train_input2 = np.load("../Data/train-input_2.npy", mmap_mode="r")
    train_output1 = np.load("../Data/train-output_1.npy", mmap_mode="r")
    train_output2 = np.load("../Data/train-output_2.npy", mmap_mode="r")
    validation_input_1 = np.load("../Data/validation-input_1.npy", mmap_mode="r")
    validation_input_2 = np.load("../Data/validation-input_2.npy", mmap_mode="r")
    validation_output_1 = np.load("../Data/validation-output_1.npy", mmap_mode="r")
    validation_output_2 = np.load("../Data/validation-output_2.npy", mmap_mode="r")

    # Set tensor
    input_1 = Input(shape=(884,), name="input_1")
    input_2 = Input(shape=(260,), name="input_2")


    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        nb_filtre = int(nb_filtre)
        nb_filtre_b = nb_filtre * 2

        x = input_1
        y = input_2

        print(nb_filtre)
        print(nb_filtre_b)

        for i in range(int(nb_layer)):
            input_1_shape = train_input1.shape
            print("output_1_shape[:1] = " + str(input_1_shape[1:]))
            # Layer 1
            if i > 0:
                C = Embedding(10000, nb_filtre_b, input_length=(884, nb_filtre_b, ))(x)
                H = Embedding(10000, nb_filtre, input_length=(884, nb_filtre_b, ))(x)
            else:
                C = Embedding(10000, nb_filtre_b, input_length=(884,))(x)
                H = Embedding(10000, nb_filtre, input_length=(884,))(x)

            Hx = concatenate([H, H])
            F = Activation('sigmoid')(Hx)

            I = Activation('sigmoid')(Hx)
            L = Activation('tanh')(Hx)
            IL = multiply([I, L])

            C = multiply([C, F])
            C = add([C, IL])

            C = Activation('tanh')(C)
            O = Activation('sigmoid')(Hx)
            x = multiply([C, O])

            #nb_filtre_b = nb_filtre_b + nb_filtre

        y = Flatten()(x)
        auxiliary_output = Dense(1, activation='sigmoid', name='output_2')(y)

        test = Embedding(10000, 32, input_length=260)(y)

        x = concatenate([x, test], axis=1)

        # And finally we add the main logistic regression layer
        # main_output = Dense(1, activation='sigmoid', name='output_1')(x)
        x = Flatten()(x)
        main_output = Dense(260, activation='softmax', name="output_1")(x)

        model = Model(inputs=[input_1, input_2], outputs=[main_output, auxiliary_output])
        model.compile(loss={'output_1': 'categorical_crossentropy', 'output_2': 'binary_crossentropy'}, loss_weights={
            'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'], optimizer='opt')

        save_dir = os.path.join(os.getcwd(), 'res_logs')
        date = datetime.today()
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        hour = date.strftime("%H")
        minute = date.strftime("%M")
        model_name = "{}{}{}{}{}_test" \
            .format(year, month, day, hour, minute)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        callbacks = TensorBoard(log_dir=filepath)

        model.fit([train_input1, train_input2],
                  [train_output1, train_output2],
                  epochs=30, batch_size=4096, callbacks=[callbacks],
                  validation_data=(
                  [validation_input_1, validation_input_2], [validation_output_1, validation_output_2]))
