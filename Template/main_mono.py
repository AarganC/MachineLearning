import keras
import os
import numpy as np
import tensorflow as tf
import math
import sys

from keras.layers import Input, concatenate, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from datetime import datetime
from tensorflow.python.keras.utils import multi_gpu_model
from keras.applications import Xception
from SLP import SLP
from MLP import MLP
from LSTM import LSTM



if __name__ == "__main__":

    # Get param
    name_param = sys.argv[1]
    print("name_param = " + name_param)
    name_modele = sys.argv[2]
    print("name_modele = " + name_modele)
    batch_size = sys.argv[3]
    print("batch_size = " + batch_size)
    epochs = sys.argv[4]
    print("epochs = " + epochs)
    activation = sys.argv[5]
    print("activation = " + activation)
    nb_layer = sys.argv[6]
    print("nb_layer = " + nb_layer)
    nb_filtre = sys.argv[7]
    print("nb_filtre = " + nb_filtre)
    final_activation = sys.argv[8]
    print("final_activation = " + final_activation)
    lera = sys.argv[8]
    print("lera = " + lera)

    # GPU config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Loading data in ram
    train_input1 = np.load("../Data/train-input_1.npy")
    train_input2 = np.load("../Data/train-input_2.npy")
    train_output1 = np.load("../Data/train-output_1.npy")
    train_output2 = np.load("../Data/train-output_2.npy")
    validation_input_1 = np.load("../Data/validation-input_1.npy")
    validation_input_2 = np.load("../Data/validation-input_2.npy")
    validation_output_1 = np.load("../Data/validation-output_1.npy")
    validation_output_2 = np.load("../Data/validation-output_2.npy")

    # Set tensor
    input_1 = Input(shape=(884,), name="input_1")
    input_2 = Input(shape=(260,), name="input_2")

    ## Modele
    if name_modele == "LSTM":
        #print(name_modele + " " + name_param)
        main_output, auxiliary_output = LSTM(input_1, input_2, nb_filtre, nb_layer)
    if name_modele == "MLP":

        concat = concatenate([input_1, input_2])

        for i in range(nb_layer):
            concat = Dense(int(nb_filtre), activation=activation)(concat)

        auxiliary_output = Dense(1, activation='sigmoid', name="output_2")(concat)
        main_output = Dense(260, activation='softmax', name="output_1")(concat)
    if name_modele == "SLP":
        concat = concatenate([input_1, input_2])
        x = Dense(int(nb_filtre), activation=activation)(concat)

        main_output = Dense(260, activation='softmax', name="output_1")(x)
        auxiliary_output = Dense(1, activation='sigmoid', name="output_2")(x)
    else:
        exit(1)

    ## Run model
    model = Model(inputs=[input_1, input_2], outputs=[main_output, auxiliary_output])

    opt = str(final_activation+'(float('+lera+'))')

    # Replicates the model on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss={'output_1': 'categorical_crossentropy', 'output_2': 'binary_crossentropy'},
                           loss_weights={'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'], optimizer='opt')

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

    parallel_model.fit([train_input1, train_input2], [train_output1, train_output2], epochs=10, batch_size=4096,
                       callbacks=[callbacks], validation_data=([validation_input_1, validation_input_2],
                                                               [validation_output_1, validation_output_2]))
