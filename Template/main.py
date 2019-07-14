import keras
import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import sys

from keras.layers import Input, Model
from keras.callbacks import TensorBoard
from datetime import datetime
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

    # GPU config
    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    keras.set_session(tf.Session(config=config))

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
    input_1 = Input(shape=(884,), name="intput_1")
    input_2 = Input(shape=(260,), name="intput_2")

    ## Modele
    if name_modele == "LSTM":
        print(name_modele + " " + name_param)
        main_output, auxiliary_output = LSTM(input_1, input_2, nb_filtre, nb_layer, nb_dropout_flag, nb_dropout_value)
    if name_modele == "MLP":
        print(name_modele + " " + name_param)
        main_output, auxiliary_output = MLP(input_1, input_2, activation, nb_layer, nb_filtre)
    if name_modele == "SLP":
        print(name_modele + " " + name_param)
        main_output, auxiliary_output = SLP(input_1, input_2, activation, nb_filtre)
    else:
        exit(1)

    ## Run model
    model = Model(inputs=[input_1, input_2], outputs=[main_output, auxiliary_output])

    model.compile(optimizer='rmsprop',
                  loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy'},
                  loss_weights={'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'])

    cb = []
    modelCheckPointCallBack = keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5')

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
    cb.append(callbacks)
    cb.append(modelCheckPointCallBack)

    nbatches_train, mod = divmod(len(train_input1)+len(train_input2),
                                 batch_size)
    nbatches_valid, mod = divmod(len(validation_input_1)+len(validation_input_2),
                                 batch_size)

    model.fit_generator([train_input1, train_input2], [train_output1, train_output2], epochs=epochs,
                        batch_size=batch_size, callbacks=[callbacks], steps_per_epoch=nbatches_train,
                        validation_steps=3 * nbatches_valid,
                        validation_data=([validation_input_1, validation_input_2], [validation_output_1, validation_output_2]))
