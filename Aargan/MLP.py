#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization, concatenate, multiply, add, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from datetime import datetime

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2 
sess = tf.Session(config=config)

train_input1 = np.load("../Data/train-input_1.npy")
train_input2 = np.load("../Data/train-input_2.npy")
train_output1 = np.load("../Data/train-output_1.npy")
train_output2 = np.load("../Data/train-output_2.npy")
validation_input_1 = np.load("../Data/validation-input_1.npy")
validation_input_2 = np.load("../Data/validation-input_2.npy")
validation_output_1 = np.load("../Data/validation-output_1.npy")
validation_output_2 = np.load("../Data/validation-output_2.npy")

output_1_shape = train_output1.shape
output_2_shape = train_output2.shape

input_1_shape = 884,
input_2_shape = 260,

input_1 = Input(shape=(884,), name="intput_1")
input_2 = Input(shape=(260,), name="intput_2")

n = 260

concat = concatenate([input_1, input_2])

for i in range(nb_layer):
    concat = Dense(n, activation=act)(concat)
output_2 = Dense(1, activation='sigmoid', name="output_2")(concat)

#y=Dense(n, activation=act)(y)
output_1 = Dense(260,activation='softmax', name="output_1")(concat)
sgd = SGD(lr=lera)


model = Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

model.compile(optimizer='adam',
              loss={'output_1': 'categorical_crossentropy', 'output_2': 'binary_crossentropy'},
              loss_weights={'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'])
model.summary()

save_dir = os.path.join(os.getcwd(), 'res_logs')
date = datetime.today()
year = date.strftime("%Y")
month = date.strftime("%m")
day = date.strftime("%d")
hour = date.strftime("%H")
minute = date.strftime("%M")
model_name = "{}{}{}{}{}_{}"     .format(year, month, day, hour, minute, name_param)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
callbacks = TensorBoard(log_dir=filepath)

model.fit([train_input1, train_input2],
          [train_output1, train_output2],
          epochs=30, batch_size=8192, callbacks=[callbacks],
         validation_data=([validation_input_1, validation_input_2], [validation_output_1, validation_output_2]))


