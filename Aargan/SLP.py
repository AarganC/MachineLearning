#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization, concatenate, multiply, add, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from datetime import datetime


# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2 
sess = tf.Session(config=config)


# In[ ]:


batch_size=128
epochs=2
lera=0.001
activation="relu"


# In[ ]:


train_input1 = np.load("../Data/train-input_1.npy")
train_input2 = np.load("../Data/train-input_2.npy")
train_output1 = np.load("../Data/train-output_1.npy")
train_output2 = np.load("../Data/train-output_2.npy")
validation_input_1 = np.load("../Data/validation-input_1.npy")
validation_input_2 = np.load("../Data/validation-input_2.npy")
validation_output_1 = np.load("../Data/validation-output_1.npy")
validation_output_2 = np.load("../Data/validation-output_2.npy")


# In[ ]:


train_input1[0]


# In[ ]:


print("train_input1" + str(train_input1))
print("train_input1 shape" + str(train_input1.shape))
print("train_input2" + str(train_input2))
print("train_input2 shape" + str(train_input2.shape))
print("train_output1" + str(train_output1))
print("train_output1 shape" + str(train_output1.shape))
print("train_output2" + str(train_output2))
print("train_output2 shape" + str(train_output2.shape))
print("validation_input_1" + str(validation_input_1))
print("validation_input_1 shape" + str(validation_input_1.shape))
print("validation_input_2" + str(validation_input_2))
print("validation_input_2 shape" + str(validation_input_2.shape))
print("validation_output_1" + str(validation_output_1))
print("validation_output_1" + str(validation_output_1.shape))
print("validation_output_2" + str(validation_output_2))
print("validation_output_2 shape" + str(validation_output_2.shape))


# In[ ]:


#input_1_shape = train_input1.shape
#input_2_shape = train_input2.shape
output_1_shape = train_output1.shape
output_2_shape = train_output2.shape

input_1_shape = 884,
input_2_shape = 260,

input_1 = Input(shape=(884,), name="intput_1")
input_2 = Input(shape=(260,), name="intput_2")


# In[ ]:


act='selu'
n=260
lera=0.01


# In[ ]:


inputs_1 = input_1
inputs_2 = input_2
concat= concatenate([input_1, input_2])

#x=Dense(n, activation=act)(x)
concat=Dense(n, activation=act)(concat)
output_2=Dense(1, activation='sigmoid', name="output_2")(concat)

#y=Dense(n, activation=act)(y)
output_1=Dense(260,activation='softmax', name="output_1")(concat)
sgd = SGD(lr=lera)


model = Model(inputs=[inputs_1, inputs_2], outputs=[output_1, output_2])

model.compile(optimizer='adam',
              loss={'output_1': 'categorical_crossentropy', 'output_2': 'binary_crossentropy'},
              loss_weights={'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'])
model.summary()


# In[ ]:




# And trained it via:
#model.fit_generator({'main_input': train_input1, 'aux_input': train_input2},
#                      {'main_output': train_output1, 'aux_output': train_output2},
#                      samples_per_epoch=10000, steps_per_epoch=(train_samples/ batch_size),)
                        
#X_out = np.concatenate([train_output1, train_output2])
#r=len(train_output1)+len(train_output2)
#X_out = np.memmap("../Data/train-output_1.npy", shape=(r), mode='r+')
#X_out[len(train_output1):] = train_output2

#r=len(train_input1)+len(train_input2)
#X_in = np.memmap("../Data/train-input_1.npy", shape=(r), mode='r+')
#X_in[len(train_input1):] = train_input2

save_dir = os.path.join(os.getcwd(), 'res_logs')
date = datetime.today()
year = date.strftime("%Y")
month = date.strftime("%m")
day = date.strftime("%d")
hour = date.strftime("%H")
minute = date.strftime("%M")
model_name = "{}{}{}{}{}_test"     .format(year, month, day, hour, minute)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
callbacks = TensorBoard(log_dir=filepath)

#model.fit_generator([train_input1, train_input2], steps_per_epoch=15, epochs=1, verbose=1, 
#              callbacks=None, validation_data=X_out, 
#              validation_steps=None, class_weight=None, max_queue_size=10, workers=3, use_multiprocessing=True, 
#              shuffle=True, initial_epoch=0)

model.fit([train_input1, train_input2],
          [train_output1, train_output2],
          epochs=30, batch_size=4096, callbacks=[callbacks],
         validation_data=([validation_input_1, validation_input_2], [validation_output_1, validation_output_2]))


# In[ ]:




