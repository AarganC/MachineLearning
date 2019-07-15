#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization, concatenate, multiply, add, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from datetime import datetime


# ## Paramètrage de l'utilisation GPU
# Les notebooks sont limité à 50% d'utilisation 
c = []
for d in ['/gpu:1', '/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
# In[3]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)


# ## Hyper Paramètre

# In[4]:


batch_size=128
epochs=2
lera=0.001
activation="relu"


# ## Preprocessing

# In[5]:


train_input1 = np.load("../Data/train-input_1.npy")
train_input2 = np.load("../Data/train-input_2.npy")
train_output1 = np.load("../Data/train-output_1.npy")
train_output2 = np.load("../Data/train-output_2.npy")
validation_input_1 = np.load("../Data/validation-input_1.npy")
validation_input_2 = np.load("../Data/validation-input_2.npy")
validation_output_1 = np.load("../Data/validation-output_1.npy")
validation_output_2 = np.load("../Data/validation-output_2.npy")


# In[6]:


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


# In[7]:


#input_1_shape = train_input1.shape
#input_2_shape = train_input2.shape
output_1_shape = train_output1.shape
output_2_shape = train_output2.shape

input_1_shape = 884,
input_2_shape = 260,

input_1 = Input(shape=(884,), name="intput_1")
input_2 = Input(shape=(260,), name="intput_2")
output_1 = Dense(260, activation='softmax', name="output_1")(input_1 )
output_2 = Dense(1, activation='sigmoid', name="output_2")(input_2)


# In[8]:


# This embedding layer will encode the input sequence
C = Embedding(10000,  32 ,  input_length = input_1_shape)(input_1)
H = Embedding(10000,  16 ,  input_length = input_1_shape)(input_1)
#lstm_out = LSTM(32, return_sequences=True, input_shape=(-1, 24000000, 884, 32))(x)


# In[9]:


# Layer 1

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


# In[10]:


y = Flatten()(x)
auxiliary_output = Dense(1, activation='sigmoid', name='output_2')(y)

test = Embedding(10000,  32 ,  input_length = (260))(input_2)

x = concatenate([x, test], axis=1)

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
#main_output = Dense(1, activation='sigmoid', name='output_1')(x)
x = Flatten()(x)
main_output = Dense(260, activation='softmax', name="output_1")(x)


# In[11]:


model = Model(inputs=[input_1, input_2], outputs=[main_output, auxiliary_output])


# In[12]:


model.compile(optimizer='rmsprop',
              loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy'},
              loss_weights={'output_1': 1.0, 'output_2': 0.001}, metrics=['accuracy'])

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


# import subprocess

# bashCommand = "gcloud compute instances stop instance-2 -q --zone europe-west4-a"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()


# In[ ]:




