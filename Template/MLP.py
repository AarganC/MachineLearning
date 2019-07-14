from keras.layers import Dense, concatenate
from keras.optimizers import SGD

def MLP(input_1, input_2, act, nb_layer, n):
    inputs_1 = input_1
    inputs_2 = input_2
    concat = concatenate([input_1, input_2])
    y = Dense(n, activation=act)(inputs_1)
    x = Dense(n, activation=act)(inputs_2)

    for i in range(nb_layer):
        x = Dense(n, activation=act)(x)
    output_2 = Dense(1, activation='sigmoid', name="output_2")(concat)

    for i in range(nb_layer):
        y = Dense(n, activation=act)(y)
    output_1 = Dense(260, activation='softmax', name="output_1")(concat)

    return output_1, output_2
