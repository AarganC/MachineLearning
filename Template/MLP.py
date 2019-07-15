from keras.layers import Dense, concatenate

def MLP(input_1, input_2, act, nb_layer, n):
    input_1 = int(input_1)
    input_2 = int(input_2)

    concat = concatenate([input_1, input_2])

    for i in range(nb_layer):
        concat = Dense(n, activation=act)(concat)

    output_2 = Dense(1, activation='sigmoid', name="output_2")(concat)
    output_1 = Dense(260, activation='softmax', name="output_1")(concat)

    return output_1, output_2
