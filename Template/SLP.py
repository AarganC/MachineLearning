from keras.layers import Dense, concatenate

def SLP(input_1, input_2, act, n):

    concat = concatenate([input_1, input_2])

    output_1 = Dense(260, activation='softmax', name="output_1")(concat)
    output_2 = Dense(1, activation='sigmoid', name="output_2")(concat)


    return output_1, output_2
