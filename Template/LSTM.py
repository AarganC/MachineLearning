import keras

from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, concatenate, Dropout, Embedding
from keras.layers import multiply, add

def LSTM (input_1, input_2, nb_filtre, nb_layer):
    nb_filtre = int(nb_filtre)
    nb_filtre_b = nb_filtre*2

    print(nb_filtre)
    print(nb_filtre_b)


    H = Embedding(10000, nb_filtre, input_length=(884,))(input_1)
    for i in range(int(nb_layer)):
        # Layer 1
        C = Embedding(10000, nb_filtre_b, input_length=(884,))(input_1)

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

        nb_filtre_b = nb_filtre_b + nb_filtre

    y = Flatten()(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='output_2')(y)

    test = Embedding(10000, 32, input_length=(260))(input_2)

    x = concatenate([x, test], axis=1)

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    # main_output = Dense(1, activation='sigmoid', name='output_1')(x)
    x = Flatten()(x)
    main_output = Dense(260, activation='softmax', name="output_1")(x)

    return main_output, auxiliary_output