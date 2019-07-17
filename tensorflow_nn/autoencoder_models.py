from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import numpy as np




def autoencode(input_size, acthidden= 'relu', actoutput = 'sigmoid'):

    input_img = Input(shape=(input_size,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(32, activation=acthidden)(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation=actoutput)(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    return autoencoder

def deep_autoencode(input_size, acthidden= 'tanh', actoutput = 'sigmoid'):

    input_img = Input(shape=(input_size,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(128, activation=acthidden)(input_img)
    encoded = Dense(64, activation=acthidden)(encoded)
    encoded = Dense(32, activation=acthidden)(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(64, activation=acthidden)(encoded)
    decoded = Dense(128, activation=acthidden)(decoded)
    decoded = Dense(784, activation=actoutput)(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    return autoencoder


def conv_autoencode(acthidden= 'tanh', actoutput = 'sigmoid'):

    input_img = Input(shape=(28, 28, 1))#, dtype = 'float64')  # adapt this if using `channels_first` image data format
    # K.cast(input_img, dtype = 'float64')
    x = Conv2D(16, (3, 3), activation=acthidden, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation=acthidden, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation=acthidden, padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation=acthidden, padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation=acthidden, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation=acthidden)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation=actoutput, padding='same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder
