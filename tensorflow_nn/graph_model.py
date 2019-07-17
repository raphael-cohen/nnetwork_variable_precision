import pickle

import keras.metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from predict_models import f1

matplotlib.use('TkAgg')


def plot_autoencode():

    floatn = "float16"
    K.set_floatx(floatn)

    with open("float16modelautoencode", 'rb') as file_pi:
        history16 = pickle.load(file_pi)

    floatn = "float32"
    K.set_floatx(floatn)

    with open("float32modelautoencode", 'rb') as file_pi:
        history32 = pickle.load(file_pi)

    floatn = "float64"
    K.set_floatx(floatn)

    with open("float64modelautoencode", 'rb') as file_pi:
        history64 = pickle.load(file_pi)

    plt.plot(history16.history['loss'])
    plt.plot(history32.history['loss'])
    plt.plot(history64.history['loss'], linestyle = '--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['16bits', '32bits', '64bits'], loc='center right')
    plt.show()


def plot_cnn_autoencode():

    floatn = "float16"
    K.set_floatx(floatn)

    with open("float16modelcnn_autoencode", 'rb') as file_pi:
        history16 = pickle.load(file_pi)

    floatn = "float32"
    K.set_floatx(floatn)

    with open("float32modelcnn_autoencode", 'rb') as file_pi:
        history32 = pickle.load(file_pi)

    floatn = "float64"
    K.set_floatx(floatn)

    with open("float64modelcnn_autoencode", 'rb') as file_pi:
        history64 = pickle.load(file_pi)

    plt.plot(history16.history['loss'])
    plt.plot(history32.history['loss'])
    plt.plot(history64.history['loss'], linestyle = '--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['16bits', '32bits', '64bits'], loc='center right')
    plt.show()


def plot_predict():

    floatn = "float16"
    K.set_floatx(floatn)

    with open("float16modelpredict", 'rb') as file_pi:
        history16 = pickle.load(file_pi)

    floatn = "float32"
    K.set_floatx(floatn)

    with open("float32modelpredict", 'rb') as file_pi:
        history32 = pickle.load(file_pi)

    floatn = "float64"
    K.set_floatx(floatn)

    with open("float64modelpredict", 'rb') as file_pi:
        history64 = pickle.load(file_pi)

    plt.plot(history16.history['val_acc'])
    plt.plot(history32.history['val_acc'])
    plt.plot(history64.history['val_acc'], linestyle = '--')
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['16bits', '32bits', '64bits'], loc='center right')
    plt.show()


def plot_cnn_predict():

    floatn = "float16"
    K.set_floatx(floatn)

    with open("float16modelcnn_predict", 'rb') as file_pi:
        history16 = pickle.load(file_pi)

    floatn = "float32"
    K.set_floatx(floatn)

    with open("float32modelcnn_predict", 'rb') as file_pi:
        history32 = pickle.load(file_pi)

    floatn = "float64"
    K.set_floatx(floatn)

    with open("float64modelcnn_predict", 'rb') as file_pi:
        history64 = pickle.load(file_pi)

    plt.plot(history16.history['val_acc'])
    plt.plot(history32.history['val_acc'])
    plt.plot(history64.history['val_acc'], linestyle = '--')
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['16bits', '32bits', '64bits'], loc='center right')
    plt.show()


def plot_cnn_denoise():

    floatn = "float16"
    K.set_floatx(floatn)

    with open("float16modelcnn_denoising", 'rb') as file_pi:
        history16 = pickle.load(file_pi)

    floatn = "float32"
    K.set_floatx(floatn)

    with open("float32modelcnn_denoising", 'rb') as file_pi:
        history32 = pickle.load(file_pi)

    floatn = "float64"
    K.set_floatx(floatn)

    with open("float64modelcnn_denoising", 'rb') as file_pi:
        history64 = pickle.load(file_pi)

    plt.plot(history16.history['loss'])
    plt.plot(history32.history['loss'])
    plt.plot(history64.history['loss'], linestyle = '--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['16bits', '32bits', '64bits'], loc='center right')
    plt.show()


if __name__ == "__main__":

    # plot_cnn_autoencode()
    plot_autoencode()
    # plot_predict()
    # plot_cnn_predict()
    # plot_cnn_denoise()
