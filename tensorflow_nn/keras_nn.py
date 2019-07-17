import matplotlib
import numpy as np
import pylab
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

import predict_models

matplotlib.use('TkAgg')


if __name__ == "__main__":

    floatn = "float16"
    K.set_floatx(floatn)
    print(K.floatx())

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(floatn) / 255.
    x_test = x_test.astype(floatn) / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    #
    # adapt this if using `channels_first` image data format
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    model = predict_models.conv_predict_model()  # 784)

    model.compile(loss='mse', optimizer='nadam', metrics=['accuracy', predict_models.f1])

    model.fit(x_train, to_categorical(y_train), validation_data=(
        x_test, to_categorical(y_test)), epochs=50, batch_size=128,
        callbacks=[TensorBoard(log_dir='C:/Users/cohen/OneDrive')])

    # model.fit(np.reshape(mnist["train_imgs"], (mnist["train_no"],input_size)).astype(np.float16),
    #           labels, epochs = 10, batch_size = 10, callbacks=[TensorBoard(log_dir='/tmp/model')])
