# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'exercise2.py'
@Time: '2020/3/24 14:59'
@Desc : '手写识别'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

class myCallBack(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('\nReached 99% accuracy so cancelling training!')
            self.model.stop_training = True

if __name__ == '__main__':
    mnist = tfk.datasets.mnist
    callbacks = myCallBack()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = tfk.Sequential([
        tfk.layers.Flatten(input_shape=(28, 28)),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
