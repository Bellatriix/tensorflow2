# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'exercise3.py'
@Time: '2020/3/24 15:10'
@Desc : '简单卷积神经网络'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

class myCallBack(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            print('\nReached 99.8% accuracy so cancelling training!')
            self.model.stop_training = True

if __name__ == '__main__':
    callbacks = myCallBack()
    mnist = tfk.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0

    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    model = tfk.Sequential([
        tfk.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Flatten(),
        tfk.layers.Dense(128, activation='relu'),
        tfk.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])