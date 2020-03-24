# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'CNN.py'
@Time: '2020/3/23 19:16'
@Desc : '简单卷积神经网络'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

'''
@Desc : '可视化卷积层和池化层'
@Parameters : none
@Returns : none
@Time : '2020/3/23 19:54'
'''
def visualConvAndPool():
    f, axarr = plt.subplots(3, 4)
    FIRST_IMAGE = 0
    SECOND_IMAGE = 7
    THIRD_IMAGE = 26
    CONVOLUTION_NUMBER = 1

    layer_ouputs = [layer.output for layer in model.layers]
    activation_model = tfk.models.Model(inputs=model.input, outputs=layer_ouputs)

    for x in range(0, 4):
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0, x].grid(False)
        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1, x].grid(False)
        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2, x].grid(False)

    plt.show()

if __name__ == '__main__':
    mnist = tfk.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(60000, 28, 28, 1)
    train_images = train_images / 255.0

    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    model = tfk.Sequential([
        tfk.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(64, (3, 3)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Flatten(),
        tfk.layers.Dense(128, activation='relu'),
        tfk.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_images, train_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)

    print(test_loss)

    #visualConvAndPool()