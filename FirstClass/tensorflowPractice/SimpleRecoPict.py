# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'SimpleRecoPict.py'
@Time: '2020/3/22 15:17'
@Desc : '使用keras自带的数据集，实现图像识别'
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

'''
@Desc : '定义回调，当损失小于0.4时，在当前迭代完成后，停止继续迭代'
'''
class myCallback(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

if __name__ == '__main__':
    callbacks = myCallback()
    # 加载keras自带数据集，数据集内容为衣服，大小为28*28
    # 加载数据用 load_data
    fashion_mnist = tfk.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 图片像素值大小为0-255，需要归一到0-1范围nei
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 构建三层模型
    model = tfk.Sequential([tfk.layers.Flatten(),
                            tfk.layers.Dense(512, activation=tf.nn.relu),
                            tfk.layers.Dense(10, activation=tf.nn.softmax)])

    # 编译训练
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=50, callbacks=[callbacks])

    # 评估模型
    print(model.evaluate(test_images, test_labels))

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])
