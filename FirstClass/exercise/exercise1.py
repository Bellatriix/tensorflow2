# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'exercise1.py'
@Time: '2020/3/24 14:53'
@Desc : '简单神经网络预测房价'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

if __name__ == '__main__':
    model = tfk.Sequential([tfk.layers.Dense(units=1, input_shape=[1])])

    model.compile(loss='mean_squared_error', optimizer='sgd')

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    model.fit(xs, ys, epochs=1000)

    print(model.predict([7.0]))
