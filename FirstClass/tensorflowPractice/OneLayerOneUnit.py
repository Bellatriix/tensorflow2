# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'OneLayerOneUnit.py'
@Time: '2020/3/22 14:43'
@Desc : 'tensorflow2练习，定义单层，且只有单个神经元的神经网络'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

if __name__ == '__main__':
    # 定义单层单个神经元的神经网络
    # Sequential表示按层一次定义
    # Dense表示，定义一层中的神经元，units为神经元个数
    layer_0 = tfk.layers.Dense(units=1, input_shape=[1])
    model = tfk.Sequential([layer_0])
    # 定义损失函数和优化器，此处损失为均方误差，优化器为随机梯度下降（sgd）
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # 定义数据集
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # 训练模型，用fit
    model.fit(xs, ys, epochs=500)

    # 预测，用predict
    print(model.predict([10.0]))

    print("Layer variables look like this: {}".format(layer_0.get_weights()))