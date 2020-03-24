# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'exercise4.py'
@Time: '2020/3/24 15:31'
@Desc : '三层卷积神经网络'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import os
import zipfile

class myCallBack(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print('\nReached 99.9% accuracy so cancelling training!')
            self.model.stop_training = True

if __name__ == '__main__':
    callbacks = myCallBack()

    model = tfk.Sequential([
        tfk.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(32, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(32, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Flatten(),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tfk.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = tfk.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(    # 设置从此目录加载图片数据
        './datasets',               # 总数据路径，其中子文件夹的名字将作为该文件夹下图片的标签
        target_size=(150, 150),     # 自动处理图片像素为设置值
        batch_size=10,              # 将图片按设置值的批次进行传入处理
        class_mode='binary'
    )

    # 未来的版本中，fit_generator将被弃用，用fit代替，如果x是generator，则不需要设置参数y
    history = model.fit(train_generator,
                        steps_per_epoch=2,    # 设置图片总数分成的小份个数，每次迭代使用一小份
                        epochs=15,            # 迭代总次数
                        verbose=1,            # 设置训练时进度的显示内容和方式
                        callbacks=[callbacks])
