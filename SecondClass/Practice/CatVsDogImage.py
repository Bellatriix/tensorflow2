# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'CatVsDogImage.py'
@Time: '2020/3/26 20:51'
@Desc : '对数据量较小的图片进行增强'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = tfk.Sequential([
        tfk.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(64, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(128, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(128, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Dropout(0.5),        # 随机去掉一些神经元，是提高准确性的方法
        tfk.layers.Flatten(),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tfk.optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_data_gen = tfk.preprocessing.image.ImageDataGenerator(    # 对图片数据进行增强
        rescale=1.0/255.0,       # 像素归一
        rotation_range=40,       # 使图片在该范围内随机旋转
        width_shift_range=0.2,   # 随机移动图片
        height_shift_range=0.2,
        shear_range=0.2,         # 使图片倾斜变形
        zoom_range=0.2,          # 图片缩放
        horizontal_flip=True,    # 水平翻转
        fill_mode='nearest'      # 填充上述操作中导致损失的像素，按最近像素处理
    )

    test_data_gen = tfk.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_data_gen.flow_from_directory(
        './train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    test_generator = test_data_gen.flow_from_directory(
        './validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=40,
        validation_data=test_generator,
        validation_steps=50,
        verbose=2
    )

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()