# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'CatVsDog.py'
@Time: '2020/3/26 16:17'
@Desc : '简单卷积神经网络实现猫狗分类'
'''

import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import matplotlib.image as mping

if __name__ == '__main__':
    model = tfk.Sequential([
        tfk.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(32, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(64, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Flatten(),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=tfk.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

    training_dir = './cats-v-dogs/training'
    # 对数据图片进行增强
    training_data_gen = tfk.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    training_data = training_data_gen.flow_from_directory(
        training_dir,
        batch_size=100,
        class_mode='binary',
        target_size=(150, 150)
    )

    test_dir = './cats-v-dogs/testing'
    test_data_gen = tfk.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_data_gen.flow_from_directory(
        test_dir,
        batch_size=100,
        class_mode='binary',
        target_size=(150, 150)
    )

    history = model.fit(
        training_data,
        epochs=40,
        validation_data=test_data,
        verbose=1
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', 'Training Accuracy')
    plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.figure()

    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.title('Training and validation loss')

    plt.show()