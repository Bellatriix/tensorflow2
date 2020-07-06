# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'TransferLearning.py'
@Time: '2020/3/27 16:50'
@Desc : '迁移学习'
'''

import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

if __name__ == '__main__':
    local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    # pre_trained_model.summary()

    last_layer = pre_trained_model.get_layer('mixed7')
    last_out = last_layer.output

    x = layers.Flatten()(last_out)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    train_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    test_data_gen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = train_data_gen.flow_from_directory('./train',
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150, 150))
    validation_generator = test_data_gen.flow_from_directory('./validation',
                                                             batch_size=20,
                                                             class_mode='binary',
                                                             target_size=(150, 150))

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=100,
                        epochs=20,
                        validation_steps=50,
                        verbose=2)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
